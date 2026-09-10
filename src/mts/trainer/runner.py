"""Trial execution + artifact writing.

This is the single place that turns "a point in the state space" into "a set of
artifacts on disk". Both backends (`surrogate`, `torch`) return the same
`BackendOutput`, so everything downstream — diagnostics, prompts, board,
knowledge — is backend-agnostic by construction.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mts.trainer.probes import (
    LayerStat,
    StepMetric,
    build_diagnostics,
    incremental_early_stop,
)
from mts.trainer.spec import TrialSpec, format_diff
from mts.trainer.surrogate import BackendOutput

ProgressFn = Callable[[str], None]
# Structured progress callback: invoked on every heartbeat tick with the
# current step / loss / ETA. The dispatcher forwards this to the board so the
# WebUI can show a running trial's progress instead of only its final state.
HeartbeatFn = Callable[[dict[str, Any]], None]

ARTIFACT_NAMES = ("spec.json", "metrics.jsonl", "layers.jsonl", "diagnostics.json", "summary.json", "log.txt")


@dataclass
class TrialResult:
    trial_id: str
    name: str
    parent: str | None
    backend: str
    device: str
    status: str                                   # completed | diverged | timeout | error | interrupted
    steps_completed: int
    duration_sec: float
    spec: dict[str, Any]
    diff: dict[str, dict[str, Any]]
    diff_text: str
    final: dict[str, float | None]
    best: dict[str, float | None]
    verdict: str
    signals: list[str]
    diagnostics: dict[str, Any]
    layer_stats: list[dict[str, Any]]
    artifacts: dict[str, str] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("diagnostics", None)
        payload.pop("layer_stats", None)
        payload["diagnostics_path"] = self.artifacts.get("diagnostics", "diagnostics.json")
        return payload


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def resolve_backend(backend: str) -> str:
    """Resolve `auto` to a backend that can actually *run* here.

    `auto` is a convenience default, so it must always pick something fast and
    available. A torch install alone is not enough: without a usable CUDA device
    the torch backend falls back to CPU training across every core, which turns a
    300-step trial into minutes and makes `mts demo` / `pytest` unrunnable. That
    also contradicts the MVP decision in docs/07 §2.2 (surrogate is what keeps
    the loop self-consistent; real GPU training is out of MVP scope).

    An explicit `backend: torch` is always honoured, CPU or not — someone who
    asks for real training by name should not be silently downgraded.
    """
    if backend != "auto":
        return backend
    try:
        import torch
    except ImportError:
        return "surrogate"
    try:
        if not torch.cuda.is_available():
            return "surrogate"
    except Exception:
        # A broken/mismatched CUDA build can raise rather than return False.
        return "surrogate"
    return "torch"


def run_trial(
    spec: TrialSpec,
    out_dir: str | Path,
    *,
    trial_id: str | None = None,
    parent_spec: TrialSpec | None = None,
    backend: str | None = None,
    device: str | None = None,
    reference_loss: float | None = None,
    progress: ProgressFn | None = None,
    heartbeat: HeartbeatFn | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> TrialResult:
    """Run one experiment and write all artifacts to `out_dir`.

    `reference_loss` is the best loss reached so far in the project. Passing it
    lets the diagnostics say "this run ended far above what we already have"
    (verdict `underfitting`) instead of only describing the run in isolation.

    `heartbeat` is a structured progress callback (step / loss / ETA) invoked
    periodically by the torch backend; the dispatcher forwards it to the board.

    `checkpoint_every` enables periodic checkpointing in the torch backend (see
    `trainer/checkpoint.py`). `resume=True` restarts from the latest checkpoint
    instead of step 0.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_name = resolve_backend(backend or spec.backend)
    device_name = device or spec.device
    trial_id = trial_id or out_dir.name

    started = time.time()
    log_lines: list[str] = []

    def note(message: str) -> None:
        log_lines.append(message)
        if progress:
            progress(message)

    note(f"trial={trial_id} backend={backend_name} device={device_name}")

    if backend_name == "surrogate":
        from mts.trainer import surrogate

        output: BackendOutput = surrogate.run(spec, out_dir=out_dir, progress=note)
    elif backend_name == "torch":
        output = _run_torch(
            spec, device_name, note,
            out_dir=out_dir,
            heartbeat=heartbeat,
            checkpoint_every=checkpoint_every,
            resume=resume,
        )
    else:
        raise ValueError(f"unknown backend: {backend_name}")

    duration = time.time() - started

    # ---- diagnostics ------------------------------------------------------
    layer_stats = _final_layer_stats(output)
    diagnostics = build_diagnostics(
        steps=output.steps,
        layer_stats=layer_stats,
        grad_values=output.grad_values,
        param_values=output.param_values,
        reference_loss=reference_loss,
    )

    status = output.status
    if status == "completed" and (diagnostics.stability.nan_or_inf or diagnostics.stability.diverged):
        status = "diverged"

    final = _final_metrics(output.steps)
    best = _best_metrics(output.steps)

    diff = spec.diff(parent_spec) if parent_spec is not None else {}
    result = TrialResult(
        trial_id=trial_id,
        name=spec.name,
        parent=parent_spec.name if parent_spec is not None else None,
        backend=backend_name,
        device=device_name,
        status=status,
        steps_completed=len(output.steps),
        duration_sec=round(duration, 3),
        spec=spec.to_dict(),
        diff=diff,
        diff_text=format_diff(diff),
        final=final,
        best=best,
        verdict=diagnostics.verdict,
        signals=diagnostics.signals,
        diagnostics=diagnostics.model_dump(mode="json"),
        layer_stats=[s.model_dump(mode="json") for s in layer_stats],
    )
    result.artifacts = write_artifacts(out_dir, result, output, log_lines)
    note(f"verdict={diagnostics.verdict} status={status} steps={len(output.steps)}")
    return result


def _final_layer_stats(output: BackendOutput) -> list[LayerStat]:
    """Use the last probe snapshot — it reflects the state at the end of training."""
    if not output.layer_snapshots:
        return []
    raw = output.layer_snapshots[-1].get("layers", [])
    stats: list[LayerStat] = []
    for item in raw:
        if isinstance(item, LayerStat):
            stats.append(item)
        elif isinstance(item, dict):
            stats.append(LayerStat.model_validate(item))
    return stats


def _final_metrics(steps: list[StepMetric]) -> dict[str, float | None]:
    if not steps:
        return {"train_loss": None, "val_loss": None, "val_acc": None}
    last = steps[-1]
    return {
        "train_loss": _finite(last.train_loss),
        "val_loss": _finite(last.val_loss),
        "val_acc": _finite(last.val_acc),
    }


def _best_metrics(steps: list[StepMetric]) -> dict[str, float | None]:
    vals = [(m.val_loss, m.step) for m in steps if m.val_loss is not None and np.isfinite(m.val_loss)]
    accs = [(m.val_acc, m.step) for m in steps if m.val_acc is not None and np.isfinite(m.val_acc)]
    best_loss = min(vals) if vals else (None, None)
    best_acc = max(accs) if accs else (None, None)
    return {
        "val_loss": best_loss[0],
        "step": best_loss[1],
        "val_acc": best_acc[0],
    }


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_artifacts(
    out_dir: Path,
    result: TrialResult,
    output: BackendOutput,
    log_lines: list[str],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "spec.json").write_text(json.dumps(result.spec, indent=2), encoding="utf-8")

    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
        for metric in output.steps:
            fh.write(json.dumps(metric.model_dump(mode="json"), separators=(",", ":")) + "\n")

    with (out_dir / "layers.jsonl").open("w", encoding="utf-8") as fh:
        for snapshot in output.layer_snapshots:
            layers = snapshot["layers"]
            payload = {
                "step": snapshot["step"],
                "layers": [
                    item.model_dump(mode="json") if isinstance(item, LayerStat) else item
                    for item in layers
                ],
            }
            fh.write(json.dumps(payload, separators=(",", ":")) + "\n")

    (out_dir / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(result.summary_dict(), indent=2), encoding="utf-8"
    )
    (out_dir / "log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {name.split(".")[0] if name != "log.txt" else "log": name for name in ARTIFACT_NAMES}


# ---------------------------------------------------------------------------
# Readers (used by the server and by agents)
# ---------------------------------------------------------------------------


def read_summary(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "summary.json").read_text(encoding="utf-8"))


def read_metrics(out_dir: str | Path, downsample: int | None = None) -> list[dict[str, Any]]:
    path = Path(out_dir) / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if downsample and len(rows) > downsample:
        rows = _downsample(rows, downsample)
    return rows


def read_layers(out_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(out_dir) / "layers.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_diagnostics(out_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(out_dir) / "diagnostics.json").read_text(encoding="utf-8"))


def _downsample(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Keep first, last and best point, then sample evenly in between."""
    if len(rows) <= target:
        return rows
    best_idx = None
    best_val = None
    for i, row in enumerate(rows):
        val = row.get("val_loss")
        if val is not None and (best_val is None or val < best_val):
            best_val, best_idx = val, i
    keep = {0, len(rows) - 1}
    if best_idx is not None:
        keep.add(best_idx)
    idx = sorted(keep | set(np.linspace(0, len(rows) - 1, target).round().astype(int).tolist()))
    return [rows[i] for i in idx]


# ---------------------------------------------------------------------------
# Torch backend (lazy import so the rest of the system works without torch)
# ---------------------------------------------------------------------------


def _run_torch(
    spec: TrialSpec,
    device_name: str,
    note: ProgressFn,
    *,
    out_dir: Path,
    heartbeat: HeartbeatFn | None = None,
    checkpoint_every: int | None = None,
    resume: bool = False,
) -> BackendOutput:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "backend=torch requires PyTorch. Install it with: uv sync --extra train"
        ) from exc

    from mts.trainer.checkpoint import Checkpointer, rng_from_json
    from mts.trainer.data import load_bundle
    from mts.trainer import model as tmodel

    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.manual_seed(spec.seed)

    bundle = load_bundle(spec, seed=spec.seed)
    vocab = max(bundle.vocab_size, spec.data.vocab_size if spec.data.dataset == "text_file" else bundle.vocab_size)

    net = tmodel.Transformer(spec.arch, vocab).to(device)
    optimizer = tmodel.build_optimizer(net, spec.optim)
    note(f"params={tmodel.count_parameters(net):,} device={device}")

    bundle_np = bundle
    rng = np.random.default_rng(spec.seed)
    max_steps = spec.budget.max_steps
    steps: list[StepMetric] = []
    snapshots: list[dict[str, Any]] = []
    grad_samples: list[np.ndarray] = []
    param_samples: list[np.ndarray] = []
    started = time.time()

    # ---- checkpoint / resume ---------------------------------------------
    ckpt: Checkpointer | None = None
    if checkpoint_every and checkpoint_every > 0:
        ckpt = Checkpointer(out_dir, every=checkpoint_every)
    start_step = 0
    if resume and ckpt is not None and ckpt.has_checkpoint():
        meta = ckpt.load(net, optimizer, device_name)
        start_step = int(meta.get("step", 0)) + 1
        rng = rng_from_json(meta.get("rng_state"), seed=spec.seed)
        note(f"resumed from checkpoint step={start_step - 1}")

    # ---- incremental early stopping -------------------------------------
    # Track a rolling window of recent val losses; if they stop improving for
    # `patience` consecutive evals (and we have a baseline), stop early and
    # report `timeout` with an `early_stopped` flag so the board can tell the
    # difference from hitting the wall-clock budget.
    early_stop_patience = getattr(spec.budget, "early_stop_patience", 0)
    best_so_far: float | None = None
    since_best = 0
    early_stopped = False

    # GPU-seconds budget (docs/08 P1-2): the hard ceiling on accelerator time.
    # On CUDA we track device wall-clock per step; on CPU we approximate GPU
    # seconds with wall-clock (single-card training: they're ~equal).
    gpu_budget = getattr(spec.budget, "gpu_seconds_budget", 0)
    gpu_seconds_used = 0.0
    gpu_budget_reached = False

    # Incremental structural early-stop (docs/08 P1-3): persist counter across
    # probe ticks so a single transient spike doesn't kill a healthy run.
    _persist_counter = 0
    _structural_stop: str | None = None

    def batch_tensor(split: str):
        x, y = bundle_np.get_batch(split, spec.data.batch_size, spec.data.seq_len, rng)
        return (
            torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device),
        )

    net.train()
    tokens_seen = 0
    for step in range(start_step, max_steps):
        lr = tmodel.lr_at(spec.optim, step, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = batch_tensor("train")
        t0 = time.time()
        _, loss = net(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), spec.optim.grad_clip or 1e9)
        grad_norm_value = float(grad_norm.item()) if torch.isfinite(grad_norm) else float("inf")
        clipped = bool(
            spec.optim.grad_clip is not None and grad_norm_value > spec.optim.grad_clip + 1e-9
        )
        optimizer.step()
        step_ms = (time.time() - t0) * 1000
        tokens_seen += int(x.numel())

        # Accumulate GPU-seconds (approximated by step wall-clock on both CUDA
        # and CPU — single-card training means they track 1:1).
        gpu_seconds_used += step_ms / 1000.0

        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            steps.append(StepMetric(step=step, train_loss=float("inf"), val_loss=float("inf"),
                                    val_acc=0.0, lr=lr, grad_norm=float("inf"), clipped=True,
                                    step_ms=step_ms, tokens=x.numel()))
            note(f"step {step}: loss diverged")
            return BackendOutput(status="diverged", steps=steps, layer_snapshots=snapshots,
                                 grad_values=_cat(grad_samples), param_values=_cat(param_samples))

        val_loss = None
        val_acc = None
        if step % spec.budget.eval_every == 0 or step == max_steps - 1:
            net.eval()
            with torch.no_grad():
                losses, correct, total = [], 0, 0
                for _ in range(spec.budget.eval_batches):
                    xv, yv = batch_tensor("val")
                    logits, vloss = net(xv, yv)
                    losses.append(float(vloss.item()))
                    pred = logits.argmax(dim=-1)
                    correct += int((pred == yv).sum().item())
                    total += int(yv.numel())
            net.train()
            val_loss = float(np.mean(losses))
            val_acc = correct / max(1, total)

            # Early stopping bookkeeping.
            if val_loss is not None and np.isfinite(val_loss):
                if best_so_far is None or val_loss < best_so_far:
                    best_so_far = val_loss
                    since_best = 0
                else:
                    since_best += 1

        steps.append(StepMetric(step=step, train_loss=loss_value, val_loss=val_loss,
                                val_acc=val_acc, lr=lr, grad_norm=grad_norm_value,
                                clipped=clipped, step_ms=step_ms, tokens=int(x.numel())))

        if step % spec.budget.probe_every == 0 or step == max_steps - 1:
            net.collect = True
            with torch.no_grad():
                xv, _ = batch_tensor("val")
                net(xv)
            net.collect = False
            probes = net.probe_blocks()
            params, grads = net.param_snapshot()
            grad_samples.append(np.asarray(grads, dtype=np.float64))
            param_samples.append(np.asarray(params, dtype=np.float64))
            layer_stats_here = _torch_layer_stats(net, probes, spec)
            snapshots.append({"step": step, "layers": layer_stats_here})

            # Incremental structural early-stop: explode / dead units / attn
            # collapse persisting across probe ticks cut the run short.
            should_stop, reason = incremental_early_stop(
                layer_stats_here, grad_norm_value, persist_counter=_persist_counter
            )
            if should_stop:
                _persist_counter += 1
                if _persist_counter >= 3:
                    _structural_stop = reason
                    note(f"step {step}: early stopped ({reason} persisted)")
                    return BackendOutput(
                        status="timeout", steps=steps, layer_snapshots=snapshots,
                        grad_values=_cat(grad_samples), param_values=_cat(param_samples),
                        extras={"early_stopped": True, "early_stop_reason": reason},
                    )
            else:
                _persist_counter = 0

        if step % max(1, max_steps // 10) == 0:
            note(f"step {step}/{max_steps} loss={loss_value:.4f} grad={grad_norm_value:.2e}")

        # Structured progress heartbeat (step / loss / ETA).
        if heartbeat and (step % max(1, max_steps // 20) == 0 or step == max_steps - 1):
            elapsed = time.time() - started
            progress_frac = (step + 1) / max(1, max_steps)
            eta = (elapsed / progress_frac) - elapsed if progress_frac > 0 else 0.0
            heartbeat({
                "step": step,
                "max_steps": max_steps,
                "train_loss": loss_value,
                "val_loss": val_loss,
                "eta_sec": round(max(0.0, eta), 1),
                "elapsed_sec": round(elapsed, 1),
                "tokens": tokens_seen,
            })

        # Periodic checkpoint.
        if ckpt is not None and ckpt.should_save(step, max_steps - 1):
            ckpt.save(net, optimizer, step=step, rng_state=rng,
                      elapsed_sec=time.time() - started, tokens=tokens_seen)

        # Early stop (val-loss plateau).
        if early_stop_patience > 0 and since_best >= early_stop_patience:
            note(f"step {step}: early stopped (no val improvement for {early_stop_patience} evals)")
            early_stopped = True
            return BackendOutput(status="timeout", steps=steps, layer_snapshots=snapshots,
                                 grad_values=_cat(grad_samples), param_values=_cat(param_samples),
                                 extras={"early_stopped": True})

        # GPU-seconds budget arbitration: hard ceiling on compute, checked
        # before the wall-clock budget (compute is the more expensive resource).
        if gpu_budget > 0 and gpu_seconds_used > gpu_budget:
            note(f"step {step}: gpu-seconds budget reached ({gpu_seconds_used:.1f}s > {gpu_budget}s)")
            gpu_budget_reached = True
            return BackendOutput(status="timeout", steps=steps, layer_snapshots=snapshots,
                                 grad_values=_cat(grad_samples), param_values=_cat(param_samples),
                                 extras={"gpu_budget_reached": True,
                                         "gpu_seconds_used": round(gpu_seconds_used, 3)})

        if time.time() - started > spec.budget.time_budget_sec:
            note(f"step {step}: time budget reached")
            return BackendOutput(status="timeout", steps=steps, layer_snapshots=snapshots,
                                 grad_values=_cat(grad_samples), param_values=_cat(param_samples))

    return BackendOutput(status="completed", steps=steps, layer_snapshots=snapshots,
                         grad_values=_cat(grad_samples), param_values=_cat(param_samples))


def _torch_layer_stats(net, probes, spec: TrialSpec) -> list[LayerStat]:
    stats: list[LayerStat] = []
    groups = net.named_layer_groups()
    grad_by_group: dict[str, float] = {}
    param_by_group: dict[str, float] = {}
    for name, params in groups:
        grads = [p.grad.detach().float() for p in params if p.grad is not None]
        grad_by_group[name] = _rms_tensors(grads)
        param_by_group[name] = _rms_tensors([p.detach().float() for p in params])
    lr_ref = spec.optim.lr
    for i, probe in enumerate(probes):
        stats.append(LayerStat(
            name=probe.name,
            depth=i,
            param_rms=param_by_group.get(probe.name, 0.0),
            grad_rms=grad_by_group.get(probe.name, 0.0),
            update_ratio=lr_ref,
            activation_rms=probe.activation_rms,
            dead_ratio=probe.dead_ratio,
            attn_entropy=probe.attn_entropy,
        ))
    return stats


def _rms_tensors(tensors) -> float:
    import torch

    if not tensors:
        return 0.0
    squares = sum(float(torch.sum(t * t).item()) for t in tensors)
    count = sum(int(t.numel()) for t in tensors)
    return math.sqrt(squares / max(1, count))


def _cat(arrays: list[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros(0)
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])
