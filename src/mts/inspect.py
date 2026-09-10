"""Model inspection tools — the read-only instruments agents point at a trial.

The trainer already writes far more about a run than its verdict: every probe
step records per-layer gradient/parameter/activation state, and the final
diagnostics carry full distributions. Until now none of it reached the agents —
they saw a single label (`verdict=unstable`) and had to guess where the problem
was.

This module is the instrument panel over that data. Every function here is a
**pure reader** over artifacts already on disk (`layers.jsonl`,
`diagnostics.json`, `metrics.jsonl`); nothing re-runs training, nothing needs a
GPU, and nothing mutates state. That makes them safe to expose over HTTP and
safe for an agent to call speculatively.

The tools answer the questions a practitioner actually asks:

  * `inspect_layers`      — what is each layer doing right now?
  * `layer_trajectory`    — how did one layer's state evolve over training?
  * `inspect_grad_flow`   — does the update signal reach the bottom of the stack?
  * `inspect_distribution`— what do the gradient/parameter histograms look like?
  * `inspect_curve`       — what did the loss/lr/grad-norm curves do?
  * `compare_trials`      — what changed between two points in the space, and
                            what did that change do to the model's internals?

`describe_tools()` renders the catalog for an LLM agent's prompt, and
`run_tool(name, ...)` dispatches by name so an agent can request a tool by
string without the caller hand-wiring every branch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from mts.trainer.runner import read_diagnostics, read_layers, read_metrics, read_summary

# Fields carried by every LayerStat row, in display order.
_LAYER_FIELDS = ("grad_rms", "param_rms", "update_ratio", "activation_rms",
                 "dead_ratio", "attn_entropy")


class InspectError(RuntimeError):
    """Raised when a trial's artifacts are missing or unreadable."""


# ---------------------------------------------------------------------------
# Artifact access
# ---------------------------------------------------------------------------


def _require_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    if not path.is_dir():
        raise InspectError(f"trial artifact directory not found: {out_dir}")
    return path


def _safe_read(reader: Callable[[Path], Any], path: Path, what: str) -> Any:
    try:
        return reader(path)
    except FileNotFoundError as exc:
        raise InspectError(f"missing {what} for {path.name}") from exc
    except (OSError, ValueError) as exc:
        raise InspectError(f"unreadable {what} for {path.name}: {exc}") from exc


# ---------------------------------------------------------------------------
# 1. Per-layer state
# ---------------------------------------------------------------------------


def inspect_layers(out_dir: str | Path, *, step: int | None = None) -> dict[str, Any]:
    """Per-layer internal state at one probe step (default: the last one).

    This is the primary "look inside the model" tool. `outliers` pre-computes the
    layers a practitioner would actually zoom in on, so an agent does not have to
    scan the table itself.
    """
    path = _require_dir(out_dir)
    snapshots = _safe_read(read_layers, path, "layers.jsonl")
    if not snapshots:
        return {"trial_dir": str(path), "step": None, "layers": [],
                "note": "no probe snapshots recorded"}

    snapshot = _pick_snapshot(snapshots, step)
    layers = snapshot.get("layers", [])
    return {
        "trial_dir": str(path),
        "step": snapshot.get("step"),
        "available_steps": [s.get("step") for s in snapshots],
        "n_layers": len(layers),
        "layers": layers,
        "summary": _layer_summary(layers),
        "outliers": _layer_outliers(layers),
    }


def _pick_snapshot(snapshots: list[dict[str, Any]], step: int | None) -> dict[str, Any]:
    if step is None:
        return snapshots[-1]
    # Nearest recorded probe step — probes are periodic, so an arbitrary step
    # rarely lands exactly on one.
    return min(snapshots, key=lambda s: abs(int(s.get("step", 0)) - step))


def _layer_summary(layers: list[dict[str, Any]]) -> dict[str, Any]:
    """Min / max / spread per field across the stack."""
    out: dict[str, Any] = {}
    for field in _LAYER_FIELDS:
        values = [_f(layer.get(field)) for layer in layers]
        values = [v for v in values if v is not None]
        if not values:
            continue
        lo, hi = min(values), max(values)
        out[field] = {
            "min": lo, "max": hi,
            "mean": sum(values) / len(values),
            # Spread is the interpretable one: 1e4 spread in grad_rms across the
            # stack is a structural problem, a 2x spread is noise.
            "spread": (hi / lo) if lo > 0 else None,
        }
    return out


def _layer_outliers(layers: list[dict[str, Any]]) -> list[str]:
    """Human-readable callouts: which specific layers look wrong, and why."""
    notes: list[str] = []
    if not layers:
        return notes

    grads = [(layer.get("name"), _f(layer.get("grad_rms"))) for layer in layers]
    grads = [(n, v) for n, v in grads if v is not None and v > 0]
    if len(grads) >= 2:
        quietest = min(grads, key=lambda kv: kv[1])
        loudest = max(grads, key=lambda kv: kv[1])
        if loudest[1] / quietest[1] >= 10:
            notes.append(
                f"gradient spread {loudest[1] / quietest[1]:.0f}x across the stack: "
                f"{quietest[0]}={quietest[1]:.2e} (quietest) .. "
                f"{loudest[0]}={loudest[1]:.2e} (loudest)"
            )

    for layer in layers:
        name = layer.get("name", "?")
        dead = _f(layer.get("dead_ratio"))
        if dead is not None and dead > 0.35:
            notes.append(f"{name}: {dead:.0%} of activations are silent")
        entropy = _f(layer.get("attn_entropy"))
        if entropy is not None and 0 < entropy < 0.35:
            notes.append(f"{name}: attention entropy {entropy:.2f} nats — near one-hot")
        act = _f(layer.get("activation_rms"))
        if act is not None and act > 1e3:
            notes.append(f"{name}: activation rms {act:.2e} — activations are blowing up")
    return notes


# ---------------------------------------------------------------------------
# 2. One layer over time
# ---------------------------------------------------------------------------


def layer_trajectory(out_dir: str | Path, layer: str | None = None,
                     *, field: str = "grad_rms") -> dict[str, Any]:
    """How one field of one layer evolved across probe steps.

    A single snapshot cannot distinguish "this layer was always quiet" from
    "this layer went quiet at step 200" — but those imply different fixes. This
    is the tool for that question. `layer=None` returns the trajectory of every
    layer for the requested field.
    """
    path = _require_dir(out_dir)
    if field not in _LAYER_FIELDS:
        raise InspectError(f"unknown layer field {field!r}; known: {list(_LAYER_FIELDS)}")
    snapshots = _safe_read(read_layers, path, "layers.jsonl")

    steps = [s.get("step") for s in snapshots]
    series: dict[str, list[float | None]] = {}
    for snapshot in snapshots:
        for entry in snapshot.get("layers", []):
            name = entry.get("name", "?")
            if layer is not None and name != layer:
                continue
            series.setdefault(name, []).append(_f(entry.get(field)))

    if layer is not None and not series:
        known = sorted({e.get("name") for s in snapshots for e in s.get("layers", [])})
        raise InspectError(f"layer {layer!r} not found; known layers: {known}")

    return {
        "trial_dir": str(path),
        "field": field,
        "steps": steps,
        "series": series,
        "trends": {name: _trend(values) for name, values in series.items()},
    }


def _trend(values: list[float | None]) -> str:
    """Classify a series as rising / falling / flat by first-vs-last thirds."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 3:
        return "insufficient data"
    third = max(1, len(clean) // 3)
    head = sum(clean[:third]) / third
    tail = sum(clean[-third:]) / third
    if head == 0:
        return "rising from zero" if tail > 0 else "flat at zero"
    ratio = tail / head
    if ratio > 2.0:
        return f"rising {ratio:.1f}x"
    if ratio < 0.5:
        return f"falling to {ratio:.2f}x"
    return "roughly flat"


# ---------------------------------------------------------------------------
# 3. Gradient flow through the stack
# ---------------------------------------------------------------------------


def inspect_grad_flow(out_dir: str | Path) -> dict[str, Any]:
    """Does the update signal reach the bottom of the stack?

    Fits the per-layer gradient magnitude to an exponential in depth. The decay
    rate is the dimensionless number that transfers across model sizes — a
    bottom/top ratio of 0.01 means the same pathology whether the model has 6
    layers or 60.
    """
    path = _require_dir(out_dir)
    snapshot = inspect_layers(path)
    layers = snapshot.get("layers", [])
    if len(layers) < 2:
        return {"trial_dir": str(path), "note": "need >= 2 layers to assess flow",
                "layers": layers}

    ordered = sorted(layers, key=lambda entry: int(entry.get("depth", 0)))
    bottom = _f(ordered[0].get("grad_rms")) or 0.0
    top = _f(ordered[-1].get("grad_rms")) or 0.0
    ratio = (bottom / top) if top > 0 else None

    # Per-step decay factor, geometric mean across adjacent pairs.
    factors = []
    for lower, upper in zip(ordered, ordered[1:]):
        a, b = _f(lower.get("grad_rms")), _f(upper.get("grad_rms"))
        if a and b and a > 0 and b > 0:
            factors.append(b / a)
    per_layer = (math.exp(sum(math.log(f) for f in factors) / len(factors))
                 if factors else None)

    if ratio is None:
        assessment = "top layer has zero gradient — the model is not training at all"
    elif ratio < 0.01:
        assessment = ("gradient starves at the bottom of the stack (<1% of the top). "
                      "Classic deep post-norm pathology: try pre/sandwich norm or "
                      "residual scaling.")
    elif ratio < 0.1:
        assessment = "noticeable gradient attenuation toward the bottom, not yet critical"
    elif ratio > 10:
        assessment = "gradient is LARGER at the bottom than the top — unusual, check init scale"
    else:
        assessment = "gradient reaches the whole stack"

    return {
        "trial_dir": str(path),
        "step": snapshot.get("step"),
        "n_layers": len(ordered),
        "bottom_layer": ordered[0].get("name"),
        "top_layer": ordered[-1].get("name"),
        "bottom_grad_rms": bottom,
        "top_grad_rms": top,
        "bottom_top_ratio": ratio,
        "per_layer_decay_factor": per_layer,
        "assessment": assessment,
        "profile": [{"name": e.get("name"), "depth": e.get("depth"),
                     "grad_rms": _f(e.get("grad_rms"))} for e in ordered],
    }


# ---------------------------------------------------------------------------
# 4. Distributions
# ---------------------------------------------------------------------------


def inspect_distribution(out_dir: str | Path, *, kind: str = "grad") -> dict[str, Any]:
    """The gradient or parameter histogram, plus a readable characterization.

    `kind` is "grad" or "param". Histograms are already computed by the probes
    (symlog bins for gradients, linear for parameters); this adds the tail/mass
    description an agent needs to act on them.
    """
    if kind not in ("grad", "param"):
        raise InspectError(f"kind must be 'grad' or 'param', got {kind!r}")
    path = _require_dir(out_dir)
    diagnostics = _safe_read(read_diagnostics, path, "diagnostics.json")
    block = diagnostics.get(kind) or {}
    hist = block.get("hist") or {}

    result = {
        "trial_dir": str(path),
        "kind": kind,
        "stats": {k: v for k, v in block.items() if k != "hist"},
        "hist": hist,
    }
    if hist:
        result["characterization"] = _describe_hist(hist, kind)
    return result


def _describe_hist(hist: dict[str, Any], kind: str) -> str:
    counts = hist.get("counts") or []
    edges = hist.get("edges") or []
    total = hist.get("total") or sum(counts)
    if not counts or not total:
        return "empty histogram"

    parts = [
        f"{total} values in [{_num(hist.get('min'))}, {_num(hist.get('max'))}], "
        f"mean={_num(hist.get('mean'))} std={_num(hist.get('std'))}"
    ]
    # Mass concentrated in a single bin means a degenerate distribution: for
    # gradients that is usually "everything is ~0" (dead) or "everything is
    # clipped to the same magnitude".
    peak = max(range(len(counts)), key=lambda i: counts[i])
    peak_share = counts[peak] / total
    if peak_share > 0.5 and peak < len(edges) - 1:
        parts.append(
            f"{peak_share:.0%} of mass in a single bin "
            f"[{_num(edges[peak])}, {_num(edges[peak + 1])}] — degenerate distribution"
        )
    if kind == "grad":
        near_zero = sum(
            c for c, lo, hi in zip(counts, edges, edges[1:])
            if abs(lo) < 1e-7 and abs(hi) < 1e-7
        )
        if near_zero / total > 0.5:
            parts.append(f"{near_zero / total:.0%} of gradients are below 1e-7 — "
                         "most of the model is receiving no update")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# 5. Training curves
# ---------------------------------------------------------------------------


def inspect_curve(out_dir: str | Path, *, downsample: int = 40) -> dict[str, Any]:
    """The loss / lr / grad-norm curves, downsampled, plus shape description.

    The shape matters as much as the endpoint: two runs ending at the same loss
    but one plateauing at step 50 and the other still descending at step 300
    call for opposite next moves.
    """
    path = _require_dir(out_dir)
    rows = _safe_read(lambda p: read_metrics(p, downsample=downsample), path,
                      "metrics.jsonl")
    train = [(r.get("step"), _f(r.get("train_loss"))) for r in rows]
    val = [(r.get("step"), _f(r.get("val_loss"))) for r in rows
           if r.get("val_loss") is not None]

    return {
        "trial_dir": str(path),
        "n_points": len(rows),
        "train_loss": train,
        "val_loss": val,
        "grad_norm": [(r.get("step"), _f(r.get("grad_norm"))) for r in rows],
        "lr": [(r.get("step"), _f(r.get("lr"))) for r in rows],
        "clipped_fraction": (
            sum(1 for r in rows if r.get("clipped")) / len(rows) if rows else 0.0
        ),
        "shape": _describe_curve([v for _, v in train if v is not None]),
        "val_shape": _describe_curve([v for _, v in val if v is not None]),
    }


def _describe_curve(values: list[float]) -> str:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 4:
        return "too few finite points to characterize"
    best = min(clean)
    quarter = max(1, len(clean) // 4)
    head, tail = clean[:quarter], clean[-quarter:]
    head_mean = sum(head) / len(head)
    tail_mean = sum(tail) / len(tail)

    if len(clean) < len(values):
        return "became non-finite mid-run (diverged)"
    if tail_mean > head_mean:
        return f"rising: started ~{head_mean:.3f}, ended ~{tail_mean:.3f} — not learning"
    # Still-descending vs plateaued: compare the last quarter's own slope
    # against the total descent.
    late_drop = tail[0] - tail[-1]
    total_drop = head_mean - tail_mean
    if total_drop <= 0:
        return "flat — no progress"
    if late_drop / max(total_drop, 1e-12) > 0.1:
        return (f"still descending at the end (best={best:.3f}) — "
                "more steps or more capacity would likely help")
    return f"plateaued near {tail_mean:.3f} (best={best:.3f})"


# ---------------------------------------------------------------------------
# 6. Cross-trial comparison
# ---------------------------------------------------------------------------


def compare_trials(out_dir_a: str | Path, out_dir_b: str | Path) -> dict[str, Any]:
    """What changed between two trials, and what it did to the model's internals.

    This is the causal tool: the spec diff is the cause, the diagnostics delta is
    the effect. Distilled knowledge is only trustworthy when both halves are
    recorded, so the agent should call this before writing a lesson.
    """
    path_a, path_b = _require_dir(out_dir_a), _require_dir(out_dir_b)
    sum_a = _safe_read(read_summary, path_a, "summary.json")
    sum_b = _safe_read(read_summary, path_b, "summary.json")

    from mts.trainer.spec import TrialSpec

    spec_delta: dict[str, Any] = {}
    try:
        spec_a = TrialSpec.from_dict(sum_a.get("spec") or {})
        spec_b = TrialSpec.from_dict(sum_b.get("spec") or {})
        spec_delta = spec_b.diff(spec_a)
    except Exception:  # noqa: BLE001 - a malformed old spec must not kill the tool
        spec_delta = {"note": "spec diff unavailable"}

    diag_a = _try(lambda: read_diagnostics(path_a)) or {}
    diag_b = _try(lambda: read_diagnostics(path_b)) or {}

    return {
        "a": {"trial_dir": str(path_a), "trial_id": sum_a.get("trial_id"),
              "verdict": sum_a.get("verdict"),
              "best_val_loss": (sum_a.get("best") or {}).get("val_loss")},
        "b": {"trial_dir": str(path_b), "trial_id": sum_b.get("trial_id"),
              "verdict": sum_b.get("verdict"),
              "best_val_loss": (sum_b.get("best") or {}).get("val_loss")},
        "spec_delta": spec_delta,
        "verdict_change": f"{sum_a.get('verdict')} -> {sum_b.get('verdict')}",
        "metric_delta": _metric_delta(sum_a, sum_b),
        "diagnostics_delta": _diag_delta(diag_a, diag_b),
    }


def _metric_delta(sum_a: dict[str, Any], sum_b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("val_loss", "val_acc"):
        a = _f((sum_a.get("best") or {}).get(key))
        b = _f((sum_b.get("best") or {}).get(key))
        if a is None or b is None:
            continue
        out[key] = {"from": a, "to": b, "delta": b - a,
                    "relative": ((b - a) / abs(a)) if a else None}
    return out


# Fields worth diffing, as (section, field) with a note on which direction is bad.
_DIAG_DELTA_FIELDS = [
    ("grad", "p99"), ("grad", "bottom_top_ratio"), ("grad", "clip_fraction"),
    ("param", "final_update_ratio"), ("param", "max_rms"),
    ("activation", "max_dead_ratio"), ("activation", "attention_entropy"),
    ("stability", "loss_spike_count"), ("stability", "generalization_gap"),
]


def _diag_delta(diag_a: dict[str, Any], diag_b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for section, field in _DIAG_DELTA_FIELDS:
        a = _f((diag_a.get(section) or {}).get(field))
        b = _f((diag_b.get(section) or {}).get(field))
        if a is None or b is None:
            continue
        entry: dict[str, Any] = {"from": a, "to": b, "delta": b - a}
        if a != 0:
            entry["fold_change"] = b / a
        out[f"{section}.{field}"] = entry
    return out


# ---------------------------------------------------------------------------
# Tool catalog — what an agent is allowed to call
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "inspect_layers": {
        "fn": inspect_layers,
        "args": "step (optional int)",
        "doc": "Per-layer grad/param/activation state at one probe step, with outlier callouts.",
    },
    "layer_trajectory": {
        "fn": layer_trajectory,
        "args": "layer (optional str), field (default grad_rms)",
        "doc": "How one layer's state evolved across probe steps — separates 'always bad' from 'went bad'.",
    },
    "inspect_grad_flow": {
        "fn": inspect_grad_flow,
        "args": "(none)",
        "doc": "Whether the update signal reaches the bottom of the stack; returns the decay profile.",
    },
    "inspect_distribution": {
        "fn": inspect_distribution,
        "args": "kind ('grad' or 'param')",
        "doc": "Gradient/parameter histogram plus a characterization of its tails and mass.",
    },
    "inspect_curve": {
        "fn": inspect_curve,
        "args": "downsample (default 40)",
        "doc": "Loss/lr/grad-norm curves and whether the run plateaued or was still descending.",
    },
    "compare_trials": {
        "fn": compare_trials,
        "args": "out_dir_b (the other trial)",
        "doc": "Spec diff vs diagnostics delta between two trials — cause paired with effect.",
    },
}


def describe_tools() -> str:
    """Render the tool catalog for an agent prompt."""
    lines = ["Available model-inspection tools (read-only, safe to call):"]
    for name, meta in TOOLS.items():
        lines.append(f"  - {name}({meta['args']}): {meta['doc']}")
    return "\n".join(lines)


def run_tool(name: str, out_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Dispatch a tool by name. Raises `InspectError` for an unknown tool."""
    if name not in TOOLS:
        raise InspectError(f"unknown tool {name!r}; known: {sorted(TOOLS)}")
    return TOOLS[name]["fn"](out_dir, **kwargs)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _f(value: Any) -> float | None:
    """Coerce to float, mapping None/non-numeric/non-finite to None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _num(value: Any) -> str:
    number = _f(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.2e}"
    return f"{number:.4g}"


def _try(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - readers are best-effort here
        return None
