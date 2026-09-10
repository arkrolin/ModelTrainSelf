"""Analytical surrogate training backend.

Why this exists
---------------
The whole point of MolelTrainSelf is the *search loop* — how agents read the
board, pick an axis, run an experiment, read the diagnostics and decide what to
do next. That loop can only be developed and tested if experiments are cheap.
Real training is not cheap.

So this module provides a **surrogate**: instead of optimizing real weights, it
maps a `TrialSpec` to a loss curve and a set of internal statistics through an
analytic model that encodes the regularities practitioners actually observe:

  * post-norm is far less tolerant of high lr than pre-norm, and needs a much
    longer warmup;
  * deep post-norm degrades into gradient starvation at the bottom of the stack,
    and residual scaling (1/sqrt(2L)) rescues it;
  * ReLU-family activations kill a large fraction of units, GELU/SiLU do not;
  * there is an optimal lr band — too low and the weights barely move, too high
    and gradients spike and clip;
  * capacity has diminishing returns and buys overfitting once data runs out.

It is not a simulator of any particular network. It is a **landscape with the
right shape**, so that the search machinery, the diagnostics, the prompts, the
board and the knowledge base can all be exercised end to end on a laptop.

Output contract is byte-for-byte compatible with the torch backend (same files,
same fields). Only `backend` in the summary differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mts.trainer.probes import LayerStat, StepMetric
from mts.trainer.spec import TrialSpec

# ---------------------------------------------------------------------------
# Coefficients of the landscape
# ---------------------------------------------------------------------------

# Intrinsic instability of each normalization placement (higher = more fragile).
NORM_INSTABILITY = {"pre": 0.6, "sandwich": 1.0, "post": 2.2, "none": 3.5}
# Learning rate each placement "wants" — the lr it is naturally calibrated for.
NORM_LR_OPT = {"pre": 3.0e-3, "sandwich": 2.0e-3, "post": 1.0e-3, "none": 3.0e-4}
# Warmup needed as a fraction of total steps.
NORM_WARMUP_RATIO = {"pre": 0.05, "sandwich": 0.10, "post": 0.25, "none": 0.30}
# How sharply instability grows with depth for each placement.
NORM_DEPTH_EXP = {"pre": 0.9, "sandwich": 1.0, "post": 1.4, "none": 1.6}
# Per-layer gradient attenuation exponent (how much signal is lost per block).
NORM_GRAD_ALPHA = {"pre": 0.25, "sandwich": 0.5, "post": 2.2, "none": 3.5}
# Per-layer activation growth (post-norm accumulates magnitude with depth).
NORM_ACT_GROWTH = {"pre": 1.02, "sandwich": 1.05, "post": 1.18, "none": 1.5}

ACT_INSTABILITY = {
    "relu": 1.0, "gelu": 1.0, "silu": 0.95, "swiglu": 1.05, "relu2": 1.7, "tanh": 0.75,
}
# Fraction of silent units at initialization-ish conditions.
ACT_DEAD = {
    "relu": 0.45, "relu2": 0.55, "gelu": 0.012, "silu": 0.006, "swiglu": 0.02, "tanh": 0.001,
}

L0_BASE = math.log(256.0)     # untrained cross-entropy scale
L_IRREDUCIBLE = 0.95          # best achievable loss for this synthetic task
TAU_BASE = 110.0              # convergence time constant at the optimal lr
INSTAB_UNSTABLE = 0.6         # instability score above which the run is unhealthy
INSTAB_DIVERGE = 1.5          # instability score above which the run blows up

ProgressFn = Callable[[str], None]


@dataclass
class BackendOutput:
    """Uniform return type for every training backend."""

    status: str                                  # completed | diverged | timeout | error
    steps: list[StepMetric] = field(default_factory=list)
    layer_snapshots: list[dict[str, Any]] = field(default_factory=list)
    grad_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    param_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Landscape evaluation
# ---------------------------------------------------------------------------


def _seed(spec: TrialSpec) -> int:
    """Deterministic per-spec seed: same spec + same seed => same run."""
    payload = spec.model_dump_json(exclude={"name", "notes"})
    return (spec.seed * 1_000_003 + zlib_crc(payload)) % (2**31 - 1)


def zlib_crc(text: str) -> int:
    import zlib

    return zlib.crc32(text.encode("utf-8"))


def instability_score(spec: TrialSpec) -> float:
    """Scalar "how likely is this configuration to misbehave" (0 = rock solid)."""
    a, o = spec.arch, spec.optim
    max_steps = max(1, spec.budget.max_steps)

    norm = a.norm_position
    depth_factor = (a.n_layer / 4.0) ** NORM_DEPTH_EXP.get(norm, 1.0)

    # lr is normalized by batch size (linear scaling) and by what this
    # normalization placement can tolerate.
    lr_ref = o.lr * (spec.data.batch_size / 32.0)
    lr_opt = NORM_LR_OPT.get(norm, 3.0e-3)
    lr_factor = max(0.0, math.log10(max(lr_ref, 1e-12) / lr_opt)) * 2.2

    # Warmup deficiency hurts the fragile placements the most.
    needed = NORM_WARMUP_RATIO.get(norm, 0.05) * max_steps
    deficit = max(0.0, 1.0 - (o.schedule.warmup_steps / needed if needed > 0 else 1.0))

    # Residual scaling is the classic rescue for deep post-norm stacks.
    resid = a.init.residual_scale
    if resid is not None:
        ideal = 1.0 / math.sqrt(2 * max(1, a.n_layer))
        resid_mitigation = 0.30 if abs(resid - ideal) <= 0.35 * ideal else 0.60
    elif a.init.scheme == "scaled":
        resid_mitigation = 0.45
    else:
        resid_mitigation = 1.0

    clip_mitigation = 0.75 if o.grad_clip else 1.0
    act_factor = ACT_INSTABILITY.get(a.activation, 1.0)
    # Normalization-free stacks have nothing to keep activations in check.
    norm_absent = 1.6 if a.norm_type == "none" else 1.0

    score = (
        NORM_INSTABILITY.get(norm, 1.0)
        * depth_factor
        * (0.25 + lr_factor)
        * act_factor
        * resid_mitigation
        * clip_mitigation
        * norm_absent
    )
    score += deficit * 2.5 * depth_factor
    return float(score)


def capacity_terms(spec: TrialSpec) -> tuple[float, float, float]:
    """Return (params, tokens, loss_floor)."""
    params = spec.param_count_estimate() + spec.data.vocab_size * spec.arch.d_model
    tokens = spec.budget.max_steps * spec.data.batch_size * spec.data.seq_len
    capacity_penalty = 0.90 / (1.0 + (params / 4.0e6) ** 0.65)
    data_penalty = 0.25 / (1.0 + (tokens / 1.2e7) ** 0.5)
    # Too high a learning rate cannot settle: it raises the achievable floor.
    lr_ref = spec.optim.lr * (spec.data.batch_size / 32.0)
    lr_opt = NORM_LR_OPT.get(spec.arch.norm_position, 3.0e-3)
    lr_penalty = 0.15 * max(0.0, math.log10(max(lr_ref, 1e-12) / lr_opt))
    floor = L_IRREDUCIBLE + capacity_penalty + data_penalty + lr_penalty
    return float(params), float(tokens), float(floor)


def convergence_tau(spec: TrialSpec) -> float:
    """Time constant of the loss decay. Larger = needs more steps to converge."""
    o = spec.optim
    lr_ref = o.lr * (spec.data.batch_size / 32.0)
    lr_opt = NORM_LR_OPT.get(spec.arch.norm_position, 3.0e-3)
    ratio = max(lr_opt / max(lr_ref, 1e-12), 1e-3)
    tau = TAU_BASE * (ratio**0.6)
    if o.name in ("sgd", "sgd_momentum"):
        tau *= 2.2          # non-adaptive optimizers converge slower here
    elif o.name == "adam":
        tau *= 1.1          # no decoupled weight decay
    return float(tau)


def generalization_gap(spec: TrialSpec, params: float, tokens: float) -> float:
    """How much worse val is than train at the end of the run."""
    o = spec.optim
    data_ratio = tokens / max(params, 1.0)
    gap = max(0.0, 0.35 / (1.0 + data_ratio) - 0.03)
    gap *= 1.0 - 0.45 * min(1.0, o.weight_decay / 0.1)
    gap *= 1.0 - 0.50 * min(1.0, spec.arch.dropout / 0.1)
    return float(min(gap, 0.8))


def dead_ratio_for(spec: TrialSpec, instab: float, rng: np.random.Generator) -> float:
    lr_ref = spec.optim.lr * (spec.data.batch_size / 32.0)
    lr_term = 1.0 + 0.35 * max(0.0, math.log10(max(lr_ref, 1e-12) / 3.0e-3))
    base = ACT_DEAD.get(spec.arch.activation, 0.02)
    return float(np.clip(base * lr_term * (1.0 + 0.15 * instab) * rng.uniform(0.85, 1.15), 0.0, 0.95))


def attention_entropy_for(spec: TrialSpec, step: int, max_steps: int, instab: float,
                          rng: np.random.Generator) -> float:
    head_dim = spec.arch.d_model / max(1, spec.arch.n_head)
    head_penalty = 0.6 * max(0.0, 1.0 - head_dim / 48.0)
    progress = step / max(1, max_steps)
    value = 0.92 - 0.22 * progress - head_penalty - 0.08 * instab + rng.normal(0, 0.02)
    return float(np.clip(value, 0.03, 0.99))


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(spec: TrialSpec, out_dir: Path | None = None, progress: ProgressFn | None = None
        ) -> BackendOutput:
    """Evaluate one point of the state space analytically."""
    rng = np.random.default_rng(_seed(spec))
    a, o, b = spec.arch, spec.optim, spec.budget
    max_steps = max(1, b.max_steps)
    vocab = max(2, spec.data.vocab_size)

    instab = instability_score(spec)
    params, tokens, loss_floor = capacity_terms(spec)
    tau = convergence_tau(spec)
    gap = generalization_gap(spec, params, tokens)
    l0 = math.log(vocab) * (L0_BASE / math.log(256.0))

    # Where the run dies, if it dies at all.
    nan_step: int | None = None
    if instab >= INSTAB_DIVERGE:
        frac = float(rng.uniform(0.08, 0.55))
        nan_step = min(max_steps - 1, int(o.schedule.warmup_steps + frac * max_steps))
    # Spikes only start once a configuration is genuinely out of its comfort zone;
    # a healthy run should see none at all.
    spike_rate = float(np.clip((instab - 0.35) * 0.05, 0.0, 0.25))

    steps: list[StepMetric] = []
    layer_snapshots: list[dict[str, Any]] = []
    grad_samples: list[np.ndarray] = []
    param_samples: list[np.ndarray] = []

    elapsed_ms = 0.0
    tokens_per_step = spec.data.batch_size * spec.data.seq_len
    step_cost_ms = 12.0 * (params / 3.0e6) ** 0.85 * (spec.data.seq_len / 128.0)

    best_train = float("inf")
    train_loss = l0

    for step in range(max_steps):
        lr = _lr_at(spec, step, max_steps)

        # ---- loss -------------------------------------------------------
        target = loss_floor + (l0 - loss_floor) * math.exp(-step / tau)
        noise = float(rng.normal(0, 0.006 + 0.004 * instab))
        train_loss = target + noise
        if step > 0 and rng.random() < spike_rate:
            train_loss *= float(rng.uniform(1.5, 3.0))
        best_train = min(best_train, train_loss)

        # ---- gradient ---------------------------------------------------
        g_base = 0.35 * math.exp(-step / max(20.0, tau * 0.8)) + 0.08
        grad_norm = abs(g_base * (1.0 + instab * float(rng.normal(0, 0.3))))
        spiked = False
        if step > 0 and rng.random() < spike_rate:
            grad_norm *= float(rng.uniform(3.0, 25.0)) * (1.0 + instab * 3.0)
            spiked = True
        clipped = False
        if o.grad_clip is not None and grad_norm > o.grad_clip:
            clipped = True

        # ---- sample distributions (for histograms) -----------------------
        if step % max(1, b.probe_every) == 0 or step == max_steps - 1:
            bulk = rng.normal(0.0, max(grad_norm, 1e-9) * 0.35, 2048)
            if spiked or instab > INSTAB_UNSTABLE:
                tail_n = int(2048 * min(0.25, 0.02 * instab))
                tail = rng.normal(0.0, max(grad_norm, 1e-9) * (2.0 + 6.0 * instab), tail_n)
                bulk = np.concatenate([bulk, tail])
            grad_samples.append(bulk)
            prms = a.init.std * (1.0 + 0.30 * math.sqrt(step / max_steps)) * (1.0 + 0.5 * instab)
            param_samples.append(rng.normal(0.0, max(prms, 1e-9), 2048))

        # ---- evaluation --------------------------------------------------
        val_loss = None
        val_acc = None
        if step % b.eval_every == 0 or step == max_steps - 1:
            gap_now = gap * (step / max_steps)
            noise_v = float(rng.normal(0, 0.02))
            val_loss = float(train_loss + gap_now + noise_v)
            val_acc = float(np.clip(math.exp(-0.85 * (val_loss - 1.0)), 0.02, 0.97))

        # ---- hard failure -------------------------------------------------
        if nan_step is not None and step >= nan_step:
            train_loss = float("inf")
            grad_norm = float("inf")
            val_loss = float("inf")
            val_acc = 0.0
            steps.append(StepMetric(step=step, train_loss=train_loss, val_loss=val_loss,
                                    val_acc=val_acc, lr=lr, grad_norm=grad_norm,
                                    clipped=True, step_ms=step_cost_ms, tokens=tokens_per_step))
            if progress:
                progress(f"step {step}: loss diverged (NaN/Inf)")
            break

        steps.append(StepMetric(
            step=step, train_loss=float(train_loss), val_loss=val_loss, val_acc=val_acc,
            lr=lr, grad_norm=float(grad_norm), clipped=clipped,
            step_ms=step_cost_ms, tokens=tokens_per_step,
        ))

        # ---- per-layer probe ---------------------------------------------
        if step % max(1, b.probe_every) == 0 or step == max_steps - 1:
            layer_snapshots.append({
                "step": step,
                "layers": _layer_stats(spec, step, max_steps, grad_norm, instab, rng),
            })

        elapsed_ms += step_cost_ms
        if elapsed_ms / 1000.0 > b.time_budget_sec:
            if progress:
                progress(f"step {step}: time budget reached")
            break

        # GPU-seconds budget arbitration (docs/08): compute time is the hard
        # ceiling. `step_cost_ms` already models per-step accelerator cost, so
        # we track it cumulatively and stop the run the instant it would exceed
        # the budget — mirroring `_run_torch`.
        if b.gpu_seconds_budget > 0 and (elapsed_ms / 1000.0) > b.gpu_seconds_budget:
            if progress:
                progress(f"step {step}: gpu-seconds budget reached")
            gpu_budget_reached = True
            break

        if progress and step % max(1, max_steps // 10) == 0:
            progress(f"step {step}/{max_steps} loss={train_loss:.4f} grad={grad_norm:.2e}")

    diverged = any(not math.isfinite(m.train_loss) for m in steps)
    timed_out = elapsed_ms / 1000.0 > b.time_budget_sec
    gpu_budget_reached = (
        b.gpu_seconds_budget > 0 and (elapsed_ms / 1000.0) > b.gpu_seconds_budget
    )
    if diverged:
        status = "diverged"
    elif timed_out or gpu_budget_reached:
        status = "timeout"
    else:
        status = "completed"

    grad_values = np.concatenate(grad_samples) if grad_samples else np.zeros(0)
    param_values = np.concatenate(param_samples) if param_samples else np.zeros(0)

    return BackendOutput(
        status=status,
        steps=steps,
        layer_snapshots=layer_snapshots,
        grad_values=grad_values,
        param_values=param_values,
        extras={
            "instability_score": instab,
            "params_estimate": params,
            "tokens_seen": tokens,
            "loss_floor": loss_floor,
            "convergence_tau": tau,
            "generalization_gap": gap,
            "gpu_seconds_used": round(elapsed_ms / 1000.0, 3),
            "gpu_budget_reached": gpu_budget_reached,
        },
    )


def _lr_at(spec: TrialSpec, step: int, max_steps: int) -> float:
    """Mirror of `mts.trainer.model.lr_at`, kept local so surrogate needs no torch."""
    o = spec.optim
    s = o.schedule
    warmup = max(1, min(s.warmup_steps, max_steps - 1)) if max_steps > 1 else 1
    if step < warmup:
        return o.lr * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup)))
    if s.kind == "constant":
        return o.lr
    if s.kind == "linear":
        return o.lr * (1 - progress) + o.lr * s.min_lr_ratio * progress
    if s.kind == "wsd":
        decay_start = 1.0 - s.decay_fraction
        if progress < decay_start:
            return o.lr
        local = (progress - decay_start) / max(1e-9, s.decay_fraction)
        return o.lr * (1 - local) + o.lr * s.min_lr_ratio * local
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return o.lr * (s.min_lr_ratio + (1 - s.min_lr_ratio) * cosine)


def _layer_stats(spec: TrialSpec, step: int, max_steps: int, grad_norm: float,
                 instab: float, rng: np.random.Generator) -> list[LayerStat]:
    """Per-block statistics. Layer 0 is the bottom of the stack."""
    a = spec.arch
    n = max(1, a.n_layer)
    norm = a.norm_position

    alpha = NORM_GRAD_ALPHA.get(norm, 1.0) * (n / 4.0) ** 0.9
    growth = NORM_ACT_GROWTH.get(norm, 1.1)
    if a.norm_type == "none":
        growth = max(growth, 1.5)

    lr_ref = spec.optim.lr * (spec.data.batch_size / 32.0)
    update_ratio = float(np.clip(lr_ref * rng.uniform(0.6, 1.0), 1e-9, 10.0))
    dead = dead_ratio_for(spec, instab, rng)
    entropy = attention_entropy_for(spec, step, max_steps, instab, rng)

    stats: list[LayerStat] = []
    for i in range(n):
        # distance from the top of the stack: layer n-1 is closest to the output
        depth_from_top = (n - 1 - i) / n
        grad_rms = max(grad_norm, 1e-12) * math.exp(-alpha * depth_from_top)
        grad_rms *= float(rng.uniform(0.85, 1.15))
        act_rms = (growth ** i) * float(rng.uniform(0.9, 1.1)) * (1.0 + 0.3 * instab)
        param_rms = a.init.std * (1.0 + 0.30 * math.sqrt(step / max_steps)) * (1.0 + 0.5 * instab)
        stats.append(LayerStat(
            name=f"block{i}",
            depth=i,
            param_rms=float(param_rms),
            grad_rms=float(grad_rms),
            update_ratio=update_ratio,
            activation_rms=float(act_rms),
            dead_ratio=float(np.clip(dead * rng.uniform(0.85, 1.15), 0.0, 0.95)),
            attn_entropy=float(np.clip(entropy + rng.normal(0, 0.02), 0.03, 0.99)),
        ))
    return stats
