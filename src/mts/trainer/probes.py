"""Diagnostic instrumentation.

Training curves alone are a weak signal: two runs can end at the same loss with
completely different internal health. The probes here turn the model's internals
into the observations agents actually reason over:

  * gradient distribution      — is the update signal vanishing, exploding, or clipped?
  * per-layer gradient flow    — is the signal reaching the bottom of the stack?
  * parameter distribution     — are weights collapsing, saturating, or dead?
  * activation statistics      — how much of the network is silent?
  * update ratio (|Δw|/|w|)    — is the optimizer actually moving the weights?

Everything is plain numpy so the same code serves the torch backend and the
analytical surrogate backend. The output shapes are stable JSON contracts —
the board, the prompts and the tests all read the same fields.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------


def symlog_bins(lo: float = 1e-8, hi: float = 1e2, n: int = 48) -> list[float]:
    """Bin edges covering +/- magnitude on a log scale, plus a zero bin.

    Gradients span many orders of magnitude, so a linear histogram is useless:
    it shows one giant spike at zero. A symmetric log histogram keeps both the
    bulk and the tails readable.
    """
    if lo <= 0 or hi <= lo:
        raise ValueError("require 0 < lo < hi")
    pos = np.logspace(math.log10(lo), math.log10(hi), n // 2)
    edges = np.concatenate([-pos[::-1], [0.0], pos])
    return [float(x) for x in edges]


def linear_bins(lo: float, hi: float, n: int = 48) -> list[float]:
    if hi <= lo:
        hi = lo + 1e-6
    return [float(x) for x in np.linspace(lo, hi, n + 1)]


def histogram(values: Sequence[float] | np.ndarray, edges: Sequence[float]) -> dict[str, Any]:
    """Histogram as a JSON-friendly payload the frontend can draw directly."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    edges_arr = np.asarray(edges, dtype=np.float64)
    counts, _ = np.histogram(arr, bins=edges_arr)
    return {
        "edges": [float(x) for x in edges_arr],
        "counts": [int(x) for x in counts],
        "total": int(arr.size),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def log_histogram(values: Sequence[float] | np.ndarray, n: int = 48) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return histogram([], symlog_bins(n=n))
    mag = np.abs(arr)
    positive = mag[mag > 0]
    lo = float(positive.min()) if positive.size else 1e-8
    hi = float(mag.max())
    if hi <= lo:
        hi = lo * 10
    return histogram(arr, symlog_bins(lo * 0.5, hi * 2, n))


def safe_rms(values: Sequence[float] | np.ndarray | None) -> float:
    if values is None:
        return 0.0
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


# ---------------------------------------------------------------------------
# Per-step metric record
# ---------------------------------------------------------------------------


class StepMetric(BaseModel):
    step: int
    train_loss: float
    val_loss: float | None = None
    val_acc: float | None = None
    lr: float = 0.0
    grad_norm: float = 0.0
    clipped: bool = False
    step_ms: float = 0.0
    tokens: int = 0


# ---------------------------------------------------------------------------
# Per-layer snapshot
# ---------------------------------------------------------------------------


class LayerStat(BaseModel):
    """One layer's internal state at a probe step."""

    name: str
    depth: int = 0
    param_rms: float = 0.0
    grad_rms: float = 0.0
    update_ratio: float = 0.0
    activation_rms: float = 0.0
    dead_ratio: float = 0.0
    attn_entropy: float = 0.0

    def compact(self) -> str:
        return (
            f"{self.name}: grad_rms={self.grad_rms:.2e} param_rms={self.param_rms:.2e} "
            f"upd={self.update_ratio:.2e} act={self.activation_rms:.2e} dead={self.dead_ratio:.1%}"
        )


# ---------------------------------------------------------------------------
# Aggregated diagnostics
# ---------------------------------------------------------------------------


class GradientDiagnostics(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    p99: float = 0.0
    max: float = 0.0
    clip_fraction: float = 0.0
    # grad_rms(bottom layer) / grad_rms(top layer).
    #   ~1.0                  -> signal reaches the whole stack
    #   << 1 (e.g. < 0.01)    -> gradient starvation at the bottom of the stack,
    #                            the classic deep post-norm pathology
    bottom_top_ratio: float = 1.0
    vanishing: bool = False
    exploding: bool = False
    hist: dict[str, Any] = Field(default_factory=dict)


class ParamDiagnostics(BaseModel):
    max_rms: float = 0.0
    final_update_ratio: float = 0.0
    max_abs: float = 0.0
    hist: dict[str, Any] = Field(default_factory=dict)


class ActivationDiagnostics(BaseModel):
    max_dead_ratio: float = 0.0
    max_rms: float = 0.0
    attention_entropy: float = 0.0
    attention_collapse: bool = False


class StabilityDiagnostics(BaseModel):
    nan_or_inf: bool = False
    diverged: bool = False
    loss_spike_count: int = 0
    final_loss: float = 0.0
    best_loss: float = 0.0
    final_val_acc: float | None = None
    # train/val gap at the end of the run
    generalization_gap: float = 0.0
    # Best loss achieved so far in this project (None for the very first run).
    # Comparing against it is how a practitioner decides "did this run even learn".
    reference_loss: float | None = None
    headroom: float | None = None


class Diagnostics(BaseModel):
    grad: GradientDiagnostics = Field(default_factory=GradientDiagnostics)
    param: ParamDiagnostics = Field(default_factory=ParamDiagnostics)
    activation: ActivationDiagnostics = Field(default_factory=ActivationDiagnostics)
    stability: StabilityDiagnostics = Field(default_factory=StabilityDiagnostics)
    verdict: str = "unknown"
    signals: list[str] = Field(default_factory=list)

    def as_prompt_block(self, limit: int = 8) -> str:
        """Compact textual rendering injected into agent prompts."""
        lines = [
            f"verdict={self.verdict}",
            f"grad: mean={self.grad.mean:.2e} p99={self.grad.p99:.2e} max={self.grad.max:.2e} "
            f"clip_frac={self.grad.clip_fraction:.1%} bottom/top={self.grad.bottom_top_ratio:.2f}",
            f"param: max_rms={self.param.max_rms:.3f} |dw|/|w|={self.param.final_update_ratio:.2e}",
            f"act: dead={self.activation.max_dead_ratio:.1%} rms={self.activation.max_rms:.2e} "
            f"attn_entropy={self.activation.attention_entropy:.2f}",
            f"stability: diverged={self.stability.diverged} spikes={self.stability.loss_spike_count} "
            f"final={self.stability.final_loss:.4f} best={self.stability.best_loss:.4f} "
            f"gap={self.stability.generalization_gap:+.4f}",
        ]
        for signal in self.signals[:limit]:
            lines.append(f"- {signal}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verdict rules — the "physics" that turns raw numbers into agent-readable state
# ---------------------------------------------------------------------------

EXPLODE_GRAD_P99 = 1e2
# bottom/top gradient ratio below this means the update signal is starving at the
# bottom of the stack. 0.01 = the first block receives <1% of the last block's grad.
VANISH_BOTTOM_TOP = 0.01
CLIP_FRACTION_HIGH = 0.25
# 35% silent units is already alarming; ReLU-family activations routinely pass 45%.
DEAD_RATIO_HIGH = 0.35
SPIKE_FACTOR = 1.5
# Absolute train/val gap that counts as overfitting on this task.
OVERFIT_GAP = 0.35
# How far above the project's best a run may end before we call it underfitting.
UNDERFIT_HEADROOM = 0.25
# Absolute validation accuracy below which a run is almost certainly underfitting
# (relative to the task's achievable ceiling). A character-level LM should clear
# ~0.3 accuracy easily; anything far below that means the model never learned.
UNDERFIT_ACC_ABS = 0.15


def _spike_count(losses: Sequence[float]) -> int:
    """Count steps where loss jumps up by more than SPIKE_FACTOR x recent level."""
    arr = np.asarray([x for x in losses if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size < 5:
        return 0
    baseline = np.minimum.accumulate(arr)  # running best
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = arr / np.maximum(baseline, 1e-12)
    return int(np.sum(ratio > SPIKE_FACTOR))


def build_diagnostics(
    *,
    steps: list[StepMetric],
    layer_stats: list[LayerStat] | None = None,
    grad_values: Sequence[float] | np.ndarray | None = None,
    param_values: Sequence[float] | np.ndarray | None = None,
    reference_loss: float | None = None,
) -> Diagnostics:
    """Aggregate a run's raw observations into one verdict object."""
    layer_stats = layer_stats or []
    finite_losses = [m.train_loss for m in steps if m.train_loss is not None and math.isfinite(m.train_loss)]
    val_losses = [m.val_loss for m in steps if m.val_loss is not None and math.isfinite(m.val_loss)]
    grad_norms = [m.grad_norm for m in steps if m.grad_norm is not None and math.isfinite(m.grad_norm)]

    nan_or_inf = bool(
        any(not math.isfinite(m.train_loss) for m in steps)
        or any(m.val_loss is not None and not math.isfinite(m.val_loss) for m in steps)
    )

    grad_arr = np.asarray([g for g in (grad_values if grad_values is not None else [])], dtype=np.float64)
    if grad_arr.size == 0 and grad_norms:
        grad_arr = np.asarray(grad_norms, dtype=np.float64)

    clip_fraction = (
        sum(1 for m in steps if m.clipped) / len(steps) if steps else 0.0
    )
    if grad_arr.size:
        finite = grad_arr[np.isfinite(grad_arr)]
        grad = GradientDiagnostics(
            mean=float(finite.mean()) if finite.size else 0.0,
            std=float(finite.std()) if finite.size else 0.0,
            p99=float(np.percentile(finite, 99)) if finite.size else 0.0,
            max=float(finite.max()) if finite.size else 0.0,
            clip_fraction=clip_fraction,
            hist=log_histogram(finite),
        )
    else:
        grad = GradientDiagnostics(clip_fraction=clip_fraction)

    if layer_stats:
        bottom = layer_stats[0].grad_rms
        top = layer_stats[-1].grad_rms
        grad.bottom_top_ratio = float(bottom / top) if top > 1e-20 else float("inf")

    grad.vanishing = (0 < grad.bottom_top_ratio < VANISH_BOTTOM_TOP) or (0 < grad.p99 < 1e-7)
    grad.exploding = grad.p99 > EXPLODE_GRAD_P99

    param_arr = np.asarray(
        [p for p in (param_values if param_values is not None else [])], dtype=np.float64
    )
    param = ParamDiagnostics()
    if layer_stats:
        param.max_rms = max((s.param_rms for s in layer_stats), default=0.0)
        param.final_update_ratio = max((s.update_ratio for s in layer_stats), default=0.0)
    if param_arr.size:
        finite = param_arr[np.isfinite(param_arr)]
        param.max_abs = float(np.abs(finite).max()) if finite.size else 0.0
        param.hist = histogram(finite, linear_bins(float(finite.min()), float(finite.max())))
        if not layer_stats:
            param.max_rms = safe_rms(finite)

    activation = ActivationDiagnostics()
    if layer_stats:
        activation.max_dead_ratio = max((s.dead_ratio for s in layer_stats), default=0.0)
        activation.max_rms = max((s.activation_rms for s in layer_stats), default=0.0)
        entropies = [s.attn_entropy for s in layer_stats if s.attn_entropy > 0]
        if entropies:
            activation.attention_entropy = float(np.mean(entropies))
            activation.attention_collapse = activation.attention_entropy < 0.35

    final_loss = finite_losses[-1] if finite_losses else float("inf")
    best_val = min(val_losses) if val_losses else final_loss
    final_acc = None
    val_accs = [m.val_acc for m in steps if m.val_acc is not None]
    if val_accs:
        final_acc = float(val_accs[-1])
    gap = 0.0
    if val_losses and finite_losses:
        gap = float(val_losses[-1] - finite_losses[-1])

    headroom = None
    if reference_loss is not None and math.isfinite(final_loss):
        headroom = float(final_loss - reference_loss)

    stability = StabilityDiagnostics(
        nan_or_inf=nan_or_inf,
        loss_spike_count=_spike_count(finite_losses),
        final_loss=float(final_loss),
        best_loss=float(best_val),
        final_val_acc=final_acc,
        generalization_gap=gap,
        reference_loss=reference_loss,
        headroom=headroom,
    )
    stability.diverged = nan_or_inf or (
        loss_growing(finite_losses) and stability.loss_spike_count > 0
    )

    diagnostics = Diagnostics(
        grad=grad, param=param, activation=activation, stability=stability
    )
    diagnostics.verdict = classify(diagnostics, finite_losses)
    diagnostics.signals = build_signals(diagnostics, layer_stats)
    return diagnostics


def loss_growing(losses: Sequence[float]) -> bool:
    """True when the tail of the curve sits clearly above the best point."""
    arr = np.asarray(losses, dtype=np.float64)
    if arr.size < 10:
        return False
    tail = arr[-max(3, arr.size // 10) :]
    return float(tail.mean()) > float(arr.min()) * 1.5 + 1e-6


def classify(d: Diagnostics, losses: Sequence[float]) -> str:
    """Ordered from most to least severe. The order itself encodes experience:

    a structural diagnosis (gradients cannot reach the bottom of the stack) is
    more actionable than a symptom (the loss wobbled), so it wins the tie.
    """
    if d.stability.nan_or_inf or d.stability.diverged:
        return "diverged"
    if d.grad.vanishing or (d.grad.max > 0 and d.grad.p99 < 1e-6):
        return "vanishing"
    if d.grad.exploding or d.grad.clip_fraction > CLIP_FRACTION_HIGH or d.stability.loss_spike_count >= 3:
        return "unstable"
    if d.activation.max_dead_ratio > DEAD_RATIO_HIGH:
        return "dying_units"
    if d.activation.attention_collapse:
        return "attention_collapse"
    if d.stability.generalization_gap > OVERFIT_GAP:
        return "overfitting"
    if d.stability.headroom is not None and d.stability.headroom > UNDERFIT_HEADROOM:
        return "underfitting"
    if d.stability.final_val_acc is not None and d.stability.final_val_acc < UNDERFIT_ACC_ABS:
        return "underfitting"
    return "healthy"


def build_signals(d: Diagnostics, layer_stats: list[LayerStat]) -> list[str]:
    """Human/agent-readable reasons behind the verdict. Ordered by severity."""
    signals: list[str] = []
    if d.stability.nan_or_inf:
        signals.append("loss became NaN/Inf — the update blew up, not just the metric")
    if d.grad.exploding:
        signals.append(
            f"gradient p99={d.grad.p99:.2e} exceeds {EXPLODE_GRAD_P99:.0e} — divergence risk"
        )
    if d.grad.vanishing:
        signals.append(
            f"bottom/top gradient ratio={d.grad.bottom_top_ratio:.2e} — gradient starves at the bottom of the stack"
        )
    if d.grad.clip_fraction > CLIP_FRACTION_HIGH:
        signals.append(
            f"{d.grad.clip_fraction:.0%} of steps hit gradient clipping — effective lr is set by the clip, not by schedule"
        )
    if d.activation.max_dead_ratio > DEAD_RATIO_HIGH:
        signals.append(
            f"{d.activation.max_dead_ratio:.0%} of activations are silent — consider a smoother activation or lower lr"
        )
    if d.activation.attention_collapse:
        signals.append(
            f"attention entropy collapsed to {d.activation.attention_entropy:.2f} nats — attention is near one-hot"
        )
    if d.stability.loss_spike_count >= 1:
        signals.append(f"{d.stability.loss_spike_count} loss spike(s) detected (>1.5x running best)")
    if d.param.final_update_ratio > 0 and d.param.final_update_ratio < 1e-4:
        signals.append(
            f"|Δw|/|w|={d.param.final_update_ratio:.2e} — weights are barely moving, lr may be too small"
        )
    if d.param.final_update_ratio > 1e-1:
        signals.append(
            f"|Δw|/|w|={d.param.final_update_ratio:.2e} — updates are large relative to weights, lr may be too high"
        )
    if d.stability.generalization_gap > OVERFIT_GAP:
        signals.append(
            f"train/val gap={d.stability.generalization_gap:+.3f} — overfitting, consider regularization or more data"
        )
    if d.stability.headroom is not None and d.stability.headroom > UNDERFIT_HEADROOM:
        signals.append(
            f"ended {d.stability.headroom:+.3f} above the project best ({d.stability.reference_loss:.4f}) "
            f"— this configuration is not learning as well as what we already have"
        )
    if (
        d.stability.final_val_acc is not None
        and d.stability.final_val_acc < UNDERFIT_ACC_ABS
        and (d.stability.headroom is None or d.stability.headroom <= UNDERFIT_HEADROOM)
    ):
        signals.append(
            f"final val accuracy={d.stability.final_val_acc:.3f} is far below the task ceiling "
            f"— the model never learned (lr too small? capacity too low?)"
        )
    if layer_stats:
        worst = max(layer_stats, key=lambda s: s.grad_rms, default=None)
        quietest = min((s for s in layer_stats if s.grad_rms > 0), key=lambda s: s.grad_rms, default=None)
        # Only worth reporting when the spread is an order of magnitude or more —
        # otherwise it is just per-layer noise.
        if (
            worst is not None
            and quietest is not None
            and worst is not quietest
            and quietest.grad_rms > 0
            and worst.grad_rms / quietest.grad_rms >= 10.0
        ):
            signals.append(
                f"gradient spread across layers: {quietest.name}={quietest.grad_rms:.2e} .. {worst.name}={worst.grad_rms:.2e}"
            )
    if not signals:
        signals.append("no pathological signal detected — this configuration is a usable baseline")
    return signals


def iter_layer_names(n_layer: int) -> Iterable[str]:
    return (f"block{i}" for i in range(n_layer))


# ---------------------------------------------------------------------------
# Incremental early-stop signal (docs/08 P1-3)
# ---------------------------------------------------------------------------
#
# `build_diagnostics` runs once, at the end of a trial. But training is long: a
# run that is clearly exploding (gradient p99 climbing, most units silent) should
# NOT be allowed to burn its whole step/time budget before being flagged. This
# function is the cheap per-probe version — it inspects the most recent snapshot
# and answers "should this run be cut short now?", returning a reason or None.
#
# It deliberately reuses the same thresholds as `classify` so the early-stop
# verdict matches what the final diagnostics would have concluded anyway.

# Consecutive probe ticks a pathology must persist before we stop the run.
# One transient spike is normal; `N` in a row means it is structural.
_EARLY_STOP_PERSIST = 3


def incremental_early_stop(
    layer_stats: list[LayerStat],
    grad_norm: float,
    *,
    persist_counter: int = 0,
) -> tuple[bool, str | None]:
    """Return (should_stop, reason) given the latest probe snapshot.

    Only considers *structural* pathologies — gradient explosion, dead units,
    attention collapse — where continuing is pure waste. Vanishing is excluded
    because a barely-moving model may still be rescued by a lr change next run
    but is not "burning GPU" the way an exploding one is.
    """
    if not math.isfinite(grad_norm):
        return True, "grad_norm non-finite"
    if grad_norm > EXPLODE_GRAD_P99 * 10:
        return True, "gradient exploding"

    if layer_stats:
        dead = max((s.dead_ratio for s in layer_stats), default=0.0)
        entropy = min((s.attn_entropy for s in layer_stats if s.attn_entropy > 0), default=1.0)
        if dead > DEAD_RATIO_HIGH + 0.2:  # >55% silent is definitively stuck
            return True, "activation units dying"
        if entropy < 0.35 and entropy > 0:
            return True, "attention collapsed"

    return False, None
