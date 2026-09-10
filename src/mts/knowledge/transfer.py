"""Cross-project knowledge transfer (docs/10 §5, P4).

Within a single project, `KnowledgeStore.match` works because a trigger's spec
paths (`arch.norm_position = post`) mean the same thing. Across projects — or
across different model scales — a raw value like `arch.n_layer = 12` or
`warmup_steps = 100` is NOT portable: what's "deep" for a 4-layer baseline is
"shallow" for a 32-layer one.

This module abstracts a lesson into a *dimensionless pathology signature* so an
experience learned on project A can be scored against project B by similarity of
the underlying failure mode, not by literal spec equality. Three pieces:

  1. `PathologySignature` — a normalized, scale-free description of the failure
     (verdict + dimensionless feature flags: relative depth, warmup ratio, etc.).
  2. `transfer_match` — scores a lesson against a target situation by signature
     similarity, decaying confidence by how few projects have validated it.
  3. `prior_prompt_block` — renders the top transferred priors as a bounded
     prompt fragment to inject into Reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mts.knowledge.store import Lesson


# ---------------------------------------------------------------------------
# Dimensionless pathology signature
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PathologySignature:
    """A scale-free fingerprint of a failure mode.

    Each flag is a normalized qualitative bucket rather than a raw spec value,
    so signatures are comparable across model scales / projects.
    """

    verdict: str = "unknown"
    # dimensionless qualitative buckets (None = not specified / not relevant)
    depth: str | None = None       # "shallow" | "deep"  (relative to a project baseline)
    norm: str | None = None        # "pre" | "post" | "sandwich" | "none"
    warmup_ratio: str | None = None  # "low" | "medium" | "high"  (warmup/max_steps)
    lr: str | None = None          # "low" | "mid" | "high" (relative scale)
    activation: str | None = None  # "smooth" | "relu-like"

    def to_dict(self) -> dict[str, str]:
        out = {"verdict": self.verdict}
        for k in ("depth", "norm", "warmup_ratio", "lr", "activation"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

    def similarity(self, other: "PathologySignature") -> float:
        """0..1 overlap of the two signatures' specified flags.

        A flag only counts when BOTH signatures specify it (a lesson that says
        nothing about `depth` neither helps nor hurts a match on depth).
        """
        a, b = self.to_dict(), other.to_dict()
        common = [k for k in a if k in b]
        if not common:
            return 0.0
        hits = sum(1 for k in common if a[k] == b[k])
        return hits / len(common)


# ---------------------------------------------------------------------------
# Signature derivation from a concrete situation
# ---------------------------------------------------------------------------

def signature_from_spec(verdict: str, spec: dict[str, Any]) -> PathologySignature:
    """Derive a dimensionless signature from a verdict + spec dict.

    The raw spec values are bucketed into qualitative flags so they are portable
    across projects. `depth` uses `n_layer` against a coarse log-scale split;
    `warmup_ratio` normalizes warmup by max_steps.
    """
    sig = PathologySignature(verdict=verdict)

    arch = spec.get("arch", {})
    optim = spec.get("optim", {})
    sched = optim.get("schedule", {})
    budget = spec.get("budget", {})

    n_layer = arch.get("n_layer")
    if isinstance(n_layer, int):
        sig.depth = "deep" if n_layer >= 12 else ("shallow" if n_layer <= 4 else "mid")

    norm = arch.get("norm_position")
    if isinstance(norm, str) and norm in ("pre", "post", "sandwich", "none"):
        sig.norm = norm

    warmup = sched.get("warmup_steps")
    max_steps = budget.get("max_steps")
    if isinstance(warmup, int) and isinstance(max_steps, int) and max_steps > 0:
        ratio = warmup / max_steps
        sig.warmup_ratio = "high" if ratio >= 0.2 else ("low" if ratio <= 0.05 else "medium")

    lr = optim.get("lr")
    if isinstance(lr, (int, float)):
        sig.lr = "high" if lr >= 1e-2 else ("low" if lr <= 1e-4 else "mid")

    act = arch.get("activation")
    if isinstance(act, str):
        sig.activation = "smooth" if act in ("gelu", "silu", "swiglu") else "relu-like"

    return sig


# ---------------------------------------------------------------------------
# Cross-project transfer matching
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ScoredPrior:
    lesson: Lesson
    score: float          # similarity (0..1)
    decayed_confidence: float  # confidence after validation-count decay


def transfer_match(
    lessons: list[Lesson],
    *,
    verdict: str,
    spec: dict[str, Any],
    top_k: int = 5,
) -> list[ScoredPrior]:
    """Score lessons against a target situation by dimensionless signature.

    A lesson's signature is read from its `trigger` (which may carry the raw
    spec paths) and/or its `tags`; the target signature is derived from `spec`.
    Confidence is decayed by how few distinct validations the lesson records.
    """
    target = signature_from_spec(verdict, spec)
    scored: list[ScoredPrior] = []
    for lesson in lessons:
        lesson_sig = _signature_from_lesson(lesson)
        sim = lesson_sig.similarity(target)
        if sim <= 0:
            continue
        validations = _validation_count(lesson)
        # Confidence decays with fewer validations: a lesson seen once is a
        # weaker prior than one confirmed across several runs.
        decay = min(1.0, validations / 3.0)
        decayed = lesson.confidence * (0.3 + 0.7 * decay)
        scored.append(ScoredPrior(lesson=lesson, score=sim, decayed_confidence=decayed))
    scored.sort(key=lambda s: (-s.score, -s.decayed_confidence))
    return scored[:top_k]


def _signature_from_lesson(lesson: Lesson) -> PathologySignature:
    sig = PathologySignature(verdict=lesson.trigger.get("verdict", "unknown")
                             if isinstance(lesson.trigger, dict) else "unknown")
    trigger = lesson.trigger if isinstance(lesson.trigger, dict) else {}
    # Map raw trigger paths onto dimensionless flags where present.
    if "arch.norm_position" in trigger:
        sig.norm = str(trigger["arch.norm_position"])
    if "arch.activation" in trigger:
        act = str(trigger["arch.activation"])
        sig.activation = "smooth" if act in ("gelu", "silu", "swiglu") else "relu-like"
    # Fall back to tags for coarse buckets.
    tags = set(lesson.tags or [])
    if sig.norm is None and "postnorm" in tags:
        sig.norm = "post"
    if sig.norm is None and "prenorm" in tags:
        sig.norm = "pre"
    return sig


def _validation_count(lesson: Lesson) -> int:
    """Number of distinct evidence lines (a proxy for cross-run validation)."""
    return len(lesson.evidence or [])


# ---------------------------------------------------------------------------
# Prior injection for Reason
# ---------------------------------------------------------------------------

def prior_prompt_block(priors: list[ScoredPrior], max_chars: int = 2000) -> str:
    """Render transferred priors as a compact prompt fragment (bounded)."""
    if not priors:
        return ""
    lines = ["## 跨项目先验（来自其他项目/模型的调参经验，仅供参考）"]
    for p in priors:
        lines.append(f"- [{p.decayed_confidence:.2f}] {p.lesson.title} "
                     f"(相似度 {p.score:.2f})")
        if p.lesson.advice:
            advice = p.lesson.advice.strip().replace("\n", " ")[:120]
            lines.append(f"    → {advice}")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n…"
    return block
