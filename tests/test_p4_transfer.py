"""P4 tests: cross-project knowledge transfer.

These pin down docs/10 §5 — the ability to carry a lesson learned on one
project/model to another by dimensionless pathology signature rather than raw
spec equality. This is MTS's biggest lever over single-project tuning scripts.
"""

from __future__ import annotations

from mts.knowledge.store import Lesson
from mts.knowledge.transfer import (
    PathologySignature,
    prior_prompt_block,
    signature_from_spec,
    transfer_match,
)


def test_signature_similarity_exact_and_disjoint():
    a = PathologySignature(verdict="diverged", norm="post", depth="deep")
    b = PathologySignature(verdict="diverged", norm="post", depth="deep")
    assert a.similarity(b) == 1.0
    c = PathologySignature(verdict="healthy", norm="pre", depth="shallow")
    assert a.similarity(c) == 0.0


def test_signature_similarity_ignores_unspecified():
    # A lesson that only specifies norm should not be penalized for depth.
    a = PathologySignature(verdict="diverged", norm="post")
    b = PathologySignature(verdict="diverged", norm="post", depth="deep", lr="high")
    assert a.similarity(b) == 1.0  # only `verdict`+`norm` are common


def test_signature_from_spec_buckets_raw_values():
    spec = {
        "arch": {"n_layer": 16, "norm_position": "post", "activation": "gelu"},
        "optim": {"lr": 3e-4, "schedule": {"warmup_steps": 10}},
        "budget": {"max_steps": 300},
    }
    sig = signature_from_spec("diverged", spec)
    assert sig.verdict == "diverged"
    assert sig.depth == "deep"        # 16 >= 12
    assert sig.norm == "post"
    assert sig.warmup_ratio == "low"  # 10/300 = 0.033 <= 0.05
    assert sig.lr == "mid"            # 3e-4 in (1e-4, 1e-2)
    assert sig.activation == "smooth"


def test_transfer_match_scores_by_similarity():
    # A lesson about post-norm divergence learned on a deep model.
    lesson = Lesson(
        slug="norm/postnorm-unstable",
        title="post-norm 在深层不稳定",
        tags=["postnorm", "stability"],
        confidence=0.8,
        trigger={"verdict": "diverged", "arch.norm_position": "post"},
        advice="换回 pre-norm 或加 residual scale",
        evidence=["t002", "t007", "t011"],  # validated 3x
    )
    # Target: a DIFFERENT project, deep post-norm, diverging.
    target_spec = {
        "arch": {"n_layer": 24, "norm_position": "post", "activation": "gelu"},
        "optim": {"lr": 3e-3, "schedule": {"warmup_steps": 30}},
        "budget": {"max_steps": 500},
    }
    priors = transfer_match([lesson], verdict="diverged", spec=target_spec)
    assert len(priors) == 1
    assert priors[0].score > 0.5
    # 3 validations → full confidence retained.
    assert priors[0].decayed_confidence > 0.7


def test_transfer_match_decays_low_validation_confidence():
    lesson = Lesson(
        slug="x", title="仅验证过一次",
        tags=[], confidence=0.9,
        trigger={"verdict": "unstable", "arch.norm_position": "post"},
        evidence=["t001"],  # only 1 validation
    )
    target = {
        "arch": {"n_layer": 16, "norm_position": "post"},
        "optim": {"lr": 1e-3, "schedule": {"warmup_steps": 50}},
        "budget": {"max_steps": 300},
    }
    priors = transfer_match([lesson], verdict="unstable", spec=target)
    assert len(priors) == 1
    # Confidence is decayed for a single validation.
    assert priors[0].decayed_confidence < lesson.confidence


def test_transfer_match_skips_irrelevant_lessons():
    lesson = Lesson(
        slug="healthy-capacity", title="健康时加大宽度",
        tags=[], confidence=0.8,
        trigger={"verdict": "healthy"},
    )
    target = {
        "arch": {"n_layer": 8, "norm_position": "pre"},
        "optim": {"lr": 3e-3, "schedule": {"warmup_steps": 50}},
        "budget": {"max_steps": 300},
    }
    # Target is unstable, not healthy → no overlap on verdict.
    priors = transfer_match([lesson], verdict="unstable", spec=target)
    assert priors == []


def test_prior_prompt_block_bounded():
    lessons = [Lesson(slug=f"l{i}", title=f"经验 {i}", confidence=0.7,
                      trigger={"verdict": "diverged"},
                      advice="降低 lr " + "x" * 200) for i in range(10)]
    spec = {"arch": {}, "optim": {}, "budget": {}}
    priors = transfer_match(lessons, verdict="diverged", spec=spec, top_k=10)
    block = prior_prompt_block(priors, max_chars=2000)
    assert len(block) <= 2000
    assert "跨项目先验" in block
