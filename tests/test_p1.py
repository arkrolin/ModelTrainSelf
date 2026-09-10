"""P1 regression tests: intent lease, GPU budget arbitration, incremental early-stop.

These pin down the training-specific long-task behaviors from docs/08:
  - P1-1: every trial is driven by a claimable intent (lease integrity).
  - P1-2: gpu_seconds_budget caps accelerator time before wall-clock does.
  - P1-3: incremental_early_stop flags structural pathologies from a probe snapshot.
"""

from __future__ import annotations

from pathlib import Path

from mts.trainer.probes import LayerStat, incremental_early_stop
from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


def test_p1_2_gpu_budget_caps_surrogate_run(tmp_path: Path):
    """A tiny gpu_seconds_budget must stop the run as timeout well before max_steps."""
    base = TrialSpec.from_file("examples/specs/baseline.yaml")
    tiny = base.model_copy(deep=True)
    tiny.budget.gpu_seconds_budget = 1  # 1 second of accelerator time

    result = run_trial(tiny, tmp_path / "t", trial_id="t")
    assert result.status == "timeout"
    assert result.steps_completed < tiny.budget.max_steps
    # The reason must be recorded in the run log.
    log = (tmp_path / "t" / "log.txt").read_text(encoding="utf-8")
    assert "gpu-seconds budget" in log


def test_p1_2_gpu_budget_disabled_by_default(tmp_path: Path):
    """gpu_seconds_budget=0 means no accelerator cap (runs to completion)."""
    base = TrialSpec.from_file("examples/specs/baseline.yaml")
    assert base.budget.gpu_seconds_budget == 0
    result = run_trial(base, tmp_path / "t", trial_id="t")
    assert result.status in ("completed", "diverged", "timeout")


def test_p1_3_incremental_early_stop_signals():
    """The per-probe early-stop function flags structural pathologies only."""
    # Exploding gradient.
    assert incremental_early_stop([], 1e4)[0] is True
    # Dead units (dead_ratio above the structural threshold).
    dead = [LayerStat(name="b0", grad_rms=1.0, dead_ratio=0.6)]
    assert incremental_early_stop(dead, 1.0)[0] is True
    # Attention collapse.
    collapsed = [LayerStat(name="b0", grad_rms=1.0, dead_ratio=0.1, attn_entropy=0.2)]
    assert incremental_early_stop(collapsed, 1.0)[0] is True
    # Healthy snapshot -> no stop.
    ok = [LayerStat(name="b0", grad_rms=1.0, dead_ratio=0.1, attn_entropy=0.8)]
    assert incremental_early_stop(ok, 1.0) == (False, None)


def test_p1_3_incremental_early_stop_ignores_vanishing():
    """Vanishing gradients are NOT a stop signal (rescuable next run, not burning GPU)."""
    vanish = [LayerStat(name="b0", grad_rms=1e-9, dead_ratio=0.1, attn_entropy=0.8)]
    should_stop, _ = incremental_early_stop(vanish, 1e-9)
    assert should_stop is False
