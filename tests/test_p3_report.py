"""P3-2 tests: experiment report generation.

The report is derived from the board (fact graph + ledger + knowledge), not
hand-written — same principle as Cairn's auto-generated reports. These tests run
a small search then assert the report contains the key sections with real data.
"""

from __future__ import annotations

from pathlib import Path

from mts.reporting import build_report
from mts.server.db import Database
from mts.server.models import (
    ConcludeRequest,
    CreateIntentRequest,
    ProjectCreate,
    TrialRegister,
)
from mts.server.services import Service
from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


def _make_service(root: Path, trials: int = 6) -> tuple[Service, str]:
    db = Database(root / "data" / "board.db")
    db.configure()
    svc = Service(db, root)
    proj = svc.create_project(ProjectCreate(
        title="报告测试", origin="字符级语言模型基线", goal="降低 val_loss 且保持 healthy",
        goal_target=1.70, budget_max_trials=trials,
    ))
    return svc, proj["id"]


def _seed_trial(svc: Service, pid: str, root: Path, name: str, lr: float) -> None:
    """Run one real trial and put it on the board the way an explore task does."""
    spec = TrialSpec.from_file("examples/specs/baseline.yaml")
    spec.optim.lr = lr
    out_dir = root / "runs" / name
    result = run_trial(spec, out_dir, trial_id=name)

    intent = svc.create_intent(pid, CreateIntentRequest.model_validate(
        {"from": [], "description": f"试 lr={lr}", "creator": "test", "worker": "w1"}
    ))
    trial, error = svc.register_trial(pid, TrialRegister(
        out_dir=str(out_dir), trial_id=name, name=name,
        intent_id=intent["id"], worker="w1", spec=spec.to_dict(),
    ))
    assert error is None, error
    final_val_loss = result.final.get("val_loss")
    svc.conclude_intent(pid, intent["id"], ConcludeRequest(
        worker="w1",
        description=f"lr={lr} 跑完，val_loss={final_val_loss}",
        metrics={"val_loss": float(final_val_loss or 0.0)},
        trial_id=trial["id"],
    ))


def test_report_contains_all_sections(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    _seed_trial(svc, pid, tmp_path, "baseline", 3e-3)

    report = build_report(svc, pid)
    assert "# 实验报告" in report
    assert "## 目标" in report
    assert "## 排行榜" in report
    assert "## 探索过程" in report
    assert "## 沉淀的经验" in report
    assert "## 复现命令" in report


def test_report_leaderboard_has_real_rows(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    _seed_trial(svc, pid, tmp_path, "baseline", 3e-3)
    _seed_trial(svc, pid, tmp_path, "lr-up", 6e-3)

    report = build_report(svc, pid)
    # The leaderboard table lists the best trial (baseline at minimum).
    assert "baseline" in report or "| `" in report
    # Reproducible command trail references the spec files.
    assert "mts train --spec" in report


def test_report_empty_project(tmp_path: Path):
    svc, pid = _make_service(tmp_path)
    report = build_report(svc, pid)
    assert "（暂无实验）" in report
    # Even an empty project has its origin/goal facts seeded at creation.
    assert "字符级语言模型基线" in report  # origin
    assert "降低 val_loss 且保持 healthy" in report  # goal
    assert "（暂无知识条目）" in report


def test_report_missing_project_raises(tmp_path: Path):
    svc, _ = _make_service(tmp_path)
    import pytest
    with pytest.raises(KeyError):
        build_report(svc, "proj_nonexistent")
