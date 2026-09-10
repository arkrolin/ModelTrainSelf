"""Tests for the trial state machine (heartbeat / claim / interrupt / stale).

This is the training-specific departure from Cairn's lease model: a running
trial reports *progress* heartbeats, and an interrupted trial is *resumable*
rather than abandoned. See docs/08.
"""

from __future__ import annotations

from pathlib import Path

from mts.server.db import Database
from mts.server.services import Service
from mts.server.models import ProjectCreate
from mts.trainer.spec import TrialSpec


def _make_service(root: Path) -> tuple[Service, str]:
    db = Database(root / "data" / "board.db")
    db.configure()
    svc = Service(db, root)
    base = TrialSpec.from_file("examples/specs/baseline.yaml")
    proj = svc.create_project(ProjectCreate(
        title="t", origin="o", goal="g", base_spec=base.to_dict(),
    ))
    return svc, proj["id"]


def _insert_running_trial(db: Database, pid: str, tid: str = "t999") -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO trials(id, project_id, name, backend, fingerprint, "
            "spec_path, out_dir, status, created_at) "
            "VALUES(?, ?, 'test', 'surrogate', 'fp', '/x/spec.json', '/x', "
            "'running', '2026-01-01T00:00:00+00:00')",
            (tid, pid),
        )


def test_running_heartbeat_interrupt_cycle(tmp_path: Path) -> None:
    svc, pid = _make_service(tmp_path)
    _insert_running_trial(svc.db, pid)

    assert svc.mark_trial_running("t999", "worker-x") is True
    assert svc.heartbeat_trial(
        "t999", "worker-x",
        {"step": 100, "max_steps": 300, "eta_sec": 42.0},
    ) is True

    t = svc.get_trial("t999")
    assert t["status"] == "running"
    assert t["progress_step"] == 100
    assert t["progress_max_steps"] == 300
    assert t["eta_sec"] == 42.0

    # A different worker must not be able to heartbeat someone else's trial.
    assert svc.heartbeat_trial("t999", "worker-y", {"step": 1}) is False

    assert svc.mark_trial_interrupted("t999") is True
    assert svc.get_trial("t999")["status"] == "interrupted"


def test_stale_running_trial_detection(tmp_path: Path) -> None:
    svc, pid = _make_service(tmp_path)
    _insert_running_trial(svc.db, pid)

    # Fresh heartbeat -> not stale.
    svc.mark_trial_running("t999", "worker-x")
    svc.heartbeat_trial("t999", "worker-x", {"step": 1})
    assert svc.list_stale_running_trials(timeout_sec=300) == []

    # Aged heartbeat -> stale (but still resumable, not deleted).
    with svc.db.connect() as conn:
        conn.execute(
            "UPDATE trials SET last_heartbeat_at='2000-01-01T00:00:00+00:00' WHERE id='t999'"
        )
    assert svc.list_stale_running_trials(timeout_sec=300) == ["t999"]


def test_terminal_status_cannot_regress(tmp_path: Path) -> None:
    svc, pid = _make_service(tmp_path)
    _insert_running_trial(svc.db, pid)
    # Move to a terminal state, then try to mark running again.
    with svc.db.connect() as conn:
        conn.execute("UPDATE trials SET status='completed' WHERE id='t999'")
    assert svc.mark_trial_running("t999", "worker-x") is False
    assert svc.mark_trial_interrupted("t999") is False
