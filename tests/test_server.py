"""Unit tests for the board server (Service layer) and the search loop."""

from __future__ import annotations

import pytest

from mts.server.db import Database
from mts.server.models import (
    ConcludeRequest,
    CreateIntentRequest,
    ProjectCreate,
    TrialRegister,
)
from mts.server.services import Service, spec_fingerprint
from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


@pytest.fixture
def service(tmp_path):
    db = Database(tmp_path / "data" / "board.db")
    db.configure()
    return Service(db, tmp_path)


def _make_project(service, **kw):
    """Create a minimal project. Origin/goal are prose: the board holds no spec."""
    return service.create_project(ProjectCreate(
        title="t", origin="o", goal="g", **kw
    ))


def test_create_project_seeds_facts(service):
    proj = _make_project(service)
    facts = service.list_facts(proj["id"])
    ids = {f["id"] for f in facts}
    assert {"origin", "goal"} <= ids


def test_fingerprint_ignores_seed(service):
    a = TrialSpec(seed=1).to_dict()
    b = TrialSpec(seed=2).to_dict()
    assert spec_fingerprint(a) == spec_fingerprint(b)


def test_register_trial_and_leaderboard(service):
    """register_trial stores a trial in the trials table."""
    proj = _make_project(service)
    tid = service.next_trial_id(proj["id"])
    out = service.root / "runs" / tid
    spec = TrialSpec()
    result = run_trial(spec, out, trial_id=tid)
    trial, err = service.register_trial(proj["id"], TrialRegister(
        out_dir=str(out), trial_id=tid, status=result.status,
        backend=result.backend, spec=result.spec,
        best_val_loss=result.best.get("val_loss"),
    ))
    assert err is None
    assert trial["id"] == tid
    # Verify the trial is retrievable
    retrieved = service.get_trial(tid)
    assert retrieved is not None
    assert retrieved["id"] == tid


def test_register_duplicate_fingerprint(service):
    proj = _make_project(service)
    spec = TrialSpec()
    out1 = service.root / "runs" / "t001"
    run_trial(spec, out1, trial_id="t001")
    service.register_trial(proj["id"], TrialRegister(
        out_dir=str(out1), trial_id="t001", spec=spec.to_dict()))
    # Second identical spec (different trial id) must collide.
    out2 = service.root / "runs" / "t002"
    run_trial(spec, out2, trial_id="t002")
    _, err = service.register_trial(proj["id"], TrialRegister(
        out_dir=str(out2), trial_id="t002", spec=spec.to_dict()))
    assert err is not None and err.startswith("duplicate:")


def test_intent_lifecycle(service):
    """An intent moves origin -> new fact through claim / lease / conclude."""
    proj = _make_project(service)
    intent = service.create_intent(proj["id"], CreateIntentRequest.model_validate({
        "from": ["origin"],
        "description": "try post-norm",
        "creator": "agent",
    }))
    # Unclaimed intents carry no worker and no outcome fact yet.
    assert intent["worker"] is None
    assert intent["to"] is None
    assert intent["from"] == ["origin"]

    claimed = service.claim_intent(proj["id"], intent["id"], "w1")
    assert claimed["worker"] == "w1"
    assert claimed["last_heartbeat_at"] is not None

    # Another worker cannot steal the lease.
    assert service.claim_intent(proj["id"], intent["id"], "w2") is None

    concluded = service.conclude_intent(proj["id"], intent["id"], ConcludeRequest(
        worker="w1",
        description="post-norm diverged",
        metrics={"val_loss": 4.2},
    ))
    assert concluded["fact"]["id"].startswith("f")
    assert concluded["fact"]["metrics"] == {"val_loss": 4.2}
    # Concluding points the intent at the fact it produced.
    assert concluded["intent"]["to"] == concluded["fact"]["id"]
    assert concluded["intent"]["concluded_at"] is not None
