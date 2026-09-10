"""Test intent and reason lease protocol with claim/heartbeat/release/conclude."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mts.server.app import create_app
from mts.server.models import ProjectCreate


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path, db_path=tmp_path / "board.db")
    return TestClient(app)


@pytest.fixture
def project(client):
    """Create a project and return its ID."""
    response = client.post("/api/projects", json={
        "title": "Test Project",
        "origin": "Test a model training pipeline",
        "goal": "Achieve 90% validation accuracy",
        "goal_metric": "val_acc",
        "goal_direction": "maximize",
        "budget_max_trials": 10,
    })
    assert response.status_code == 200
    return response.json()["id"]


# =====================================================================
# Intent lease tests
# =====================================================================

def test_intent_claim_success(client, project):
    """Worker can claim an unclaimed intent."""
    # Create an intent
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    assert response.status_code == 200
    intent_id = response.json()["id"]

    # Claim the intent
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/claim",
        json={"worker": "w1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["worker"] == "w1"
    assert data["last_heartbeat_at"] is not None


def test_intent_claim_conflict(client, project):
    """Second worker cannot claim already-claimed intent."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    # First worker claims
    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    # Second worker cannot claim
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/claim",
        json={"worker": "w2"}
    )
    assert response.status_code == 409


def test_intent_heartbeat_success(client, project):
    """Worker holding lease can send heartbeat."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/heartbeat",
        json={"worker": "w1"}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_intent_heartbeat_wrong_worker(client, project):
    """Worker not holding lease cannot send heartbeat."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/heartbeat",
        json={"worker": "w2"}
    )
    assert response.status_code == 403


def test_intent_release_success(client, project):
    """Worker can release a held lease."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/release",
        json={"worker": "w1"}
    )
    assert response.status_code == 200

    # After release, another worker can claim
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/claim",
        json={"worker": "w2"}
    )
    assert response.status_code == 200


def test_intent_release_wrong_worker(client, project):
    """Worker not holding lease cannot release."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/release",
        json={"worker": "w2"}
    )
    assert response.status_code == 403


def test_intent_conclude_creates_fact(client, project):
    """Conclude creates a fact and concludes the intent."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/conclude",
        json={
            "worker": "w1",
            "description": "Baseline achieved 85% accuracy",
            "metrics": {"val_acc": 0.85, "val_loss": 0.32},
            "trial_id": "t001"
        }
    )
    assert response.status_code == 200
    data = response.json()

    # Check fact was created
    assert data["fact"]["description"] == "Baseline achieved 85% accuracy"
    assert data["fact"]["metrics"]["val_acc"] == 0.85
    assert data["fact"]["trial_id"] == "t001"

    # Check intent was concluded
    assert data["intent"]["to"] is not None
    assert data["intent"]["concluded_at"] is not None


def test_intent_conclude_empty_metrics_allowed(client, project):
    """Conclude can omit metrics."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/conclude",
        json={
            "worker": "w1",
            "description": "Observation only",
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["fact"]["metrics"] == {}


def test_intent_lazy_expiry(client, project, monkeypatch):
    """Intent lease expires after timeout if no heartbeat (lazy check)."""
    # Set a tiny timeout for testing
    response = client.put("/api/settings", json={"intent_timeout": 1})
    assert response.status_code == 200

    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    # Wait for lease to expire
    time.sleep(1.5)

    # Heartbeat from original worker should fail (lease expired)
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/heartbeat",
        json={"worker": "w1"}
    )
    assert response.status_code == 403

    # Another worker can now claim
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/claim",
        json={"worker": "w2"}
    )
    assert response.status_code == 200


# =====================================================================
# Reason lease tests
# =====================================================================

def test_reason_claim_success(client, project):
    """Worker can claim reason lease."""
    response = client.post(
        f"/api/projects/{project}/reason/claim",
        json={"worker": "w1", "trigger": "initial"}
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_reason_claim_conflict(client, project):
    """Second worker cannot claim already-held reason lease."""
    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    response = client.post(
        f"/api/projects/{project}/reason/claim",
        json={"worker": "w2", "trigger": "graph_changed"}
    )
    assert response.status_code == 409


def test_reason_heartbeat_success(client, project):
    """Worker holding reason lease can send heartbeat."""
    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    response = client.post(
        f"/api/projects/{project}/reason/heartbeat",
        json={"worker": "w1"}
    )
    assert response.status_code == 200


def test_reason_heartbeat_wrong_worker(client, project):
    """Worker not holding reason lease cannot heartbeat."""
    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    response = client.post(
        f"/api/projects/{project}/reason/heartbeat",
        json={"worker": "w2"}
    )
    assert response.status_code == 403


def test_reason_release_success(client, project):
    """Worker can release reason lease."""
    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    response = client.post(
        f"/api/projects/{project}/reason/release",
        json={"worker": "w1"}
    )
    assert response.status_code == 200

    # After release, another worker can claim
    response = client.post(
        f"/api/projects/{project}/reason/claim",
        json={"worker": "w2", "trigger": "graph_changed"}
    )
    assert response.status_code == 200


def test_reason_release_wrong_worker(client, project):
    """Worker not holding reason lease cannot release."""
    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    response = client.post(
        f"/api/projects/{project}/reason/release",
        json={"worker": "w2"}
    )
    assert response.status_code == 403


def test_reason_lazy_expiry(client, project):
    """Reason lease expires after timeout if no heartbeat."""
    # Set a tiny timeout
    client.put("/api/settings", json={"reason_timeout": 1})

    client.post(f"/api/projects/{project}/reason/claim", json={"worker": "w1", "trigger": "initial"})

    # Wait for expiry
    time.sleep(1.5)

    # Heartbeat should fail
    response = client.post(
        f"/api/projects/{project}/reason/heartbeat",
        json={"worker": "w1"}
    )
    assert response.status_code == 403

    # Another worker can claim
    response = client.post(
        f"/api/projects/{project}/reason/claim",
        json={"worker": "w2", "trigger": "graph_changed"}
    )
    assert response.status_code == 200


def test_inactive_project_rejects_intent_operations(client, project):
    """Operations on intents fail when project is not active."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    # Stop the project
    client.patch(f"/api/projects/{project}/status", params={"status": "stopped"})

    # Claim should fail with 403
    response = client.post(
        f"/api/projects/{project}/intents/{intent_id}/claim",
        json={"worker": "w1"}
    )
    assert response.status_code == 403


def test_inactive_project_rejects_reason_operations(client, project):
    """Reason operations fail when project is not active."""
    # Stop the project
    client.patch(f"/api/projects/{project}/status", params={"status": "stopped"})

    # Claim should fail with 403
    response = client.post(
        f"/api/projects/{project}/reason/claim",
        json={"worker": "w1", "trigger": "initial"}
    )
    assert response.status_code == 403
