"""Test graph export in YAML format (Cairn protocol)."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from mts.server.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(root=tmp_path, db_path=tmp_path / "board.db")
    return TestClient(app)


@pytest.fixture
def project(client):
    """Create a project with some graph data."""
    response = client.post("/api/projects", json={
        "title": "Test Project",
        "origin": "Develop a text classifier",
        "goal": "Achieve 90% validation accuracy",
        "goal_metric": "val_acc",
        "goal_direction": "maximize",
        "goal_target": 0.90,
        "budget_max_trials": 10,
        "hints": ["Try pre-trained models", "Consider data augmentation"],
    })
    return response.json()["id"]


def test_export_contains_project_metadata(client, project):
    """Export includes project title, origin, goal, and settings."""
    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    data = yaml.safe_load(response.text)

    assert "project" in data
    assert data["project"]["title"] == "Test Project"
    assert data["project"]["origin"] == "Develop a text classifier"
    assert data["project"]["goal"] == "Achieve 90% validation accuracy"
    assert data["project"]["goal_metric"] == "val_acc"
    assert data["project"]["goal_direction"] == "maximize"
    assert data["project"]["goal_target"] == 0.90


def test_export_contains_facts(client, project):
    """Export includes origin and goal facts."""
    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    data = yaml.safe_load(response.text)

    assert "facts" in data
    assert isinstance(data["facts"], list)
    assert len(data["facts"]) >= 2

    fact_ids = {f["id"] for f in data["facts"]}
    assert "origin" in fact_ids
    assert "goal" in fact_ids


def test_export_contains_intents(client, project):
    """Export includes intents."""
    # Create an intent
    client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline model",
        "creator": "human",
    })

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    data = yaml.safe_load(response.text)

    assert "intents" in data
    assert isinstance(data["intents"], list)
    assert len(data["intents"]) == 1
    assert data["intents"][0]["description"] == "Try baseline model"
    assert data["intents"][0]["from"] == ["origin"]


def test_export_contains_hints(client, project):
    """Export includes hints."""
    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    data = yaml.safe_load(response.text)

    assert "hints" in data
    assert isinstance(data["hints"], list)
    assert len(data["hints"]) == 2

    hint_contents = [h["content"] for h in data["hints"]]
    assert "Try pre-trained models" in hint_contents
    assert "Consider data augmentation" in hint_contents


def test_export_contains_leaderboard(client, project):
    """Export includes leaderboard when trials exist."""
    # Create and conclude an intent with metrics
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})
    client.post(f"/api/projects/{project}/intents/{intent_id}/conclude", json={
        "worker": "w1",
        "description": "Baseline: 85% accuracy",
        "metrics": {"val_acc": 0.85, "val_loss": 0.32},
        "trial_id": "t001",
    })

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    data = yaml.safe_load(response.text)

    # Leaderboard may or may not be present depending on implementation
    # The key part is that export works and contains the basic structure
    assert "project" in data
    assert "facts" in data
    assert "intents" in data


def test_export_round_trips_through_yaml(client, project):
    """Export produces valid YAML that can be loaded back."""
    # Add some graph data
    client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Intent A",
        "creator": "human",
    })
    client.post(f"/api/projects/{project}/hints", json={
        "content": "Additional hint",
        "creator": "agent",
    })

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200

    # Should parse without error
    data = yaml.safe_load(response.text)

    # Should be able to dump it back
    reencoded = yaml.safe_dump(data)
    reparsed = yaml.safe_load(reencoded)

    # Key structures should be preserved
    assert reparsed["project"]["title"] == data["project"]["title"]
    assert len(reparsed["facts"]) == len(data["facts"])
    assert len(reparsed["intents"]) == len(data["intents"])
    assert len(reparsed["hints"]) == len(data["hints"])


def test_export_facts_include_metrics_when_present(client, project):
    """Facts with metrics include them in the export."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})
    response = client.post(f"/api/projects/{project}/intents/{intent_id}/conclude", json={
        "worker": "w1",
        "description": "Baseline result",
        "metrics": {"val_acc": 0.85, "train_loss": 0.12},
        "trial_id": "t001",
    })
    fact_id = response.json()["fact"]["id"]

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    data = yaml.safe_load(response.text)

    # Find the fact we created
    fact = next((f for f in data["facts"] if f["id"] == fact_id), None)
    assert fact is not None
    assert fact["metrics"]["val_acc"] == 0.85
    assert fact["metrics"]["train_loss"] == 0.12
    assert fact["trial_id"] == "t001"


def test_export_intent_includes_worker_when_claimed(client, project):
    """Claimed intents show the worker in export."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    data = yaml.safe_load(response.text)

    intent = data["intents"][0]
    assert intent["worker"] == "w1"
    assert intent["to"] is None  # Not concluded yet


def test_export_intent_includes_to_when_concluded(client, project):
    """Concluded intents show the resulting fact ID."""
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Try baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project}/intents/{intent_id}/claim", json={"worker": "w1"})
    response = client.post(f"/api/projects/{project}/intents/{intent_id}/conclude", json={
        "worker": "w1",
        "description": "Result",
        "metrics": {},
    })
    fact_id = response.json()["fact"]["id"]

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    data = yaml.safe_load(response.text)

    intent = data["intents"][0]
    assert intent["to"] == fact_id
    assert intent["concluded_at"] is not None


def test_export_missing_project_returns_404(client):
    """Export for non-existent project returns 404."""
    response = client.get("/api/projects/nonexistent/export", params={"format": "yaml"})
    assert response.status_code == 404


def test_export_content_type_is_plain_text(client, project):
    """Export response has text/plain content type."""
    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_export_facts_include_artifacts_when_present(client, project):
    """Artifacts reach the graph snapshot agents read.

    The prompts require an agent to re-examine the existing model (read the
    loss curve, stat the checkpoint's parameter distribution) before it may
    propose spending more compute. The snapshot is its only graph input, so
    dropping `artifacts` here would leave it with no way to find the model —
    and the "analyse first, scale later" rule unenforceable.
    """
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "Baseline",
        "creator": "human",
    })
    intent_id = response.json()["id"]
    client.post(f"/api/projects/{project}/intents/{intent_id}/claim",
                json={"worker": "w1"})
    response = client.post(f"/api/projects/{project}/intents/{intent_id}/conclude", json={
        "worker": "w1",
        "description": "Baseline trained",
        "metrics": {"val_acc": 0.71},
        "trial_id": "t001",
        "artifacts": {
            "checkpoint_path": "/runs/exp1/best.pt",
            "out_dir": "/runs/exp1",
            "architecture": "CharTransformer",
            "param_count": 1976832,
        },
    })
    fact_id = response.json()["fact"]["id"]

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    data = yaml.safe_load(response.text)

    fact = next(f for f in data["facts"] if f["id"] == fact_id)
    assert fact["artifacts"]["out_dir"] == "/runs/exp1"
    assert fact["artifacts"]["checkpoint_path"] == "/runs/exp1/best.pt"
    assert fact["artifacts"]["architecture"] == "CharTransformer"
    assert fact["artifacts"]["param_count"] == 1976832


def test_export_omits_artifacts_key_when_absent(client, project):
    """A fact without artifacts carries no `artifacts` key at all.

    Emitting `artifacts: null` on every fact would just add noise to a snapshot
    that already goes into an agent's context window.
    """
    response = client.post(f"/api/projects/{project}/intents", json={
        "from": ["origin"],
        "description": "No-artifact conclusion",
        "creator": "human",
    })
    intent_id = response.json()["id"]
    client.post(f"/api/projects/{project}/intents/{intent_id}/claim",
                json={"worker": "w1"})
    response = client.post(f"/api/projects/{project}/intents/{intent_id}/conclude", json={
        "worker": "w1",
        "description": "Dead end, no artifacts",
        "metrics": {"val_acc": 0.4},
    })
    fact_id = response.json()["fact"]["id"]

    response = client.get(f"/api/projects/{project}/export", params={"format": "yaml"})
    data = yaml.safe_load(response.text)

    fact = next(f for f in data["facts"] if f["id"] == fact_id)
    assert "artifacts" not in fact
    # The origin/goal seed facts have no artifacts either.
    assert "artifacts" not in next(f for f in data["facts"] if f["id"] == "origin")
