"""Test payload validation for reason, explore, and bootstrap tasks."""

from __future__ import annotations

import pytest

from mts.dispatcher.contracts import (
    validate_reason_payload,
    validate_explore_payload,
    validate_bootstrap_execute_payload,
    validate_bootstrap_conclude_payload,
)


# =====================================================================
# Reason payload tests
# =====================================================================

def test_reason_complete():
    payload = {"complete": {"from": ["f1"], "description": "Goal reached"}}
    kind, data = validate_reason_payload(payload, open_intents_empty=False, max_intents=3)
    assert kind == "complete"
    assert data["from"] == ["f1"]
    assert data["description"] == "Goal reached"


def test_reason_intents():
    payload = {
        "intents": [
            {"from": ["f1"], "description": "Try A"},
            {"from": ["f2"], "description": "Try B"},
        ]
    }
    kind, data = validate_reason_payload(payload, open_intents_empty=True, max_intents=3)
    assert kind == "intents"
    assert len(data) == 2
    assert data[0]["description"] == "Try A"


def test_reason_singular_intent_backward_compat():
    """Accept singular 'intent' key from LLMs."""
    payload = {"intent": {"from": ["f1"], "description": "Try A"}}
    kind, data = validate_reason_payload(payload, open_intents_empty=True, max_intents=3)
    assert kind == "intents"
    assert len(data) == 1
    assert data[0]["description"] == "Try A"


def test_reason_intents_capped_at_max():
    payload = {
        "intents": [
            {"from": ["f1"], "description": f"Intent {i}"}
            for i in range(10)
        ]
    }
    kind, data = validate_reason_payload(payload, open_intents_empty=True, max_intents=3)
    assert kind == "intents"
    assert len(data) == 3


def test_reason_noop_when_open_intents_exist():
    """When open intents exist, agent can return no new intents (noop)."""
    payload = {"accepted": True, "data": {}}
    kind, data = validate_reason_payload(payload, open_intents_empty=False, max_intents=3)
    assert kind == "noop"
    assert data is None


def test_reason_noop_when_empty_intents_and_open_exist():
    """Empty intents array when open intents exist is a noop."""
    payload = {"intents": []}
    kind, data = validate_reason_payload(payload, open_intents_empty=False, max_intents=3)
    assert kind == "noop"
    assert data is None


def test_reason_rejected():
    payload = {"accepted": False}
    kind, data = validate_reason_payload(payload, open_intents_empty=False, max_intents=3)
    assert kind == "rejected"
    assert data is None


def test_reason_wrapped_accepted_true():
    payload = {
        "accepted": True,
        "data": {"intents": [{"from": ["f1"], "description": "Try A"}]}
    }
    kind, data = validate_reason_payload(payload, open_intents_empty=True, max_intents=3)
    assert kind == "intents"
    assert len(data) == 1


def test_reason_empty_intents_when_open_intents_empty_fails():
    payload = {"intents": []}
    with pytest.raises(ValueError, match="intents must not be empty"):
        validate_reason_payload(payload, open_intents_empty=True, max_intents=3)


def test_reason_no_intents_when_open_intents_empty_fails():
    """When open intents are empty, payload must provide intents or be wrapped."""
    payload = {}
    with pytest.raises(ValueError, match="accepted must be true or false"):
        validate_reason_payload(payload, open_intents_empty=True, max_intents=3)


def test_reason_complete_and_intents_conflict():
    """Complete and intents cannot both be present."""
    payload = {
        "accepted": True,
        "data": {
            "complete": {"from": ["f1"], "description": "Done"},
            "intents": [{"from": ["f2"], "description": "Try A"}]
        }
    }
    with pytest.raises(ValueError, match="complete and intents cannot coexist"):
        validate_reason_payload(payload, open_intents_empty=False, max_intents=3)


def test_reason_invalid_complete_missing_from():
    """Complete payload must have 'from' field."""
    payload = {"accepted": True, "data": {"complete": {"description": "Done"}}}
    with pytest.raises(ValueError, match="invalid complete payload"):
        validate_reason_payload(payload, open_intents_empty=False, max_intents=3)


def test_reason_invalid_intent_missing_description():
    """Intent must have description field."""
    payload = {"intents": [{"from": ["f1"]}]}
    with pytest.raises(ValueError, match="invalid intent at index 0"):
        validate_reason_payload(payload, open_intents_empty=True, max_intents=3)


def test_reason_ambiguous_payload_requires_accepted():
    """Payload that doesn't look like reason data must have accepted field."""
    payload = {"random": "data"}
    with pytest.raises(ValueError, match="accepted must be true or false"):
        validate_reason_payload(payload, open_intents_empty=False, max_intents=3)


# =====================================================================
# Explore payload tests
# =====================================================================

def test_explore_fact_with_metrics():
    payload = {
        "description": "Tried A, got B",
        "metrics": {"val_acc": 0.85, "train_loss": 0.12},
        "trial_id": "t001"
    }
    kind, outcome = validate_explore_payload(payload)
    assert kind == "fact"
    assert outcome.description == "Tried A, got B"
    assert outcome.metrics["val_acc"] == 0.85
    assert outcome.trial_id == "t001"


def test_explore_fact_without_metrics():
    """Metrics are optional, empty dict by default."""
    payload = {"description": "Observation"}
    kind, outcome = validate_explore_payload(payload)
    assert kind == "fact"
    assert outcome.description == "Observation"
    assert outcome.metrics == {}
    assert outcome.trial_id is None


def test_explore_rejected():
    payload = {"accepted": False}
    kind, outcome = validate_explore_payload(payload)
    assert kind == "rejected"
    assert outcome is None


def test_explore_wrapped_accepted_true():
    payload = {
        "accepted": True,
        "data": {"description": "Result", "metrics": {"val_loss": 0.3}}
    }
    kind, outcome = validate_explore_payload(payload)
    assert kind == "fact"
    assert outcome.description == "Result"
    assert outcome.metrics["val_loss"] == 0.3


def test_explore_empty_description_fails():
    payload = {"description": "   ", "metrics": {"val_loss": 0.2}}
    with pytest.raises(ValueError, match="description is required"):
        validate_explore_payload(payload)


def test_explore_missing_description_fails():
    """Description is required in explore payload."""
    payload = {"accepted": True, "data": {"metrics": {"val_loss": 0.2}}}
    with pytest.raises(ValueError, match="description is required"):
        validate_explore_payload(payload)


def test_explore_invalid_metrics_not_dict():
    payload = {"description": "Test", "metrics": "not a dict"}
    with pytest.raises(ValueError, match="metrics must be an object"):
        validate_explore_payload(payload)


def test_explore_invalid_metric_boolean():
    payload = {"description": "Test", "metrics": {"val_acc": True}}
    with pytest.raises(ValueError, match="metric 'val_acc' must be numeric, not boolean"):
        validate_explore_payload(payload)


def test_explore_invalid_metric_string():
    payload = {"description": "Test", "metrics": {"val_loss": "low"}}
    with pytest.raises(ValueError, match="metric 'val_loss' must be numeric"):
        validate_explore_payload(payload)


def test_explore_invalid_metric_nan():
    payload = {"description": "Test", "metrics": {"val_loss": float('nan')}}
    with pytest.raises(ValueError, match="metric 'val_loss' must be finite"):
        validate_explore_payload(payload)


def test_explore_invalid_metric_inf():
    payload = {"description": "Test", "metrics": {"val_loss": float('inf')}}
    with pytest.raises(ValueError, match="metric 'val_loss' must be finite"):
        validate_explore_payload(payload)


def test_explore_trial_id_must_be_string():
    """trial_id must be a string if provided - bare payload needs accepted wrapper."""
    payload = {"accepted": True, "data": {"description": "Test", "trial_id": 123}}
    with pytest.raises(ValueError, match="trial_id must be a string"):
        validate_explore_payload(payload)


# =====================================================================
# Bootstrap execute payload tests
# =====================================================================

def test_bootstrap_execute_complete():
    payload = {
        "fact": {
            "description": "Baseline implemented",
            "metrics": {"val_acc": 0.75},
            "trial_id": "t001"
        },
        "complete": {
            "description": "Ready for optimization"
        }
    }
    kind, data = validate_bootstrap_execute_payload(payload)
    assert kind == "complete"
    assert data["fact"].description == "Baseline implemented"
    assert data["fact"].metrics["val_acc"] == 0.75
    assert data["complete_description"] == "Ready for optimization"


def test_bootstrap_execute_rejected():
    payload = {"accepted": False}
    kind, data = validate_bootstrap_execute_payload(payload)
    assert kind == "rejected"
    assert data is None


def test_bootstrap_execute_wrapped():
    payload = {
        "accepted": True,
        "data": {
            "fact": {"description": "Done", "metrics": {}},
            "complete": {"description": "Ready"}
        }
    }
    kind, data = validate_bootstrap_execute_payload(payload)
    assert kind == "complete"
    assert data["fact"].description == "Done"


def test_bootstrap_execute_missing_complete_fails():
    """Bootstrap execute requires both fact and complete."""
    payload = {
        "accepted": True,
        "data": {"fact": {"description": "Done", "metrics": {}}}
    }
    with pytest.raises(ValueError, match="complete is required"):
        validate_bootstrap_execute_payload(payload)


def test_bootstrap_execute_missing_fact_fails():
    """Bootstrap execute requires both fact and complete."""
    payload = {
        "accepted": True,
        "data": {"complete": {"description": "Ready"}}
    }
    with pytest.raises(ValueError, match="fact is required"):
        validate_bootstrap_execute_payload(payload)


# =====================================================================
# Bootstrap conclude payload tests
# =====================================================================

def test_bootstrap_conclude_fact():
    payload = {
        "fact": {
            "description": "Baseline works",
            "metrics": {"val_loss": 0.3},
            "trial_id": "t001"
        }
    }
    kind, outcome = validate_bootstrap_conclude_payload(payload)
    assert kind == "fact"
    assert outcome.description == "Baseline works"
    assert outcome.metrics["val_loss"] == 0.3


def test_bootstrap_conclude_with_complete_ignored():
    """Complete field is allowed but ignored in conclude."""
    payload = {
        "fact": {"description": "Done", "metrics": {}},
        "complete": {"description": "This is ignored"}
    }
    kind, outcome = validate_bootstrap_conclude_payload(payload)
    assert kind == "fact"
    assert outcome.description == "Done"


def test_bootstrap_conclude_rejected():
    payload = {"accepted": False}
    kind, outcome = validate_bootstrap_conclude_payload(payload)
    assert kind == "rejected"
    assert outcome is None


def test_bootstrap_conclude_extra_keys_fails():
    """Bootstrap conclude only allows fact and optional complete - exact key check."""
    payload = {
        "accepted": True,
        "data": {
            "fact": {"description": "Done", "metrics": {}},
            "unexpected": "field"
        }
    }
    with pytest.raises(ValueError, match="unexpected keys"):
        validate_bootstrap_conclude_payload(payload)


def test_bootstrap_conclude_missing_fact_fails():
    """Bootstrap conclude requires fact."""
    payload = {"accepted": True, "data": {"complete": {"description": "Ready"}}}
    with pytest.raises(ValueError, match="fact is required"):
        validate_bootstrap_conclude_payload(payload)
