from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mts.dispatcher.output_parser import extract_json_object


@dataclass(slots=True)
class ExploreOutcome:
    description: str
    metrics: dict[str, float]
    trial_id: str | None = None
    artifacts: dict[str, Any] | None = None


def parse_json_output(stdout: str) -> dict[str, Any]:
    return extract_json_object(stdout)


def _unwrap_wrapped_payload(payload: dict[str, Any]) -> tuple[bool | None, dict[str, Any] | None]:
    accepted = payload.get("accepted")
    if accepted is False:
        return False, None
    if accepted is True:
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("data must be an object")
        return True, data
    return None, None


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _looks_like_reason_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    if keys == {"complete"}:
        complete = payload["complete"]
        return isinstance(complete, dict) and "from" in complete and "description" in complete
    if keys == {"intents"}:
        return isinstance(payload["intents"], list)
    if keys == {"intent"}:
        intent = payload["intent"]
        return isinstance(intent, dict) and "from" in intent and "description" in intent
    return False


def _looks_like_bootstrap_execute_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or set(payload) != {"fact", "complete"}:
        return False
    return _is_dict(payload.get("fact")) and _is_dict(payload.get("complete"))


def _looks_like_bootstrap_conclude_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    if keys not in ({"fact"}, {"fact", "complete"}):
        return False
    return _is_dict(payload.get("fact"))


def _looks_like_explore_data(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = set(payload)
    return "description" in keys and ("metrics" in keys or keys == {"description"})


def _validate_metrics(metrics: Any) -> dict[str, float]:
    """Validate and coerce metrics dict to float values."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    validated: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            raise ValueError(f"metric '{key}' must be numeric, not boolean")
        if not isinstance(value, (int, float)):
            raise ValueError(f"metric '{key}' must be numeric")
        float_value = float(value)
        if math.isnan(float_value) or math.isinf(float_value):
            raise ValueError(f"metric '{key}' must be finite")
        validated[key] = float_value
    return validated


def validate_reason_payload(
    payload: dict[str, Any], open_intents_empty: bool, max_intents: int,
) -> tuple[str, dict[str, Any] | list[dict[str, Any]] | None]:
    accepted, data = _unwrap_wrapped_payload(payload)
    if accepted is False:
        return "rejected", None
    if accepted is None:
        if not _looks_like_reason_data(payload):
            raise ValueError("accepted must be true or false")
        data = payload
    if not isinstance(data, dict):
        raise ValueError("accepted must be true or false")
    complete = data.get("complete")
    intents = data.get("intents")
    # backward compat: accept singular "intent" key from LLMs
    if intents is None:
        singular = data.get("intent")
        if isinstance(singular, dict):
            intents = [singular]
    if complete is not None:
        if intents is not None:
            raise ValueError("complete and intents cannot coexist")
        if not isinstance(complete, dict) or "from" not in complete or "description" not in complete:
            raise ValueError("invalid complete payload")
        return "complete", complete
    if intents is not None:
        if not isinstance(intents, list):
            raise ValueError("intents must be an array")
        for i, intent in enumerate(intents):
            if not isinstance(intent, dict) or "from" not in intent or "description" not in intent:
                raise ValueError(f"invalid intent at index {i}")
        if not intents and open_intents_empty:
            raise ValueError("intents must not be empty when open_intents is empty")
        intents = intents[:max_intents]
        if not intents:
            return "noop", None
        return "intents", intents
    if open_intents_empty:
        raise ValueError("intents is required when open_intents is empty")
    return "noop", None


def validate_explore_payload(payload: dict[str, Any]) -> tuple[str, ExploreOutcome | None]:
    accepted, data = _unwrap_wrapped_payload(payload)
    if accepted is False:
        return "rejected", None
    if accepted is None:
        if not _looks_like_explore_data(payload):
            raise ValueError("accepted must be true or false")
        data = payload
    if not isinstance(data, dict):
        raise ValueError("accepted must be true or false")
    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")

    metrics_raw = data.get("metrics", {})
    metrics = _validate_metrics(metrics_raw)

    trial_id = data.get("trial_id")
    if trial_id is not None and not isinstance(trial_id, str):
        raise ValueError("trial_id must be a string")

    artifacts = data.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        raise ValueError("artifacts must be an object")

    return "fact", ExploreOutcome(
        description=description.strip(),
        metrics=metrics,
        trial_id=trial_id,
        artifacts=artifacts,
    )


def validate_bootstrap_execute_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    accepted, data = _unwrap_wrapped_payload(payload)
    if accepted is False:
        return "rejected", None
    if accepted is None:
        if not _looks_like_bootstrap_execute_data(payload):
            raise ValueError("accepted must be true or false")
        data = payload
    if not isinstance(data, dict):
        raise ValueError("accepted must be true or false")

    fact = data.get("fact")
    if not isinstance(fact, dict):
        raise ValueError("fact is required")
    fact_description = fact.get("description")
    if not isinstance(fact_description, str) or not fact_description.strip():
        raise ValueError("fact.description is required")

    metrics_raw = fact.get("metrics", {})
    metrics = _validate_metrics(metrics_raw)

    trial_id = fact.get("trial_id")
    if trial_id is not None and not isinstance(trial_id, str):
        raise ValueError("fact.trial_id must be a string")

    artifacts = fact.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        raise ValueError("fact.artifacts must be an object")

    complete = data.get("complete")
    if complete is None:
        raise ValueError("complete is required")
    if not isinstance(complete, dict):
        raise ValueError("complete must be an object")
    complete_description = complete.get("description")
    if not isinstance(complete_description, str) or not complete_description.strip():
        raise ValueError("complete.description is required")

    return "complete", {
        "fact": ExploreOutcome(
            description=fact_description.strip(),
            metrics=metrics,
            trial_id=trial_id,
            artifacts=artifacts,
        ),
        "complete_description": complete_description.strip(),
    }


def validate_bootstrap_conclude_payload(payload: dict[str, Any]) -> tuple[str, ExploreOutcome | None]:
    accepted, data = _unwrap_wrapped_payload(payload)
    if accepted is False:
        return "rejected", None
    if accepted is None:
        if not _looks_like_bootstrap_conclude_data(payload):
            raise ValueError("accepted must be true or false")
        data = payload
    if not isinstance(data, dict):
        raise ValueError("accepted must be true or false")
    extra_keys = set(data) - {"fact", "complete"}
    if extra_keys:
        raise ValueError("unexpected keys in conclude payload")
    fact = data.get("fact")
    if not isinstance(fact, dict):
        raise ValueError("fact is required")
    fact_description = fact.get("description")
    if not isinstance(fact_description, str) or not fact_description.strip():
        raise ValueError("fact.description is required")

    metrics_raw = fact.get("metrics", {})
    metrics = _validate_metrics(metrics_raw)

    trial_id = fact.get("trial_id")
    if trial_id is not None and not isinstance(trial_id, str):
        raise ValueError("fact.trial_id must be a string")

    artifacts = fact.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        raise ValueError("fact.artifacts must be an object")

    return "fact", ExploreOutcome(
        description=fact_description.strip(),
        metrics=metrics,
        trial_id=trial_id,
        artifacts=artifacts,
    )
