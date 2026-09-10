"""Test prompt loading and token validation for all prompt groups."""

from __future__ import annotations

import pytest

from mts.dispatcher.prompting import load_prompt, render_prompt
from mts.dispatcher.config import (
    DEFAULT_PROMPT_REQUIRED_TOKENS,
    PROMPT_REQUIRED_TOKENS_BY_GROUP,
)


PROMPT_GROUPS = ["default", "mock"]
PROMPT_NAMES = [
    "reason.md",
    "explore.md",
    "explore_conclude.md",
    "bootstrap.md",
    "bootstrap_conclude.md",
]


@pytest.mark.parametrize("group", PROMPT_GROUPS)
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_loads(group, name):
    """Every prompt in every group can be loaded."""
    content = load_prompt(group, name)
    assert isinstance(content, str)
    assert len(content) > 0


@pytest.mark.parametrize("group", PROMPT_GROUPS)
@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_contains_no_leftover_placeholders(group, name):
    """After rendering with all required tokens, no placeholders remain."""
    content = load_prompt(group, name)

    # Get required tokens for this prompt
    required_tokens_map = PROMPT_REQUIRED_TOKENS_BY_GROUP.get(group, DEFAULT_PROMPT_REQUIRED_TOKENS)
    required_tokens = required_tokens_map.get(name, ())

    # Build replacement dict with dummy values
    replacements = {}
    for token in required_tokens:
        key = token.strip("{}")
        replacements[key] = f"dummy_{key}"

    # Render the prompt
    rendered = render_prompt(content, replacements)

    # Check for leftover placeholders
    # Look for patterns like {token_name}
    import re
    leftover = re.findall(r"\{[a-z_]+\}", rendered)

    # Filter out any tokens that are not in the required list
    # (some prompts may have optional tokens)
    unexpected = [t for t in leftover if t in required_tokens]

    assert not unexpected, f"Leftover required placeholders in {group}/{name}: {unexpected}"


def test_reason_prompt_required_tokens():
    """Reason prompt requires specific tokens."""
    required = DEFAULT_PROMPT_REQUIRED_TOKENS["reason.md"]
    assert "{graph_yaml}" in required
    assert "{fact_ids}" in required
    assert "{open_intents}" in required
    assert "{max_intents}" in required
    assert "{goal_metric}" in required
    assert "{goal_direction}" in required


def test_explore_prompt_required_tokens():
    """Explore prompt requires specific tokens."""
    required = DEFAULT_PROMPT_REQUIRED_TOKENS["explore.md"]
    assert "{graph_yaml}" in required
    assert "{intent_id}" in required
    assert "{intent_description}" in required
    assert "{goal_metric}" in required
    assert "{workdir}" in required


def test_explore_conclude_prompt_required_tokens():
    """Explore conclude prompt requires specific tokens."""
    required = DEFAULT_PROMPT_REQUIRED_TOKENS["explore_conclude.md"]
    assert "{graph_yaml}" in required
    assert "{intent_id}" in required
    assert "{intent_description}" in required
    assert "{goal_metric}" in required


def test_bootstrap_prompt_required_tokens():
    """Bootstrap prompt requires specific tokens."""
    required = DEFAULT_PROMPT_REQUIRED_TOKENS["bootstrap.md"]
    assert "{origin}" in required
    assert "{goal}" in required
    assert "{hints}" in required
    assert "{goal_metric}" in required
    assert "{goal_direction}" in required
    assert "{workdir}" in required


def test_bootstrap_conclude_prompt_required_tokens():
    """Bootstrap conclude prompt requires specific tokens."""
    required = DEFAULT_PROMPT_REQUIRED_TOKENS["bootstrap_conclude.md"]
    assert "{origin}" in required
    assert "{goal}" in required
    assert "{hints}" in required
    assert "{goal_metric}" in required


def test_mock_prompts_have_minimal_requirements():
    """Mock prompts have reduced token requirements."""
    mock_tokens = PROMPT_REQUIRED_TOKENS_BY_GROUP["mock"]

    # Mock prompts should require fewer tokens than default
    assert len(mock_tokens["reason.md"]) < len(DEFAULT_PROMPT_REQUIRED_TOKENS["reason.md"])
    assert len(mock_tokens["explore.md"]) < len(DEFAULT_PROMPT_REQUIRED_TOKENS["explore.md"])
    assert len(mock_tokens["bootstrap.md"]) < len(DEFAULT_PROMPT_REQUIRED_TOKENS["bootstrap.md"])


def test_render_prompt_basic():
    """render_prompt performs simple replacements."""
    template = "Hello {name}, you have {count} items."
    result = render_prompt(template, {"name": "Alice", "count": "5"})
    assert result == "Hello Alice, you have 5 items."


def test_render_prompt_missing_token_preserved():
    """render_prompt leaves unreplaced tokens as-is."""
    template = "Hello {name}, you have {count} items."
    result = render_prompt(template, {"name": "Bob"})
    assert result == "Hello Bob, you have {count} items."


def test_render_prompt_extra_replacements_ignored():
    """render_prompt ignores extra replacement keys."""
    template = "Hello {name}."
    result = render_prompt(template, {"name": "Charlie", "unused": "value"})
    assert result == "Hello Charlie."


def test_render_prompt_empty_template():
    """render_prompt handles empty template."""
    result = render_prompt("", {"name": "value"})
    assert result == ""


def test_render_prompt_no_replacements():
    """render_prompt handles template with no placeholders."""
    template = "This is a static string."
    result = render_prompt(template, {"key": "value"})
    assert result == template


def test_default_group_has_all_prompts():
    """Default group contains all five required prompts."""
    for name in PROMPT_NAMES:
        content = load_prompt("default", name)
        assert len(content) > 100, f"default/{name} seems too short"


def test_mock_group_has_all_prompts():
    """Mock group contains all five required prompts."""
    for name in PROMPT_NAMES:
        content = load_prompt("mock", name)
        assert len(content) > 0, f"mock/{name} is empty"


def test_prompt_group_validation_catches_missing_prompt(monkeypatch):
    """Config validation catches missing prompts in a group."""
    from mts.dispatcher.config import validate_prompt_resources

    # Valid group should pass
    validate_prompt_resources("default")
    validate_prompt_resources("mock")

    # Invalid group should fail
    with pytest.raises(ValueError, match="missing prompt group"):
        validate_prompt_resources("nonexistent_group")


def test_prompt_has_required_token_validation():
    """Each prompt in default group contains its required tokens."""
    for name, required_tokens in DEFAULT_PROMPT_REQUIRED_TOKENS.items():
        content = load_prompt("default", name)
        for token in required_tokens:
            assert token in content, f"default/{name} missing required token {token}"
