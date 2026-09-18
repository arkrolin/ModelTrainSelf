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


# ---------------------------------------------------------------------------
# Analyse-before-spend discipline
#
# The default prompts require an agent to re-examine the existing model before
# it may propose or run anything that costs more compute. These are contract
# tests: the wording may be reworded freely, but if the rule itself or the
# plumbing it depends on disappears, the agents silently go back to blindly
# scaling up training, which is exactly what these prompts exist to prevent.
# ---------------------------------------------------------------------------

# Field names the prompts tell the agent to read out of each inspect tool's
# result. Kept in sync with mts.inspect by test_inspect_tool_fields_exist.
INSPECT_FIELDS_CITED = {
    "inspect_curve": ("shape", "val_shape", "clipped_fraction"),
    "inspect_distribution": ("characterization", "stats"),
    "inspect_layers": ("outliers", "summary"),
    "layer_trajectory": ("trends",),
    "inspect_grad_flow": ("assessment", "bottom_top_ratio", "per_layer_decay_factor"),
    "compare_trials": ("verdict_change", "spec_delta", "metric_delta"),
}


@pytest.mark.parametrize("name", ["reason.md", "explore.md"])
def test_prompt_requires_analysis_before_scaling_cost(name):
    """The two prompts that decide/spend compute must carry the rule."""
    content = load_prompt("default", name)
    assert "强制前置步骤" in content, f"{name} lost the mandatory pre-analysis section"
    # The artifacts field is the agent's only route to a past model.
    assert "artifacts" in content
    assert "checkpoint_path" in content
    assert "out_dir" in content
    # The three things the user asked to be analysed.
    assert "参数分布" in content
    assert "metrics.jsonl" in content or "训练曲线" in content
    assert "梯度" in content


@pytest.mark.parametrize("name", ["reason.md", "explore.md"])
def test_prompt_forbids_blind_cost_increase(name):
    """Scaling compute without evidence must be explicitly forbidden."""
    content = load_prompt("default", name)
    assert "禁止" in content, f"{name} no longer forbids anything"
    # Naming the specific levers matters: a generic "be careful" does not stop
    # an agent from proposing "train for 10x more steps".
    assert "训练步数" in content
    assert "batch" in content.lower()


@pytest.mark.parametrize("name", ["reason.md", "explore.md"])
def test_prompt_cites_inspect_tools_with_readable_fields(name):
    """Every inspect tool the prompt names also says which field to read.

    Without the field name an agent gets a few thousand downsampled curve
    points back and no indication that `shape` already holds the verdict.
    """
    content = load_prompt("default", name)
    for tool, fields in INSPECT_FIELDS_CITED.items():
        assert tool in content, f"{name} stopped mentioning {tool}"
        assert any(f in content for f in fields), (
            f"{name} names {tool} but none of its verdict fields {fields}"
        )


def test_inspect_tool_fields_exist():
    """Fields the prompts tell agents to read are really returned.

    Guards against the prompts pointing at a key that was renamed in
    mts.inspect — the agent would just see `None` and lose the evidence it is
    required to base its decision on.
    """
    from mts.inspect import TOOLS

    assert set(INSPECT_FIELDS_CITED) <= set(TOOLS), (
        f"prompts cite unknown tools: {set(INSPECT_FIELDS_CITED) - set(TOOLS)}"
    )


def test_reason_prompt_has_workdir_for_analysis_scripts():
    """reason.md needs a workdir: it now writes analysis scripts.

    The token must be both declared as required and present in the template,
    otherwise the rendered prompt ships a literal `{workdir}` to the agent.
    """
    assert "{workdir}" in DEFAULT_PROMPT_REQUIRED_TOKENS["reason.md"]
    assert "{workdir}" in load_prompt("default", "reason.md")


def test_reason_task_supplies_workdir():
    """The reason task actually fills {workdir}.

    Declaring the token is not enough — tasks/reason.py has to pass it, or
    validate_prompt_resources passes while the agent still sees the literal.
    """
    import inspect as _inspect

    from mts.dispatcher.tasks import reason as reason_task

    src = _inspect.getsource(reason_task)
    assert '"workdir"' in src, "reason.py no longer passes workdir to render_prompt"


@pytest.mark.parametrize("name", ["bootstrap.md", "bootstrap_conclude.md"])
def test_bootstrap_prompts_request_artifacts(name):
    """Bootstrap must hand back artifacts; every later analysis starts there."""
    content = load_prompt("default", name)
    assert "artifacts" in content
    assert "out_dir" in content
    assert "checkpoint_path" in content


def test_conclude_prompts_do_not_ask_for_more_commands():
    """Conclude is a hard stop — it must not order fresh analysis runs.

    The analysis requirement belongs to the execute phases. Asking for it here
    would fight the "stop immediately, run nothing" contract that keeps a
    timed-out task from hanging forever.
    """
    for name in ("explore_conclude.md", "bootstrap_conclude.md"):
        content = load_prompt("default", name)
        assert "不要再运行任何命令" in content or "不需要再运行任何命令" in content
        assert "强制前置步骤" not in content, (
            f"{name} must not demand pre-analysis; it is a stop phase"
        )
