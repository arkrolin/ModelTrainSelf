"""Pydantic models for the board API.

These are the wire types the REST API speaks. They mirror Cairn's protocol
with MTS-specific extensions for metrics and trials.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# =====================================================================
# Settings
# =====================================================================

class Settings(BaseModel):
    intent_timeout: int
    reason_timeout: int
    max_trials: int


# =====================================================================
# Core graph types (Cairn protocol)
# =====================================================================

class Fact(BaseModel):
    id: str
    description: str
    metrics: dict[str, float] = Field(default_factory=dict)
    trial_id: str | None = None
    artifacts: dict[str, Any] | None = None
    created_at: str


class Intent(BaseModel):
    id: str
    from_: list[str] = Field(alias="from")
    to: str | None = None
    description: str
    creator: str
    worker: str | None = None
    last_heartbeat_at: str | None = None
    created_at: str
    concluded_at: str | None = None

    model_config = {"populate_by_name": True}


class Hint(BaseModel):
    id: str
    content: str
    creator: str
    created_at: str


class ProjectReason(BaseModel):
    worker: str
    trigger: str
    started_at: str
    last_heartbeat_at: str


class ProjectMeta(BaseModel):
    id: str
    title: str
    status: Literal["active", "stopped", "completed"]
    bootstrap_enabled: bool
    origin: str
    goal: str
    goal_metric: str
    goal_target: float | None
    goal_direction: str
    budget_max_trials: int
    created_at: str
    reason: ProjectReason | None = None
    search_config: "SearchConfigOut | None" = None


class ProjectSummary(ProjectMeta):
    fact_count: int
    intent_count: int
    working_intent_count: int
    unclaimed_intent_count: int
    hint_count: int
    trial_count: int
    best_metric: float | None


class ProjectDetail(BaseModel):
    project: ProjectMeta
    facts: list[Fact]
    intents: list[Intent]
    hints: list[Hint]


# =====================================================================
# Request types
# =====================================================================

class ProjectCreate(BaseModel):
    title: str
    origin: str
    goal: str
    goal_metric: str = "val_loss"
    goal_target: float | None = None
    goal_direction: str = "minimize"
    budget_max_trials: int = 60
    hints: list[str] = Field(default_factory=list)
    bootstrap_enabled: bool = True


class ProjectUpdate(BaseModel):
    """Partial update of a project's own fields, editable after creation.

    Every field is optional: only what the WebUI actually sends gets written, so
    a form that touches one input does not clobber the rest. `goal_target` uses a
    sentinel-free `None` because clearing the threshold is a legitimate edit — the
    service distinguishes "absent" from "explicitly null" via `exclude_unset`.
    """

    title: str | None = Field(default=None, min_length=1)
    origin: str | None = Field(default=None, min_length=1)
    goal: str | None = Field(default=None, min_length=1)
    goal_metric: str | None = Field(default=None, min_length=1)
    goal_target: float | None = None
    goal_direction: Literal["minimize", "maximize"] | None = None
    budget_max_trials: int | None = Field(default=None, ge=1, le=10000)
    bootstrap_enabled: bool | None = None


class CreateIntentRequest(BaseModel):
    from_: list[str] = Field(alias="from")
    description: str
    creator: str
    worker: str | None = None

    model_config = {"populate_by_name": True}


class HeartbeatRequest(BaseModel):
    worker: str


class ConcludeRequest(BaseModel):
    worker: str
    description: str
    metrics: dict[str, float] | None = None
    trial_id: str | None = None
    artifacts: dict[str, Any] | None = None


class CompleteRequest(BaseModel):
    from_: list[str] = Field(alias="from")
    description: str
    worker: str

    model_config = {"populate_by_name": True}


class ReasonClaimRequest(BaseModel):
    worker: str
    trigger: str


class HintCreate(BaseModel):
    content: str
    creator: str = "human"


# =====================================================================
# Response types
# =====================================================================

class ConcludeResponse(BaseModel):
    fact: Fact
    intent: Intent


class CompleteResponse(BaseModel):
    project: ProjectMeta
    fact: Fact
    intent: Intent


# =====================================================================
# MTS-specific trial types (not part of Cairn protocol)
# =====================================================================

class TrialRegister(BaseModel):
    """Dispatcher posts this to register a finished experiment."""

    out_dir: str
    trial_id: str | None = None
    name: str | None = None
    parent_id: str | None = None
    intent_id: str | None = None
    fact_id: str | None = None
    worker: str | None = None
    backend: str = "surrogate"
    fingerprint: str | None = None
    status: str = "completed"
    steps_done: int = 0
    duration_sec: float = 0.0
    diff_json: dict[str, Any] = Field(default_factory=dict)
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    final_val_acc: float | None = None
    best_val_acc: float | None = None
    verdict: str | None = None
    signals: list[str] = Field(default_factory=list)
    spec: dict[str, Any] | None = None


class TrialHeartbeat(BaseModel):
    """Progress heartbeat from a running trial's worker."""

    worker: str | None = None
    step: int = 0
    max_steps: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    eta_sec: float | None = None


class TrialOut(BaseModel):
    id: str
    project_id: str
    name: str
    parent_id: str | None = None
    intent_id: str | None = None
    fact_id: str | None = None
    worker: str | None = None
    backend: str
    status: str
    steps_done: int
    duration_sec: float
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    best_val_loss: float | None = None
    final_val_acc: float | None = None
    best_val_acc: float | None = None
    verdict: str | None = None
    signal_count: int
    top_signal: str | None = None
    diff_text: str = ""
    created_at: str


class ObservationCreate(BaseModel):
    """A measurement an agent took with an inspection tool.

    `payload` carries the tool's structured return value so the finding stays
    auditable — a later reader can check the claim against the numbers instead of
    trusting the prose.
    """

    trial_id: str | None = None
    observer: str = "agent"
    tool: str = "manual"
    finding: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ObservationOut(BaseModel):
    id: str
    trial_id: str | None = None
    observer: str
    tool: str
    finding: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class WorkerRequirement(BaseModel):
    """Worker requirement specified by a project.

    Projects specify what type of worker they need and how many,
    the dispatcher will automatically assign matching workers from its pool.
    """

    worker_type: str = Field(default="claudecode", description="Worker type: claudecode, mock, etc.")
    count: int = Field(default=2, ge=1, le=32, description="Number of workers needed")
    provider_id: str | None = Field(default=None, description="Optional: use a specific provider")


class WorkerSpec(BaseModel):
    """DEPRECATED: Legacy format for backward compatibility.

    One entry in a project's worker roster. This is kept for migration purposes.
    New code should use WorkerRequirement instead.
    """

    name: str
    driver: str = "mock"
    provider_id: str | None = None


class SearchConfigIn(BaseModel):
    """Search settings a user configures in the WebUI.

    Projects specify worker requirements (type + count), and the dispatcher
    automatically assigns matching workers from its YAML-configured pool.
    """

    max_trials: int = Field(default=12, ge=1, le=1000)
    # Defaults to None, not WorkerRequirement(): the dispatcher reads this as a hard
    # constraint and filters its worker pool by worker_type. With a default_factory a
    # project that never saved a config would still claim to require claudecode, which
    # silently filters out every other worker in the pool and starves dispatch.
    # None == "no preference, any worker in the pool will do".
    worker_requirement: WorkerRequirement | None = None

    # Legacy field for backward compatibility
    max_workers: int | None = Field(default=None, ge=1, le=32, deprecated=True)
    workers: list[WorkerSpec] = Field(default_factory=list, deprecated=True)


class SearchConfigOut(SearchConfigIn):
    project_id: str
    updated_at: str | None = None


class DispatchStart(BaseModel):
    """Request to start a search run in-process (no shell, no YAML)."""

    config: SearchConfigIn | None = None


# =====================================================================
# LLM providers
# =====================================================================

ProviderKind = Literal["openai", "anthropic", "claudecode"]


class ProviderIn(BaseModel):
    """An LLM endpoint configured from the WebUI.

    `kind` selects the wire protocol: "openai" posts to {base_url}/v1/chat/completions,
    "anthropic" posts to {base_url}/v1/messages, "claudecode" drives the local
    `claude` CLI. Any OpenAI-compatible server (vLLM, Ollama, LM Studio, DeepSeek,
    DashScope, OpenRouter) works with kind="openai".
    """

    name: str = Field(min_length=1, max_length=100)
    kind: ProviderKind = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    is_default: bool = False
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    headers: dict[str, str] = Field(default_factory=dict)


class ProviderUpdate(BaseModel):
    """Partial update. api_key=None keeps the stored key; "" clears it."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    kind: ProviderKind | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    is_default: bool | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=200000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    headers: dict[str, str] | None = None


class ProviderOut(BaseModel):
    """Read shape. Never carries the raw key — only a masked hint."""

    id: str
    name: str
    kind: str
    base_url: str
    api_key_masked: str
    has_api_key: bool
    model: str
    is_default: bool
    max_tokens: int
    temperature: float | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ProviderTestResult(BaseModel):
    ok: bool
    latency_ms: float | None = None
    model: str | None = None
    reply: str | None = None
    error: str | None = None


class KnowledgeCreate(BaseModel):
    slug: str
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class KnowledgeOut(BaseModel):
    slug: str
    title: str
    tags: list[str]
    path: str
    confidence: float
    hits: int
    created_at: str
    updated_at: str
