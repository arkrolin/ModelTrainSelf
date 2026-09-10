"""Test DispatcherLoop decision tree with a fake client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, MagicMock
import tempfile
import pytest

from mts.dispatcher.config import (
    DispatchConfig,
    RuntimeConfig,
    TasksConfig,
    ReasonTaskConfig,
    ExploreTaskConfig,
    BootstrapTaskConfig,
    WorkerConfig,
    LocalConfig,
)
from mts.dispatcher.models import ReasonCheckpoint
from mts.dispatcher.scheduler.loop import DispatcherLoop
from mts.server.models import (
    ProjectSummary,
    ProjectDetail,
    ProjectMeta,
    Fact,
    Intent,
)


@pytest.fixture
def config_yaml(tmp_path):
    """Create a minimal dispatch config file."""
    config_path = tmp_path / "dispatch.yaml"
    config_path.write_text("""
server: http://localhost:8000
runtime:
  max_workers: 2
  max_running_projects: 1
  max_project_workers: 1
  interval: 1
  healthcheck_timeout: 5
  worker_healthcheck: disabled
  execution: local
  prompt_group: mock
tasks:
  bootstrap:
    timeout: 10
    conclude_timeout: 10
  reason:
    timeout: 10
    max_intents: 3
  explore:
    timeout: 10
    conclude_timeout: 10
local:
  workspace_root: null
  completed_action: keep
workers:
  - name: mock1
    type: mock
    task_types: [bootstrap, reason, explore]
    max_running: 2
    priority: 0
""")
    return config_path


@pytest.fixture
def stub_executor(monkeypatch):
    """Replace the loop's thread pool so no task body actually runs.

    Every dispatch path claims the lease *synchronously* and only then submits
    the task, so the decision tree is fully observable from the client mock.
    Letting the real pool run would start heartbeat threads and mock
    subprocesses that have nothing to do with the decision under test.
    """

    def install(loop):
        executor = Mock()
        executor.submit = Mock(return_value=Mock(done=Mock(return_value=False)))
        executor.shutdown = Mock()
        monkeypatch.setattr(loop, "executor", executor)
        return executor

    return install


@pytest.fixture
def fake_client():
    """Create a fake MTSClient that records calls."""
    client = Mock()
    client.calls = []

    def record_call(method, *args, **kwargs):
        client.calls.append((method, args, kwargs))

    # Set up mock methods
    client.get_settings = Mock(return_value=Mock(intent_timeout=60, reason_timeout=60))
    client.list_projects = Mock(return_value=[])
    client.get_project = Mock()
    client.export_project = Mock(return_value="project: {}\nfacts: []\nintents: []\nhints: []")
    client.create_intent = Mock()
    client.claim_intent = Mock()
    client.claim_reason = Mock()
    client.heartbeat = Mock()
    client.reason_heartbeat = Mock()
    client.release = Mock()
    client.release_reason = Mock()
    client.conclude = Mock()
    client.complete = Mock()
    client.close = Mock()

    return client


def _make_summary(
    project_id: str,
    status: str = "active",
    fact_count: int = 2,
    intent_count: int = 0,
    working_intent_count: int = 0,
    unclaimed_intent_count: int = 0,
    hint_count: int = 0,
    trial_count: int = 0,
    best_metric: float | None = None,
) -> ProjectSummary:
    """Helper to create ProjectSummary."""
    return ProjectSummary(
        id=project_id,
        title="Test Project",
        status=status,
        bootstrap_enabled=True,
        origin="Test origin",
        goal="Test goal",
        goal_metric="val_loss",
        goal_target=None,
        goal_direction="minimize",
        budget_max_trials=10,
        created_at="2026-09-09T00:00:00Z",
        fact_count=fact_count,
        intent_count=intent_count,
        working_intent_count=working_intent_count,
        unclaimed_intent_count=unclaimed_intent_count,
        hint_count=hint_count,
        trial_count=trial_count,
        best_metric=best_metric,
    )


def _make_detail(
    project_id: str,
    facts: list[Fact] | None = None,
    intents: list[Intent] | None = None,
    bootstrap_enabled: bool = True,
) -> ProjectDetail:
    """Helper to create ProjectDetail."""
    if facts is None:
        facts = [
            Fact(id="origin", description="Origin fact", created_at="2026-09-09T00:00:00Z"),
            Fact(id="goal", description="Goal fact", created_at="2026-09-09T00:00:00Z"),
        ]
    if intents is None:
        intents = []

    return ProjectDetail(
        project=ProjectMeta(
            id=project_id,
            title="Test Project",
            status="active",
            bootstrap_enabled=bootstrap_enabled,
            origin="Test origin",
            goal="Test goal",
            goal_metric="val_loss",
            goal_target=None,
            goal_direction="minimize",
            budget_max_trials=10,
            created_at="2026-09-09T00:00:00Z",
        ),
        facts=facts,
        intents=intents,
        hints=[],
    )


def test_fresh_project_dispatches_bootstrap(config_yaml, fake_client, stub_executor, monkeypatch):
    """Fresh project with bootstrap enabled dispatches bootstrap task."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)
    executor = stub_executor(loop)

    # Set up state: fresh project (only origin/goal facts, no intents)
    summary = _make_summary("p1")
    detail = _make_detail("p1", bootstrap_enabled=True)

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail
    fake_client.create_intent.return_value = Mock(
        ok=True,
        status_code=200,
        data={
            "id": "i1",
            "from": ["origin"],
            "to": None,
            "description": "bootstrap",
            "creator": "dispatcher.bootstrap",
            "worker": None,
            "last_heartbeat_at": None,
            "created_at": "2026-09-09T01:00:00Z",
            "concluded_at": None,
        },
    )
    fake_client.claim_intent.return_value = Mock(ok=True, status_code=200)

    loop.run(once=True)

    # Should have created a bootstrap intent, claimed it, and submitted the task.
    # Acquisition goes through claim, not heartbeat: the intent it just created
    # is unclaimed, and heartbeat only renews an existing holder.
    assert fake_client.create_intent.called
    assert fake_client.claim_intent.called
    assert executor.submit.called


def test_graph_changed_dispatches_reason(config_yaml, fake_client, stub_executor, monkeypatch):
    """Project with new facts dispatches reason task."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)
    executor = stub_executor(loop)

    # Initialize checkpoint with old state
    loop.reason_checkpoints["p1"] = ReasonCheckpoint(fact_count=2, hint_count=0, open_intent_count=0)

    # Set up state: project has new facts
    summary = _make_summary("p1", fact_count=3, intent_count=0)
    detail = _make_detail("p1", bootstrap_enabled=False)
    detail.facts.append(Fact(id="f1", description="New fact", created_at="2026-09-09T01:00:00Z"))

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail
    fake_client.claim_reason.return_value = Mock(ok=True, status_code=200)

    loop.run(once=True)

    # Should have claimed reason and submitted the task
    assert fake_client.claim_reason.called
    call_args = fake_client.claim_reason.call_args
    assert call_args[0][0] == "p1"  # project_id
    assert "facts" in call_args[0][2]  # trigger should mention facts
    assert executor.submit.called


def test_unclaimed_intent_dispatches_explore(config_yaml, fake_client, stub_executor, monkeypatch):
    """Project with unclaimed intent dispatches explore task."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)
    executor = stub_executor(loop)

    # Set up state: project has an unclaimed intent
    summary = _make_summary("p1", intent_count=1, unclaimed_intent_count=1)
    intent = Intent(
        id="i1",
        from_=["origin"],
        to=None,
        description="Try baseline",
        creator="human",
        worker=None,
        created_at="2026-09-09T01:00:00Z",
    )
    detail = _make_detail("p1", bootstrap_enabled=False, intents=[intent])

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail
    fake_client.claim_intent.return_value = Mock(ok=True, status_code=200)

    loop.run(once=True)

    # Should have claimed the open intent and submitted the task. Claim is the
    # acquire op (200 free / 409 taken); heartbeat would 403 here because this
    # worker does not hold the lease yet.
    assert fake_client.claim_intent.called
    call_args = fake_client.claim_intent.call_args
    assert call_args[0][0] == "p1"  # project_id
    assert call_args[0][1] == "i1"  # intent_id
    assert executor.submit.called


def test_nothing_to_do_skips_dispatch(config_yaml, fake_client, stub_executor, monkeypatch):
    """A settled non-initial project with an unchanged graph dispatches nothing.

    The project must carry a third fact: a project holding only origin/goal is
    an *initial* project, and an initial project always earns reason("initial")
    regardless of any checkpoint, which is a different branch of the tree.
    """
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)
    executor = stub_executor(loop)

    # Checkpoint matches the current graph exactly, so no trigger fires.
    loop.reason_checkpoints["p1"] = ReasonCheckpoint(fact_count=3, hint_count=0, open_intent_count=0)

    # Non-initial project, every intent already concluded.
    summary = _make_summary("p1", fact_count=3, intent_count=1)
    detail = _make_detail(
        "p1",
        bootstrap_enabled=False,
        facts=[
            Fact(id="origin", description="Origin fact", created_at="2026-09-09T00:00:00Z"),
            Fact(id="goal", description="Goal fact", created_at="2026-09-09T00:00:00Z"),
            Fact(id="f001", description="Baseline measured", created_at="2026-09-09T01:00:00Z"),
        ],
        intents=[
            Intent(
                id="i001",
                from_=["origin"],
                to="f001",
                description="Measure the baseline",
                creator="human",
                worker="w1",
                created_at="2026-09-09T00:30:00Z",
                concluded_at="2026-09-09T01:00:00Z",
            )
        ],
    )

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail

    loop.run(once=True)

    # Should not have dispatched anything. It has to be claim_intent that stays
    # untouched, not heartbeat: nothing calls heartbeat at dispatch time, so
    # asserting on it would pass even if a task had been handed out.
    assert not fake_client.claim_reason.called
    assert not fake_client.claim_intent.called
    assert not fake_client.create_intent.called
    assert not executor.submit.called


def test_budget_exhausted_skips_dispatch(config_yaml, fake_client, monkeypatch):
    """Project at budget limit does not dispatch."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)

    # Set up state: project at budget
    summary = _make_summary("p1", trial_count=10, unclaimed_intent_count=1)
    summary.budget_max_trials = 10
    intent = Intent(
        id="i1",
        from_=["origin"],
        to=None,
        description="Try something",
        creator="human",
        worker=None,
        created_at="2026-09-09T01:00:00Z",
    )
    detail = _make_detail("p1", bootstrap_enabled=False, intents=[intent])

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail

    # Run once
    loop.run(once=True)

    # Should not dispatch. claim_intent is the acquire op, so an untouched
    # claim_intent is what proves no explore task was handed out.
    assert not fake_client.claim_intent.called


def test_inactive_project_skips_dispatch(config_yaml, fake_client, monkeypatch):
    """Stopped or completed projects are not dispatched."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)

    # Set up state: stopped project with work
    summary = _make_summary("p1", status="stopped", unclaimed_intent_count=1)

    fake_client.list_projects.return_value = [summary]

    # Run once
    loop.run(once=True)

    # Should not dispatch
    assert not fake_client.get_project.called


def test_claimed_intent_already_running_locally_skips(config_yaml, fake_client, monkeypatch):
    """Intent already running in this dispatcher is not re-dispatched."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)

    # Add a running task for this intent
    from mts.dispatcher.models import RunningTask
    from mts.dispatcher.runtime.cancellation import TaskCancellation

    loop.futures = {
        Mock(done=Mock(return_value=False)): RunningTask(
            project_id="p1",
            task_type="explore",
            worker_name="mock1",
            cancellation=TaskCancellation(),
            intent_id="i1",
        )
    }

    # Set up state: project has the intent
    summary = _make_summary("p1", intent_count=1, unclaimed_intent_count=1)
    intent = Intent(
        id="i1",
        from_=["origin"],
        to=None,
        description="Try baseline",
        creator="human",
        worker=None,
        created_at="2026-09-09T01:00:00Z",
    )
    detail = _make_detail("p1", bootstrap_enabled=False, intents=[intent])

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail

    # Run once
    loop.run(once=True)

    # Should not dispatch again
    assert not fake_client.claim_intent.called


def test_no_available_worker_skips_dispatch(config_yaml, fake_client, monkeypatch):
    """When all workers are busy, dispatch is skipped."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)

    # Fill up all workers
    from mts.dispatcher.models import RunningTask
    from mts.dispatcher.runtime.cancellation import TaskCancellation

    loop.futures = {
        Mock(done=Mock(return_value=False)): RunningTask(
            project_id="p1",
            task_type="explore",
            worker_name="mock1",
            cancellation=TaskCancellation(),
            intent_id="i_other",
        ),
        Mock(done=Mock(return_value=False)): RunningTask(
            project_id="p2",
            task_type="reason",
            worker_name="mock1",
            cancellation=TaskCancellation(),
        ),
    }

    # Set up state: project has work
    summary = _make_summary("p1", unclaimed_intent_count=1)
    intent = Intent(
        id="i1",
        from_=["origin"],
        to=None,
        description="Try baseline",
        creator="human",
        worker=None,
        created_at="2026-09-09T01:00:00Z",
    )
    detail = _make_detail("p1", bootstrap_enabled=False, intents=[intent])

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail

    # Run once
    loop.run(once=True)

    # Should not dispatch (all workers busy)
    assert not fake_client.claim_intent.called


def test_reason_lease_held_skips_reason_dispatch(config_yaml, fake_client, monkeypatch):
    """When reason is already held, reason dispatch is skipped."""
    loop = DispatcherLoop(config_yaml)
    monkeypatch.setattr(loop, "client", fake_client)

    # Initialize checkpoint showing change
    loop.reason_checkpoints["p1"] = Mock(fact_count=2, hint_count=0, open_intent_count=0)

    # Set up state: project changed, but reason is held
    summary = _make_summary("p1", fact_count=3)
    detail = _make_detail("p1", bootstrap_enabled=False)
    detail.facts.append(Fact(id="f1", description="New", created_at="2026-09-09T01:00:00Z"))
    detail.project.reason = Mock(worker="other_worker", trigger="initial")

    fake_client.list_projects.return_value = [summary]
    fake_client.get_project.return_value = detail

    # Run once
    loop.run(once=True)

    # Should not claim reason (already held)
    assert not fake_client.claim_reason.called
