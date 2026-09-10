from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from mts.dispatcher.config import DispatchConfig, WorkerConfig
from mts.dispatcher.contracts import ExploreOutcome
from mts.dispatcher.protocol.client import MTSClient
from mts.dispatcher.runtime.activity import ActivityWatcher
from mts.dispatcher.runtime.cancellation import TaskCancellation
from mts.dispatcher.runtime.heartbeat import HeartbeatLease
from mts.dispatcher.runtime.local_backend import LocalBackend
from mts.dispatcher.runtime.process import ProcessResult

PROCESS_COMMUNICATE_GRACE_SECONDS = 15
LOG_PREVIEW_LIMIT = 1200
LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class ConcludeWriteResult:
    status: str
    fact_id: str | None = None


def preview(text: str, limit: int = LOG_PREVIEW_LIMIT) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def did_timeout(result: ProcessResult) -> bool:
    return not result.cancelled and (
        result.timed_out or result.returncode in (124, 137)
    )


def cancel_reason(
    result: ProcessResult, cancellation: TaskCancellation | None = None
) -> str | None:
    if result.cancelled:
        return result.cancel_reason or "cancelled"
    if cancellation is not None:
        return cancellation.reason
    return None


def communicate_timeout(
    timeout_seconds: int, grace_seconds: int = PROCESS_COMMUNICATE_GRACE_SECONDS
) -> int:
    return timeout_seconds + grace_seconds


def task_healthcheck_enabled(config: DispatchConfig) -> bool:
    if config.runtime.execution == "local":
        return False
    return config.runtime.worker_healthcheck == "startup_and_task"


def write_graph_snapshot_reference(
    backend: LocalBackend,
    workdir: str,
    graph_yaml: str,
    *,
    phase: str,
) -> str:
    """Write graph YAML to a file under the workdir and return reference text.

    Unlike Cairn which writes to /tmp, MTS writes to <workdir>/.mts-prompts/<phase>-<hex>/graph.yaml
    so the agent can read it and a human can audit it.
    """
    from pathlib import Path

    snapshot_dir = Path(workdir) / ".mts-prompts" / f"{phase}-{uuid.uuid4().hex[:12]}"
    path = snapshot_dir / "graph.yaml"
    backend.write_text_file(workdir, str(path), graph_yaml)
    return (
        "The graph YAML snapshot is stored in this file:\n\n"
        f"{path}\n\n"
        "Before using the graph, read the entire file and treat its contents as the YAML snapshot "
        "for this Graph section."
    )


def run_worker_process(
    backend: LocalBackend,
    workdir: str,
    worker: WorkerConfig,
    argv: list[str],
    *,
    phase: str,
    timeout_seconds: int,
    lease: HeartbeatLease | None = None,
    cancellation: TaskCancellation | None = None,
    project_id: str | None = None,
    session: str | None = None,
    intent_id: str | None = None,
) -> ProcessResult:
    LOG.info(
        "starting worker process workdir=%s worker=%s phase=%s timeout=%ss",
        workdir,
        worker.name,
        phase,
        timeout_seconds,
    )
    # Driver-supplied env (e.g. the CLI escape hatch claudecode needs under root)
    # goes underneath worker.env, so an explicit config entry always wins.
    from mts.dispatcher.workers.registry import get_driver

    try:
        driver_env = get_driver(worker.type).process_env(worker)
    except KeyError:
        driver_env = {}
    process = backend.build_exec_process(
        workdir,
        {**driver_env, **worker.env},
        argv,
        timeout_seconds=timeout_seconds,
    )
    process.start()
    if lease is not None:
        lease.attach_process(process)
    if cancellation is not None:
        cancellation.attach_process(process)
    # 旁路观测：读 CLI 自己写的 session transcript，把 agent 的动作喂给 WebUI。
    # 没有 session（mock driver）时 watcher 自动空转。
    watcher = (
        ActivityWatcher(
            session,
            project_id=project_id or "",
            worker=worker.name,
            phase=phase,
            intent_id=intent_id,
            workdir=workdir,
        )
        if project_id
        else None
    )
    try:
        if watcher is not None:
            watcher.start()
        return process.communicate(timeout=communicate_timeout(timeout_seconds))
    finally:
        if watcher is not None:
            watcher.stop()
        if lease is not None:
            lease.attach_process(None)
        if cancellation is not None:
            cancellation.attach_process(None)


def project_allows_conclude_fallback(
    client: MTSClient, project_id: str, *, worker_name: str, intent_id: str
) -> bool:
    project = client.get_project(project_id)
    if project.project.status == "active":
        return True
    LOG.info(
        "skip conclude fallback because project is no longer active project=%s intent=%s worker=%s status=%s",
        project_id,
        intent_id,
        worker_name,
        project.project.status,
    )
    return False


def best_effort_release_reason(
    client: MTSClient, project_id: str, worker_name: str
) -> None:
    response = client.release_reason(project_id, worker_name)
    if not response.ok and response.status_code not in (403, 409):
        LOG.warning(
            "reason release failed project=%s worker=%s status=%s",
            project_id,
            worker_name,
            response.status_code,
        )
    elif response.ok:
        LOG.info("released reason project=%s worker=%s", project_id, worker_name)
    else:
        LOG.info(
            "reason release skipped project=%s worker=%s status=%s",
            project_id,
            worker_name,
            response.status_code,
        )


def write_conclude_result(
    client: MTSClient,
    project_id: str,
    intent_id: str,
    worker_name: str,
    outcome: ExploreOutcome,
    *,
    source: str,
    phase_ms: int,
    total_ms: int | None = None,
) -> str:
    """MTS extension: post outcome with metrics and trial_id to /conclude."""
    return write_conclude_result_with_fact_id(
        client,
        project_id,
        intent_id,
        worker_name,
        outcome,
        source=source,
        phase_ms=phase_ms,
        total_ms=total_ms,
    ).status


def write_conclude_result_with_fact_id(
    client: MTSClient,
    project_id: str,
    intent_id: str,
    worker_name: str,
    outcome: ExploreOutcome,
    *,
    source: str,
    phase_ms: int,
    total_ms: int | None = None,
) -> ConcludeWriteResult:
    """MTS extension: post outcome with metrics and trial_id to /conclude."""
    response = client.conclude(
        project_id,
        intent_id,
        worker_name,
        outcome.description,
        metrics=outcome.metrics,
        trial_id=outcome.trial_id,
    )
    if response.ok:
        fact_id: str | None = None
        if isinstance(response.data, dict):
            fact = response.data.get("fact")
            if isinstance(fact, dict):
                candidate = fact.get("id")
                if isinstance(candidate, str) and candidate:
                    fact_id = candidate
        if total_ms is None:
            LOG.info(
                "intent concluded project=%s intent=%s worker=%s source=%s phase_ms=%s",
                project_id,
                intent_id,
                worker_name,
                source,
                phase_ms,
            )
        else:
            LOG.info(
                "intent concluded project=%s intent=%s worker=%s source=%s phase_ms=%s total_ms=%s",
                project_id,
                intent_id,
                worker_name,
                source,
                phase_ms,
                total_ms,
            )
        return ConcludeWriteResult(status="success", fact_id=fact_id)
    if response.status_code == 403:
        LOG.info(
            "project became inactive during conclude project=%s intent=%s worker=%s",
            project_id,
            intent_id,
            worker_name,
        )
    else:
        LOG.warning(
            "conclude write failed project=%s intent=%s worker=%s status=%s body=%s",
            project_id,
            intent_id,
            worker_name,
            response.status_code,
            response.text,
        )
    best_effort_release(client, project_id, intent_id, worker_name)
    return ConcludeWriteResult(status="failed", fact_id=None)


def best_effort_release(
    client: MTSClient, project_id: str, intent_id: str, worker_name: str
) -> None:
    response = client.release(project_id, intent_id, worker_name)
    if not response.ok and response.status_code not in (403, 409):
        LOG.warning(
            "release failed project=%s intent=%s worker=%s status=%s",
            project_id,
            intent_id,
            worker_name,
            response.status_code,
        )
    elif response.ok:
        LOG.info(
            "released intent project=%s intent=%s worker=%s",
            project_id,
            intent_id,
            worker_name,
        )
    else:
        LOG.info(
            "release skipped project=%s intent=%s worker=%s status=%s",
            project_id,
            intent_id,
            worker_name,
            response.status_code,
        )
