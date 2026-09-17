"""DispatcherManager: 在后台线程里跑 DispatcherLoop。

WebUI 的「启动调度」按钮和 CLI 的 `mts dispatch` 走同一条代码路径：把项目存储的
SearchConfigIn 合成成一份 DispatchConfig，写到临时 YAML，再交给 DispatcherLoop。
配置解析只有一处真源（DispatchConfig.load），prompt 全部来自 prompts/*.md。

单例模式，由 FastAPI app 持有。每个项目同时只能运行一个调度循环。
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from mts.dispatcher.config import DispatchConfig
from mts.dispatcher.runtime.activity import get_activity_bus
from mts.dispatcher.scheduler.loop import DispatcherLoop
from mts.server.models import SearchConfigIn
from mts.server.services import Service

LOG = logging.getLogger(__name__)

LOG_RING_SIZE = 1000
DEFAULT_BOARD_URL = "http://127.0.0.1:8000"
DEFAULT_TASKS: dict[str, dict[str, int]] = {
    "bootstrap": {"timeout": 1800, "conclude_timeout": 300},
    "reason": {"timeout": 600, "max_intents": 3},
    "explore": {"timeout": 7200, "conclude_timeout": 600},
}


# 每个 driver 能用哪些 provider kind。不匹配的组合不是"配置得差一点"，而是凭证
# 注入到了错误的环境变量上：kind=openai 的端点被塞进 ANTHROPIC_BASE_URL，claude
# CLI 拿着一个 OpenAI 协议地址去调，必然失败——而且要等满 bootstrap 的 1800s
# 超时才暴露出来。所以在启动前就要拦住。
_DRIVER_PROVIDER_KINDS: dict[str, frozenset[str]] = {
    "claudecode": frozenset({"anthropic", "claudecode"}),
    "llm": frozenset({"openai", "anthropic"}),
}


def provider_kind_matches(driver: str, kind: str) -> bool:
    """driver 能否使用这种 kind 的 provider。未知 driver 一律放行。"""
    allowed = _DRIVER_PROVIDER_KINDS.get(driver)
    return True if allowed is None else kind in allowed


def provider_env(provider: dict[str, Any] | None, driver: str) -> dict[str, str]:
    """Translate a stored provider row into the env vars its driver reads.

    The `llm` driver reads LLM_* and switches wire protocol on LLM_API_FORMAT, so
    an OpenAI-compatible endpoint needs nothing more than these four keys. The
    `claudecode` driver shells out to the `claude` CLI, which reads ANTHROPIC_*.
    Returns {} when there is no provider, leaving the host environment in charge.
    """
    if provider is None or driver == "mock":
        return {}

    kind = provider.get("kind") or "openai"
    base_url = (provider.get("base_url") or "").rstrip("/")
    api_key = provider.get("api_key") or ""
    model = provider.get("model") or ""
    extra = provider.get("extra") or {}

    if driver == "claudecode":
        env = {}
        if base_url:
            env["ANTHROPIC_BASE_URL"] = base_url
        if api_key:
            env["ANTHROPIC_AUTH_TOKEN"] = api_key
        if model:
            env["ANTHROPIC_MODEL"] = model
        return env

    env: dict[str, str] = {"LLM_API_FORMAT": "anthropic" if kind == "anthropic" else "openai"}
    if base_url:
        env["LLM_BASE_URL"] = base_url
    if api_key:
        env["LLM_API_KEY"] = api_key
    if model:
        env["LLM_MODEL"] = model
    if extra.get("max_tokens"):
        env["LLM_MAX_TOKENS"] = str(extra["max_tokens"])
    if extra.get("temperature") is not None:
        env["LLM_TEMPERATURE"] = str(extra["temperature"])
    headers = extra.get("headers") or {}
    if headers:
        import json as _json

        env["LLM_EXTRA_HEADERS"] = _json.dumps(headers)
    return env


def build_dispatch_config(
    config: SearchConfigIn,
    *,
    server: str,
    workspace_root: Path | None = None,
    prompt_group: str = "default",
    resolve_provider: Any = None,
) -> dict[str, Any]:
    """把项目的 worker roster 合成成 DispatchConfig 的 dict 形态。

    凭证解析：worker 指定 provider_id 时从 providers 表取凭证注入 env；未指定时
    `resolve_provider(None)` 给出默认 provider；两者都没有就回退到宿主机环境变量
    （ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL），保持旧行为。
    """
    workers: list[dict[str, Any]] = []
    max_workers = 2

    # worker_requirement 是新格式，workers 是旧格式。只填了旧 workers 的配置必须走
    # 旧分支，否则 driver="llm" 会被顶替成 claudecode，注入的凭证环境变量也跟着错
    # （LLM_* vs ANTHROPIC_*）。
    if config.worker_requirement is None and config.workers:
        max_workers = max(1, config.max_workers or len(config.workers))
        for spec in config.workers:
            env = {}
            if resolve_provider is not None and spec.driver != "mock":
                provider = resolve_provider(getattr(spec, "provider_id", None))
                env = provider_env(provider, spec.driver)
            workers.append(
                {
                    "name": spec.name,
                    "type": spec.driver,
                    "task_types": ["bootstrap", "reason", "explore"],
                    "max_running": 1,
                    "priority": 0,
                    "env": env,
                }
            )
    elif config.worker_requirement is not None:
        req = config.worker_requirement
        max_workers = max(1, req.count)
        env = {}
        if resolve_provider is not None and req.worker_type != "mock":
            provider = resolve_provider(req.provider_id)
            if provider is not None:
                kind = provider.get("kind") or "openai"
                if not provider_kind_matches(req.worker_type, kind):
                    raise RuntimeError(
                        f"provider '{provider.get('name') or provider.get('id')}' is kind="
                        f"{kind}, which worker_type={req.worker_type} cannot use. "
                        f"Pick a provider of kind "
                        f"{'/'.join(sorted(_DRIVER_PROVIDER_KINDS[req.worker_type]))}, "
                        f"or change the project's worker type."
                    )
            env = provider_env(provider, req.worker_type)
        # 项目只声明「要几个什么类型的 worker」，名字由这里合成。
        for i in range(max_workers):
            workers.append(
                {
                    "name": f"{req.worker_type}-{i + 1}",
                    "type": req.worker_type,
                    "task_types": ["bootstrap", "reason", "explore"],
                    "max_running": 1,
                    "priority": 0,
                    "env": env,
                }
            )

    if not workers:
        raise RuntimeError("No workers configured: add at least one worker to the project roster")

    local: dict[str, Any] = {"completed_action": "keep"}
    if workspace_root is not None:
        local["workspace_root"] = str(workspace_root)

    return {
        "server": server,
        "runtime": {
            "max_workers": max_workers,
            "max_running_projects": 1,
            "max_project_workers": max_workers,
            "interval": 3,
            "healthcheck_timeout": 15,
            "worker_healthcheck": "disabled",
            "execution": "local",
            "prompt_group": prompt_group,
        },
        "tasks": DEFAULT_TASKS,
        "local": local,
        "common_env": {},
        "workers": workers,
    }


class _RingHandler(logging.Handler):
    """把 mts.dispatcher.* 的日志抄进环形缓冲区，供 WebUI 轮询。"""

    def __init__(self, ring: deque[str]) -> None:
        super().__init__()
        self.ring = ring
        self.setFormatter(
            logging.Formatter(fmt="[%(asctime)s] %(levelname)s %(name)s %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.ring.append(self.format(record))
        except Exception:  # noqa: BLE001 - a log sink must never raise
            pass


class DispatcherManager:
    """管理项目的后台调度循环：启动、停止、状态查询、日志。"""

    def __init__(self, service: Service, root: Path, *, server: str | None = None):
        self.service = service
        self.root = root
        self.runs_dir = root / "runs"
        # DispatcherLoop only talks to the board over HTTP, so it needs the URL this
        # server is reachable at. create_app() does not know the port uvicorn bound,
        # hence the env override; pass server= explicitly when the caller does know.
        self.server = server or os.environ.get("MTS_BOARD_URL") or DEFAULT_BOARD_URL
        # project_id -> {"run_id", "thread", "loop", "status", "logs", "config", ...}
        self._active: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    def start(self, project_id: str, config: SearchConfigIn | None = None) -> str:
        """启动调度循环。该项目已有运行中的循环时抛出 RuntimeError。"""
        with self._lock:
            entry = self._active.get(project_id)
            if entry is not None and entry["status"] in ("running", "stopping"):
                raise RuntimeError(f"Project {project_id} already has a running dispatcher")

            proj = self.service.get_project(project_id)
            if proj is None:
                raise RuntimeError(f"Project {project_id} not found")

            if config is None:
                config = SearchConfigIn.model_validate(self.service.get_search_config(project_id))

            payload = build_dispatch_config(
                config,
                server=self.server,
                workspace_root=self.runs_dir / project_id,
                resolve_provider=self._resolve_provider,
            )
            config_path = self._write_config(project_id, payload)
            # 提前校验，让配置错误在 HTTP 400 里回给前端而不是死在后台线程。
            DispatchConfig.load(config_path)

            # 按项目起的循环只许调度这个项目：WebUI 每个项目一个 DispatcherManager
            # entry，不收窄的话两个项目各点一次「启动搜索」就会互相抢 intent。
            loop = DispatcherLoop(config_path, project_id=project_id)

            # 活动流按项目留在环形缓冲里，跨 run 不会自己消失。不清的话新一轮的
            # 界面开头挂的是上一轮的 agent 事件，看起来像新 run 一启动就干完了活。
            # 清掉同时把 seq 归零，所以前端 startSearch 也要跟着重置 activitySeq，
            # 否则新一轮的低位 seq 会被旧的高水位全部过滤掉。
            get_activity_bus().clear(project_id)

            run_id = f"run_{self.service.next_trial_id(project_id)}"
            logs: deque[str] = deque(maxlen=LOG_RING_SIZE)
            handler = _RingHandler(logs)
            logging.getLogger("mts.dispatcher").addHandler(handler)

            record: dict[str, Any] = {
                "run_id": run_id,
                "loop": loop,
                "status": "running",
                "logs": logs,
                "handler": handler,
                "config": config,
                "config_path": config_path,
                "started_at": time.time(),
                "thread": None,
                "result": None,
                "error": None,
            }
            thread = threading.Thread(
                target=self._run_loop,
                args=(project_id, loop),
                name=f"dispatch-{project_id}",
                daemon=True,
            )
            record["thread"] = thread
            self._active[project_id] = record

        logs.append(f"dispatcher starting project={project_id} config={config_path}")
        thread.start()
        return run_id

    def _run_loop(self, project_id: str, loop: DispatcherLoop) -> None:
        try:
            loop.run()
            # A loop that exited because the user pressed stop is not "completed".
            self._finish(project_id, "stopped" if loop.stop_requested else "completed")
        except Exception as exc:  # noqa: BLE001 - surface any loop failure to the UI
            LOG.exception("dispatcher loop failed project=%s", project_id)
            self._finish(project_id, "error", error=f"{type(exc).__name__}: {exc}")

    def _finish(self, project_id: str, status: str, *, error: str | None = None) -> None:
        with self._lock:
            entry = self._active.get(project_id)
            if entry is None:
                return
            entry["status"] = status
            entry["error"] = error
            handler = entry.pop("handler", None)
            entry["logs"].append(f"dispatcher {status}" + (f": {error}" if error else ""))
        if handler is not None:
            logging.getLogger("mts.dispatcher").removeHandler(handler)

    def forget(self, project_id: str) -> None:
        """丢掉项目的调度记录。项目被删除时用，不是 stop 的替代。

        必须自己摘掉日志 handler：后台线程收尾时调的 `_finish` 找不到记录会直接
        返回，handler 就一直挂在 mts.dispatcher logger 上，继续往一个没人再读的
        队列里写。
        """
        with self._lock:
            entry = self._active.pop(project_id, None)
            handler = entry.pop("handler", None) if entry else None
        if handler is not None:
            logging.getLogger("mts.dispatcher").removeHandler(handler)

    def stop(self, project_id: str) -> None:
        """请求停止调度循环：取消所有在跑的任务，循环在下一轮退出。"""
        with self._lock:
            entry = self._active.get(project_id)
            if entry is None or entry["status"] not in ("running", "stopping"):
                raise RuntimeError(f"No active dispatcher for project {project_id}")
            entry["status"] = "stopping"
            loop: DispatcherLoop = entry["loop"]
            entry["logs"].append("stop requested")

        # Break the poll loop first, then cancel in-flight tasks: cancelling alone
        # leaves `run()` spinning, which is why the stop button looked like a no-op.
        loop.request_stop()
        for task in list(loop.futures.values()):
            task.cancellation.cancel("dispatcher stopped")

    # -- introspection ----------------------------------------------------

    def status(self, project_id: str) -> dict[str, Any]:
        """查询调度状态。WebUI 读 running / elapsed_sec / active_agents。"""
        with self._lock:
            entry = self._active.get(project_id)
            if entry is None:
                return {"status": "idle", "running": False}
            loop: DispatcherLoop = entry["loop"]
            snapshot = {
                "run_id": entry["run_id"],
                "status": entry["status"],
                "running": entry["status"] in ("running", "stopping"),
                "elapsed_sec": time.time() - entry["started_at"],
                "active_agents": len(loop.futures),
                "config": entry["config"].model_dump() if entry.get("config") else None,
                "error": entry.get("error"),
            }
        snapshot["trials_completed"] = len(self.service.list_trials(project_id))
        return snapshot

    def logs(self, project_id: str, limit: int = 100) -> dict[str, Any]:
        """返回最新 limit 行日志，WebUI 直接渲染 `output`。"""
        with self._lock:
            entry = self._active.get(project_id)
            if entry is None:
                return {"output": "", "lines": []}
            lines = list(entry["logs"])[-limit:]
        return {"output": "\n".join(lines), "lines": lines}

    # -- helpers ----------------------------------------------------------

    def _resolve_provider(self, provider_id: str | None) -> dict[str, Any] | None:
        """provider_id → row with api_key; falls back to the default provider."""
        if provider_id:
            provider = self.service.get_provider_secret(provider_id)
            if provider is not None:
                return provider
            LOG.warning("provider %s not found, falling back to default", provider_id)
        return self.service.get_default_provider_secret()

    def _write_config(self, project_id: str, payload: dict[str, Any]) -> Path:
        directory = Path(tempfile.mkdtemp(prefix=f"mts-dispatch-{project_id}-"))
        path = directory / "dispatch.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path


_manager: DispatcherManager | None = None
_manager_lock = threading.Lock()


def get_dispatcher_manager(service: Service, root: Path, *, server: str | None = None) -> DispatcherManager:
    """DEPRECATED：进程级单例，只在第一次调用时绑定 service/root。

    `create_app` 现在自己持有 manager（app.py:dispatcher_manager），因为同进程建
    第二个 app（测试、`mts demo` 后再 `mts serve`）时这里会把第一个 app 的库和
    runs 目录交给第二个 app 用。保留此函数仅为兼容外部调用方。
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = DispatcherManager(service, root, server=server)
        return _manager
