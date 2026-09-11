"""FastAPI application for the board.

Serves the REST API (facts / intents / trials / knowledge) and the static WebUI.
Run via `mts serve` or `uvicorn mts.server.app:create_app --factory`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from mts.server.db import Database
from mts.server.models import (
    CompleteRequest,
    ConcludeRequest,
    CreateIntentRequest,
    DispatchStart,
    HeartbeatRequest,
    HintCreate,
    KnowledgeCreate,
    ObservationCreate,
    ProjectCreate,
    ProjectUpdate,
    ProviderIn,
    ProviderUpdate,
    ReasonClaimRequest,
    SearchConfigIn,
    TrialHeartbeat,
    TrialRegister,
)
from mts.server.provider_probe import probe_provider
from mts.server.services import Service

STATIC_DIR = Path(__file__).parent / "static"


def create_app(root: str | Path | None = None, db_path: str | Path | None = None) -> FastAPI:
    root = Path(root) if root else Path.cwd()
    db_path = Path(db_path) if db_path else root / "data" / "board.db"

    db = Database(db_path)
    db.configure()
    service = Service(db, root)

    app = FastAPI(title="ModelTrainSelf Board", version="0.1.0")

    def require_active_project(pid: str) -> None:
        """404 when the project is unknown, 403 when it is not active.

        Every lease write goes through this. A worker that gets 403 treats it
        as a hard failure and kills its attached process, so a human stopping
        a project is what cancels the agents already running on it.
        """
        status = service.project_status(pid)
        if status is None:
            raise HTTPException(404, "project not found")
        if status != "active":
            raise HTTPException(403, f"project is {status}")

    # =====================================================================
    # Settings
    # =====================================================================

    @app.get("/api/settings")
    def get_settings():
        return db.get_settings()

    @app.put("/api/settings")
    def put_settings(payload: dict[str, Any]):
        return db.put_settings(**payload)

    # =====================================================================
    # Projects
    # =====================================================================

    @app.get("/api/projects")
    def list_projects():
        return service.list_projects()

    @app.post("/api/projects")
    def create_project(payload: ProjectCreate):
        return service.create_project(payload)

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        proj = service.get_project(pid)
        if proj is None:
            raise HTTPException(404, "project not found")
        facts = service.list_facts(pid)
        intents = service.list_intents(pid)
        hints = service.list_hints(pid)
        search_config = service.get_search_config(pid)
        return {
            "project": {
                "id": proj["id"],
                "title": proj["title"],
                "status": proj["status"],
                "bootstrap_enabled": proj["bootstrap_enabled"],
                "origin": proj["origin"],
                "goal": proj["goal"],
                "goal_metric": proj["goal_metric"],
                "goal_target": proj["goal_target"],
                "goal_direction": proj["goal_direction"],
                "budget_max_trials": proj["budget_max_trials"],
                "created_at": proj["created_at"],
                "reason": service._project_reason_from_row(proj),
                "search_config": search_config,
            },
            "facts": facts,
            "intents": intents,
            "hints": hints,
        }

    @app.patch("/api/projects/{pid}")
    def update_project(pid: str, payload: ProjectUpdate):
        """就地修改项目配置（目标、指标、预算等），只写入本次提交的字段。"""
        proj = service.update_project(pid, payload.model_dump(exclude_unset=True))
        if proj is None:
            raise HTTPException(404, "project not found")
        return proj

    @app.patch("/api/projects/{pid}/status")
    def set_status(pid: str, status: str = Query(..., pattern="^(active|stopped|completed)$")):
        proj = service.set_project_status(pid, status)
        if proj is None:
            raise HTTPException(404, "project not found")
        return proj

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        """删除项目及其所有关联数据。"""
        proj = service.get_project(pid)
        if proj is None:
            raise HTTPException(404, "project not found")

        # 删除数据库记录
        success = service.delete_project(pid)
        if not success:
            raise HTTPException(500, "failed to delete project")

        return {"ok": True, "id": pid}

    @app.get("/api/projects/{pid}/facts")
    def list_facts(pid: str):
        return service.list_facts(pid)

    # =====================================================================
    # Intents
    # =====================================================================

    @app.get("/api/projects/{pid}/intents")
    def list_intents(pid: str):
        return service.list_intents(pid)

    @app.post("/api/projects/{pid}/intents")
    def create_intent(pid: str, payload: CreateIntentRequest):
        require_active_project(pid)
        return service.create_intent(pid, payload)

    @app.post("/api/projects/{pid}/intents/{iid}/claim")
    def claim_intent(pid: str, iid: str, payload: HeartbeatRequest):
        require_active_project(pid)
        intent = service.claim_intent(pid, iid, payload.worker)
        if intent is None:
            raise HTTPException(409, "intent already claimed by another worker")
        return intent

    @app.post("/api/projects/{pid}/intents/{iid}/heartbeat")
    def heartbeat_intent(pid: str, iid: str, payload: HeartbeatRequest):
        require_active_project(pid)
        if not service.heartbeat_intent(pid, iid, payload.worker):
            raise HTTPException(403, "lease lost")
        return {"ok": True}

    @app.post("/api/projects/{pid}/intents/{iid}/release")
    def release_intent(pid: str, iid: str, payload: HeartbeatRequest):
        require_active_project(pid)
        if not service.release_intent(pid, iid, payload.worker):
            raise HTTPException(403, "release failed")
        return {"ok": True}

    @app.post("/api/projects/{pid}/intents/{iid}/conclude")
    def conclude_intent(pid: str, iid: str, payload: ConcludeRequest):
        require_active_project(pid)
        result = service.conclude_intent(pid, iid, payload)
        if result is None:
            raise HTTPException(404, "intent not found")
        return result

    # =====================================================================
    # Project completion (Cairn protocol)
    # =====================================================================

    @app.post("/api/projects/{pid}/complete")
    def complete_project(pid: str, payload: CompleteRequest):
        require_active_project(pid)
        result = service.complete_project(pid, payload.from_, payload.description, payload.worker)
        if result is None:
            raise HTTPException(404, "project not found")
        return result

    # =====================================================================
    # Reason lease (Cairn protocol)
    # =====================================================================

    @app.post("/api/projects/{pid}/reason/claim")
    def claim_reason(pid: str, payload: ReasonClaimRequest):
        require_active_project(pid)
        result = service.claim_reason(pid, payload.worker, payload.trigger)
        if result == "conflict":
            raise HTTPException(409, "reason lease already held by another worker")
        return {"ok": True}

    @app.post("/api/projects/{pid}/reason/heartbeat")
    def heartbeat_reason(pid: str, payload: HeartbeatRequest):
        require_active_project(pid)
        if not service.heartbeat_reason(pid, payload.worker):
            raise HTTPException(403, "reason lease not held by this worker")
        return {"ok": True}

    @app.post("/api/projects/{pid}/reason/release")
    def release_reason(pid: str, payload: HeartbeatRequest):
        require_active_project(pid)
        if not service.release_reason(pid, payload.worker):
            raise HTTPException(403, "reason lease not held by this worker")
        return {"ok": True}

    # =====================================================================
    # Hints
    # =====================================================================

    @app.get("/api/projects/{pid}/hints")
    def list_hints(pid: str):
        return service.list_hints(pid)

    @app.post("/api/projects/{pid}/hints")
    def add_hint(pid: str, payload: HintCreate):
        return service.add_hint(pid, payload.content, payload.creator)

    @app.delete("/api/projects/{pid}/hints/{hid}")
    def delete_hint(pid: str, hid: str):
        if not service.delete_hint(pid, hid):
            raise HTTPException(404, "hint not found")
        return {"status": "deleted", "id": hid}

    # =====================================================================
    # Export (Cairn protocol YAML + legacy JSON)
    # =====================================================================

    @app.get("/api/projects/{pid}/export")
    def export_graph(pid: str):
        """Export project graph as YAML (Cairn protocol)."""
        proj = service.get_project(pid)
        if proj is None:
            raise HTTPException(404, "project not found")
        yaml_text = service.export_graph_yaml(pid)
        return Response(content=yaml_text, media_type="text/plain")

    @app.get("/api/projects/{pid}/export.yaml")
    def export_context(pid: str):
        """Legacy export (returns JSON with yaml field for WebUI compatibility)."""
        proj = service.get_project(pid)
        if proj is None:
            raise HTTPException(404, "project not found")
        yaml_text = service.export_graph_yaml(pid)
        return JSONResponse({"yaml": yaml_text})

    # =====================================================================
    # Report (human-facing Markdown)
    # =====================================================================

    @app.get("/api/projects/{pid}/report.md")
    def report_markdown(pid: str):
        """Human-readable experiment report derived from the graph + ledger."""
        from mts.reporting import build_report

        proj = service.get_project(pid)
        if proj is None:
            raise HTTPException(404, "project not found")
        try:
            return JSONResponse({"markdown": build_report(service, pid)})
        except KeyError:
            raise HTTPException(404, "project not found")

    # =====================================================================
    # Trials
    # =====================================================================

    # NOTE: The following trials endpoints are kept for backward compatibility
    # with the dispatcher's heartbeat mechanism. Most trials-table machinery
    # is unreachable from the autonomous-agent flow (agents conclude intents
    # → facts, not trials). The register_trial/list_trials/get_trial endpoints
    # exist but are not called by the current agent prompt templates.

    @app.post("/api/projects/{pid}/trials")
    def register_trial(pid: str, payload: TrialRegister):
        trial, err = service.register_trial(pid, payload)
        if err is not None:
            if err.startswith("duplicate:"):
                raise HTTPException(409, f"duplicate experiment (same as {err.split(':')[1]})")
            raise HTTPException(400, err)
        return trial

    @app.get("/api/projects/{pid}/trials")
    def list_trials(pid: str, verdict: str | None = None, status: str | None = None):
        return service.list_trials(pid, verdict, status)

    @app.get("/api/trials/{tid}")
    def get_trial(tid: str):
        trial = service.get_trial(tid)
        if trial is None:
            raise HTTPException(404, "trial not found")
        return trial

    @app.post("/api/trials/{tid}/heartbeat")
    def heartbeat_trial(tid: str, payload: TrialHeartbeat):
        trial = service.get_trial(tid)
        if trial is None:
            raise HTTPException(404, "trial not found")
        ok = service.heartbeat_trial(
            tid, payload.worker,
            progress={
                "step": payload.step,
                "max_steps": payload.max_steps,
                "train_loss": payload.train_loss,
                "val_loss": payload.val_loss,
                "eta_sec": payload.eta_sec,
            },
        )
        if not ok:
            raise HTTPException(409, "trial not owned by this worker")
        return {"ok": True}

    @app.post("/api/trials/{tid}/interrupt")
    def interrupt_trial(tid: str):
        trial = service.get_trial(tid)
        if trial is None:
            raise HTTPException(404, "trial not found")
        service.mark_trial_interrupted(tid)
        return {"ok": True, "status": "interrupted"}

    @app.get("/api/trials/{tid}/metrics")
    def get_metrics(tid: str, downsample: int | None = None):
        trial = service.get_trial(tid)
        if trial is None:
            raise HTTPException(404, "trial not found")
        from mts.trainer.runner import read_metrics

        return read_metrics(trial["out_dir"], downsample=downsample)

    # NOTE: no /error, /code or /metrics-summary endpoints here. They would have to
    # resolve per-trial artifacts (train.py, error.txt, metrics.jsonl) out of the agent
    # workdir, but LocalBackend hands every task of a project the SAME directory
    # (local_backend.py:_project_dir), so there is no per-trial artifact path to read,
    # and nothing in the agent flow writes those filenames anyway. Agents report their
    # results as metrics on the fact they conclude — that is the contract the UI uses.

    # =====================================================================
    # Observations
    # =====================================================================

    @app.get("/api/projects/{pid}/observations")
    def list_observations(pid: str, trial_id: str | None = None, limit: int = 100):
        return service.list_observations(pid, trial_id=trial_id, limit=limit)

    @app.post("/api/projects/{pid}/observations")
    def add_observation(pid: str, payload: ObservationCreate):
        return service.add_observation(pid, payload)

    @app.get("/api/projects/{pid}/observations/{oid}")
    def get_observation(pid: str, oid: str):
        obs = service.get_observation(pid, oid)
        if obs is None:
            raise HTTPException(404, "observation not found")
        return obs

    # =====================================================================
    # Knowledge
    # =====================================================================

    @app.get("/api/knowledge")
    def list_knowledge(tag: str | None = None, q: str | None = None):
        return service.list_knowledge(tag, q)

    @app.get("/api/knowledge/{slug:path}")
    def get_knowledge(slug: str):
        note = service.get_knowledge(slug)
        if note is None:
            raise HTTPException(404, "knowledge note not found")
        return note

    @app.post("/api/knowledge")
    def upsert_knowledge(payload: KnowledgeCreate):
        return service.upsert_knowledge(payload)

    # =====================================================================
    # LLM providers
    # =====================================================================

    @app.get("/api/providers")
    def list_providers():
        """所有已配置的 LLM 端点。api_key 只回掩码，不回明文。"""
        return service.list_providers()

    @app.post("/api/providers")
    def create_provider(payload: ProviderIn):
        if payload.kind != "claudecode" and not payload.base_url:
            raise HTTPException(400, "base_url is required for openai/anthropic providers")
        try:
            return service.create_provider(payload)
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"provider name '{payload.name}' already exists")

    @app.get("/api/providers/{provider_id}")
    def get_provider(provider_id: str):
        provider = service.get_provider(provider_id)
        if provider is None:
            raise HTTPException(404, "provider not found")
        return provider

    @app.patch("/api/providers/{provider_id}")
    def update_provider(provider_id: str, payload: ProviderUpdate):
        try:
            provider = service.update_provider(provider_id, payload)
        except sqlite3.IntegrityError:
            raise HTTPException(409, f"provider name '{payload.name}' already exists")
        if provider is None:
            raise HTTPException(404, "provider not found")
        return provider

    @app.delete("/api/providers/{provider_id}")
    def delete_provider(provider_id: str):
        if not service.delete_provider(provider_id):
            raise HTTPException(404, "provider not found")
        return {"ok": True, "id": provider_id}

    @app.post("/api/providers/{provider_id}/test")
    def test_provider(provider_id: str):
        """用存储的凭证发一次最小请求，验证端点可用。"""
        provider = service.get_provider_secret(provider_id)
        if provider is None:
            raise HTTPException(404, "provider not found")
        return probe_provider(
            kind=provider["kind"],
            base_url=provider["base_url"],
            api_key=provider["api_key"],
            model=provider["model"],
            headers=(provider.get("extra") or {}).get("headers") or {},
        )

    @app.post("/api/providers/test")
    def test_provider_draft(payload: ProviderIn):
        """在保存之前测试一份表单填的配置。"""
        return probe_provider(
            kind=payload.kind,
            base_url=payload.base_url,
            api_key=payload.api_key,
            model=payload.model,
            headers=payload.headers,
        )

    # =====================================================================
    # Search config
    # =====================================================================

    @app.get("/api/projects/{pid}/search-config")
    def get_search_config(pid: str):
        return service.get_search_config(pid)

    @app.put("/api/projects/{pid}/search-config")
    def put_search_config(pid: str, payload: SearchConfigIn):
        return service.put_search_config(pid, payload)

    # =====================================================================
    # Dispatcher
    # =====================================================================

    @app.post("/api/projects/{pid}/dispatch/start")
    def start_dispatch(pid: str, payload: DispatchStart):
        """启动搜索任务（异步后台运行）。"""
        from mts.server.dispatcher_manager import get_dispatcher_manager

        mgr = get_dispatcher_manager(service, root)
        try:
            run_id = mgr.start(pid, payload.config)
            return {"run_id": run_id, "status": "running"}
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"{type(exc).__name__}: {exc}")

    @app.post("/api/projects/{pid}/dispatch/stop")
    def stop_dispatch(pid: str):
        """停止正在运行的搜索任务。"""
        from mts.server.dispatcher_manager import get_dispatcher_manager

        mgr = get_dispatcher_manager(service, root)
        try:
            mgr.stop(pid)
            return {"status": "stopped"}
        except RuntimeError as exc:
            raise HTTPException(400, str(exc))

    @app.get("/api/projects/{pid}/dispatch/status")
    def dispatch_status(pid: str):
        """查询搜索任务状态。"""
        from mts.server.dispatcher_manager import get_dispatcher_manager

        mgr = get_dispatcher_manager(service, root)
        return mgr.status(pid)

    @app.get("/api/projects/{pid}/dispatch/logs")
    def dispatch_logs(pid: str, limit: int = 100):
        """获取搜索任务日志（最新 limit 条）。"""
        from mts.server.dispatcher_manager import get_dispatcher_manager

        mgr = get_dispatcher_manager(service, root)
        return mgr.logs(pid, limit)

    @app.get("/api/projects/{pid}/activity")
    def get_activity(pid: str, after_seq: int = 0, limit: int = 200):
        """获取 agent 活动流：after_seq 之后最旧的 limit 条，按 seq 升序。

        `latest_seq` 是项目当前的全局最大 seq，不是本批的末尾。调用方要按本批实际
        收到的最大 seq 翻页，用 latest_seq 会跳过被 limit 截断的那部分；它只用来
        判断「后面还有没有」（latest_seq > 本批末尾 → 可以立刻再拉一次）。
        """
        from mts.dispatcher.runtime.activity import get_activity_bus

        bus = get_activity_bus()
        events = bus.events(pid, after_seq=after_seq, limit=limit)
        return {"events": [e.to_dict() for e in events], "latest_seq": bus.latest_seq(pid)}

    # =====================================================================
    # Inspection tools
    # =====================================================================

    @app.get("/api/inspect/tools")
    def list_inspect_tools():
        """列出所有可用的检查工具。"""
        from mts.inspect import TOOLS
        return {"tools": [{"name": k, **v} for k, v in TOOLS.items()]}

    @app.post("/api/projects/{pid}/trials/{tid}/inspect/{tool}")
    def run_inspect_tool(pid: str, tid: str, tool: str, params: dict[str, Any] | None = None):
        """对指定 trial 运行检查工具。"""
        from mts.inspect import InspectError, run_tool

        trial = service.get_trial(tid)
        if trial is None:
            raise HTTPException(404, "trial not found")

        spec_path = trial.get("spec_path")
        if not spec_path:
            raise HTTPException(400, "trial has no spec_path")

        out_dir = str(Path(spec_path).parent)
        try:
            result = run_tool(tool, out_dir, **(params or {}))
            return result
        except InspectError as exc:
            raise HTTPException(400, str(exc))

    # =====================================================================
    # Static files
    # =====================================================================

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse({"message": "ModelTrainSelf Board API", "docs": "/docs"})

    return app


def run_server(host: str, port: int, db_path: str | None, runs_dir: str | None,
               reload: bool) -> None:
    """Entry point used by `mts serve`."""
    import uvicorn

    root = Path(runs_dir) if runs_dir else Path.cwd()
    app = create_app(root=root, db_path=db_path)
    uvicorn.run(app, host=host, port=port, reload=reload)
