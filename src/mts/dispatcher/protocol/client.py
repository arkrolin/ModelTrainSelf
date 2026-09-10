from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging
import threading

from pydantic import TypeAdapter
import requests
from requests.adapters import HTTPAdapter

from mts.server.models import ProjectDetail, ProjectSummary, Settings

LOG = logging.getLogger(__name__)


class ProtocolError(RuntimeError):
    def __init__(self, message: str, status_code: int, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


@dataclass(slots=True)
class ApiResult:
    ok: bool
    status_code: int | None
    text: str
    data: Any = None


class MTSClient:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._summary_adapter = TypeAdapter(list[ProjectSummary])
        self._local = threading.local()
        self._sessions: dict[int, requests.Session] = {}
        self._sessions_lock = threading.Lock()

    def close(self) -> None:
        with self._sessions_lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()

    def get_settings(self) -> Settings:
        response = self._session().get(self._url("/api/settings"), timeout=self._timeout)
        response.raise_for_status()
        return Settings.model_validate(response.json())

    def list_projects(self) -> list[ProjectSummary]:
        response = self._session().get(self._url("/api/projects"), timeout=self._timeout)
        response.raise_for_status()
        return self._summary_adapter.validate_python(response.json())

    def get_project(self, project_id: str) -> ProjectDetail:
        response = self._session().get(self._url(f"/api/projects/{project_id}"), timeout=self._timeout)
        response.raise_for_status()
        return ProjectDetail.model_validate(response.json())

    def export_project(self, project_id: str) -> str:
        response = self._session().get(
            self._url(f"/api/projects/{project_id}/export"),
            params={"format": "yaml"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.text

    def create_intent(
        self,
        project_id: str,
        from_: list[str],
        description: str,
        creator: str,
        worker: str | None = None,
    ) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/intents",
            json={"from": from_, "description": description, "creator": creator, "worker": worker},
        )

    def claim_intent(self, project_id: str, intent_id: str, worker: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/intents/{intent_id}/claim",
            json={"worker": worker},
        )

    def heartbeat(self, project_id: str, intent_id: str, worker: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/intents/{intent_id}/heartbeat",
            json={"worker": worker},
        )

    def release(self, project_id: str, intent_id: str, worker: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/intents/{intent_id}/release",
            json={"worker": worker},
        )

    def conclude(
        self,
        project_id: str,
        intent_id: str,
        worker: str,
        description: str,
        metrics: dict[str, float] | None = None,
        trial_id: str | None = None,
    ) -> ApiResult:
        payload: dict[str, Any] = {"worker": worker, "description": description}
        if metrics is not None:
            payload["metrics"] = metrics
        if trial_id is not None:
            payload["trial_id"] = trial_id
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/intents/{intent_id}/conclude",
            json=payload,
        )

    def complete(
        self,
        project_id: str,
        from_: list[str],
        description: str,
        worker: str,
    ) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/complete",
            json={"from": from_, "description": description, "worker": worker},
        )

    def claim_reason(self, project_id: str, worker: str, trigger: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/reason/claim",
            json={"worker": worker, "trigger": trigger},
        )

    def reason_heartbeat(self, project_id: str, worker: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/reason/heartbeat",
            json={"worker": worker},
        )

    def release_reason(self, project_id: str, worker: str) -> ApiResult:
        return self._request_json(
            "POST",
            f"/api/projects/{project_id}/reason/release",
            json={"worker": worker},
        )

    def _request_json(self, method: str, path: str, json: dict[str, Any]) -> ApiResult:
        try:
            response = self._session().request(
                method,
                self._url(path),
                json=json,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            LOG.warning("request failed method=%s path=%s error=%s", method, path, exc)
            return ApiResult(ok=False, status_code=None, text=str(exc))

        data: Any | None = None
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                data = response.json()
            except Exception:
                pass

        ok = 200 <= response.status_code < 300
        return ApiResult(ok=ok, status_code=response.status_code, data=data, text=response.text)

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is not None:
            return session

        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, pool_block=False)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self._local.session = session
        with self._sessions_lock:
            self._sessions[threading.get_ident()] = session
        return session
