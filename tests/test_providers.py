"""Provider settings: CRUD, key masking, and credential injection into workers.

The security-relevant invariant these lock down: the raw api_key must reach the
worker env (so the driver can call the endpoint) and must never reach an HTTP
response body (so the UI can display a provider without leaking its key).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import yaml
from fastapi.testclient import TestClient

from mts.dispatcher.config import DispatchConfig
from mts.server.app import create_app
from mts.server.db import Database
from mts.server.dispatcher_manager import build_dispatch_config, provider_env
from mts.server.models import SearchConfigIn
from mts.server.services import Service

SECRET = "sk-secret-abcdef123456"


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(root=tmp_path, db_path=tmp_path / "board.db"))


@pytest.fixture
def service(tmp_path, client):
    # client first: create_app runs the migrations this Service reads.
    return Service(Database(tmp_path / "board.db"), tmp_path)


def _make(client, **overrides):
    payload = {
        "name": "fake-oai",
        "kind": "openai",
        "base_url": "http://127.0.0.1:9/",
        "api_key": SECRET,
        "model": "test-model",
        **overrides,
    }
    response = client.post("/api/providers", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------- CRUD


def test_create_masks_key_and_strips_trailing_slash(client):
    provider = _make(client)
    assert provider["api_key_masked"] == "sk-s...3456"
    assert provider["has_api_key"] is True
    assert provider["base_url"] == "http://127.0.0.1:9"
    assert "api_key" not in provider


def test_list_and_get_never_return_the_raw_key(client):
    _make(client)
    assert SECRET not in client.get("/api/providers").text
    listed = client.get("/api/providers").json()
    assert SECRET not in client.get(f"/api/providers/{listed[0]['id']}").text


def test_duplicate_name_is_rejected(client):
    _make(client)
    assert client.post(
        "/api/providers",
        json={"name": "fake-oai", "base_url": "http://x"},
    ).status_code == 409


def test_openai_provider_requires_base_url(client):
    assert client.post("/api/providers", json={"name": "no-url"}).status_code == 400
    # claudecode drives the local CLI, which supplies its own endpoint.
    assert client.post(
        "/api/providers",
        json={"name": "cc", "kind": "claudecode"},
    ).status_code == 200


def test_patch_without_api_key_keeps_the_stored_one(client, service):
    provider = _make(client)
    updated = client.patch(
        f"/api/providers/{provider['id']}", json={"model": "v2"}
    ).json()
    assert updated["model"] == "v2"
    assert service.get_provider_secret(provider["id"])["api_key"] == SECRET


def test_patch_with_empty_api_key_clears_it(client, service):
    provider = _make(client)
    updated = client.patch(f"/api/providers/{provider['id']}", json={"api_key": ""}).json()
    assert updated["has_api_key"] is False
    assert service.get_provider_secret(provider["id"])["api_key"] == ""


def test_only_one_default_survives(client):
    first = _make(client, name="a", is_default=True)
    second = _make(client, name="b", is_default=True)
    by_id = {p["id"]: p for p in client.get("/api/providers").json()}
    assert by_id[second["id"]]["is_default"] is True
    assert by_id[first["id"]]["is_default"] is False


def test_delete_and_missing_ids_404(client):
    provider = _make(client)
    assert client.delete(f"/api/providers/{provider['id']}").status_code == 200
    assert client.get(f"/api/providers/{provider['id']}").status_code == 404
    assert client.delete(f"/api/providers/{provider['id']}").status_code == 404
    assert client.post(f"/api/providers/{provider['id']}/test").status_code == 404


# ------------------------------------------------------- env injection


def test_provider_env_maps_openai_to_llm_keys():
    env = provider_env(
        {
            "kind": "openai",
            "base_url": "http://h:1",
            "api_key": SECRET,
            "model": "m",
            "extra": {"max_tokens": 8192, "temperature": 0.3, "headers": {"X-T": "a"}},
        },
        "llm",
    )
    assert env["LLM_API_FORMAT"] == "openai"
    assert env["LLM_BASE_URL"] == "http://h:1"
    assert env["LLM_API_KEY"] == SECRET
    assert env["LLM_MODEL"] == "m"
    assert env["LLM_MAX_TOKENS"] == "8192"
    assert env["LLM_TEMPERATURE"] == "0.3"
    assert json.loads(env["LLM_EXTRA_HEADERS"]) == {"X-T": "a"}


def test_provider_env_maps_anthropic_format_and_claudecode_keys():
    row = {"kind": "anthropic", "base_url": "http://h", "api_key": SECRET, "model": "m", "extra": {}}
    assert provider_env(row, "llm")["LLM_API_FORMAT"] == "anthropic"
    assert provider_env(row, "claudecode") == {
        "ANTHROPIC_BASE_URL": "http://h",
        "ANTHROPIC_AUTH_TOKEN": SECRET,
        "ANTHROPIC_MODEL": "m",
    }


def test_provider_env_is_empty_without_a_provider_or_for_mock():
    row = {"kind": "openai", "base_url": "http://h", "api_key": SECRET, "model": "m", "extra": {}}
    assert provider_env(None, "llm") == {}
    assert provider_env(row, "mock") == {}


def test_dispatch_config_carries_the_key_and_still_validates(tmp_path, client, service):
    provider = _make(client, is_default=True)
    config = SearchConfigIn.model_validate(
        {"workers": [{"name": "agent-1", "driver": "llm", "provider_id": provider["id"]}]}
    )
    payload = build_dispatch_config(
        config,
        server="http://127.0.0.1:8000",
        resolve_provider=lambda pid: (
            service.get_provider_secret(pid) if pid else service.get_default_provider_secret()
        ),
    )
    assert payload["workers"][0]["env"]["LLM_API_KEY"] == SECRET

    path = tmp_path / "dispatch.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    loaded = DispatchConfig.load(path)
    assert loaded.workers[0].env["LLM_API_KEY"] == SECRET


def test_unset_provider_id_falls_back_to_the_default_provider(client, service):
    _make(client, name="explicit")
    _make(client, name="the-default", is_default=True, api_key="sk-default-key-9999")
    payload = build_dispatch_config(
        SearchConfigIn.model_validate({"workers": [{"name": "a", "driver": "llm"}]}),
        server="http://127.0.0.1:8000",
        resolve_provider=lambda pid: (
            service.get_provider_secret(pid) if pid else service.get_default_provider_secret()
        ),
    )
    assert payload["workers"][0]["env"]["LLM_API_KEY"] == "sk-default-key-9999"


def test_search_config_round_trips_provider_id(client):
    provider = _make(client)
    project = client.post(
        "/api/projects",
        json={"title": "p", "origin": "o", "goal": "g", "goal_metric": "val_loss"},
    ).json()
    saved = client.put(
        f"/api/projects/{project['id']}/search-config",
        json={"workers": [{"name": "a", "driver": "llm", "provider_id": provider["id"]}]},
    ).json()
    assert saved["workers"][0]["provider_id"] == provider["id"]
    reread = client.get(f"/api/projects/{project['id']}/search-config").json()
    assert reread["workers"][0]["provider_id"] == provider["id"]


# ------------------------------------------------------------- probing


class _FakeOpenAI(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append({"auth": self.headers.get("Authorization"), "body": body})
        blob = json.dumps(
            {"model": body.get("model"), "choices": [{"message": {"content": "pong"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_endpoint():
    _FakeOpenAI.seen = []
    server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", _FakeOpenAI
    server.shutdown()


def test_probe_draft_config_before_saving(client, fake_endpoint):
    base, handler = fake_endpoint
    result = client.post(
        "/api/providers/test",
        json={"name": "d", "base_url": base, "api_key": SECRET, "model": "m"},
    ).json()
    assert result["ok"] is True
    assert result["reply"] == "pong"
    assert result["latency_ms"] > 0
    assert handler.seen[-1]["auth"] == f"Bearer {SECRET}"


def test_probe_saved_provider_uses_stored_key(client, fake_endpoint):
    base, handler = fake_endpoint
    provider = _make(client, base_url=base)
    assert client.post(f"/api/providers/{provider['id']}/test").json()["ok"] is True
    assert handler.seen[-1]["auth"] == f"Bearer {SECRET}"


def test_probe_reports_failure_without_raising(client):
    result = client.post(
        "/api/providers/test",
        json={"name": "d", "base_url": "http://127.0.0.1:1", "api_key": "k", "model": "m"},
    ).json()
    assert result["ok"] is False
    assert result["error"]


def test_probe_rejects_incomplete_config(client):
    missing_url = client.post("/api/providers/test", json={"name": "d", "model": "m"}).json()
    assert missing_url["ok"] is False and "base_url" in missing_url["error"]
    missing_model = client.post(
        "/api/providers/test", json={"name": "d", "base_url": "http://h"}
    ).json()
    assert missing_model["ok"] is False and "model" in missing_model["error"]
