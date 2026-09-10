"""End-to-end test with the mock driver: no network, no GPU, no credentials.

Everything here is real except the agent CLI: a real board server on a real
socket, the real dispatcher loop, real claim/heartbeat/conclude traffic over
HTTP. Two harness details are load-bearing and easy to get wrong:

* The dispatcher's client speaks real HTTP, so the board must be served by a
  real uvicorn socket. A bare ``TestClient`` answers in-process only, and the
  dispatcher would get ECONNREFUSED.
* ``DispatcherLoop.run(once=True)`` shuts its executor down in a ``finally``,
  which makes a loop single-use. Each tick builds a fresh one, and that
  shutdown doubles as "wait for the task this tick dispatched" — so the whole
  test stays synchronous and needs no sleeps.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import requests
import uvicorn
import yaml

from mts.dispatcher.scheduler.loop import BOOTSTRAP_INTENT_CREATOR, DispatcherLoop
from mts.server.app import create_app


def find_free_port() -> int:
    """Ask the OS for a free port. Racy in principle, fine for a test host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


# Every allowed outcome of a phase has to appear with a weight, because the
# config loader requires the weights of each phase to sum to exactly 1.0.
_PHASE_OUTCOMES: dict[str, tuple[str, ...]] = {
    "reason": ("complete", "intent", "noop", "rejected", "invalid_json",
               "invalid_payload", "command_fail"),
    "bootstrap": ("complete", "fact", "rejected", "invalid_json",
                  "invalid_payload", "command_fail"),
    "bootstrap_conclude": ("fact", "rejected", "invalid_json",
                           "invalid_payload", "command_fail"),
    "explore_execute": ("fact", "rejected", "invalid_json",
                        "invalid_payload", "command_fail"),
    "explore_conclude": ("fact", "rejected", "invalid_json",
                         "invalid_payload", "command_fail"),
}


def _phase(name: str, winner: str, *, rules: list[dict[str, Any]] | None = None) -> str:
    """A mock phase that always picks ``winner``, as a JSON env-var value.

    Delays stay small but non-zero so the lease/heartbeat plumbing is genuinely
    exercised rather than optimised away by an instant return.
    """
    behavior: dict[str, Any] = {
        "delay": [0.05, 0.15],
        "outcomes": {
            outcome: ("1.0" if outcome == winner else "0.0")
            for outcome in _PHASE_OUTCOMES[name]
        },
    }
    if rules:
        behavior["rules"] = rules
    return json.dumps(behavior, ensure_ascii=False)


def write_dispatch_config(
    path: Path,
    *,
    port: int,
    workspace: Path,
    reason_completes_at: int,
) -> Path:
    """Write a dispatch config whose mock agents drive a deterministic cycle.

    ``reason_completes_at`` is the fact count at which reason stops proposing
    intents and completes the project instead, which is how each test picks
    how much of the cycle it wants to watch. It counts the facts reason is
    actually offered, which excludes ``goal`` -- the prompt is built from
    ``allowed_fact_ids``, and ``goal`` is never a legal intent source.
    """
    config = {
        "server": f"http://127.0.0.1:{port}",
        "runtime": {
            "max_workers": 2,
            "max_running_projects": 2,
            "max_project_workers": 2,
            "interval": 1,
            "healthcheck_timeout": 5,
            "worker_healthcheck": "disabled",
            "execution": "local",
            "prompt_group": "mock",
        },
        "tasks": {
            "bootstrap": {"timeout": 30, "conclude_timeout": 30},
            # max_intents=1 pins the mock's randint(1, max_intents) to exactly
            # one intent per reason pass, which makes the tick sequence
            # deterministic instead of 1-or-2 per pass.
            "reason": {"timeout": 30, "max_intents": 1},
            "explore": {"timeout": 30, "conclude_timeout": 30},
        },
        "local": {"workspace_root": str(workspace), "completed_action": "keep"},
        "workers": [
            {
                "name": "mock_bootstrap",
                "type": "mock",
                "task_types": ["bootstrap"],
                "max_running": 1,
                "priority": 0,
                "env": {
                    # 'fact' rather than 'complete': bootstrap should seed the
                    # graph and hand off, not finish the project on tick 1.
                    "MOCK_BOOTSTRAP": _phase("bootstrap", "fact"),
                    "MOCK_BOOTSTRAP_CONCLUDE": _phase("bootstrap_conclude", "fact"),
                },
            },
            {
                "name": "mock_reason",
                "type": "mock",
                "task_types": ["reason"],
                "max_running": 1,
                "priority": 0,
                "env": {
                    "MOCK_REASON": _phase(
                        "reason",
                        "intent",
                        rules=[{"force": "complete",
                                "fact_ids_gte": reason_completes_at}],
                    ),
                },
            },
            {
                "name": "mock_explore",
                "type": "mock",
                "task_types": ["explore"],
                "max_running": 1,
                "priority": 0,
                "env": {
                    "MOCK_EXPLORE_EXECUTE": _phase("explore_execute", "fact"),
                    "MOCK_EXPLORE_CONCLUDE": _phase("explore_conclude", "fact"),
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


class Board:
    """Thin HTTP handle on the live board, so tests read what agents read."""

    def __init__(self, base_url: str, tmp_path: Path, workspace: Path) -> None:
        self.base_url = base_url
        self.tmp_path = tmp_path
        self.workspace = workspace
        self.config_path = tmp_path / "dispatch.yaml"

    def create_project(self, **payload: Any) -> str:
        response = requests.post(f"{self.base_url}/api/projects", json=payload, timeout=10)
        assert response.status_code == 200, response.text
        return str(response.json()["id"])

    def detail(self, pid: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/projects/{pid}", timeout=10)
        assert response.status_code == 200, response.text
        return dict(response.json())

    def post(self, path: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", json=payload, timeout=10)

    def export(self, pid: str) -> str:
        response = requests.get(
            f"{self.base_url}/api/projects/{pid}/export",
            params={"format": "yaml"},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        return response.text


@pytest.fixture
def board(tmp_path):
    """Serve the board on a real socket for the duration of one test.

    The dispatcher's client is a plain ``requests`` session, so an in-process
    ``TestClient`` is not enough: it needs a real listener to connect to.
    """
    port = find_free_port()
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    app = create_app(root=tmp_path, db_path=tmp_path / "board.db")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{base_url}/api/settings", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.05)
    else:
        server.should_exit = True
        pytest.fail(f"board server did not come up on {base_url}")

    try:
        yield Board(base_url, tmp_path, workspace)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def tick(config_path: Path) -> None:
    """Run exactly one dispatcher pass, then wait for whatever it dispatched.

    A loop is single-use: ``run(once=True)`` closes its executor in a
    ``finally``, and that shutdown waits for the submitted task. So a fresh
    loop per tick gives a synchronous "dispatch one thing and let it finish"
    step, and any exception in the task body is re-raised here rather than
    being swallowed by a future nobody reaps.
    """
    loop = DispatcherLoop(config_path)
    loop.run(once=True)
    for future in list(loop.futures):
        future.result()


def run_until_completed(config_path: Path, board: Board, pid: str, *, max_ticks: int) -> dict[str, Any]:
    """Tick the dispatcher until the project completes, or fail with the graph."""
    detail = board.detail(pid)
    for _ in range(max_ticks):
        if detail["project"]["status"] == "completed":
            return detail
        tick(config_path)
        detail = board.detail(pid)
    pytest.fail(
        f"project {pid} not completed after {max_ticks} ticks: "
        f"status={detail['project']['status']} "
        f"facts={[f['id'] for f in detail['facts']]} "
        f"intents={[(i['id'], i['creator'], i['to']) for i in detail['intents']]}"
    )


def test_e2e_mock_full_cycle(board):
    """Bootstrap seeds the graph, reason proposes, explore executes, reason ends it.

    The cycle is driven to real completion rather than to a fact-count
    threshold, so the terminal ``to == "goal"`` intent is covered too.
    """
    write_dispatch_config(
        board.config_path,
        port=int(board.base_url.rsplit(":", 1)[1]),
        workspace=board.workspace,
        reason_completes_at=4,
    )
    pid = board.create_project(
        title="Mock Training Test",
        origin=(
            "Test a simple training pipeline. Available resources: mock dataset "
            "at /tmp/mock_data.csv, mock model checkpoint at /tmp/mock_model.pt. "
            "The agent should inspect these and decide how to use them."
        ),
        goal="Achieve validation loss below 0.30",
        goal_metric="val_loss",
        goal_direction="minimize",
        goal_target=0.30,
        budget_max_trials=10,
        hints=[
            "Start with a baseline run to establish performance",
            "Monitor for overfitting",
        ],
        bootstrap_enabled=True,
    )

    detail = run_until_completed(board.config_path, board, pid, max_ticks=15)

    facts = detail["facts"]
    intents = detail["intents"]
    fact_ids = [fact["id"] for fact in facts]
    assert "origin" in fact_ids
    assert "goal" in fact_ids

    # Bootstrap ran: its intent is the one the dispatcher itself authored, and
    # it concluded into a fact of its own.
    bootstrap = [i for i in intents if i["creator"] == BOOTSTRAP_INTENT_CREATOR]
    assert len(bootstrap) == 1, f"expected exactly one bootstrap intent, got {bootstrap}"
    assert bootstrap[0]["to"] not in (None, "goal")

    # Reason ran: it authored intents of its own, from a real worker name.
    proposed = [i for i in intents if i["creator"] == "mock_reason"]
    assert proposed, f"reason proposed nothing: {intents}"

    # Explore ran: reason's intents were claimed and concluded into facts.
    explored = [i for i in proposed if i["to"] is not None and i["to"] != "goal"]
    assert explored, f"no proposed intent was explored to a fact: {proposed}"

    # Reason ended it: exactly one terminal intent, pointing at the goal fact.
    terminal = [i for i in intents if i["to"] == "goal"]
    assert len(terminal) == 1, f"expected exactly one terminal intent, got {terminal}"
    assert "goal" not in terminal[0]["from"], "goal is never a legal intent source"

    # origin + goal + bootstrap fact + at least one explore fact.
    assert len(fact_ids) >= 4, f"cycle produced too few facts: {fact_ids}"

    # Every agent-written fact carries the project's goal metric.
    written = [f for f in facts if f["id"] not in ("origin", "goal")]
    assert written, "no agent-written facts"
    for fact in written:
        assert "val_loss" in (fact.get("metrics") or {}), fact

    export = yaml.safe_load(board.export(pid))
    assert export["project"]["title"] == "Mock Training Test"
    assert len(export["facts"]) == len(facts)
    assert len(export["intents"]) == len(intents)


def test_e2e_mock_minimal(board):
    """A project with bootstrap disabled starts at origin/goal with one open intent."""
    pid = board.create_project(
        title="Minimal Test",
        origin="Test origin with resource path: /tmp/data",
        goal="Test goal",
        goal_metric="val_loss",
        goal_direction="minimize",
        budget_max_trials=1,
        bootstrap_enabled=False,
    )
    response = board.post(f"/api/projects/{pid}/intents", {
        "from": ["origin"], "description": "Manual intent", "creator": "test",
    })
    assert response.status_code == 200, response.text

    detail = board.detail(pid)
    assert {f["id"] for f in detail["facts"]} == {"origin", "goal"}
    assert len(detail["intents"]) == 1
    assert detail["intents"][0]["to"] is None


def test_e2e_mock_reason_completes_project(board):
    """Reason completes a project whose graph already satisfies its rule.

    Bootstrap is off and a fact is seeded by hand, so the first dispatched task
    is reason, and its rule fires immediately: this isolates the completion
    path from the rest of the cycle.
    """
    write_dispatch_config(
        board.config_path,
        port=int(board.base_url.rsplit(":", 1)[1]),
        workspace=board.workspace,
        reason_completes_at=2,
    )
    pid = board.create_project(
        title="Complete Test",
        origin="Test with early completion",
        goal="Test goal",
        goal_metric="val_loss",
        goal_direction="minimize",
        budget_max_trials=10,
        bootstrap_enabled=False,
    )

    # Seed one concluded intent by hand: origin -> f001.
    response = board.post(f"/api/projects/{pid}/intents", {
        "from": ["origin"], "description": "seed", "creator": "test",
    })
    assert response.status_code == 200, response.text
    iid = response.json()["id"]
    assert board.post(f"/api/projects/{pid}/intents/{iid}/claim", {"worker": "test"}).status_code == 200
    concluded = board.post(f"/api/projects/{pid}/intents/{iid}/conclude", {
        "worker": "test", "description": "seed fact", "metrics": {"val_loss": 0.25},
    })
    assert concluded.status_code == 200, concluded.text

    # origin + f001 are the two facts reason is offered; the rule fires at 2.
    detail = run_until_completed(board.config_path, board, pid, max_ticks=5)

    assert detail["project"]["status"] == "completed"
    terminal = [i for i in detail["intents"] if i["to"] == "goal"]
    assert len(terminal) == 1, f"expected one terminal intent, got {terminal}"
    assert terminal[0]["creator"] == "mock_reason"
