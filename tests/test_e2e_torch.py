"""Real PyTorch end-to-end test: board + dispatcher + real GPU training.

Unlike test_e2e_mock, this bypasses the agent layer (which is blocked by
unavailable DeepSeek models and the root-as-user constraint) and proves the
training substrate directly: HTTP board + real GPU trials + metric registration.

Minimal agent simulation: the test acts as a scripted worker, claiming intents,
running `mts train --backend torch`, parsing the real summary.json, and
concluding with the actual metrics torch produced.

Skips gracefully when CUDA is unavailable.
"""
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml

from mts.server.app import create_app
from mts.server.models import ConcludeRequest, CreateIntentRequest, ProjectCreate, TrialRegister


@pytest.fixture(scope="module")
def has_cuda():
    """Check if CUDA is available; skip the whole module if not."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def workspace(tmp_path):
    """Isolated workspace for one test."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def board_server(tmp_path):
    """Real uvicorn server on a real socket; yields (port, base_url)."""
    app = create_app(root=tmp_path, db_path=tmp_path / "board.db")

    # Find a free port by binding to 0, reading back the assigned port, then
    # releasing it for uvicorn to claim. Race-y but good enough for tests.
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll /api/settings until the server answers or 20s elapses.
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    ready = False
    while time.time() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(f"{base_url}/api/settings", timeout=2) as _:
                ready = True
                break
        except Exception:
            time.sleep(0.1)

    if not ready:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError(f"Board server on port {port} did not become ready in 20s")

    yield port, base_url

    server.should_exit = True
    thread.join(timeout=10)


def http_post(url, payload):
    """POST JSON and return the response dict."""
    import urllib.request
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def http_get(url):
    """GET and return the response dict."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def test_e2e_torch_real_gpu_training(has_cuda, workspace, board_server, tmp_path):
    """End-to-end: create project via HTTP, run real PyTorch trial, register metrics."""
    if not has_cuda:
        pytest.skip("No CUDA available; skipping real PyTorch E2E")

    port, base_url = board_server

    # ---- 1. Create project from the prose-only YAML --------------------
    project_yaml = Path(__file__).parent / "fixtures" / "project_torch_e2e.yaml"
    assert project_yaml.exists(), f"Missing {project_yaml}"

    with open(project_yaml) as f:
        proj_def = yaml.safe_load(f)

    project = http_post(
        f"{base_url}/api/projects",
        {
            "title": proj_def["title"],
            "origin": proj_def["origin"],
            "goal": proj_def["goal"],
            "goal_metric": proj_def.get("goal_metric"),
            "goal_target": proj_def.get("goal_target"),
            "goal_direction": proj_def.get("goal_direction"),
            "budget_max_trials": proj_def.get("budget_max_trials", 10),
            "hints": proj_def.get("hints", []),
        },
    )
    project_id = project["id"]

    # ---- 2. Create the first intent (simulating bootstrap or reason) ----
    intent = http_post(
        f"{base_url}/api/projects/{project_id}/intents",
        {
            "from": ["origin"],
            "description": "基线配置：2层 transformer / lr 3e-4 / 40步",
            "creator": "test_worker",
        },
    )
    intent_id = intent["id"]

    # ---- 3. Claim the intent -----------------------------------------
    claim = http_post(
        f"{base_url}/api/projects/{project_id}/intents/{intent_id}/claim",
        {"worker": "test_worker"},
    )
    assert claim["worker"] == "test_worker"

    # ---- 4. Run a real PyTorch trial ----------------------------------
    # Build a minimal spec from the origin prose. In a real agent this would be
    # parsed from the text; here we hardcode what the YAML describes.
    spec = {
        "backend": "torch",
        "device": "cuda:0",
        "dataset": "synthetic_lm",
        "arch": {
            "kind": "transformer_decoder",
            "num_layers": 2,
            "d_model": 128,
            "num_heads": 2,
        },
        "train": {
            "lr": 3e-4,
            "batch_size": 16,
            "num_steps": 40,
        },
    }

    trial_dir = workspace / "trial_001"
    trial_dir.mkdir()
    spec_file = trial_dir / "spec.json"
    with open(spec_file, "w") as f:
        json.dump(spec, f, indent=2)

    # Run training directly in-process (subprocess approach failed silently).
    # The CLI is a click command; invoke it programmatically.
    from mts.trainer.cli import train as train_cmd
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        train_cmd,
        [
            "--spec", str(spec_file),
            "--out-dir", str(trial_dir),
            "--backend", "torch",
        ],
    )

    print(f"\n=== Training output (exit_code={result.exit_code}) ===")
    print(f"output:\n{result.output}")
    if result.exception:
        print(f"exception: {result.exception}")
        import traceback
        traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

    assert result.exit_code == 0, f"train command failed: {result.output}"

    # Parse summary.json to get the real metrics torch produced.
    summary_file = trial_dir / "summary.json"
    assert summary_file.exists(), f"Missing {summary_file}"
    with open(summary_file) as f:
        summary = json.load(f)

    # Core assertions: backend=torch, CUDA device, real loss.
    assert summary["backend"] == "torch", f"Wrong backend: {summary['backend']}"
    assert summary["device"].startswith("cuda"), f"Not CUDA: {summary['device']}"
    assert "final" in summary, "Missing final metrics"
    assert "val_loss" in summary["final"], "Missing val_loss"
    val_loss = summary["final"]["val_loss"]
    assert isinstance(val_loss, (int, float)), "val_loss not numeric"
    assert val_loss < 100, f"val_loss {val_loss} too high"

    # ---- 5. Register the trial with the board -------------------------
    trial_payload = {
        "out_dir": str(trial_dir.resolve()),
        "trial_id": trial_dir.name,
        "intent_id": intent_id,
        "worker": "test_worker",
        "backend": summary["backend"],
        "status": summary["status"],
        "steps_done": summary.get("steps_completed", 0),
        "duration_sec": summary["duration_sec"],
        "final_val_loss": val_loss,
        "final_val_acc": summary["final"].get("val_acc"),
        "best_val_loss": summary["best"].get("val_loss"),
        "best_val_acc": summary["best"].get("val_acc"),
    }
    registered = http_post(f"{base_url}/api/projects/{project_id}/trials", trial_payload)
    # Registration succeeded; the response shape may vary but we got 200.
    print(f"Trial registered: {registered}")

    # ---- 6. Conclude the intent with the real metrics -----------------
    conclude_payload = {
        "worker": "test_worker",
        "description": f"基线训练完成: val_loss={val_loss:.3f}, backend={summary['backend']}, device={summary['device']}, verdict={summary['verdict']}",
        "metrics": {
            "val_loss": val_loss,
            "val_acc": summary["final"].get("val_acc", 0.0),
        },
    }
    conclude_result = http_post(
        f"{base_url}/api/projects/{project_id}/intents/{intent_id}/conclude",
        conclude_payload,
    )

    # The intent now points to a newly created fact.
    fact = conclude_result["fact"]
    assert fact["metrics"]["val_loss"] == val_loss
    concluded_intent = conclude_result["intent"]
    assert concluded_intent["to"] == fact["id"]
    assert concluded_intent["concluded_at"] is not None

    # ---- 7. Verify the project graph contains the new fact ------------
    proj_response = http_get(f"{base_url}/api/projects/{project_id}")
    proj_state = proj_response["project"]
    facts = proj_response["facts"]
    # Fact count is at least 3: origin, goal, plus the one we just created.
    fact_ids = [f["id"] for f in facts]
    assert fact["id"] in fact_ids, "Concluded fact not in project graph"

    # ---- 8. Verify export (returns YAML, parse separately) -------------
    import urllib.request
    with urllib.request.urlopen(f"{base_url}/api/projects/{project_id}/export", timeout=10) as resp:
        export_yaml = resp.read().decode("utf-8")
    # Export succeeded and contains our trial_id
    assert trial_dir.name in export_yaml, f"Trial {trial_dir.name} not in export"
    assert str(val_loss)[:6] in export_yaml, f"val_loss {val_loss} not in export"

    # Success: the entire protocol stack works with real PyTorch training.
    print(f"\n✓ Real PyTorch E2E passed: val_loss={val_loss:.4f}, "
          f"backend={summary['backend']}, device={summary['device']}, "
          f"verdict={summary['verdict']}, duration={summary['duration_sec']:.1f}s")
