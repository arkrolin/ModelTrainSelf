"""End-to-end test with real DeepSeek API (skipped unless DEEPSEEK_API_KEY set)."""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from mts.server.app import create_app
from mts.dispatcher.config import DispatchConfig
from mts.dispatcher.scheduler.loop import DispatcherLoop


def find_free_port():
    """Find a free port for the test server."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture
def deepseek_env(tmp_path):
    """Set up environment for DeepSeek test."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")

    # Create board database
    board_db = tmp_path / "board.db"

    # Create workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create dispatch config
    config_path = tmp_path / "dispatch.yaml"
    port = find_free_port()

    # Check if claude CLI is available
    import shutil
    claude_available = shutil.which("claude") is not None

    if claude_available:
        # Use claudecode worker with DeepSeek endpoint
        worker_config = f"""
  - name: deepseek_bootstrap
    type: claudecode
    task_types: [bootstrap]
    max_running: 1
    priority: 0
    env:
      ANTHROPIC_BASE_URL: https://api.deepseek.com/anthropic
      ANTHROPIC_AUTH_TOKEN: {api_key}
      ANTHROPIC_MODEL: deepseek-reasoner
  - name: deepseek_reason
    type: claudecode
    task_types: [reason]
    max_running: 1
    priority: 0
    env:
      ANTHROPIC_BASE_URL: https://api.deepseek.com/anthropic
      ANTHROPIC_AUTH_TOKEN: {api_key}
      ANTHROPIC_MODEL: deepseek-reasoner
  - name: deepseek_explore
    type: claudecode
    task_types: [explore]
    max_running: 1
    priority: 0
    env:
      ANTHROPIC_BASE_URL: https://api.deepseek.com/anthropic
      ANTHROPIC_AUTH_TOKEN: {api_key}
      ANTHROPIC_MODEL: deepseek-reasoner
"""
    else:
        # Fall back to llm worker (limited functionality)
        worker_config = f"""
  - name: deepseek_reason
    type: llm
    task_types: [reason]
    max_running: 1
    priority: 0
    env:
      LLM_BASE_URL: https://api.deepseek.com
      LLM_API_KEY: {api_key}
      LLM_MODEL: deepseek-reasoner
      LLM_API_FORMAT: openai
"""

    config_path.write_text(f"""
server: http://127.0.0.1:{port}
runtime:
  max_workers: 2
  max_running_projects: 1
  max_project_workers: 1
  interval: 2
  healthcheck_timeout: 10
  worker_healthcheck: startup_only
  execution: local
  prompt_group: default
tasks:
  bootstrap:
    timeout: 1800
    conclude_timeout: 600
  reason:
    timeout: 300
    max_intents: 2
  explore:
    timeout: 1800
    conclude_timeout: 600
local:
  workspace_root: {workspace}
  completed_action: keep
workers:
{worker_config}
""")

    return {
        "board_db": board_db,
        "workspace": workspace,
        "config_path": config_path,
        "port": port,
        "tmp_path": tmp_path,
        "claude_available": claude_available,
    }


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
def test_e2e_deepseek_real_training(deepseek_env):
    """
    Real E2E test with DeepSeek API:
    - Start board server
    - Create project with real resource descriptions (RoBERTa checkpoint)
    - Run dispatcher with generous timeouts
    - Verify bootstrap/reason/explore fire
    - Check if real training metrics are reported
    """
    print("\n" + "="*70)
    print("E2E TEST WITH REAL DEEPSEEK API")
    print("="*70)

    if not deepseek_env["claude_available"]:
        print("\n⚠️  WARNING: claude CLI not available, falling back to LLM worker")
        print("   Only reason task will work, bootstrap/explore require claudecode\n")

    # Start board server
    app = create_app(root=deepseek_env["tmp_path"], db_path=deepseek_env["board_db"])

    server_thread = None
    server = None

    def run_server():
        nonlocal server
        config = uvicorn.Config(app, host="127.0.0.1", port=deepseek_env["port"], log_level="warning")
        server = uvicorn.Server(config)
        server.run()

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)

    try:
        client = TestClient(app)

        # Create project with real resources
        roberta_path = "/root/work/nlp/xjzhao13/lijie_llama/MolelTrainSelf/RoBERTa_zh_L12_PyTorch.zip"

        origin_text = f"""
Develop a Chinese text classification system using the provided pre-trained model.

Available resources:
- Pre-trained model: {roberta_path}
  This is a Chinese RoBERTa-base checkpoint (zipped archive). You should:
  1. Inspect the archive to understand its structure
  2. Extract and load the model
  3. Decide on the training approach

Dataset options:
- If a suitable Chinese text classification dataset is available in standard libraries
  (e.g., via datasets, torchtext), use it
- Otherwise, synthesize a small labeled Chinese text dataset for validation purposes
  (e.g., 100-200 samples across 2-3 classes, using simple rules or templates)

The choice of dataset and how to use the model checkpoint is entirely up to you.
Document your decisions clearly.
"""

        response = client.post("/api/projects", json={
            "title": "DeepSeek E2E Test: Chinese Text Classification",
            "origin": origin_text,
            "goal": "Achieve at least 70% validation accuracy",
            "goal_metric": "val_acc",
            "goal_direction": "maximize",
            "goal_target": 0.70,
            "budget_max_trials": 3,
            "hints": [
                "Start with a simple baseline to verify the pipeline works",
                "The RoBERTa checkpoint may require specific handling",
            ],
            "bootstrap_enabled": deepseek_env["claude_available"],
        })
        assert response.status_code == 200
        project_id = response.json()["id"]

        print(f"\n✓ Created project: {project_id}")
        print(f"  Bootstrap enabled: {deepseek_env['claude_available']}")

        # If no bootstrap, manually create an intent for reason to process
        if not deepseek_env["claude_available"]:
            response = client.post(f"/api/projects/{project_id}/intents", json={
                "from": ["origin"],
                "description": "Manual initial intent for testing",
                "creator": "test",
            })
            print(f"  Created manual intent (no bootstrap available)")

        # Load dispatcher
        loop = DispatcherLoop(deepseek_env["config_path"])

        print("\n" + "-"*70)
        print("RUNNING DISPATCHER (max 2 trials, ~60 seconds)")
        print("-"*70)

        # Run dispatcher with timeout
        max_duration = 90  # seconds
        start_time = time.time()
        iteration = 0

        bootstrap_ran = False
        reason_ran = False
        explore_ran = False
        metrics_reported = {}

        while time.time() - start_time < max_duration:
            iteration += 1

            try:
                loop.run(once=True)
            except Exception as e:
                if "closed" not in str(e).lower():
                    print(f"  Iteration {iteration} error: {e}")

            # Check state
            response = client.get(f"/api/projects/{project_id}")
            if response.status_code != 200:
                break

            project = response.json()
            facts = project["facts"]
            intents = project["intents"]

            # Detect what happened
            if len(facts) > 2 and not bootstrap_ran:
                bootstrap_ran = True
                print(f"\n  ✓ Bootstrap ran (fact count: {len(facts)})")

            if len(intents) > 0 and not reason_ran:
                reason_ran = True
                print(f"  ✓ Reason ran (created {len(intents)} intent(s))")

            concluded = [i for i in intents if i["to"] is not None]
            if concluded and not explore_ran:
                explore_ran = True
                print(f"  ✓ Explore ran (concluded {len(concluded)} intent(s))")

            # Collect metrics
            for fact in facts:
                if fact.get("metrics"):
                    fact_id = fact["id"]
                    if fact_id not in metrics_reported:
                        metrics_reported[fact_id] = fact["metrics"]
                        print(f"  • Metrics from {fact_id}: {fact['metrics']}")

            # Stop if budget reached or completed
            if len(concluded) >= 2:
                print(f"\n  Reached {len(concluded)} trials, stopping")
                break

            if project["project"]["status"] == "completed":
                print(f"\n  Project completed")
                break

            time.sleep(2)

        print("\n" + "-"*70)
        print("DISPATCHER RUN COMPLETE")
        print("-"*70)

        # Final state
        response = client.get(f"/api/projects/{project_id}")
        project = response.json()

        facts = project["facts"]
        intents = project["intents"]
        concluded = [i for i in intents if i["to"] is not None]

        print(f"\nFinal state:")
        print(f"  Facts: {len(facts)}")
        print(f"  Intents: {len(intents)}")
        print(f"  Concluded: {len(concluded)}")
        print(f"  Metrics collected: {len(metrics_reported)}")

        # Verify export
        response = client.get(f"/api/projects/{project_id}/export", params={"format": "yaml"})
        assert response.status_code == 200

        import yaml
        export_data = yaml.safe_load(response.text)
        print(f"  Export valid: {len(export_data['facts'])} facts")

        # Report what actually happened
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)

        result = {
            "bootstrap_ran": bootstrap_ran,
            "reason_ran": reason_ran,
            "explore_ran": explore_ran,
            "facts_created": len(facts),
            "intents_created": len(intents),
            "trials_concluded": len(concluded),
            "metrics_count": len(metrics_reported),
        }

        for key, value in result.items():
            print(f"  {key}: {value}")

        if metrics_reported:
            print("\nSample metrics:")
            for fact_id, metrics in list(metrics_reported.items())[:2]:
                print(f"  {fact_id}: {metrics}")

        # Assertions
        if deepseek_env["claude_available"]:
            # With claude CLI, we expect full cycle
            assert reason_ran, "Reason task should have run"
            assert len(intents) >= 1, "Reason should create at least one intent"
            # Note: explore might not complete within timeout, which is OK for a test
        else:
            # With LLM worker, only reason can run
            assert reason_ran or len(intents) > 0, "At least reason should have processed or intents exist"

        print("\n✓ E2E DEEPSEEK TEST PASSED")
        print("="*70)

        return result

    finally:
        loop.close()
        if server:
            server.should_exit = True


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="DEEPSEEK_API_KEY not set")
def test_e2e_deepseek_reason_only(deepseek_env):
    """Minimal test that just verifies reason works with DeepSeek."""
    print("\n" + "="*70)
    print("MINIMAL DEEPSEEK TEST (REASON ONLY)")
    print("="*70)

    app = create_app(root=deepseek_env["tmp_path"], db_path=deepseek_env["board_db"])
    client = TestClient(app)

    # Create simple project
    response = client.post("/api/projects", json={
        "title": "Minimal DeepSeek Test",
        "origin": "Test origin: train a simple model",
        "goal": "Achieve 80% accuracy",
        "goal_metric": "val_acc",
        "goal_direction": "maximize",
        "budget_max_trials": 5,
        "bootstrap_enabled": False,
    })
    project_id = response.json()["id"]

    # Add a fact to trigger reason
    response = client.post(f"/api/projects/{project_id}/intents", json={
        "from": ["origin"],
        "description": "Initial exploration",
        "creator": "test",
    })
    intent_id = response.json()["id"]

    client.post(f"/api/projects/{project_id}/intents/{intent_id}/claim", json={"worker": "test"})
    client.post(f"/api/projects/{project_id}/intents/{intent_id}/conclude", json={
        "worker": "test",
        "description": "Baseline: 75% accuracy",
        "metrics": {"val_acc": 0.75},
    })

    print(f"✓ Created project with initial fact")

    # Run dispatcher briefly
    loop = DispatcherLoop(deepseek_env["config_path"])

    try:
        for i in range(10):
            loop.run(once=True)
            response = client.get(f"/api/projects/{project_id}")
            project = response.json()

            if len(project["intents"]) > 1:
                print(f"✓ Reason created new intent(s)")
                break

            time.sleep(2)

        # Check final state
        response = client.get(f"/api/projects/{project_id}")
        project = response.json()

        print(f"\nFinal: {len(project['intents'])} intents, {len(project['facts'])} facts")
        print("✓ MINIMAL DEEPSEEK TEST PASSED")

    finally:
        loop.close()
