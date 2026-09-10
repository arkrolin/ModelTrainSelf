"""真实 DeepSeek Agent E2E 测试

完整端到端测试：
1. 启动真实的 Board 看板服务器
2. 启动 Dispatcher，配置真实的 DeepSeek Flash worker
3. 创建项目（prose-only YAML）
4. Agent 自动执行 bootstrap → 生成训练配置
5. Agent 运行真实的 PyTorch GPU 训练
6. Agent conclude → 创建 fact
7. 验证整个协议栈：Board + Dispatcher + Agent + Trainer
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn
import yaml

from mts.server.app import create_app


@pytest.fixture
def has_cuda():
    """检查 CUDA 是否可用"""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.fixture
def has_deepseek_api():
    """检查 DeepSeek API 配置是否存在"""
    return all(
        [
            os.getenv("LLM_API_KEY"),
            os.getenv("LLM_BASE_URL"),
        ]
    )


@pytest.fixture
def workspace(tmp_path):
    """创建测试工作区"""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def board_server(tmp_path):
    """启动真实的 Board 看板服务器"""
    app = create_app(root=tmp_path, db_path=tmp_path / "board.db")

    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

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
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(url):
    """GET and return the response dict."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


@pytest.mark.skipif(
    "not (os.getenv('LLM_API_KEY') and os.getenv('LLM_BASE_URL'))",
    reason="需要 DeepSeek API 配置：LLM_API_KEY 和 LLM_BASE_URL",
)
def test_e2e_deepseek_agent_bootstrap(
    has_cuda, workspace, board_server, tmp_path, has_deepseek_api
):
    """真实 DeepSeek Agent E2E：Bootstrap 任务生成训练配置"""
    if not has_cuda:
        pytest.skip("No CUDA available; skipping real PyTorch E2E")
    if not has_deepseek_api:
        pytest.skip("DeepSeek API not configured")

    port, base_url = board_server

    # ---- 1. 创建 Dispatcher 配置文件 --------------------------------
    dispatcher_config = {
        "runtime": {
            "max_workers": 2,
            "max_running_projects": 1,
            "max_project_workers": 1,
            "interval": 2,
            "healthcheck_timeout": 30,
            "worker_healthcheck": "startup_only",
            "execution": "local",
            "prompt_group": "default",
        },
        "tasks": {
            "bootstrap": {"timeout": 300, "conclude_timeout": 60},
            "reason": {"timeout": 180, "max_intents": 3},
            "explore": {"timeout": 300, "conclude_timeout": 60},
        },
        "local": {"workspace_root": str(workspace), "completed_action": "keep"},
        "workers": [
            {
                "name": "deepseek-bootstrap",
                "type": "llm",
                "task_types": ["bootstrap"],
                "max_running": 1,
                "priority": 10,
                "env": {
                    "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-chat"),
                    "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
                    "LLM_API_KEY": os.getenv("LLM_API_KEY"),
                    "LLM_API_FORMAT": "openai",
                },
            }
        ],
    }

    config_file = tmp_path / "dispatcher_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(dispatcher_config, f)

    print(f"\n=== Dispatcher 配置 ===")
    print(f"配置文件: {config_file}")
    print(f"LLM_MODEL: {dispatcher_config['workers'][0]['env']['LLM_MODEL']}")
    print(f"LLM_BASE_URL: {dispatcher_config['workers'][0]['env']['LLM_BASE_URL']}")

    # ---- 2. 创建项目（prose-only YAML）-------------------------------
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
            "budget_max_trials": proj_def.get("budget_max_trials", 5),
            "hints": proj_def.get("hints", []),
        },
    )
    project_id = project["id"]
    print(f"\n=== 项目创建成功 ===")
    print(f"项目 ID: {project_id}")
    print(f"标题: {project['title']}")

    # ---- 3. 启动 Dispatcher ------------------------------------------
    # Dispatcher 会自动发现项目并分配 bootstrap 任务给 DeepSeek agent
    dispatcher_proc = subprocess.Popen(
        [
            "/root/anaconda3/envs/mts/bin/python",
            "-m",
            "mts.dispatcher.runtime.local",
            "--config",
            str(config_file),
            "--board-url",
            base_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(f"\n=== Dispatcher 启动 ===")
    print(f"PID: {dispatcher_proc.pid}")

    try:
        # ---- 4. 等待 agent 完成 bootstrap 任务 -----------------------
        # Bootstrap 任务会：
        # 1. 读取 origin/goal/hints
        # 2. 生成训练配置 (spec.json)
        # 3. 运行训练
        # 4. 注册 trial
        # 5. 创建 fact

        print(f"\n=== 等待 DeepSeek Agent 完成 Bootstrap ===")
        max_wait = 300  # 5 分钟超时
        start_time = time.time()
        bootstrap_completed = False

        while time.time() - start_time < max_wait:
            # 检查项目状态
            proj_response = http_get(f"{base_url}/api/projects/{project_id}")
            facts = proj_response.get("facts", [])

            # 如果有新的 fact（除了 origin 和 goal），说明 bootstrap 完成了
            non_system_facts = [
                f for f in facts if f.get("role") not in ["origin", "goal"]
            ]

            if non_system_facts:
                print(f"\n✓ Bootstrap 完成！创建了 {len(non_system_facts)} 个新 fact")
                bootstrap_completed = True
                break

            # 读取 dispatcher 输出
            if dispatcher_proc.poll() is not None:
                # Dispatcher 意外退出
                print("\n✗ Dispatcher 意外退出")
                break

            time.sleep(2)

        if not bootstrap_completed:
            # 收集 dispatcher 日志
            dispatcher_proc.terminate()
            dispatcher_proc.wait(timeout=5)
            pytest.fail(
                f"Bootstrap 未在 {max_wait}s 内完成。可能的原因：\n"
                f"1. DeepSeek API 调用失败\n"
                f"2. Agent prompt 解析错误\n"
                f"3. 训练执行失败"
            )

        # ---- 5. 验证结果 --------------------------------------------
        proj_response = http_get(f"{base_url}/api/projects/{project_id}")
        facts = proj_response.get("facts", [])
        intents = proj_response.get("intents", [])

        print(f"\n=== 最终状态 ===")
        print(f"Facts: {len(facts)}")
        print(f"Intents: {len(intents)}")

        # 找到 bootstrap 创建的 fact
        bootstrap_facts = [
            f for f in facts if f.get("role") not in ["origin", "goal"]
        ]
        assert len(bootstrap_facts) >= 1, "应该至少有 1 个 bootstrap fact"

        latest_fact = bootstrap_facts[-1]
        print(f"\n=== Bootstrap Fact ===")
        print(f"ID: {latest_fact['id']}")
        print(f"描述: {latest_fact.get('description', 'N/A')}")
        print(f"指标: {latest_fact.get('metrics', {})}")

        # 验证指标
        metrics = latest_fact.get("metrics", {})
        assert "val_loss" in metrics, "应该有 val_loss 指标"
        val_loss = metrics["val_loss"]
        assert isinstance(val_loss, (int, float)), "val_loss 应该是数字"
        assert val_loss < 100, f"val_loss {val_loss} 异常高"

        print(f"\n✓ 真实 DeepSeek Agent E2E 测试通过！")
        print(f"  - val_loss: {val_loss:.4f}")
        print(f"  - Agent: DeepSeek {dispatcher_config['workers'][0]['env']['LLM_MODEL']}")
        print(f"  - 训练后端: PyTorch + CUDA")

    finally:
        # 清理 dispatcher
        if dispatcher_proc.poll() is None:
            dispatcher_proc.terminate()
            try:
                dispatcher_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dispatcher_proc.kill()
                dispatcher_proc.wait()


@pytest.mark.skipif(
    "not (os.getenv('LLM_API_KEY') and os.getenv('LLM_BASE_URL'))",
    reason="需要 DeepSeek API 配置：LLM_API_KEY 和 LLM_BASE_URL",
)
def test_e2e_deepseek_agent_reason_explore(
    has_cuda, workspace, board_server, tmp_path, has_deepseek_api
):
    """真实 DeepSeek Agent E2E：完整的 Reason + Explore 循环"""
    if not has_cuda:
        pytest.skip("No CUDA available; skipping real PyTorch E2E")
    if not has_deepseek_api:
        pytest.skip("DeepSeek API not configured")

    port, base_url = board_server

    # ---- 1. 创建 Dispatcher 配置（支持所有任务类型）-----------------
    dispatcher_config = {
        "runtime": {
            "max_workers": 3,
            "max_running_projects": 1,
            "max_project_workers": 2,
            "interval": 2,
            "healthcheck_timeout": 30,
            "worker_healthcheck": "startup_only",
            "execution": "local",
            "prompt_group": "default",
        },
        "tasks": {
            "bootstrap": {"timeout": 300, "conclude_timeout": 60},
            "reason": {"timeout": 180, "max_intents": 3},
            "explore": {"timeout": 300, "conclude_timeout": 60},
        },
        "local": {"workspace_root": str(workspace), "completed_action": "keep"},
        "workers": [
            {
                "name": "deepseek-bootstrap",
                "type": "llm",
                "task_types": ["bootstrap"],
                "max_running": 1,
                "priority": 10,
                "env": {
                    "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-chat"),
                    "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
                    "LLM_API_KEY": os.getenv("LLM_API_KEY"),
                    "LLM_API_FORMAT": "openai",
                },
            },
            {
                "name": "deepseek-reason",
                "type": "llm",
                "task_types": ["reason"],
                "max_running": 1,
                "priority": 10,
                "env": {
                    "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-chat"),
                    "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
                    "LLM_API_KEY": os.getenv("LLM_API_KEY"),
                    "LLM_API_FORMAT": "openai",
                },
            },
            {
                "name": "deepseek-explore",
                "type": "llm",
                "task_types": ["explore"],
                "max_running": 1,
                "priority": 10,
                "env": {
                    "LLM_MODEL": os.getenv("LLM_MODEL", "deepseek-chat"),
                    "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
                    "LLM_API_KEY": os.getenv("LLM_API_KEY"),
                    "LLM_API_FORMAT": "openai",
                },
            },
        ],
    }

    config_file = tmp_path / "dispatcher_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(dispatcher_config, f)

    # ---- 2. 创建项目 ----------------------------------------------
    project_yaml = Path(__file__).parent / "fixtures" / "project_torch_e2e.yaml"
    with open(project_yaml) as f:
        proj_def = yaml.safe_load(f)

    project = http_post(
        f"{base_url}/api/projects",
        {
            "title": "DeepSeek Agent 完整循环测试",
            "origin": proj_def["origin"],
            "goal": proj_def["goal"],
            "goal_metric": proj_def.get("goal_metric"),
            "goal_target": proj_def.get("goal_target"),
            "goal_direction": proj_def.get("goal_direction"),
            "budget_max_trials": 3,  # 限制试验次数
            "hints": proj_def.get("hints", []),
        },
    )
    project_id = project["id"]

    print(f"\n=== 项目创建：完整循环测试 ===")
    print(f"项目 ID: {project_id}")
    print(f"预算: 最多 3 次试验")

    # ---- 3. 启动 Dispatcher --------------------------------------
    dispatcher_proc = subprocess.Popen(
        [
            "/root/anaconda3/envs/mts/bin/python",
            "-m",
            "mts.dispatcher.runtime.local",
            "--config",
            str(config_file),
            "--board-url",
            base_url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(f"\n=== Dispatcher 启动（3 个 worker）===")

    try:
        # ---- 4. 等待完整的循环执行 -------------------------------
        # 预期流程：
        # Bootstrap → Reason → Explore → Reason → Explore → ...
        # 直到达到预算限制或找到满意的解

        max_wait = 600  # 10 分钟超时
        start_time = time.time()
        last_fact_count = 0

        print(f"\n=== 监控 Agent 执行 ===")

        while time.time() - start_time < max_wait:
            proj_response = http_get(f"{base_url}/api/projects/{project_id}")
            facts = proj_response.get("facts", [])
            intents = proj_response.get("intents", [])
            project_status = proj_response["project"]["status"]

            # 打印进度
            if len(facts) > last_fact_count:
                print(
                    f"[{int(time.time() - start_time)}s] Facts: {len(facts)}, Intents: {len(intents)}, Status: {project_status}"
                )
                last_fact_count = len(facts)

            # 检查是否完成
            if project_status == "completed":
                print(f"\n✓ 项目完成！")
                break

            # 检查是否有足够的试验
            non_system_facts = [
                f for f in facts if f.get("role") not in ["origin", "goal"]
            ]
            if len(non_system_facts) >= 3:
                print(f"\n✓ 达到预算限制（3 次试验）")
                break

            if dispatcher_proc.poll() is not None:
                print("\n✗ Dispatcher 意外退出")
                break

            time.sleep(3)

        # ---- 5. 验证最终结果 -------------------------------------
        proj_response = http_get(f"{base_url}/api/projects/{project_id}")
        facts = proj_response.get("facts", [])
        intents = proj_response.get("intents", [])

        print(f"\n=== 最终结果 ===")
        print(f"总 Facts: {len(facts)}")
        print(f"总 Intents: {len(intents)}")

        non_system_facts = [
            f for f in facts if f.get("role") not in ["origin", "goal"]
        ]
        print(f"训练试验: {len(non_system_facts)}")

        assert len(non_system_facts) >= 1, "应该至少完成 1 次试验"

        # 打印所有试验的指标
        print(f"\n=== 试验指标 ===")
        for i, fact in enumerate(non_system_facts, 1):
            metrics = fact.get("metrics", {})
            val_loss = metrics.get("val_loss", "N/A")
            print(f"试验 {i}: val_loss={val_loss}")

        print(f"\n✓ DeepSeek Agent 完整循环测试通过！")
        print(f"  - 完成试验: {len(non_system_facts)}")
        print(f"  - Agent: DeepSeek {dispatcher_config['workers'][0]['env']['LLM_MODEL']}")

    finally:
        if dispatcher_proc.poll() is None:
            dispatcher_proc.terminate()
            try:
                dispatcher_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                dispatcher_proc.kill()
                dispatcher_proc.wait()
