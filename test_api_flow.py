#!/usr/bin/env python
"""测试 ModelTrainSelf API 完整流程：创建项目 → 配置 → 启动搜索。

需要先手动启动服务器：
    export LLM_BASE_URL="https://api.deepseek.com"
    export LLM_API_KEY="sk-691200919b534eec9544f421e98f0a07"
    export LLM_MODEL="deepseek-chat"
    cd /root/work/nlp/xjzhao13/lijie_llama/MolelTrainSelf/ModelTrainSelf
    uvicorn mts.server.app:create_app --factory --host 0.0.0.0 --port 8848
"""

import time
import requests

BASE_URL = "http://localhost:8848"

def test_api_flow():
    print("=== 1. 检查服务器状态 ===")
    try:
        resp = requests.get(f"{BASE_URL}/api/projects", timeout=5)
        print(f"✓ 服务器在线，现有项目数: {len(resp.json())}")
    except requests.RequestException as e:
        print(f"✗ 服务器未启动: {e}")
        print("\n请先启动服务器：")
        print("  /tmp/start_mts_server.sh")
        return

    print("\n=== 2. 创建项目 ===")
    project_data = {
        "title": "RoBERTa 随机二分类搜索实验",
        "origin": "examples/specs/roberta_random_clf.yaml",
        "goal": "优化随机特征二分类任务的验证损失",
        "goal_metric": "val_loss",
        "goal_target": 0.68,  # 随机猜测的交叉熵
        "goal_direction": "minimize",
    }
    resp = requests.post(f"{BASE_URL}/api/projects", json=project_data)
    if resp.status_code != 200:
        print(f"✗ 创建项目失败: {resp.status_code} {resp.text}")
        return
    project = resp.json()
    pid = project["id"]
    print(f"✓ 项目已创建: {pid} - {project['title']}")

    print("\n=== 3. 配置搜索参数 ===")
    search_config = {
        "max_trials": 8,
        "n_agents": 2,
        "max_workers": 2,
        "seed": 42,
        "backend": "torch",
        "workers": [
            {"name": "agent-llm-1", "driver": "llm"},
            {"name": "agent-llm-2", "driver": "llm"},
        ],
    }
    resp = requests.put(f"{BASE_URL}/api/projects/{pid}/search-config", json=search_config)
    if resp.status_code != 200:
        print(f"✗ 配置失败: {resp.status_code} {resp.text}")
        return
    config = resp.json()
    print(f"✓ 搜索配置已保存:")
    print(f"  - max_trials: {config['max_trials']}")
    print(f"  - workers: {len(config['workers'])} × {config['workers'][0]['driver']}")
    print(f"  - backend: {config['backend']}")

    print("\n=== 4. 启动搜索任务 ===")
    resp = requests.post(f"{BASE_URL}/api/projects/{pid}/dispatch/start", json={})
    if resp.status_code != 200:
        print(f"✗ 启动失败: {resp.status_code} {resp.text}")
        return
    result = resp.json()
    run_id = result["run_id"]
    print(f"✓ 搜索任务已启动: {run_id}")

    print("\n=== 5. 监控任务状态 ===")
    for i in range(10):
        time.sleep(3)
        resp = requests.get(f"{BASE_URL}/api/projects/{pid}/dispatch/status")
        status = resp.json()
        print(f"  [{i*3:3d}s] status={status['status']}", end="")

        # 获取日志
        resp = requests.get(f"{BASE_URL}/api/projects/{pid}/dispatch/logs?limit=3")
        logs = resp.json()
        if logs:
            last_event = logs[-1].get("event", "?")
            print(f"  last_event={last_event}", end="")

        print()

        if status["status"] in ("completed", "error"):
            print(f"\n任务结束: {status['status']}")
            if status.get("result"):
                print(f"  trials: {status['result'].get('trials')}")
                print(f"  best_val_loss: {status['result'].get('best_val_loss')}")
            if status.get("error"):
                print(f"  error: {status['error']}")
            break

    print("\n=== 6. 查询排行榜 ===")
    resp = requests.get(f"{BASE_URL}/api/projects/{pid}/leaderboard?metric=val_loss&top=5")
    if resp.status_code == 200:
        leaderboard = resp.json()
        print(f"✓ Top-{len(leaderboard)} trials:")
        for rank, trial in enumerate(leaderboard, 1):
            print(f"  {rank}. {trial['trial_id']}: loss={trial.get('val_loss', 'N/A'):.4f} "
                  f"verdict={trial.get('verdict', '?')}")

    print("\n=== 测试完成 ===")
    print(f"项目 ID: {pid}")
    print(f"WebUI: http://localhost:8848/")


if __name__ == "__main__":
    test_api_flow()
