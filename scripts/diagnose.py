#!/usr/bin/env python3
"""快速诊断脚本：检查 MolelTrainSelf 项目完成度"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_imports():
    """检查关键模块是否可导入"""
    checks = []

    # 名字按源码实际导出写，不按文档措辞猜。
    modules = [
        ("mts.trainer.spec", "TrialSpec"),
        ("mts.trainer.surrogate", "run"),
        ("mts.trainer.probes", "Diagnostics"),
        ("mts.trainer.runner", "run_trial"),
        ("mts.trainer.runner", "resolve_backend"),
        ("mts.server.db", "Database"),
        ("mts.server.services", "Service"),
        ("mts.knowledge.store", "KnowledgeStore"),
        ("mts.dispatcher.config", "DispatchConfig"),
        ("mts.dispatcher.scheduler.loop", "DispatcherLoop"),
        ("mts.dispatcher.runtime.device", "DevicePool"),
        ("mts.reporting", "build_report"),
    ]

    for module_path, attr in modules:
        try:
            module = __import__(module_path, fromlist=[attr])
            getattr(module, attr)
            checks.append((module_path, attr, "✅"))
        except Exception as e:
            checks.append((module_path, attr, f"❌ {e}"))

    return checks

def check_files():
    """检查关键文件是否存在"""
    root = Path(__file__).parent.parent
    files = [
        "src/mts/cli.py",
        "src/mts/demo.py",
        "src/mts/trainer/spec.py",
        "src/mts/trainer/surrogate.py",
        "src/mts/trainer/runner.py",
        "src/mts/server/app.py",
        "src/mts/server/db.py",
        "src/mts/server/static/index.html",
        "src/mts/server/static/app.js",
        "src/mts/knowledge/store.py",
        "src/mts/dispatcher/config.py",
        "src/mts/dispatcher/scheduler/loop.py",
        "src/mts/dispatcher/workers/base.py",
        "src/mts/dispatcher/runtime/device.py",
        "src/mts/dispatcher/runtime/local_process.py",
        "docs/07-MVP范围与实施计划.md",
        "docs/10-迭代探测报告.md",
    ]

    checks = []
    for f in files:
        path = root / f
        if path.exists():
            size = path.stat().st_size
            checks.append((f, f"✅ {size} bytes"))
        else:
            checks.append((f, "❌ 不存在"))

    return checks

def check_commands():
    """检查命令是否可用"""
    import subprocess

    commands = [
        ["mts", "--help"],
        ["mts", "train", "--help"],
        ["mts", "serve", "--help"],
        ["mts", "demo", "--help"],
    ]

    checks = []
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
                text=True
            )
            if result.returncode == 0:
                checks.append((" ".join(cmd), "✅"))
            else:
                checks.append((" ".join(cmd), f"❌ exit {result.returncode}"))
        except Exception as e:
            checks.append((" ".join(cmd), f"❌ {e}"))

    return checks

def main():
    print("=" * 70)
    print("MolelTrainSelf 项目诊断报告")
    print("=" * 70)

    print("\n【1. 模块导入检查】")
    for module, attr, status in check_imports():
        print(f"  {status:20} {module}.{attr}")

    print("\n【2. 关键文件检查】")
    for file, status in check_files():
        print(f"  {status:20} {file}")

    print("\n【3. 命令可用性检查】")
    for cmd, status in check_commands():
        print(f"  {status:20} {cmd}")

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)

if __name__ == "__main__":
    main()
