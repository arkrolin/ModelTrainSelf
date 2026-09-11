#!/usr/bin/env python
"""训练脚本：运行基线实验并输出指标到 metrics.json"""

import json
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


def main():
    # 工作目录
    proj_dir = Path(__file__).parent

    # 加载基线配置
    baseline_path = proj_dir.parent / "examples" / "specs" / "baseline.yaml"
    print(f"加载基线配置: {baseline_path}")
    spec = TrialSpec.from_file(baseline_path)

    # 输出目录
    out_dir = proj_dir / "run_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {out_dir}")
    print(f"配置: {spec.name}")
    print(f"  - 层数: {spec.arch.n_layer}")
    print(f"  - d_model: {spec.arch.d_model}")
    print(f"  - 学习率: {spec.optim.lr}")
    print(f"  - 最大步数: {spec.budget.max_steps}")
    print(f"  - Norm位置: {spec.arch.norm_position}")
    print(f"  - 后端: {spec.backend}")
    print()

    # 运行训练
    def progress(msg: str):
        print(f"[TRAIN] {msg}")

    print("开始训练...")
    result = run_trial(
        spec,
        out_dir,
        trial_id="baseline",
        parent_spec=None,
        backend=None,  # 使用配置中的 backend
        device=None,   # 使用配置中的 device
        reference_loss=None,
        progress=progress,
    )

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f"Trial ID: {result.trial_id}")
    print(f"状态: {result.status}")
    print(f"判决: {result.verdict}")
    print(f"步数: {result.steps_completed}")
    print(f"耗时: {result.duration_sec:.2f}秒")
    print(f"\n最终指标:")
    print(f"  train_loss: {result.final.get('train_loss')}")
    print(f"  val_loss: {result.final.get('val_loss')}")
    print(f"  val_acc: {result.final.get('val_acc')}")
    print(f"\n最佳指标:")
    print(f"  val_loss: {result.best.get('val_loss')} (step {result.best.get('step')})")
    print(f"  val_acc: {result.best.get('val_acc')}")
    print(f"\n信号: {', '.join(result.signals) if result.signals else '无'}")

    # 输出 metrics.json
    metrics_output = {
        "trial_id": result.trial_id,
        "name": result.name,
        "status": result.status,
        "verdict": result.verdict,
        "steps_completed": result.steps_completed,
        "duration_sec": result.duration_sec,
        "train_loss": result.final.get("train_loss"),
        "val_loss": result.final.get("val_loss"),
        "val_acc": result.final.get("val_acc"),
        "best_val_loss": result.best.get("val_loss"),
        "best_val_loss_step": result.best.get("step"),
        "best_val_acc": result.best.get("val_acc"),
        "signals": result.signals,
        "backend": result.backend,
        "device": result.device,
    }

    metrics_path = proj_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2, ensure_ascii=False)

    print(f"\n指标已写入: {metrics_path}")
    print(f"详细产物位于: {out_dir}")

    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
