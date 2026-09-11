#!/usr/bin/env python
"""i002 探索：训练预算与学习率扫描实验"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec


def run_experiment(lr: float, max_steps: int, warmup_steps: int, trial_id: str, out_dir: Path) -> Dict[str, Any]:
    """运行单个实验配置"""

    # 加载基线配置
    baseline_path = Path(__file__).parent.parent / "examples" / "specs" / "baseline.yaml"
    spec = TrialSpec.from_file(baseline_path)

    # 修改配置：学习率和训练步数
    spec.optim.lr = lr
    spec.budget.max_steps = max_steps
    spec.budget.eval_every = max(25, max_steps // 40)  # 大约40个评估点
    spec.budget.probe_every = spec.budget.eval_every

    # 调整warmup
    spec.optim.schedule.warmup_steps = warmup_steps

    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"实验: {trial_id}")
    print(f"{'='*70}")
    print(f"  学习率: {lr}")
    print(f"  最大步数: {max_steps}")
    print(f"  Warmup步数: {warmup_steps}")
    print(f"  评估间隔: {spec.budget.eval_every}")
    print(f"  输出目录: {out_dir}")
    print(f"{'='*70}\n")

    # 运行训练
    def progress(msg: str):
        print(f"[{trial_id}] {msg}")

    result = run_trial(
        spec,
        out_dir,
        trial_id=trial_id,
        parent_spec=None,
        backend=None,
        device=None,
        reference_loss=None,
        progress=progress,
    )

    print(f"\n{'='*70}")
    print(f"实验完成: {trial_id}")
    print(f"{'='*70}")
    print(f"  状态: {result.status}")
    print(f"  判决: {result.verdict}")
    print(f"  步数: {result.steps_completed}")
    print(f"  耗时: {result.duration_sec:.2f}秒")
    print(f"  train_loss: {result.final.get('train_loss', 'N/A')}")
    print(f"  val_loss: {result.final.get('val_loss', 'N/A')}")
    print(f"  best_val_loss: {result.best.get('val_loss', 'N/A')} @ step {result.best.get('step', 'N/A')}")
    print(f"  信号: {', '.join(result.signals) if result.signals else '无'}")
    print(f"{'='*70}\n")

    return {
        "trial_id": trial_id,
        "lr": lr,
        "max_steps": max_steps,
        "warmup_steps": warmup_steps,
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


def main():
    parser = argparse.ArgumentParser(description="i002 训练预算与学习率扫描")
    parser.add_argument("--mode", choices=["lr_scan", "budget_scan", "single"], required=True,
                      help="实验模式：lr_scan=学习率扫描, budget_scan=训练预算扫描, single=单次实验")
    parser.add_argument("--lr", type=float, help="学习率（single模式）")
    parser.add_argument("--steps", type=int, help="训练步数（single模式）")
    args = parser.parse_args()

    proj_dir = Path(__file__).parent
    results = []

    if args.mode == "lr_scan":
        # 学习率扫描：用5000步测试不同学习率
        lr_values = [1e-3, 3e-3, 5e-3, 1e-2]
        max_steps = 5000
        warmup_steps = 200  # 4% warmup

        for lr in lr_values:
            trial_id = f"i002_lr{lr:.0e}_s{max_steps}"
            out_dir = proj_dir / f"run_{trial_id}"

            try:
                result = run_experiment(lr, max_steps, warmup_steps, trial_id, out_dir)
                results.append(result)
            except Exception as e:
                print(f"实验 {trial_id} 失败: {e}")
                results.append({
                    "trial_id": trial_id,
                    "lr": lr,
                    "max_steps": max_steps,
                    "status": "failed",
                    "error": str(e)
                })

    elif args.mode == "budget_scan":
        # 训练预算扫描：用最优学习率测试不同步数
        # 从 lr_scan 结果中选择最优lr，这里先用3e-3
        best_lr = 3e-3
        step_values = [3000, 5000, 10000]

        for max_steps in step_values:
            warmup_steps = int(max_steps * 0.04)  # 4% warmup
            trial_id = f"i002_lr{best_lr:.0e}_s{max_steps}"
            out_dir = proj_dir / f"run_{trial_id}"

            try:
                result = run_experiment(best_lr, max_steps, warmup_steps, trial_id, out_dir)
                results.append(result)
            except Exception as e:
                print(f"实验 {trial_id} 失败: {e}")
                results.append({
                    "trial_id": trial_id,
                    "lr": best_lr,
                    "max_steps": max_steps,
                    "status": "failed",
                    "error": str(e)
                })

    elif args.mode == "single":
        # 单次实验
        if args.lr is None or args.steps is None:
            print("错误：single 模式需要 --lr 和 --steps 参数")
            return 1

        warmup_steps = int(args.steps * 0.04)
        trial_id = f"i002_lr{args.lr:.0e}_s{args.steps}"
        out_dir = proj_dir / f"run_{trial_id}"

        result = run_experiment(args.lr, args.steps, warmup_steps, trial_id, out_dir)
        results.append(result)

    # 保存汇总结果
    summary_path = proj_dir / f"i002_summary_{args.mode}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "experiments": results,
            "summary": {
                "total": len(results),
                "completed": sum(1 for r in results if r.get("status") == "completed"),
                "failed": sum(1 for r in results if r.get("status") == "failed"),
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"所��实验完成！汇总结果已保存到: {summary_path}")
    print(f"{'='*70}\n")

    # 输出最佳结果
    completed = [r for r in results if r.get("status") == "completed" and r.get("best_val_loss") is not None]
    if completed:
        best = min(completed, key=lambda r: r["best_val_loss"])
        print(f"最佳结果:")
        print(f"  Trial: {best['trial_id']}")
        print(f"  LR: {best['lr']}")
        print(f"  Steps: {best['max_steps']}")
        print(f"  Best Val Loss: {best['best_val_loss']:.4f} @ step {best['best_val_loss_step']}")
        print(f"  Verdict: {best['verdict']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
