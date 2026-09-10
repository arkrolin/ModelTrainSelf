"""Scratch calibration: check that the surrogate landscape has the right shape.

Not part of the test suite — run it when tuning the surrogate coefficients.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mts.trainer.runner import run_trial
from mts.trainer.spec import TrialSpec
from mts.trainer.surrogate import instability_score

BASE = TrialSpec.from_file(Path(__file__).resolve().parents[1] / "examples/specs/baseline.yaml")

CASES: list[tuple[str, dict]] = [
    ("baseline (pre, lr 3e-3, 4L)", {}),
    ("post-norm, lr 3e-3", {"arch.norm_position": "post"}),
    ("post-norm, lr 1e-3", {"arch.norm_position": "post", "optim.lr": 1.0e-3}),
    ("post-norm, lr 3e-3, warmup 120", {"arch.norm_position": "post",
                                        "optim.schedule.warmup_steps": 120}),
    ("post-norm 12L, lr 1e-3", {"arch.norm_position": "post", "arch.n_layer": 12,
                                "optim.lr": 1.0e-3}),
    ("post-norm 12L + residual scale", {"arch.norm_position": "post", "arch.n_layer": 12,
                                        "optim.lr": 1.0e-3,
                                        "arch.init.residual_scale": 1.0 / (2 * 12) ** 0.5}),
    ("pre 8L", {"arch.n_layer": 8}),
    ("pre 16L", {"arch.n_layer": 16}),
    ("relu activation", {"arch.activation": "relu"}),
    ("relu2 activation", {"arch.activation": "relu2"}),
    ("swiglu activation", {"arch.activation": "swiglu"}),
    ("lr 3e-2 (too high)", {"optim.lr": 3.0e-2}),
    ("lr 1e-4 (too low)", {"optim.lr": 1.0e-4}),
    ("no norm", {"arch.norm_position": "none"}),
    ("no grad clip", {"optim.grad_clip": None}),
    ("16 heads (head_dim 16)", {"arch.n_head": 16}),
    ("dropout 0.2", {"arch.dropout": 0.2}),
    ("wd 0.0", {"optim.weight_decay": 0.0}),
    ("sgd", {"optim.name": "sgd"}),
]

out_root = Path(__file__).resolve().parents[1] / "runs" / "_calibration"
out_root.mkdir(parents=True, exist_ok=True)

rows = []
for idx, (label, changes) in enumerate(CASES, start=1):
    spec = BASE.derive(**changes) if changes else BASE
    trial_id = f"cal{idx:02d}"
    result = run_trial(spec, out_root / trial_id, trial_id=trial_id, parent_spec=BASE)
    grad = result.diagnostics["grad"]
    rows.append({
        "label": label,
        "instab": round(instability_score(spec), 3),
        "verdict": result.verdict,
        "status": result.status,
        "val": result.best.get("val_loss"),
        "acc": result.best.get("val_acc"),
        "p99": grad["p99"],
        "clip": round(grad["clip_fraction"], 2),
        "b/t": grad["bottom_top_ratio"],
        "dead": round(result.diagnostics["activation"]["max_dead_ratio"], 3),
        "ent": round(result.diagnostics["activation"]["attention_entropy"], 2),
        "spk": result.diagnostics["stability"]["loss_spike_count"],
    })

width = max(len(r["label"]) for r in rows)
print(f"{'case'.ljust(width)} | instab | verdict    | val_loss | acc   | grad_p99  | clip | bot/top  | dead  | ent  | spk")
print("-" * (width + 95))
for r in rows:
    val = f"{r['val']:.3f}" if r["val"] is not None else "  n/a"
    print(f"{r['label'].ljust(width)} | {r['instab']:6.3f} | {r['verdict']:<9} | {val:>8} | "
          f"{r['acc']:.3f} | {r['p99']:9.2e} | {r['clip']:.2f} | {r['b/t']:8.4f} | "
          f"{r['dead']:.3f} | {r['ent']:.2f} | {r['spk']}")

print()
print(json.dumps({"rows": rows}, indent=2)[:200])
