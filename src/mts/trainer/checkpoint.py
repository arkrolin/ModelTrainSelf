"""Checkpoint save/load for long-running training trials.

Training a single trial can run for hours or days. Losing that to a process
death is pure waste, so the torch backend periodically persists a checkpoint
under ``runs/{tid}/ckpt/`` and can resume from it. This module is imported
lazily (only by the torch backend), so the server / dispatcher / surrogate path
never requires torch.

A checkpoint contains:

  * ``model.pt``       — ``state_dict`` of the model
  * ``optimizer.pt``   — ``state_dict`` of the optimizer (Adam m/v etc.)
  * ``ckpt.json``      — metadata: step, rng state, elapsed, token count

The on-disk layout is deliberately simple and human-inspectable, matching the
rest of the artifact convention (spec.json / metrics.jsonl / summary.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


CKPT_DIRNAME = "ckpt"
MODEL_FILE = "model.pt"
OPTIM_FILE = "optimizer.pt"
META_FILE = "ckpt.json"


class Checkpointer:
    """Periodic checkpoint writer bound to a single trial's out_dir.

    ``every`` is the number of steps between checkpoints. Saving is atomic-ish:
    each save goes to a fresh subdirectory ``ckpt/step-{n}`` so a crash mid-write
    never corrupts the previous good checkpoint; ``latest`` is updated only
    after a successful write. Keeping a short ring (``keep``) bounds disk usage.
    """

    def __init__(self, out_dir: str | Path, *, every: int = 500, keep: int = 2):
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / CKPT_DIRNAME
        self.every = max(1, every)
        self.keep = max(1, keep)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def should_save(self, step: int, last_step: int) -> bool:
        return step > 0 and (step % self.every == 0 or step == last_step)

    def save(self, model: Any, optimizer: Any, *, step: int, rng_state: Any,
             elapsed_sec: float, tokens: int) -> Path:
        import torch

        slot = self.ckpt_dir / f"step-{step:07d}"
        slot.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), slot / MODEL_FILE)
        torch.save(optimizer.state_dict(), slot / OPTIM_FILE)
        (slot / META_FILE).write_text(
            json.dumps({
                "step": step,
                "elapsed_sec": round(elapsed_sec, 3),
                "tokens": tokens,
                "rng_state": _rng_to_json(rng_state),
                "saved_at": time.time(),
            }, indent=2),
            encoding="utf-8",
        )
        self._write_latest(slot)
        self._prune()
        return slot

    def load(self, model: Any, optimizer: Any, device: str) -> dict[str, Any]:
        """Restore model + optimizer from the latest checkpoint, returning meta."""
        import torch

        latest = self.ckpt_dir / "latest"
        if not (latest / MODEL_FILE).exists():
            raise FileNotFoundError(f"no checkpoint under {self.ckpt_dir}")
        model.load_state_dict(torch.load(latest / MODEL_FILE, map_location=device, weights_only=False))
        optimizer.load_state_dict(torch.load(latest / OPTIM_FILE, map_location=device, weights_only=False))
        meta = json.loads((latest / META_FILE).read_text(encoding="utf-8"))
        return meta

    def has_checkpoint(self) -> bool:
        return (self.ckpt_dir / "latest" / MODEL_FILE).exists()

    def latest_step(self) -> int:
        if not self.has_checkpoint():
            return 0
        meta = json.loads((self.ckpt_dir / "latest" / META_FILE).read_text(encoding="utf-8"))
        return int(meta.get("step", 0))

    def _write_latest(self, slot: Path) -> None:
        latest = self.ckpt_dir / "latest"
        # Use a temporary symlink-ish marker: write a small pointer file rather
        # than relying on symlink support (Windows). Simpler and portable.
        (latest / META_FILE).write_text(
            (slot / META_FILE).read_text(encoding="utf-8"), encoding="utf-8"
        )
        (latest / MODEL_FILE).write_bytes((slot / MODEL_FILE).read_bytes())
        (latest / OPTIM_FILE).write_bytes((slot / OPTIM_FILE).read_bytes())

    def _prune(self) -> None:
        slots = sorted(
            (p for p in self.ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("step-")),
            key=lambda p: p.name,
        )
        for stale in slots[:-self.keep] if len(slots) > self.keep else []:
            import shutil

            shutil.rmtree(stale, ignore_errors=True)


def _rng_to_json(state: Any) -> Any:
    """Best-effort serialization of a numpy Generator / RandomState / int seed."""
    if state is None:
        return None
    if isinstance(state, int):
        return state
    try:
        import numpy as np

        if isinstance(state, np.random.Generator):
            return {"kind": "Generator", "bit_generator": state.bit_generator.state}
        if isinstance(state, np.random.RandomState):
            return {"kind": "RandomState", "state": state.get_state()}
    except Exception:
        pass
    return None


def rng_from_json(data: Any, seed: int = 0) -> Any:
    """Reconstruct an rng from ``_rng_to_json`` output; falls back to a seed."""
    if data is None:
        import numpy as np

        return np.random.default_rng(seed)
    import numpy as np

    if isinstance(data, int):
        return np.random.default_rng(data)
    if isinstance(data, dict) and data.get("kind") == "Generator":
        try:
            gen = np.random.Generator(np.random.PCG64())
            gen.bit_generator.state = data["bit_generator"]
            return gen
        except Exception:
            return np.random.default_rng(seed)
    return np.random.default_rng(seed)
