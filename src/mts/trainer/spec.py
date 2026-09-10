"""Search-space definition for deep-model training.

A `TrialSpec` is one point in the state space that agents search over. It is
deliberately flat, JSON/YAML serializable and diffable — agents derive a child
spec from a parent spec by editing a handful of keys, and the board renders the
diff so everyone sees exactly what changed between two trials.

Three buckets:
  * `arch`  — network architecture design (depth, width, norm placement, act...)
  * `optim` — optimization hyperparameters (lr, schedule, clip, weight decay...)
  * `data` / `budget` — the measurement apparatus, held fixed across a search
    unless an agent explicitly suspects the measurement itself.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations. These are the axes agents are allowed to move along.
# ---------------------------------------------------------------------------

NormPosition = Literal["pre", "post", "sandwich", "none"]
NormType = Literal["layernorm", "rmsnorm", "none"]
Activation = Literal["gelu", "relu", "relu2", "silu", "swiglu", "tanh"]
PosEmbedding = Literal["learned", "rope", "none"]
InitScheme = Literal["gpt2", "xavier", "kaiming", "scaled", "small"]
OptimizerName = Literal["adamw", "adam", "sgd", "sgd_momentum"]
ScheduleKind = Literal["constant", "cosine", "linear", "wsd"]
# "auto" picks torch when it is importable, otherwise falls back to the surrogate
# landscape — so a fresh checkout runs end to end on a machine with no GPU stack.
Backend = Literal["auto", "torch", "surrogate"]


# ---------------------------------------------------------------------------
# Spec models
# ---------------------------------------------------------------------------


class InitSpec(BaseModel):
    scheme: InitScheme = "gpt2"
    std: float = Field(default=0.02, gt=0, le=1.0)
    # Scale the residual branch by this factor (e.g. 1/sqrt(2 * n_layer)).
    # `null` means "use the scheme default" — this is the knob agents reach for
    # when a deep post-norm model turns out to be unstable.
    residual_scale: float | None = Field(default=None, gt=0, le=1.0)
    # Scale initial output-projection / attn-projection weights (GPT-2 style).
    scale_projection: bool = True


class ArchSpec(BaseModel):
    kind: Literal["transformer_decoder", "roberta"] = "transformer_decoder"
    n_layer: int = Field(default=4, ge=1, le=64)
    d_model: int = Field(default=256, ge=16, le=8192)
    n_head: int = Field(default=4, ge=1, le=128)
    d_ff: int = Field(default=1024, ge=16, le=32768)
    norm_position: NormPosition = "pre"
    norm_type: NormType = "layernorm"
    activation: Activation = "gelu"
    pos_embedding: PosEmbedding = "learned"
    tie_embeddings: bool = True
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    attn_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    attn_bias: bool = False
    mlp_bias: bool = False
    init: InitSpec = Field(default_factory=InitSpec)
    # RoBERTa specific fields
    pretrained: str | None = None
    num_labels: int | None = None
    freeze_encoder: bool = False

    @model_validator(mode="after")
    def validate_heads(self) -> "ArchSpec":
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        return self


class ScheduleSpec(BaseModel):
    kind: ScheduleKind = "cosine"
    warmup_steps: int = Field(default=50, ge=0)
    min_lr_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    # WSD (warmup-stable-decay) only: fraction of steps spent in decay.
    decay_fraction: float = Field(default=0.2, gt=0.0, le=1.0)


class OptimSpec(BaseModel):
    name: OptimizerName = "adamw"
    lr: float = Field(default=3.0e-3, gt=0, le=10.0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1.0e-8, gt=0)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    grad_clip: float | None = Field(default=1.0, gt=0)
    schedule: ScheduleSpec = Field(default_factory=ScheduleSpec)

    @field_validator("betas")
    @classmethod
    def validate_betas(cls, value: tuple[float, float]) -> tuple[float, float]:
        if not (0.0 <= value[0] < 1.0 and 0.0 <= value[1] < 1.0):
            raise ValueError("betas must be in [0, 1)")
        return value


class DataSpec(BaseModel):
    dataset: Literal["synthetic_lm", "text_file", "custom"] = "synthetic_lm"
    path: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    seq_len: int = Field(default=128, ge=8, le=8192)
    batch_size: int = Field(default=32, ge=1, le=4096)
    num_workers: int = Field(default=0, ge=0)
    # Vocabulary is derived from the dataset; `vocab_size` is a cap.
    vocab_size: int = Field(default=256, ge=16, le=100000)


class BudgetSpec(BaseModel):
    max_steps: int = Field(default=300, ge=1, le=1_000_000)
    max_epochs: int = Field(default=1, ge=1, le=1000)
    time_budget_sec: int = Field(default=900, ge=10)
    eval_every: int = Field(default=25, ge=1)
    probe_every: int = Field(default=25, ge=1)
    eval_batches: int = Field(default=10, ge=1)
    # Early stopping: stop the trial if val_loss fails to improve for this many
    # consecutive evaluations. 0 disables it (run to the step/time budget).
    early_stop_patience: int = Field(default=0, ge=0, le=10000)
    # GPU-seconds budget: the hard ceiling on accelerator time (compute is the
    # real money for training). 0 means "no explicit GPU budget" (fall back to
    # wall-clock time_budget_sec only). On CPU backends this is approximated by
    # wall-clock seconds. See docs/08 for the arbitration between the three
    # budgets (steps / wall-clock / gpu-seconds).
    gpu_seconds_budget: int = Field(default=0, ge=0, le=10_000_000)


class ResourceSpec(BaseModel):
    """The accelerator resources a trial needs (docs/09 P2-1).

    The dispatcher uses this to place a trial on a device. `mem_bytes=0` means
    "let the backend estimate from param count × precision", which keeps the
    surrogate path (and any no-GPU machine) zero-config.
    """

    device: str = "auto"       # auto | cuda:0 | cpu
    mem_bytes: int = Field(default=0, ge=0)   # estimated accelerator memory
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"

    def estimate_mem_bytes(self, param_count: int) -> int:
        """Estimate accelerator memory from parameter count when `mem_bytes=0`.

        Coarse but order-correct: params × bytes-per-param × (weights + grads +
        optimizer states + activation headroom). fp32→4B, bf16/fp16→2B, with a
        ~3× multiplier for optimizer moments and activations.
        """
        if self.mem_bytes > 0:
            return self.mem_bytes
        per_param = 4 if self.precision == "fp32" else 2
        return int(param_count * per_param * 3)


class TrialSpec(BaseModel):
    """One point in the training state space."""

    name: str = "trial"
    parent: str | None = None
    seed: int = 0
    backend: Backend = "auto"
    device: str = "auto"
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    arch: ArchSpec = Field(default_factory=ArchSpec)
    optim: OptimSpec = Field(default_factory=OptimSpec)
    data: DataSpec = Field(default_factory=DataSpec)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    resource: ResourceSpec = Field(default_factory=ResourceSpec)
    notes: str = ""

    # ---- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialSpec":
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str) -> "TrialSpec":
        return cls.model_validate(json.loads(text))

    @classmethod
    def from_file(cls, path: str | Path) -> "TrialSpec":
        raw = Path(path).read_text(encoding="utf-8")
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            import yaml

            return cls.model_validate(yaml.safe_load(raw) or {})
        return cls.from_json(raw)

    def derive(self, **changes: Any) -> "TrialSpec":
        """Return a copy with dotted-path overrides applied.

        `spec.derive(**{"arch.norm_position": "post", "optim.lr": 1e-3})`
        is the primitive agents use to walk the state space one axis at a time.
        """
        payload = copy.deepcopy(self.to_dict())
        for dotted, value in changes.items():
            target = payload
            keys = dotted.split(".")
            for key in keys[:-1]:
                if key not in target or not isinstance(target[key], dict):
                    raise KeyError(f"unknown spec path: {dotted}")
                target = target[key]
            if keys[-1] not in target:
                raise KeyError(f"unknown spec path: {dotted}")
            target[keys[-1]] = value
        return TrialSpec.from_dict(payload)

    def diff(self, other: "TrialSpec") -> dict[str, dict[str, Any]]:
        """Return {dotted_path: {"from": parent_value, "to": self_value}} for every
        changed leaf, i.e. the changes this spec makes *relative to* `other` (the
        parent). Rendered as `parent -> self` by `format_diff`.

        Identity / measurement fields (`name`, `seed`, `notes`, `parent`, `backend`,
        `device`, `precision`) are excluded — they do not describe a *search move*.
        """

        def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
            out: dict[str, Any] = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
            elif isinstance(obj, (list, tuple)):
                out[prefix] = list(obj)
            else:
                out[prefix] = obj
            return out

        def _excluded(key: str) -> bool:
            # Top-level identity/measurement fields, plus the whole `resource`
            # subtree (placement, not a search move).
            top = key.split(".")[0]
            return top in {"name", "seed", "notes", "parent", "backend",
                           "device", "precision", "resource"}

        parent = {k: v for k, v in flatten(other.to_dict()).items() if not _excluded(k)}
        child = {k: v for k, v in flatten(self.to_dict()).items() if not _excluded(k)}
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(set(parent) | set(child)):
            if parent.get(key) != child.get(key):
                changes[key] = {"from": parent.get(key), "to": child.get(key)}
        return changes

    def param_count_estimate(self) -> int:
        """Non-embedding parameter estimate, good enough for budgeting."""
        a = self.arch
        per_layer = (
            4 * a.d_model * a.d_model  # qkv + out projection
            + (3 * a.d_model * a.d_ff if a.activation == "swiglu" else 2 * a.d_model * a.d_ff)
        )
        return a.n_layer * per_layer


# ---------------------------------------------------------------------------
# Human-readable rendering (used in prompts and the board)
# ---------------------------------------------------------------------------

_AXIS_LABELS: dict[str, str] = {
    "arch.n_layer": "depth",
    "arch.d_model": "width",
    "arch.n_head": "heads",
    "arch.d_ff": "ffn_width",
    "arch.norm_position": "norm_pos",
    "arch.norm_type": "norm_type",
    "arch.activation": "act",
    "arch.pos_embedding": "pos_emb",
    "arch.dropout": "dropout",
    "arch.init.scheme": "init",
    "arch.init.residual_scale": "res_scale",
    "optim.name": "optimizer",
    "optim.lr": "lr",
    "optim.weight_decay": "wd",
    "optim.grad_clip": "clip",
    "optim.schedule.kind": "schedule",
    "optim.schedule.warmup_steps": "warmup",
    "data.batch_size": "batch",
    "data.seq_len": "seq_len",
    "budget.max_steps": "steps",
}


def format_diff(diff: dict[str, dict[str, Any]]) -> str:
    """Render a spec diff as one compact line, e.g. `norm_pos pre->post, lr 3e-3->1e-3`."""
    if not diff:
        return "no change"
    parts: list[str] = []
    for path, change in diff.items():
        label = _AXIS_LABELS.get(path, path)
        old, new = change["from"], change["to"]
        parts.append(f"{label} {_short(old)}->{_short(new)}")
    return ", ".join(parts)


def _short(value: Any) -> str:
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) < 1e-3 or abs(value) >= 1e4:
            return f"{value:.1e}"
        return f"{value:g}"
    return str(value)
