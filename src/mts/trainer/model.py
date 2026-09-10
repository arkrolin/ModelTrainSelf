"""A small, fully configurable decoder-only Transformer.

The point of this model is not to be the best architecture — it is to make every
axis an agent might want to move (norm placement, activation, residual scaling,
position encoding, init scheme) a real, independently settable knob, and to
expose the internals the probes need.

This module imports torch eagerly. It is only ever imported by the `torch`
training backend, so hosts without a GPU stack (server, dispatcher, tests) never
touch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mts.trainer.spec import ArchSpec, OptimSpec


# ---------------------------------------------------------------------------
# Normalization / activation
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SquaredReLU(nn.Module):
    """ReLU^2 — a useful stress test: it amplifies activation outliers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.square(F.relu(x))


def make_norm(spec: ArchSpec, dim: int) -> nn.Module:
    if spec.norm_type == "layernorm":
        return nn.LayerNorm(dim)
    if spec.norm_type == "rmsnorm":
        return RMSNorm(dim)
    return nn.Identity()


def make_activation(name: str) -> nn.Module | None:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "relu":
        return nn.ReLU()
    if name == "relu2":
        return SquaredReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "swiglu":
        return None  # gating needs a 2x projection, handled inside MLP
    raise ValueError(f"unknown activation: {name}")


# ---------------------------------------------------------------------------
# Rotary position embedding
# ---------------------------------------------------------------------------


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: (B, H, T, Dh); cos,sin: (T, Dh)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ---------------------------------------------------------------------------
# Attention / MLP / Block
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, spec: ArchSpec):
        super().__init__()
        self.n_head = spec.n_head
        self.d_model = spec.d_model
        self.head_dim = spec.d_model // spec.n_head
        self.qkv = nn.Linear(spec.d_model, 3 * spec.d_model, bias=spec.attn_bias)
        self.proj = nn.Linear(spec.d_model, spec.d_model, bias=spec.attn_bias)
        self.attn_dropout = nn.Dropout(spec.attn_dropout)
        self.resid_dropout = nn.Dropout(spec.dropout)
        self.entropy: float = 0.0

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None, collect: bool):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if rope is not None:
            q, k = apply_rotary(q, k, *rope)
        if collect:
            with torch.no_grad():
                scale = math.sqrt(self.head_dim)
                scores = (q @ k.transpose(-2, -1)) / scale
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(causal, float("-inf"))
                probs = torch.softmax(scores.float(), dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                norm = math.log(T) if T > 1 else 1.0
                # 1.0 = uniform attention, 0.0 = one-hot (collapsed).
                self.entropy = float((entropy / norm).mean().item())
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, spec: ArchSpec):
        super().__init__()
        self.act_name = spec.activation
        self.swiglu = spec.activation == "swiglu"
        if self.swiglu:
            self.gate_up = nn.Linear(spec.d_model, 2 * spec.d_ff, bias=spec.mlp_bias)
        else:
            self.fc = nn.Linear(spec.d_model, spec.d_ff, bias=spec.mlp_bias)
            self.act = make_activation(spec.activation)
        self.proj = nn.Linear(spec.d_ff, spec.d_model, bias=spec.mlp_bias)
        self.dropout = nn.Dropout(spec.dropout)
        self.dead_ratio: float = 0.0
        self.act_rms: float = 0.0

    def forward(self, x: torch.Tensor, collect: bool):
        if self.swiglu:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            h = F.silu(gate) * up
        else:
            h = self.act(self.fc(x))
        if collect:
            with torch.no_grad():
                flat = h.detach().float()
                self.act_rms = float(flat.pow(2).mean().sqrt().item())
                self.dead_ratio = float((flat.abs() < 1e-6).float().mean().item())
        return self.dropout(self.proj(h))


class Block(nn.Module):
    def __init__(self, spec: ArchSpec, index: int):
        super().__init__()
        self.index = index
        self.spec = spec
        self.ln1 = make_norm(spec, spec.d_model)
        self.attn = CausalSelfAttention(spec)
        self.ln2 = make_norm(spec, spec.d_model)
        self.mlp = MLP(spec)
        self.resid_scale = spec.init.residual_scale
        if self.resid_scale is None and spec.init.scheme == "scaled":
            self.resid_scale = 1.0 / math.sqrt(2 * spec.n_layer)

    def _res(self, branch: torch.Tensor) -> torch.Tensor:
        return branch if self.resid_scale is None else branch * self.resid_scale

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        collect: bool,
    ) -> torch.Tensor:
        pos = self.spec.norm_position
        if pos == "pre":
            x = x + self._res(self.attn(self.ln1(x), rope, collect))
            x = x + self._res(self.mlp(self.ln2(x), collect))
            return x
        if pos == "sandwich":
            h = self.ln1(x)
            x = x + self._res(self.attn(h, rope, collect))
            x = self.ln1(x + self._res(self.mlp(x, collect)))
            return x
        if pos == "post":
            x = self.ln1(x + self._res(self.attn(x, rope, collect)))
            x = self.ln2(x + self._res(self.mlp(x, collect)))
            return x
        # none: bare residual stack, no normalization anywhere
        x = x + self._res(self.attn(x, rope, collect))
        return x + self._res(self.mlp(x, collect))


# ---------------------------------------------------------------------------
# Probe snapshot
# ---------------------------------------------------------------------------


@dataclass
class BlockProbe:
    name: str
    attn_entropy: float = 0.0
    dead_ratio: float = 0.0
    activation_rms: float = 0.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Transformer(nn.Module):
    def __init__(self, spec: ArchSpec, vocab_size: int):
        super().__init__()
        self.spec = spec
        self.vocab_size = vocab_size
        self.collect = False

        self.embed = nn.Embedding(vocab_size, spec.d_model)
        self.blocks = nn.ModuleList([Block(spec, i) for i in range(spec.n_layer)])
        # Pre-norm stacks need a final norm before the head; post-norm stacks
        # already normalize inside each block, but keeping ln_f is standard and
        # keeps the two variants comparable.
        self.ln_f = make_norm(spec, spec.d_model)

        if spec.pos_embedding == "learned":
            self.pos = nn.Embedding(spec_max_len(spec), spec.d_model)
        else:
            self.pos = None
        if spec.pos_embedding == "rope":
            self.rope = Rotary(spec.d_model // spec.n_head)
        else:
            self.rope = None

        self.lm_head = nn.Linear(spec.d_model, vocab_size, bias=False)
        if spec.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(lambda m: self._init_module(m))
        self._apply_init_overrides()

    # ---- init ------------------------------------------------------------

    def _init_module(self, module: nn.Module) -> None:
        spec = self.spec
        if isinstance(module, nn.Linear):
            if spec.init.scheme == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif spec.init.scheme == "kaiming":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:  # gpt2 / scaled / small
                std = spec.init.std
                if spec.init.scheme == "small":
                    std = std * 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=spec.init.std)

    def _apply_init_overrides(self) -> None:
        """GPT-2 style: down-scale residual projections by depth."""
        if not self.spec.init.scale_projection:
            return
        scale = 1.0 / math.sqrt(2 * self.spec.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=self.spec.init.std * scale)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=self.spec.init.std * scale)

    # ---- forward ---------------------------------------------------------

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape
        device = idx.device
        x = self.embed(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(T, device=device))
        rope = None
        if self.rope is not None:
            rope = self.rope(T, device)

        for block in self.blocks:
            x = block(x, rope, self.collect)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    # ---- probes ----------------------------------------------------------

    def probe_blocks(self) -> list[BlockProbe]:
        return [
            BlockProbe(
                name=f"block{i}",
                attn_entropy=block.attn.entropy,
                dead_ratio=block.mlp.dead_ratio,
                activation_rms=block.mlp.act_rms,
            )
            for i, block in enumerate(self.blocks)
        ]

    @torch.no_grad()
    def param_snapshot(self) -> tuple[list[float], list[float]]:
        """Flattened (params, grads) samples for histogram building."""
        params: list[float] = []
        grads: list[float] = []
        for _, p in self.named_parameters():
            if p.numel() == 0:
                continue
            flat = p.detach().float().flatten()
            params.extend(_sample(flat).tolist())
            if p.grad is not None:
                g = p.grad.detach().float().flatten()
                grads.extend(_sample(g).tolist())
        return params, grads

    def named_layer_groups(self) -> list[tuple[str, list[nn.Parameter]]]:
        groups: list[tuple[str, list[nn.Parameter]]] = []
        for i, block in enumerate(self.blocks):
            groups.append((f"block{i}", [p for p in block.parameters()]))
        head = [p for p in self.lm_head.parameters()]
        if not self.spec.tie_embeddings:
            head = head + [p for p in self.embed.parameters()]
        groups.append(("head", head))
        return groups


def _sample(flat: torch.Tensor, cap: int = 8192) -> torch.Tensor:
    if flat.numel() <= cap:
        return flat.cpu()
    idx = torch.linspace(0, flat.numel() - 1, cap).long()
    return flat[idx].cpu()


def spec_max_len(spec: ArchSpec) -> int:
    return 4096


# ---------------------------------------------------------------------------
# Optimizer / schedule
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, spec: OptimSpec) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim < 2 or "norm" in name or "ln_" in name) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": spec.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if spec.name == "adamw":
        return torch.optim.AdamW(groups, lr=spec.lr, betas=spec.betas, eps=spec.eps)
    if spec.name == "adam":
        return torch.optim.Adam(groups, lr=spec.lr, betas=spec.betas, eps=spec.eps)
    if spec.name == "sgd":
        return torch.optim.SGD(groups, lr=spec.lr)
    if spec.name == "sgd_momentum":
        return torch.optim.SGD(groups, lr=spec.lr, momentum=spec.momentum)
    raise ValueError(f"unknown optimizer: {spec.name}")


def lr_at(spec: OptimSpec, step: int, max_steps: int) -> float:
    s = spec.schedule
    warmup = max(1, min(s.warmup_steps, max_steps - 1)) if max_steps > 1 else 1
    if step < warmup:
        return spec.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    if s.kind == "constant":
        return spec.lr
    if s.kind == "linear":
        return spec.lr * (1 - progress) + spec.lr * s.min_lr_ratio * progress
    if s.kind == "wsd":
        decay_start = 1.0 - s.decay_fraction
        if progress < decay_start:
            return spec.lr
        local = (progress - decay_start) / max(1e-9, s.decay_fraction)
        return spec.lr * (1 - local) + spec.lr * s.min_lr_ratio * local
    # cosine
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return spec.lr * (s.min_lr_ratio + (1 - s.min_lr_ratio) * cosine)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
