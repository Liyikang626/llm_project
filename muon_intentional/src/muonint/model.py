"""A compact GPT whose matrix layers can report their in/out activations.

The activation reporting exists solely so the intentional step size can be
computed. For a linear layer h = W x the update induces

    dh = dW x        (exactly, not to first order)

so a target on ||dh||/||h|| translates into a closed-form step size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    mlp_ratio: float = 3.0        # SwiGLU hidden = mlp_ratio * d_model
    seq_len: int = 512
    tied_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MuonLinear(nn.Module):
    """Linear layer that can report (subsampled) inputs and ||output||_F."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)
        self.capture = False
        self.sub_idx: Optional[torch.Tensor] = None
        self.record = None          # (x_sub_bf16, n_rows, ||h||_F)

    def forward(self, x):
        h = F.linear(x, self.weight)
        if self.capture:
            xf = x.reshape(-1, x.shape[-1])
            n_rows = xf.shape[0]
            if self.sub_idx is not None and self.sub_idx.numel() < n_rows:
                xs = xf.index_select(0, self.sub_idx)
            else:
                xs = xf
            self.record = (xs.detach().to(torch.bfloat16), n_rows,
                           float(h.detach().float().norm()))
        return h


def precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv)                       # (T, hd/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    out = torch.stack((o1, o2), dim=-1).flatten(-2)
    return out


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d, h = cfg.d_model, int(cfg.mlp_ratio * cfg.d_model)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.wq, self.wk, self.wv = MuonLinear(d, d), MuonLinear(d, d), MuonLinear(d, d)
        self.wo = MuonLinear(d, d)
        self.w_gate, self.w_up = MuonLinear(d, h), MuonLinear(d, h)
        self.w_down = MuonLinear(h, d)
        self.n_head = cfg.n_head
        self.head_dim = d // cfg.n_head

    def set_capture(self, on: bool, sub_idx=None):
        for m in (self.wq, self.wk, self.wv, self.wo, self.w_gate, self.w_up, self.w_down):
            m.capture = on
            m.sub_idx = sub_idx

    def collect(self, out: Dict[int, tuple]):
        for m in (self.wq, self.wk, self.wv, self.wo, self.w_gate, self.w_up, self.w_down):
            if m.record is not None:
                out[id(m.weight)] = m.record
                m.record = None

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.wo(a)
        h = self.n2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)
        cos, sin = precompute_rope(cfg.seq_len, cfg.d_model // cfg.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._n_sub = 512
        self._sub_idx = None
        self.layer_inputs = {}
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("wo.weight") or n.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        # nn.Embedding's default init is N(0, 1); left alone it makes the
        # initial logits enormous (initial loss ~127 instead of ~ln(50257)=10.8).
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.embed.weight.numel()
        return dict(total=total, embedding=emb, non_embedding=total - emb)

    def set_capture(self, on: bool):
        self._capture = on
        for b in self.blocks:
            b.set_capture(on, getattr(self, "_sub_idx", None))

    def forward(self, idx, targets=None, capture: bool = False):
        B, T = idx.shape
        if capture:
            total = B * T
            n_sub = min(self._n_sub, total)
            self._sub_idx = torch.randperm(total, device=idx.device)[:n_sub]
            self.set_capture(True)
        else:
            self.set_capture(False)

        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        x = self.norm_f(x)
        logits = x @ self.embed.weight.t()          # tied head

        self.layer_inputs = {}
        if capture:
            for b in self.blocks:
                b.collect(self.layer_inputs)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss
