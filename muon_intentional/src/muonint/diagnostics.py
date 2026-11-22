"""Stage A: does the intentional step have anything to fix?

A1  exactness      -- confirm that for a linear layer the induced output change
                      equals the intended one exactly (no first-order error).
A2/A3 headroom     -- under *fixed-eta* Muon, how heterogeneous and how noisy is
                      the realised per-layer relative output change
                      ||dH_l||/||H_l|| ?  If it is already nearly uniform across
                      layers and stable in time, an intentional rule that pins it
                      to eta has almost nothing left to do.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import numpy as np
import torch

from .model import GPT, GPTConfig
from .common import autocast
from .optim import zeropower_via_newtonschulz5


# --------------------------------------------------------------------------
# A1
# --------------------------------------------------------------------------
def test_exactness(seed: int = 0) -> Dict[str, float]:
    """||dH||/||H|| must equal eta to float precision for a single linear layer."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for (m, n) in [(512, 512), (1408, 512), (512, 1408)]:
        W = torch.randn(m, n, generator=g)
        X = torch.randn(2048, n, generator=g)
        M = torch.randn(m, n, generator=g)
        O = zeropower_via_newtonschulz5(M)
        O = O * math.sqrt(max(1.0, m / n))
        H = X @ W.t()
        eta = 0.01
        alpha = eta * H.norm() / (X @ O.t()).norm()
        dH = X @ (-alpha * O).t()
        out[f"{m}x{n}"] = float(dH.norm() / H.norm())
    out["target_eta"] = 0.01
    out["max_rel_err"] = max(abs(v - 0.01) / 0.01 for k, v in out.items() if k != "target_eta")
    return out


# --------------------------------------------------------------------------
# A2 / A3
# --------------------------------------------------------------------------
@torch.no_grad()
def _noop():
    pass


def run_layer_heterogeneity(
    data_path: str,
    batch_sizes=(8, 64, 256),
    steps: int = 300,
    seq_len: int = 256,
    d_model: int = 256,
    n_layer: int = 4,
    n_head: int = 4,
    lr: float = 0.02,
    seed: int = 0,
    out_path: str = "/home/15Cu2Plus/muon_intentional/results/diag/heterogeneity.json",
    device="cuda",
) -> Dict:
    """Train a tiny GPT with classic Muon and record ||dH_l||/||H_l|| per layer.

    The recorded quantity is exactly what a fixed step size produces; the
    intentional rule would by construction hold it at a constant eta.
    """
    from .data import TokenBatcher
    from .optim import MuonIntentional, split_muon_params

    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "done":
            return prev

    results = {}
    for bs in batch_sizes:
        torch.manual_seed(seed)
        cfg = GPTConfig(vocab_size=50257, n_layer=n_layer, n_head=n_head,
                        d_model=d_model, seq_len=seq_len)
        model = GPT(cfg).to(device)
        model._n_sub = 256
        batcher = TokenBatcher(data_path, bs, seq_len, seed=seed)
        muon, adam = split_muon_params(model.named_parameters())
        opt = MuonIntentional(muon, adam, mode="fixed", lr=lr, adam_lr=3e-4)
        names = {id(p): n for n, p in model.named_parameters()}
        opt.register_names(names)
        model.train()

        per_step_ratio: List[List[float]] = []
        losses = []
        for step in range(steps):
            x, y = batcher.train_batch(device)
            opt.zero_grad(set_to_none=True)
            with autocast(device):
                _, loss = model(x, y, capture=True)
            loss.backward()
            opt.set_layer_inputs(model.layer_inputs)
            opt.step()
            names_l = list(opt.stats.rel_delta_h.keys())
            if step >= steps // 5:            # skip warm-up transient
                per_step_ratio.append([opt.stats.rel_delta_h[k] for k in names_l])
            losses.append(float(loss.item()))
        R = torch.tensor(per_step_ratio)      # (T, L)
        layer_names = list(opt.stats.rel_delta_h.keys())
        # collapse q/k/v/wo/gate/up/down of the same block for readability
        mean_by_layer = R.mean(0)
        spread_step = R.max(1).values / (R.min(1).values + 1e-12)          # across layers, per step
        cv_temporal = R.std(0) / (R.mean(0) + 1e-12)                       # per layer, over time
        results[f"bs{bs}"] = dict(
            layer_names=layer_names,
            mean_by_layer=mean_by_layer.tolist(),
            final_loss=losses[-1], first_loss=losses[0],
            across_layer_spread=dict(
                mean=float(spread_step.mean()), p90=float(spread_step.quantile(0.9)),
                max=float(spread_step.max()),
            ),
            across_layer_cv=float(mean_by_layer.std() / (mean_by_layer.mean() + 1e-12)),
            temporal_cv_mean=float(cv_temporal.mean()),
            ratio_mean=float(R.mean()), ratio_median=float(R.median()),
            ratio_p10=float(R.quantile(0.1)), ratio_p90=float(R.quantile(0.9)),
        )

    out = dict(status="done", results=results)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out
