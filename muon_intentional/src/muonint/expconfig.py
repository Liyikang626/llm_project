"""Single source of truth for the experiment matrix.

Both `muon_intentional_pilot.ipynb` (interactive) and
`scripts/run_experiments.py` (headless, under Slurm) import from here, so the
two always produce the *same* result-file tags and can share one cache.
"""

from __future__ import annotations

import os

ROOT = "/home/15Cu2Plus/muon_intentional"
DATA = os.path.join(ROOT, "data", "climbmix_400m.u16")

ARMS = ["adamw", "muon", "orscale", "intent_act",
        "intent_kl_global", "intent_kl_layer"]

ARM_LABEL = {
    "adamw": "AdamW（参照锚点）",
    "muon": "Muon 原始（基线）",
    "orscale": "Muon+OrScale（参数空间逐层）",
    "intent_act": "逐层 Activation 控制",
    "intent_kl_global": "全局 Logit-KL 控制",
    "intent_kl_layer": "逐层 Logit-KL 控制（本方案）",
}

# 图内文字用英文：集群没有 CJK 字体
ARM_SHORT = {
    "adamw": "AdamW", "muon": "Muon", "orscale": "Muon+OrScale",
    "intent_act": "Act-wise", "intent_kl_global": "KL-global",
    "intent_kl_layer": "KL-layer",
}

SWEEP_PARAM = {
    "adamw": "lr", "muon": "lr", "orscale": "lr", "intent_act": "lr",
    "intent_kl_global": "kl_eps", "intent_kl_layer": "kl_eps",
}

# ε 的范围必须让换算出的实际 η 覆盖 Muon 的 lr 区间，间距取 √4 = 2 倍 η。
SWEEP = {
    "adamw": [3e-4, 6e-4, 1e-3, 2e-3],
    "muon": [0.01, 0.02, 0.04, 0.08],
    "orscale": [0.01, 0.02, 0.04, 0.08],
    "intent_act": [0.005, 0.01, 0.02, 0.04],
    "intent_kl_global": [3e-5, 1.25e-4, 5e-4, 2e-3, 8e-3, 3e-2],
    "intent_kl_layer": [3e-5, 1.25e-4, 5e-4, 2e-3, 8e-3, 3e-2],
}

_BASE_P = dict(
    a_steps=300, a_d=256, a_L=4, a_H=4, a_seq=256, a_batches=(8, 64, 256),
    b1_d=384, b1_L=6, b1_tokens=15_000_000, b1_batch=32, b1_seq=512,
    b2_d=512, b2_L=8, b2_tokens=100_000_000, b2_batch=32, b2_seq=512, b2_seeds=(0, 1),
    b3_d=320, b3_L=10, b3_tokens=60_000_000, b3_batch=8, b3_seq=512, b3_seeds=(0, 1),
    c_steps=250, c_seeds=(0, 1, 2), c_eval=100,
    c_prompts=8, c_group=6, c_max_new=192,
)

_QUICK_P = dict(
    a_steps=12, a_d=128, a_L=2, a_H=2, a_seq=128, a_batches=(8, 32),
    b1_d=128, b1_L=2, b1_tokens=12_000, b1_batch=8, b1_seq=128,
    b2_d=128, b2_L=2, b2_tokens=16_000, b2_batch=8, b2_seq=128, b2_seeds=(0,),
    b3_d=96, b3_L=3, b3_tokens=16_000, b3_batch=4, b3_seq=128, b3_seeds=(0,),
    c_steps=4, c_seeds=(0,), c_eval=8, c_prompts=2, c_group=2, c_max_new=12,
)


def params(mode: str = "full") -> dict:
    """Experiment sizes. 'quick' shrinks everything to a plumbing check."""
    p = dict(_BASE_P)
    if mode == "quick":
        p.update(_QUICK_P)
    return p


def sweep_for(mode: str = "full") -> dict:
    """Sweep grid. quick keeps 2 candidates per arm."""
    return {k: (list(v)[:2] if mode == "quick" else list(v)) for k, v in SWEEP.items()}


def paths(mode: str = "full") -> dict:
    """Result dirs, partitioned by mode so quick and full never mix."""
    res = os.path.join(ROOT, "results")
    return dict(
        RES=res,
        DIAG=os.path.join(res, "diag", mode),
        PRE=os.path.join(res, "pretrain", mode),
        RLD=os.path.join(res, "rl", mode),
    )


def make_cfg(arm: str, value: float, d: int, L: int, tokens: int, batch: int,
             seq: int, seed: int, tag_prefix: str, out_dir: str, adam_lr: float = 3e-4):
    """Build a RunConfig, routing `value` to lr or kl_eps depending on the arm."""
    from .pretrain import RunConfig

    kw = dict(name=f"{tag_prefix}_{arm}", opt_mode=arm, adam_lr=adam_lr,
              n_layer=L, n_head=max(2, L), d_model=d, seq_len=seq,
              batch_size=batch, tokens=tokens, seed=seed,
              log_every=10 ** 9, eval_every=10 ** 9,
              data_path=DATA, out_dir=out_dir)
    if SWEEP_PARAM[arm] == "kl_eps":
        kw.update(lr=0.02, kl_eps=value, kl_every=20, kl_check_every=200)
    else:
        kw.update(lr=value)
    return RunConfig(**kw)
