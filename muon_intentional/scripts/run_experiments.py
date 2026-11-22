#!/usr/bin/env python
"""Headless driver for the pilot experiments (use under Slurm, not on head).

    python -I scripts/run_experiments.py --stage b1
    python -I scripts/run_experiments.py --stage b2
    python -I scripts/run_experiments.py --stage all

Results land in results/<stage>/<mode>/ as one JSON per configuration, and the
notebook reads the same files, so after a Slurm run the notebook's analysis
cells hit the cache and finish instantly.

Every run is resumable: an existing `status="done"` file is reused.  A diverged
configuration is stored as `status="diverged"` and *is* retried, so a fixed
run can never be silently skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/home/15Cu2Plus/muon_intentional/src")

import numpy as np  # noqa: E402

from muonint import expconfig as X  # noqa: E402
from muonint.common import save_json, load_json  # noqa: E402
from muonint.data import build_token_cache  # noqa: E402
from muonint.pretrain import run as run_pretrain  # noqa: E402


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _one(arm, value, P, out_dir, seeds, tag_prefix, device):
    """Run one (arm, value) cell of the sweep grid; returns (losses, etas)."""
    vals, etas = [], []
    for sd in seeds:
        c = X.make_cfg(arm, value, P["b1_d"], P["b1_L"], P["b1_tokens"],
                       P["b1_batch"], P["b1_seq"], sd, tag_prefix, out_dir)
        r = run_pretrain(c, device=device, verbose=False)
        if r.get("status") != "done":
            print(f"      seed={sd}: DIVERGED at step {r.get('diverged_at_step')}", flush=True)
        vals.append(float(r.get("final_val_loss", float("nan"))))
        kd = r.get("last_kl_diag") or {}
        if kd.get("sched_seen"):
            for k in ("eta_med", "eta"):
                if kd.get(k):
                    etas.append(kd[k])
                    break
    return vals, etas


def best_of(table):
    """Best finite value per arm. A diverged (nan) point must never win."""
    out = {}
    for arm, rows in table.items():
        good = [(v, l) for v, l in rows if np.isfinite(l)]
        if not good:
            raise RuntimeError(f"{arm}: 所有候选都发散了")
        if len(good) < len(rows):
            print(f"   ! {arm}: {len(rows)-len(good)}/{len(rows)} 个候选发散，已排除")
        out[arm] = min(good, key=lambda t: t[1])[0]
    return out


def stage_a(P, paths, device):
    from muonint.diagnostics import run_layer_heterogeneity
    t0 = time.time()
    os.makedirs(paths["DIAG"], exist_ok=True)
    d = run_layer_heterogeneity(
        X.DATA, batch_sizes=P["a_batches"], steps=P["a_steps"], seq_len=P["a_seq"],
        d_model=P["a_d"], n_layer=P["a_L"], n_head=P["a_H"], device=device,
        out_path=os.path.join(paths["DIAG"], "heterogeneity.json"))
    for k, v in d["results"].items():
        print(f"  {k}: across_layer_cv={v['across_layer_cv']:.3f} "
              f"p90={v['across_layer_spread']['p90']:.2f} temporal_cv={v['temporal_cv_mean']:.3f}")
    print(f"Stage A {time.time()-t0:.0f}s", flush=True)


def stage_b1(P, paths, device, mode):
    t0 = time.time()
    grid = X.sweep_for(mode)
    table = {}
    print("B1 sweep", flush=True)
    for arm in X.ARMS:
        rows = []
        for val in grid[arm]:
            vals, etas = _one(arm, val, P, paths["PRE"], (0,), "b1", device)
            m = float(np.mean(vals))
            rows.append((val, m))
            unit = "ε" if X.SWEEP_PARAM[arm] == "kl_eps" else "lr/η"
            tag = "  ← 发散" if not np.isfinite(m) else ""
            eta = f"  实测η中位≈{np.mean(etas):.4f}" if etas else ""
            print(f"   {arm:16s} {unit}={val:<9g} val_loss={m:.4f}{tag}{eta}", flush=True)
        table[arm] = rows
    save_json(os.path.join(paths["PRE"], "b1_sweep.json"), table)
    best = best_of(table)
    save_json(os.path.join(paths["PRE"], "b1_best.json"), best)
    print("选定:", best, flush=True)
    print(f"B1 {(time.time()-t0)/60:.1f} min", flush=True)
    return best


def stage_b2(P, paths, device):
    t0 = time.time()
    best = load_json(os.path.join(paths["PRE"], "b1_best.json"))
    if best is None:
        raise RuntimeError("先跑 --stage b1（缺 b1_best.json）")
    print(f"B2 main with {best}", flush=True)
    b2 = {}
    for arm in X.ARMS:
        vals = []
        for sd in P["b2_seeds"]:
            c = X.make_cfg(arm, best[arm], P["b2_d"], P["b2_L"], P["b2_tokens"],
                           P["b2_batch"], P["b2_seq"], sd, "b2", paths["PRE"])
            c.log_every = 50
            c.eval_every = max(50, (P["b2_tokens"] // (P["b2_batch"] * P["b2_seq"])) // 8)
            r = run_pretrain(c, device=device, verbose=False)
            if r.get("status") != "done":
                print(f"   {arm:16s} seed={sd} DIVERGED at {r.get('diverged_at_step')}", flush=True)
            vals.append(float(r.get("final_val_loss", float("nan"))))
        b2[arm] = vals
        finite = [v for v in vals if np.isfinite(v)]
        m = np.mean(finite) if finite else float("nan")
        s = np.std(finite) if len(finite) > 1 else 0.0
        print(f"   {arm:16s} {m:.4f} ± {s:.4f}", flush=True)
    save_json(os.path.join(paths["PRE"], "b2_summary.json"), b2)
    print(f"B2 {(time.time()-t0)/60:.1f} min", flush=True)


def stage_b3(P, paths, device, only_arms=None):
    """only_arms: 只处理这些臂（用于把剩余工作拆到多个并行作业）。
    结果会与已有的 b3_summary.json 合并，不会被部分结果覆盖。"""
    t0 = time.time()
    arms = [a for a in X.ARMS if (only_arms is None or a in only_arms)]
    print(f"B3 arms={arms}", flush=True)
    best = load_json(os.path.join(paths["PRE"], "b1_best.json"))
    grid = X.sweep_for("full")
    T = dict(d=P["b3_d"], L=P["b3_L"], tokens=P["b3_tokens"],
             batch=P["b3_batch"], seq=P["b3_seq"])

    def sweep_subset(sub, tag):
        table = {}
        for arm in sub:          # iterate the subset, not all arms
            rows = []
            for val in sub[arm]:
                vals = []
                for sd in P["b3_seeds"]:
                    c = X.make_cfg(arm, val, T["d"], T["L"], T["tokens"],
                                   T["batch"], T["seq"], sd, tag, paths["PRE"])
                    r = run_pretrain(c, device=device, verbose=False)
                    vals.append(float(r.get("final_val_loss", float("nan"))))
                rows.append((val, float(np.mean(vals))))
                print(f"   {tag} {arm:16s} {val:<9g} {rows[-1][1]:.4f}", flush=True)
            table[arm] = rows
        return table

    transfer = sweep_subset({a: [best[a]] for a in arms}, "b3_transfer")
    # The "reuse the b1 optimum" cell of the retune grid is exactly the transfer
    # run; re-running it under a different tag prefix would waste 12 runs.
    reuse = {a: dict(transfer[a]) for a in arms}
    retuned = {}
    for a in arms:
        rows = [(best[a], reuse[a][best[a]])]        # already measured above
        others = [v for v in grid[a] if v != best[a]][:2]
        sub_rows = sweep_subset({a: others}, "b3_retuned")[a]
        retuned[a] = rows + sub_rows   # sweep_subset already printed these
    best3 = best_of(retuned)
    penalty = {}
    for a in arms:
        t = dict(transfer[a])[best[a]]
        r = dict(retuned[a])[best3[a]]
        penalty[a] = dict(transferred=t, retuned=r, gap=t - r)
    sp = os.path.join(paths["PRE"], "b3_summary.json")
    merged = load_json(sp) or {}
    merged.update(penalty)                      # 合并，不覆盖其它作业的结果
    merged["__complete__"] = all(a in merged for a in X.ARMS)
    save_json(sp, merged)
    penalty = merged
    for a in arms:
        p = penalty[a]
        print(f"   {a:16s} 迁移={p['transferred']:.4f} 重调={p['retuned']:.4f} gap={p['gap']:+.4f}")
    print(f"B3 {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["a", "b1", "b2", "b3", "all"])
    ap.add_argument("--mode", default="full", choices=["full", "quick"])
    ap.add_argument("--arms", default=None,
                    help="逗号分隔，只跑这些臂（b3 拆并行作业用）")
    args = ap.parse_args()

    device = _device()
    P = X.params(args.mode)
    paths = X.paths(args.mode)
    for k in ("DIAG", "PRE", "RLD"):
        os.makedirs(paths[k], exist_ok=True)

    print(f"stage={args.stage} mode={args.mode} device={device}", flush=True)
    build_token_cache(X.DATA, 400_000_000)

    t0 = time.time()
    if args.stage in ("a", "all"):
        stage_a(P, paths, device)
    if args.stage in ("b1", "all"):
        stage_b1(P, paths, device, args.mode)
    if args.stage in ("b2", "all"):
        stage_b2(P, paths, device)
    if args.stage in ("b3", "all"):
        only = [a.strip() for a in args.arms.split(",")] if args.arms else None
        stage_b3(P, paths, device, only_arms=only)
    print(f"\nTOTAL {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
