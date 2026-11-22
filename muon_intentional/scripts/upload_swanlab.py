#!/usr/bin/env python
"""把已完成的实验结果上传到 SwanLab。

只上传：Stage A（逐层异质性诊断）、B1（扫参）、B2（主实验）。
B3 尚未跑完，不上传。

每次训练配置 = 一个 SwanLab run；run 名形如  b2/intent_kl_global/seed0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("http_proxy", "http://10.53.146.96:18889")
os.environ.setdefault("https_proxy", "http://10.53.146.96:18889")

ROOT = "/home/15Cu2Plus/muon_intentional"
sys.path.insert(0, os.path.join(ROOT, "src"))
PROJECT = "llm_from_scratch"
WORKSPACE = "liyikang24"
LOGDIR = os.path.join(ROOT, "swanlab_logs")

import swanlab  # noqa: E402
from muonint import expconfig as X  # noqa: E402


def _load(path):
    with open(path) as f:
        return json.load(f)


def _run(name, group, cfg, scalars, curve=None, tags=None):
    swanlab.init(project=PROJECT, workspace=WORKSPACE, name=name, group=group,
                 config=cfg, mode="cloud", log_dir=LOGDIR, tags=tags or [])
    try:
        if curve:
            for step, val in curve:
                swanlab.log({"val_loss": val}, step=step)
        swanlab.log(scalars)
    finally:
        swanlab.finish()


def upload_stage_a(dry=False, limit=0):
    p = os.path.join(X.paths("full")["DIAG"], "heterogeneity.json")
    if not os.path.exists(p):
        print("  stageA: 缺文件，跳过"); return 0
    d = _load(p)
    n = 0
    for k, v in list(d["results"].items())[: limit or None]:
        cfg = dict(batch_size=k, steps=v.get("n", None),
                   across_layer_cv=v["across_layer_cv"],
                   temporal_cv=v["temporal_cv_mean"],
                   ratio_median=v["ratio_median"])
        sc = {"across_layer_cv": v["across_layer_cv"],
              "across_layer_p90": v["across_layer_spread"]["p90"],
              "temporal_cv": v["temporal_cv_mean"],
              "ratio_median": v["ratio_median"]}
        print(f"  stageA/{k}")
        if not dry:
            _run(f"stageA/{k}", "stageA", cfg, sc, tags=["diag"])
        n += 1
    return n


def upload_b1(dry=False, limit=0):
    pre = X.paths("full")["PRE"]
    files = sorted(glob.glob(os.path.join(pre, "*b1_*.json")))
    files = [f for f in files if "b1_sweep" not in f and "b1_best" not in f]
    files = files[: limit or None]
    n = 0
    for f in files:
        d = _load(f)
        c = d["cfg"]
        st = d.get("status")
        # 每个 arm 实际扫的是哪个超参，由 SWEEP_PARAM 决定（RunConfig 对所有
        # arm 都带一个 kl_eps 默认值，所以不能靠"有没有 kl_eps"判断）
        prm = X.SWEEP_PARAM[c["opt_mode"]]
        swept = c["kl_eps"] if prm == "kl_eps" else c["lr"]
        name = f"b1/{c['opt_mode']}/{prm}={swept:g}"
        cfg = dict(arm=c["opt_mode"], lr=c.get("lr"), kl_eps=c.get("kl_eps"),
                   d_model=c["d_model"], n_layer=c["n_layer"], batch_size=c["batch_size"],
                   seq_len=c["seq_len"], tokens=c["tokens"], seed=c["seed"], status=st)
        sc = {}
        if st == "done":
            sc["final_val_loss"] = d["final_val_loss"]
            sc["wall_min"] = d["wall_sec"] / 60
            sc["tokens_per_sec"] = d["tokens_per_sec"]
        else:
            sc["diverged_at_step"] = d.get("diverged_at_step", -1)
        print(f"  {name} -> {sc}")
        if not dry:
            _run(name, f"b1/{c['opt_mode']}", cfg, sc, tags=["b1", st])
        n += 1
    return n


def upload_b2(dry=False, limit=0):
    pre = X.paths("full")["PRE"]
    files = sorted(glob.glob(os.path.join(pre, "*b2_*.json")))
    files = [f for f in files if "b2_summary" not in f]
    files = files[: limit or None]
    n = 0
    for f in files:
        d = _load(f)
        c = d["cfg"]
        st = d.get("status")
        name = f"b2/{c['opt_mode']}/seed{c['seed']}"
        cfg = dict(arm=c["opt_mode"], lr=c.get("lr"), kl_eps=c.get("kl_eps"),
                   d_model=c["d_model"], n_layer=c["n_layer"], batch_size=c["batch_size"],
                   seq_len=c["seq_len"], tokens=c["tokens"], seed=c["seed"], status=st)
        curve = [(h["tokens"], h["val_loss"]) for h in d.get("history", [])
                 if "val_loss" in h and "tokens" in h]
        sc = {}
        if st == "done":
            sc["final_val_loss"] = d["final_val_loss"]
            sc["wall_min"] = d["wall_sec"] / 60
            sc["tokens_per_sec"] = d["tokens_per_sec"]
        else:
            sc["diverged_at_step"] = d.get("diverged_at_step", -1)
        print(f"  {name} curve={len(curve)}pts -> final={sc.get('final_val_loss', 'DIVERGED')}")
        if not dry:
            _run(name, f"b2/{c['opt_mode']}", cfg, sc, curve=curve, tags=["b2", st])
        n += 1
    return n


def _cap(fn, dry, limit):
    return fn(dry, limit)


def main():
    global PROJECT, WORKSPACE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只打印，不上传")
    ap.add_argument("--only", choices=["a", "b1", "b2"], default=None)
    ap.add_argument("--limit", type=int, default=0, help="每个阶段最多上传几个（0=全部）")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--workspace", default=WORKSPACE)
    args = ap.parse_args()
    PROJECT, WORKSPACE = args.project, args.workspace
    print(f"目标: workspace={WORKSPACE} project={PROJECT}")
    os.makedirs(LOGDIR, exist_ok=True)
    total = 0
    lim = args.limit
    if args.only in (None, "a"):
        print("== Stage A =="); total += _cap(upload_stage_a, args.dry, lim)
    if args.only in (None, "b1"):
        print("== B1 扫参 =="); total += _cap(upload_b1, args.dry, lim)
    if args.only in (None, "b2"):
        print("== B2 主实验 =="); total += _cap(upload_b2, args.dry, lim)
    print(f"\n共 {total} 个 run{'（dry-run，未上传）' if args.dry else ''}")


if __name__ == "__main__":
    main()
