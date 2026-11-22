"""Small-scale LM pretraining loop for the optimizer comparison."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import TokenBatcher
from .common import autocast
from .model import GPT, GPTConfig
from .optim import MuonIntentional, PlainAdamW, build_optimizer, split_muon_params


def kl_diag_at_nominal(diag: dict, sched: float) -> dict:
    """Rescale the measured per-step eta back to the nominal (sched = 1) budget."""
    if not diag:
        return {}
    out = dict(diag)
    if sched and sched > 0:
        import math as _m
        k = 1.0 / _m.sqrt(sched)
        for key in ("eta_med", "eta"):
            if key in out and isinstance(out[key], (int, float)):
                out[key] = out[key] * k
        out["sched_seen"] = sched
    return out


@dataclass
class RunConfig:
    name: str = "muon"
    opt_mode: str = "fixed"        # fixed | orscale | intent_l | intent_t | adamw
    lr: float = 0.02               # eta for intentional modes, lr otherwise
    adam_lr: float = 3e-4
    momentum: float = 0.95
    nesterov: bool = True
    weight_decay: float = 0.0
    alpha_clip_ratio: float = 20.0
    # model
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    seq_len: int = 512
    # optimisation budget
    batch_size: int = 32
    tokens: int = 150_000_000
    warmup_frac: float = 0.02
    lr_schedule: str = "cosine"    # cosine | const
    seed: int = 0
    # io
    data_path: str = "/home/15Cu2Plus/muon_intentional/data/climbmix_400m.u16"
    out_dir: str = "/home/15Cu2Plus/muon_intentional/results/pretrain"
    log_every: int = 50
    eval_every: int = 500
    n_sub: int = 512               # tokens used to estimate ||X O^T||_F
    kl_every: int = 20             # recompute KL-rule step sizes every N steps
    kl_probe_seqs: int = 2         # probe batch = kl_probe_seqs * seq_len tokens
    kl_eps: float = 1e-3           # per-step KL budget (nats/token), total
    kl_check_every: int = 0        # 0 = off; else measure the joint-update KL

    @property
    def steps(self) -> int:
        return max(1, self.tokens // (self.batch_size * self.seq_len))

    @property
    def tag(self) -> str:
        # every field that changes the run's behaviour must appear here, or two
        # different configurations silently share one cache file.  kl_eps was
        # missing, which made the whole epsilon sweep return the first result.
        keys = ("name", "opt_mode", "lr", "kl_eps", "n_layer", "n_head", "d_model",
                "batch_size", "tokens", "seed", "seq_len", "adam_lr", "weight_decay",
                "n_sub", "kl_every")
        return "_".join(f"{k}={getattr(self, k)}" for k in keys)


def lr_at(step: int, total: int, base: float, warmup_frac: float, schedule: str) -> float:
    w = max(1, int(total * warmup_frac))
    if step < w:
        return base * (step + 1) / w
    if schedule == "const":
        return base
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1.0 + math.cos(math.pi * p))


def build_model(cfg: RunConfig, device) -> GPT:
    mc = GPTConfig(vocab_size=50257, n_layer=cfg.n_layer, n_head=cfg.n_head,
                   d_model=cfg.d_model, seq_len=cfg.seq_len)
    m = GPT(mc).to(device)
    m._n_sub = cfg.n_sub
    return m


def _summ(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    t = torch.tensor(vals)
    return dict(mean=float(t.mean()), std=float(t.std() if len(vals) > 1 else 0.0),
                min=float(t.min()), max=float(t.max()),
                p10=float(t.quantile(0.1)), p90=float(t.quantile(0.9)))


@torch.no_grad()
def evaluate(model, batcher, n_batches=8, device="cuda") -> Dict[str, float]:
    model.eval()
    losses = []
    for x, y in batcher.val_batches(n_batches):
        x, y = x.to(device), y.to(device)
        with autocast(device):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return {"val_loss": sum(losses) / len(losses)}


def run(cfg: RunConfig, device="cuda", verbose=True) -> Dict:
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.tag + ".json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "done" and math.isfinite(prev.get("final_val_loss", float("nan"))):
            if verbose:
                print(f"[skip] {cfg.tag}")
            return prev

    model = build_model(cfg, device)
    nparam = model.num_params()
    batcher = TokenBatcher(cfg.data_path, cfg.batch_size, cfg.seq_len, seed=cfg.seed)

    is_kl = cfg.opt_mode in ("intent_kl_layer", "intent_kl_global")
    if cfg.opt_mode == "adamw":
        opt = PlainAdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = build_optimizer(model, cfg, force_mode="ext" if is_kl else None)

    probe_x = None
    if is_kl:
        from .kl_tuner import measure_step_sizes, check_joint_kl
        probe_x = batcher.val_batches(1, batch_size=cfg.kl_probe_seqs,
                                      seq_len=cfg.seq_len)[0][0].to(device)
        kl_layerwise = cfg.opt_mode == "intent_kl_layer"
    # the AdamW-side lr is held fixed across Muon arms; make it explicit
    if isinstance(opt, MuonIntentional):
        opt.param_groups[1]["adam_lr"] = cfg.adam_lr
        opt.param_groups[1]["weight_decay"] = cfg.weight_decay

    total = cfg.steps
    hist = []
    t0 = time.time()
    model.train()
    # "intent_act" is the experiment-facing name of the internal "intent_l" rule
    capture = cfg.opt_mode in ("intent_l", "intent_act")

    kl_diag = {}
    kl_diag_nominal = {}
    kl_best_sched = -1.0
    for step in range(total):
        cur_lr = lr_at(step, total, cfg.lr, cfg.warmup_frac, cfg.lr_schedule)
        if isinstance(opt, MuonIntentional):
            opt.param_groups[0]["lr"] = cur_lr
        else:
            opt.param_groups[0]["lr"] = cur_lr

        # the Muon direction is the momentum buffer, which only exists once a
        # step has been taken -- so step 0 cannot measure.  Measuring at step 1
        # gives the rule its first real values.
        if is_kl and (step % cfg.kl_every == 0 or step == 1):
            # eps follows the same schedule as a tuned lr would, so the arm is
            # comparable to Muon at the same point in training.
            sched = cur_lr / max(cfg.lr, 1e-12)
            alphas, kdiag = measure_step_sizes(
                model, opt, probe_x, eps=cfg.kl_eps * sched,
                layerwise=kl_layerwise, probe_eta=1e-2)
            opt.set_external_alphas(alphas)
            kl_diag = kdiag
            # Report the step size at nominal budget (sched = 1).  The last
            # measurement happens after the cosine schedule has decayed eps to
            # ~0, so its eta is meaningless -- keep the one at peak schedule.
            # step 0 has no momentum buffer yet, so it yields {"status":
            # "no_momentum"} -- and its sched is exactly 1.0, which would
            # otherwise shadow every later (valid) measurement.
            if (kdiag.get("eta_med") or kdiag.get("eta")) and sched > kl_best_sched:
                kl_best_sched = sched
                kl_diag_nominal = dict(kdiag)

        x, y = batcher.train_batch(device)
        opt.zero_grad(set_to_none=True)
        with autocast(device):
            _, loss = model(x, y, capture=capture)
        if not torch.isfinite(loss):
            # Diverged. Stop immediately instead of burning the token budget,
            # and record it as "diverged" so a re-run is not skipped.
            res = dict(status="diverged", diverged_at_step=step,
                       cfg=asdict(cfg), history=hist,
                       last_kl_diag=kl_diag_at_nominal(kl_diag_nominal, kl_best_sched))
            with open(out_path, "w") as f:
                json.dump(res, f, indent=1)
            if verbose:
                print(f"[DIVERGED] {cfg.tag} at step {step} "
                      f"({step * cfg.batch_size * cfg.seq_len / 1e6:.2f}M tokens)")
            return res
        loss.backward()

        gnorm = float(torch.sqrt(sum((p.grad.float().pow(2).sum() for p in model.parameters()
                                      if p.grad is not None), torch.zeros((), device=device))))
        if not math.isfinite(gnorm):
            res = dict(status="diverged", diverged_at_step=step, reason="nonfinite_grad",
                       cfg=asdict(cfg), history=hist,
                       last_kl_diag=kl_diag_at_nominal(kl_diag_nominal, kl_best_sched))
            with open(out_path, "w") as f:
                json.dump(res, f, indent=1)
            if verbose:
                print(f"[DIVERGED] {cfg.tag} at step {step} (non-finite gradient)")
            return res
        if capture:
            opt.set_layer_inputs(model.layer_inputs)
        opt.step()

        if step % cfg.log_every == 0:
            rec = dict(step=step, loss=float(loss.item()), lr=cur_lr, grad_norm=gnorm,
                       tokens=step * cfg.batch_size * cfg.seq_len,
                       elapsed=time.time() - t0)
            if getattr(opt, "stats", None) and opt.stats.rel_delta_h:
                by_layer = {}
                for k, v in opt.stats.rel_delta_h.items():
                    by_layer.setdefault(k.split(".")[0], []).append(v)
                rec["rel_delta_h"] = {k: float(torch.tensor(v).mean()) for k, v in by_layer.items()}
                rec["rel_delta_h_spread"] = float(torch.tensor(list(opt.stats.rel_delta_h.values())).max()
                                                  / (torch.tensor(list(opt.stats.rel_delta_h.values())).min() + 1e-12))
            if getattr(opt, "stats", None) and opt.stats.alpha:
                a = torch.tensor(list(opt.stats.alpha.values()))
                rec["alpha_med"] = float(a.median())
                rec["alpha_spread"] = float(a.max() / (a.min() + 1e-12))
            if is_kl and kl_diag:
                rec["kl_diag"] = {k: v for k, v in kl_diag.items() if k != "status"}
                if cfg.kl_check_every and step % cfg.kl_check_every == 0 and step > 0:
                    rec["kl_check"] = check_joint_kl(model, opt, probe_x, alphas)
            hist.append(rec)
            if verbose and step % (cfg.log_every * 5) == 0:
                print(f"  {cfg.tag} step {step}/{total} loss {rec['loss']:.4f} "
                      f"lr {cur_lr:.4g} gn {gnorm:.2f}")

        if (step + 1) % cfg.eval_every == 0 or step == total - 1:
            ev = evaluate(model, batcher, device=device)
            ev["step"] = step + 1
            ev["tokens"] = (step + 1) * cfg.batch_size * cfg.seq_len
            hist.append(ev)
            if verbose:
                print(f"  [eval] step {step+1} val_loss {ev['val_loss']:.4f}")

    wall = time.time() - t0
    result = dict(status="done", cfg=asdict(cfg), num_params=nparam,
                  last_kl_diag=kl_diag_at_nominal(kl_diag_nominal, kl_best_sched),
                  final_val_loss=hist[-1]["val_loss"], wall_sec=wall,
                  tokens_per_sec=(total * cfg.batch_size * cfg.seq_len) / wall,
                  history=hist)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    if verbose:
        print(f"[done] {cfg.tag} val_loss={result['final_val_loss']:.4f} "
              f"{wall:.0f}s ({result['tokens_per_sec']/1e3:.0f}k tok/s)")
    return result
