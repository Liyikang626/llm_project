"""Per-layer sensitivity to a unit Muon update, in two different spaces.

For every Muon-optimised matrix W_l we apply the *unit* direction D_l (the
orthogonalised momentum times Muon's shape factor) and measure two things:

  activation sensitivity
      a_l = ||dH_l||_F / ||H_l||_F          with dW_l = -D_l   (eta = 1)
      exact and linear in eta, because H = W X is linear in W.

  function sensitivity
      s_l = 1/2 Var_p[v_l]                  v_l = dz/deta at eta = 0
      z are the next-token logits; the variance is over the vocab under the
      current predictive distribution p.  This is exactly the denominator a
      layer-wise logit-KL rule would invert:

          KL(p || p') ~ 1/2 eta^2 Var_p[v_l]  =>  eta_l = sqrt(2 eps / (2 s_l))

The two rules equalise different quantities:

      activation rule :  alpha_l ~ 1 / a_l
      function   rule :  eta_l   ~ 1 / sqrt(s_l)

so the two views disagree exactly when  log s_l - 2 log a_l  varies across
layers.  `compare_views()` reports that spread, which is the number that
decides whether activation-space normalisation can stand in for a
function-space step size.

Numeric note: at a near-uniform p the induced KL is ~1e-13, far below fp32
resolution, so the probe runs the model in float64 and estimates v_l by a
finite difference on the logits (never by subtracting two near-equal KLs).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .model import MuonLinear
from .optim import zeropower_via_newtonschulz5

MATRIX_ORDER = ["wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"]


def fmt_tokens(t: float) -> str:
    return f"{t/1e6:.1f}M" if t >= 1e6 else f"{t/1e3:.0f}k"
SHORT = {"wq": "q", "wk": "k", "wv": "v", "wo": "o",
         "w_gate": "gate", "w_up": "up", "w_down": "down"}


# --------------------------------------------------------------------------
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank) — numpy only, no scipy."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------
def muon_directions(model, opt) -> Dict[int, torch.Tensor]:
    """Unit Muon direction for every matrix that has a momentum buffer."""
    out = {}
    if not hasattr(opt, "param_groups") or not opt.param_groups:
        return out
    for p in list(opt.param_groups[0]["params"]):
        buf = opt.state.get(p, {}).get("momentum_buffer")
        if buf is None:
            continue
        with torch.no_grad():
            D = zeropower_via_newtonschulz5(buf, steps=5)
            m, n = p.shape
            D = D * math.sqrt(max(1.0, m / n))
        out[id(p)] = D
    return out


def ordered_params(model) -> List[torch.nn.Parameter]:
    by_block = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[2] in MATRIX_ORDER:
            by_block.setdefault(int(parts[1]), {})[parts[2]] = p
    out = []
    for b in sorted(by_block):
        for mat in MATRIX_ORDER:
            if mat in by_block[b]:
                out.append(by_block[b][mat])
    return out


def layer_labels(model) -> Dict[int, str]:
    labels = {}
    for name, p in model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[2] in SHORT:
            labels[id(p)] = f"b{parts[1]}.{SHORT[parts[2]]}"
    return labels


class _Grab:
    """Capture each Muon layer's input and output for the probe batch."""

    def __init__(self, model, wanted):
        self.rec: Dict[int, tuple] = {}
        self._h = []
        for mod in model.modules():
            if isinstance(mod, MuonLinear) and id(mod.weight) in wanted:
                self._h.append(mod.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        self.rec[id(mod.weight)] = (inp[0].detach(), out.detach())

    def close(self):
        for h in self._h:
            h.remove()
        self._h = []


# --------------------------------------------------------------------------
@torch.no_grad()
def measure_sensitivities(model, opt, x, etas=(1e-2, 3e-3)) -> Dict:
    """a_l and s_l for every Muon matrix, on the fixed probe batch x."""
    dirs = muon_directions(model, opt)
    params = [p for p in ordered_params(model) if id(p) in dirs]
    labels = layer_labels(model)
    orig_dtype = next(model.parameters()).dtype

    # ---- activation sensitivity -------------------------------------------
    grab = _Grab(model, {id(p) for p in params})
    try:
        model(x, targets=None, capture=False)
        act_io = {k: (v[0].float(), v[1].float()) for k, v in grab.rec.items()}
    finally:
        grab.close()

    act = {}
    for p in params:
        if id(p) not in act_io:
            continue
        X, H = act_io[id(p)]
        Xf = X.reshape(-1, X.shape[-1])
        D = dirs[id(p)].to(Xf.dtype)
        dH = (Xf @ D.t()).norm().item()
        act[id(p)] = dH / (H.norm().item() + 1e-12)

    # ---- function sensitivity ---------------------------------------------
    model.to(torch.float64)
    try:
        base = model(x, targets=None, capture=False)[0]        # (B,T,V)
        lp = F.log_softmax(base, dim=-1)
        p = lp.exp()
        var_eps = 1e-30

        func = {e: {} for e in etas}
        kl_true = {e: {} for e in etas}
        for p_param in params:
            D = dirs[id(p_param)]
            old = p_param.data
            for e in etas:
                p_param.data = (old - e * D.to(old.dtype)).contiguous()
                try:
                    z = model(x, targets=None, capture=False)[0]
                finally:
                    p_param.data = old
                w = (z - base) / e                            # v_l, first order
                # Var_p[w] = E[w^2] - (E[w])^2 , per token then averaged
                m1 = (p * w).sum(-1)
                m2 = (p * w * w).sum(-1)
                func[e][id(p_param)] = 0.5 * float((m2 - m1 * m1).mean())
                lp2 = F.log_softmax(z, dim=-1)
                kl_true[e][id(p_param)] = float((p * (lp - lp2)).sum(-1).mean())
    finally:
        model.to(orig_dtype)

    return {
        "labels": [labels[id(p)] for p in params],
        "activation": np.array([act.get(id(p), np.nan) for p in params]),
        "function": {e: np.array([func[e].get(id(p), np.nan) for p in params]) for e in etas},
        "kl_true": {e: np.array([kl_true[e].get(id(p), np.nan) for p in params]) for e in etas},
        "etas": list(etas),
    }


# --------------------------------------------------------------------------
def compare_views(act: np.ndarray, func: np.ndarray) -> Dict[str, float]:
    """How much do the activation view and the function view disagree?

    The activation rule gives alpha_l ~ 1/a_l and the function rule gives
    eta_l ~ 1/sqrt(s_l), so agreement means log(s_l) - 2 log(a_l) is constant.
    `cv_step_ratio` is the spread of that log-ratio: it is the decisive number.
    """
    m = np.isfinite(act) & np.isfinite(func) & (act > 0) & (func > 0)
    a, s = act[m], func[m]
    la, ls = np.log(a), np.log(s)
    gap = ls - 2 * la
    return dict(
        n=int(m.sum()),
        spearman=spearman(a, s),
        pearson_log=float(np.corrcoef(la, ls)[0, 1]),
        cv_activation=float(np.std(la)),
        cv_function=float(np.std(ls)),
        cv_step_ratio=float(np.std(gap)),
        step_ratio_p90_over_p10=float(np.exp(np.percentile(gap, 90) - np.percentile(gap, 10))),
    )


def summarize(res: Dict) -> Dict:
    """Use the smaller eta (closest to the linear regime); validate it."""
    etas = sorted(res["etas"])
    small, big = etas[0], etas[-1]
    f_small, f_big = res["function"][small], res["function"][big]
    good = np.isfinite(f_small) & np.isfinite(f_big) & (f_small > 0)
    linearity = float(np.median(f_big[good] / f_small[good])) if good.sum() else float("nan")

    # does the quadratic KL model match the measured KL at the small eta?
    kl_meas = res["kl_true"][small]
    kl_pred = small ** 2 * f_small
    ok = np.isfinite(kl_meas) & np.isfinite(kl_pred) & (kl_pred > 0)
    kl_ratio = float(np.median(kl_meas[ok] / kl_pred[ok])) if ok.sum() else float("nan")

    stats = compare_views(res["activation"], f_small)
    stats["linearity_ratio"] = linearity     # ~1 => still in the linear regime
    stats["kl_pred_over_meas"] = kl_ratio    # ~1 => quadratic KL model valid
    return stats


# --------------------------------------------------------------------------
def train_with_probes(cfg, ckpt_fracs=(0.02, 0.5, 1.0), probe_seqs=2,
                      etas=(1e-2, 3e-3), device="cuda", verbose=True):
    """Train with Muon and measure both sensitivities at a few checkpoints.

    Probes happen *after* an optimizer step, so the Muon momentum buffer (i.e.
    the direction the optimizer is actually following) already exists.
    """
    from .common import autocast
    from .data import TokenBatcher
    from .optim import build_optimizer
    from .pretrain import build_model, lr_at

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, device)
    batcher = TokenBatcher(cfg.data_path, cfg.batch_size, cfg.seq_len, seed=cfg.seed)
    opt = build_optimizer(model, cfg)
    probe_x = batcher.val_batches(1, batch_size=probe_seqs)[0][0].to(device)

    total = cfg.steps
    ckpt_steps = sorted({max(1, int(round(f * total))) for f in ckpt_fracs})

    results, loss = {}, torch.tensor(float("nan"))
    model.train()
    for step in range(total):
        cur = lr_at(step, total, cfg.lr, cfg.warmup_frac, cfg.lr_schedule)
        opt.param_groups[0]["lr"] = cur
        x, y = batcher.train_batch(device)
        opt.zero_grad(set_to_none=True)
        with autocast(device):
            _, loss = model(x, y)
        loss.backward()
        opt.step()

        if (step + 1) in ckpt_steps:
            if verbose:
                print(f"  probe @ {fmt_tokens((step+1)*cfg.batch_size*cfg.seq_len)} tokens, "
                      f"loss {loss.item():.3f}")
            results[step + 1] = measure_sensitivities(model, opt, probe_x, etas=etas)
            model.train()

    results["final_loss"] = float(loss.item())
    results["tokens"] = {k: k * cfg.batch_size * cfg.seq_len
                         for k in results if isinstance(k, int)}
    return model, results
