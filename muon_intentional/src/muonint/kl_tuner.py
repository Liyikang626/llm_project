"""Layer-wise Logit-KL intentional step sizes.

For a Muon direction D_l the step is Delta W_l = -eta_l D_l.  We choose eta_l
so that perturbing *this layer alone* moves the next-token distribution by a
prescribed KL:

    KL(p || p') ~ 1/2 eta^2 * Var_p[dz/deta]
    =>  eta_l = sqrt(2 eps_l / (2 s_l)) = sqrt(eps_l / s_l)

with s_l = 1/2 Var_p[v_l] measured by a finite difference on the logits.

Two allocations are supported, and they differ in one thing only — how the
per-step KL budget `eps` is spread:

    geomean / global :  one scalar eta from the *joint* sensitivity, i.e. all
                        layers are perturbed together, so the cross terms are
                        included by construction.
    layerwise        :  eps_l = eps / n_layers, each layer perturbed alone.

Comparing the two isolates "per-layer allocation" from "KL as the target",
which is the contrast the pilot needs.

Cost: one extra forward on a small probe batch per measured layer.  With a
512-token probe and 7*L layers this is ~1.3 training steps per measurement,
so recomputing every `every` steps keeps the overhead near 5%.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .optim import zeropower_via_newtonschulz5


def muon_directions(opt) -> Dict[int, torch.Tensor]:
    """Unit Muon direction for every matrix with a momentum buffer."""
    out = {}
    if not hasattr(opt, "param_groups") or not opt.param_groups:
        return out
    for p in list(opt.param_groups[0]["params"]):
        buf = opt.state.get(p, {}).get("momentum_buffer")
        if buf is None:
            continue
        D = zeropower_via_newtonschulz5(buf, steps=5)
        m, n = p.shape
        out[id(p)] = D * math.sqrt(max(1.0, m / n))
    return out


@torch.no_grad()
def _probe_stats(model, probe_x, base_logits, dirs, groups, probe_eta, orig_dtype):
    """s = 1/2 Var_p[dz/deta] for each perturbation group.

    groups: list of (name, {param_id: param}, scale_multiplier)
    All members of a group are perturbed together, so a single group gives the
    joint sensitivity (cross terms included).
    """
    lp = F.log_softmax(base_logits, dim=-1)
    p = lp.exp()
    out = {}
    for name, members, _ in groups:
        saved = {}
        for pid, p_param in members.items():
            saved[pid] = p_param.data
            p_param.data = (saved[pid] - probe_eta * dirs[pid].to(saved[pid].dtype)).contiguous()
        try:
            z = model(probe_x, targets=None, capture=False)[0].float()
        finally:
            for pid, p_param in members.items():
                p_param.data = saved[pid]
        w = (z - base_logits) / probe_eta
        m1 = (p * w).sum(-1)
        m2 = (p * w * w).sum(-1)
        out[name] = 0.5 * float((m2 - m1 * m1).mean())
    return out


@torch.no_grad()
def _joint_kl(model, dirs, params, alphas, probe_x, base, lpa, pa) -> float:
    saved = {}
    for p_param in params:
        saved[id(p_param)] = p_param.data
        p_param.data = (saved[id(p_param)]
                        - alphas[id(p_param)] * dirs[id(p_param)].to(saved[id(p_param)].dtype)).contiguous()
    try:
        z = model(probe_x, targets=None, capture=False)[0].float()
    finally:
        for p_param in params:
            p_param.data = saved[id(p_param)]
    lpb = F.log_softmax(z, dim=-1)
    return float((pa * (lpa - lpb)).sum(-1).mean())


@torch.no_grad()
def measure_step_sizes(model, opt, probe_x, eps, layerwise=True,
                       probe_eta=1e-2, max_ratio=30.0,
                       calibrate=True) -> Tuple[Dict[int, float], Dict]:
    """Return {param_id: eta} plus diagnostics for one step.

    eps is the per-step KL budget in nats (per token, averaged).
    """
    dirs = muon_directions(opt)
    if not dirs:
        return {}, {"status": "no_momentum"}
    params = [p for p in opt.param_groups[0]["params"] if id(p) in dirs]
    orig_dtype = next(model.parameters()).dtype

    base = model(probe_x, targets=None, capture=False)[0].float()

    if layerwise:
        groups = [(str(i), {id(p): p}, 1.0) for i, p in enumerate(params)]
    else:
        groups = [("all", {id(p): p for p in params}, 1.0)]

    s_map = _probe_stats(model, probe_x, base, dirs, groups, probe_eta, orig_dtype)

    diag = {"mode": "layerwise" if layerwise else "global"}
    if layerwise:
        n = len(params)
        eps_l = eps / max(1, n)
        floor = 1e-12
        raw = {id(p): math.sqrt(eps_l / max(s_map[str(i)], floor))
               for i, p in enumerate(params)}
        vals = torch.tensor(list(raw.values()))
        cap = max_ratio * float(vals.median())
        alphas = {k: min(v, cap) for k, v in raw.items()}
        diag.update(
            n=n, eps_l=eps_l,
            s_sum=float(sum(s_map.values())),
            s_max=float(max(s_map.values())),
            s_min=float(min(s_map.values())),
            eta_med=float(vals.median()),
            eta_spread=float(vals.max() / (vals.min() + 1e-30)),
            frac_capped=float(sum(1 for v in raw.values() if v >= cap) / n),
            # what the joint KL would be if per-layer terms simply added up:
            kl_additive=float(eps_l * n),
            # the joint KL actually implied by summing the per-layer variances:
            kl_sum_measured=float(sum(s_map.values()) * vals.median() ** 2),
        )
    else:
        s_all = max(s_map["all"], 1e-12)
        a = math.sqrt(eps / s_all)
        alphas = {id(p): a for p in params}
        diag.update(n=len(params), s_joint=s_all, eta=a, kl_target=eps)

    # ---- calibration -----------------------------------------------------
    # Per-layer targets do NOT add: all layers move along the same descent
    # direction, so their logit changes are positively correlated and the
    # joint KL can be several times larger than the sum.  Measure the joint
    # update and rescale so the *total* KL actually hits eps.  Shape comes
    # from the sensitivity rule; the scalar comes from measurement.
    if calibrate:
        lpa = F.log_softmax(base, dim=-1)
        pa = lpa.exp()
        kl0 = _joint_kl(model, dirs, params, alphas, probe_x, base, lpa, pa)
        if kl0 > 1e-20:
            diag["kl_before_calib"] = kl0
            # At the step sizes this rule produces (eta ~ 0.1) the measured KL
            # is strongly super-quadratic in the scale, so the analytic
            # sqrt(eps/kl) correction overshoots.  Fit KL ~ scale^p in log-log
            # and secant to KL = eps; monotone, so this converges in a few
            # evaluations regardless of the true curvature.
            base_alphas = dict(alphas)
            s_prev, k_prev = 1.0, kl0
            s_cur = math.sqrt(eps / kl0)
            kl = kl0
            p_exp = 2.0
            for _ in range(4):
                alphas = {k: v * s_cur for k, v in base_alphas.items()}
                kl = _joint_kl(model, dirs, params, alphas, probe_x, base, lpa, pa)
                if abs(math.log(max(kl, 1e-30) / eps)) < 0.05:
                    break
                if kl > 0 and s_cur != s_prev and k_prev > 0:
                    p_exp = max(1.0, min(8.0,
                        (math.log(kl) - math.log(k_prev)) /
                        (math.log(s_cur) - math.log(s_prev))))
                s_next = math.exp(math.log(s_cur) +
                                  (math.log(eps) - math.log(max(kl, 1e-30))) / p_exp)
                s_prev, k_prev = s_cur, kl
                s_cur = s_next
            alphas = {k: v * s_cur for k, v in base_alphas.items()}
            kl = _joint_kl(model, dirs, params, alphas, probe_x, base, lpa, pa)
            diag.update(calib_scale=s_cur, calib_exponent=p_exp,
                        kl_after_calib=kl, kl_achieved_ratio=kl / eps)

    return alphas, diag


@torch.no_grad()
def check_joint_kl(model, opt, probe_x, alphas, probe_eta=None) -> Dict[str, float]:
    """Apply the chosen per-layer alphas *together* and measure the real KL.

    This is the number that says whether "sum of per-layer targets" is the same
    as "KL of the combined update".
    """
    dirs = muon_directions(opt)
    params = [p for p in opt.param_groups[0]["params"] if id(p) in dirs]
    if not params:
        return {}
    base = model(probe_x, targets=None, capture=False)[0].float()
    lpa = F.log_softmax(base, dim=-1)
    pa = lpa.exp()

    saved = {}
    for p in params:
        saved[id(p)] = p.data
        p.data = (saved[id(p)] - alphas[id(p)] * dirs[id(p)].to(saved[id(p)].dtype)).contiguous()
    try:
        z = model(probe_x, targets=None, capture=False)[0].float()
    finally:
        for p in params:
            p.data = saved[id(p)]

    lpb = F.log_softmax(z, dim=-1)
    kl_joint = float((pa * (lpa - lpb)).sum(-1).mean())

    # first-order prediction summed over layers
    kl_sum = 0.0
    for p in params:
        D = dirs[id(p)]
        pid = id(p)
        sv = saved[pid]
        p.data = (sv - alphas[pid] * D.to(sv.dtype)).contiguous()
        try:
            zi = model(probe_x, targets=None, capture=False)[0].float()
        finally:
            p.data = sv
        lpi = F.log_softmax(zi, dim=-1)
        kl_sum += float((pa * (lpa - lpi)).sum(-1).mean())

    return dict(kl_joint=kl_joint, kl_sum=kl_sum,
                cross_term_ratio=kl_joint / (kl_sum + 1e-30))
