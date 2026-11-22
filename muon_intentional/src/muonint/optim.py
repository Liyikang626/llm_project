"""Optimizers: Muon, Muon+OrScale-style trust ratio, and Muon+Intentional.

All Muon-family optimizers share
    * matrix-valued (2D, fan_in >= 2) params -> orthogonalised momentum direction
    * everything else (embedding, head, norm gains) -> AdamW
so that the *only* thing that differs between arms is how the step magnitude
for the matrix params is chosen. That is the variable under study.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Newton-Schulz orthogonalisation (Muon's core primitive)
# --------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate UV^T (the polar factor) of a matrix / batch of matrices.

    Uses the quintic Newton-Schulz iteration with the coefficients from the
    Muon reference implementation. Returns a matrix whose nonzero singular
    values are all ~1, i.e. ||out||_op ~ 1.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


# --------------------------------------------------------------------------
# Param bookkeeping
# --------------------------------------------------------------------------
EMBED_HINTS = ("embed", "lm_head", "tok_embeddings", "shared")


# display / experiment names -> internal rule implemented below
MODE_ALIAS = {
    "muon": "fixed",
    "intent_act": "intent_l",
    "intent_kl_layer": "ext",
    "intent_kl_global": "ext",
}


def split_muon_params(params, embed_hints=EMBED_HINTS) -> tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Muon takes hidden-layer matrices; AdamW takes embeddings and 1-D params.

    Accepts either an iterable of tensors or of (name, tensor) pairs. Embedding /
    head weights must never be orthogonalised: their rows are token codes, not a
    linear map between two feature spaces.
    """
    muon, adam = [], []
    for item in params:
        if isinstance(item, tuple):
            name, p = item
        else:
            name, p = "", item
        if not p.requires_grad:
            continue
        is_emb = any(h in name.lower() for h in embed_hints)
        if p.ndim >= 2 and min(p.shape) >= 2 and not is_emb:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


# --------------------------------------------------------------------------
# The optimizer
# --------------------------------------------------------------------------
@dataclass
class IntentionalStats:
    """Per-step diagnostics, filled in by the Muon-in-the-loop optimizer."""

    alpha: Dict[str, float] = field(default_factory=dict)          # raw step size per layer
    rel_delta_h: Dict[str, float] = field(default_factory=dict)    # ||dH||/||H|| actually applied
    impl_alpha: Dict[str, float] = field(default_factory=dict)     # alpha a fixed-lr Muon would imply
    grad_muon_norm: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class MuonIntentional(torch.optim.Optimizer):
    """Muon direction with a swappable step-size rule.

    mode
    ----
    "fixed"      : classic Muon, W <- W - lr * shape_scale * O.
    "orscale"    : per-layer LARS/LAMB-style trust ratio in *parameter* space,
                   r_l = ||W_l||_F / ||D_l||_F, clipped and anchored at 1.
                   This is the strongest parameter-space competitor to the
                   intentional rule (cf. OrScale, 2605.07815).
    "intent_l"   : per-layer step chosen so that the layer's *output* changes by
                   a prescribed relative amount:   ||dH_l||_F = eta * ||H_l||_F.
                   Exact, because for h = W x the map W -> h is linear.
                   Needs `layer_inputs` (captured by the model) to be set.
    "intent_t"   : one global step chosen so the *predicted* change in the batch
                   mean token log-probability equals eta nats:
                   alpha = eta / |sum_l <g_l, O_l>|.

    Only Muon-eligible (2D) params are steered by the mode; the AdamW group has
    its own lr, held fixed across arms so that the comparison is about the
    matrix layers only (this mirrors the "matched update RMS" convention of the
    Moonlight/Muon recipes).
    """

    def __init__(
        self,
        params_muon: List[nn.Parameter],
        params_adam: List[nn.Parameter],
        mode: str = "intent_l",
        lr: float = 0.02,              # step size / eta, depending on mode
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adam_lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        orscale_clip: tuple[float, float] = (0.01, 100.0),
        alpha_clip_ratio: float = 20.0,
        eps: float = 1e-12,
    ):
        assert mode in ("fixed", "orscale", "intent_l", "intent_t", "ext")
        self.mode = mode
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.orstyle_clip = orscale_clip
        self.alpha_clip_ratio = alpha_clip_ratio
        self.eps = eps

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        adam_lr=adam_lr, betas=adam_betas, adam_eps=adam_eps, step=0)
        groups = [dict(params=params_muon, is_muon=True),
                  dict(params=params_adam, is_muon=False)]
        super().__init__(groups, defaults)

        # running state for the "anchored at 1" variants
        self._ratio_ema: Dict[int, float] = {}
        self._alpha_ema: Dict[int, float] = {}
        self._layer_inputs: Dict[int, tuple] = {}
        self._ext_alpha: Dict[int, float] = {}      # param_id -> step size, set from outside
        self._names: Dict[int, str] = {}
        self.stats = IntentionalStats()

    # -- external wiring ---------------------------------------------------
    def set_layer_inputs(self, layer_inputs: Optional[Dict[int, tuple]]):
        """`layer_inputs[param_id] = (x_sub, n_rows, h_norm)` captured in forward."""
        self._layer_inputs = layer_inputs or {}

    def _name(self, p: torch.nn.Parameter) -> str:
        return self._names.get(id(p), f"p{id(p) % 10000}") if hasattr(self, "_names") else "?"

    def register_names(self, mapping: Dict[int, str]):
        self._names = mapping

    def set_external_alphas(self, alphas: Dict[int, float]):
        """Step sizes decided outside the optimizer (e.g. by a KL budget rule)."""
        self._ext_alpha = dict(alphas)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_group, adam_group = self.param_groups[0], self.param_groups[1]
        for g in self.param_groups:
            g["step"] += 1
        step = muon_group["step"]
        self.stats = IntentionalStats()

        # ---------------- 1. Muon directions -----------------------------
        dirs: List[torch.Tensor] = []
        grads: List[torch.Tensor] = []
        for p in muon_group["params"]:
            if p.grad is None:
                dirs.append(None)
                grads.append(None)
                continue
            gr = p.grad
            grads.append(gr.detach())
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.mul_(muon_group["momentum"]).add_(gr)
            use = gr.add(buf, alpha=muon_group["momentum"]) if self.nesterov else buf
            O = zeropower_via_newtonschulz5(use, steps=self.ns_steps)
            m, n = p.shape
            O = O * math.sqrt(max(1.0, m / n))   # per-shape scale, as in Muon
            dirs.append(O)

        if muon_group["weight_decay"] > 0:
            for p in muon_group["params"]:
                if p.grad is not None:
                    p.mul_(1.0 - muon_group["lr"] * muon_group["weight_decay"])

        # ---------------- 2. step magnitudes -----------------------------
        if self.mode == "fixed":
            alphas = [muon_group["lr"]] * len(dirs)

        elif self.mode == "orscale":
            raw = []
            for p, D in zip(muon_group["params"], dirs):
                if D is None:
                    raw.append(None)
                    continue
                num = p.norm()
                den = D.norm() + self.eps
                raw.append((num / den).item())
            alphas = self._anchored(raw, muon_group["lr"])

        elif self.mode == "intent_l":
            raw = []
            for p, D in zip(muon_group["params"], dirs):
                if D is None:
                    raw.append(None)
                    continue
                info = self._layer_inputs.get(id(p))
                if info is None:                      # no capture -> skip this step
                    raw.append(None)
                    continue
                x_sub, n_rows, h_norm = info
                # ||dH||_F = alpha * ||X O^T||_F ; estimate the latter on x_sub
                xo_norm = self._xo_norm(x_sub, D, n_rows)
                raw.append(muon_group["lr"] * h_norm / (xo_norm + self.eps))
            alphas = self._clamp_alpha(raw)

        elif self.mode == "ext":
            alphas = [self._ext_alpha.get(id(p)) for p in muon_group["params"]]

        elif self.mode == "intent_t":
            dot = 0.0
            for D, gr in zip(dirs, grads):
                if D is None:
                    continue
                dot += torch.sum(gr * D).item()
            a = muon_group["lr"] / (abs(dot) + self.eps)
            alphas = [a] * len(dirs)

        else:  # pragma: no cover
            raise ValueError(self.mode)

        # ---------------- 3. apply ---------------------------------------
        pred_gain = 0.0
        for i, (p, D, a) in enumerate(zip(muon_group["params"], dirs, alphas)):
            if D is None or a is None:
                continue
            if grads[i] is not None:
                # -<g, dW> with dW = -a*D  ->  +a<g, D> : predicted change of the
                # scalar whose gradient is g (in the loss's own units).
                pred_gain += a * float(torch.sum(grads[i] * D))
            p.add_(D, alpha=-a)
            nm = self._name(p)
            self.stats.alpha[nm] = float(a)
            info = self._layer_inputs.get(id(p))
            if info is not None:
                x_sub, n_rows, h_norm = info
                xo = self._xo_norm(x_sub, D, n_rows)
                self.stats.rel_delta_h[nm] = float(a * xo / (h_norm + self.eps))

        self.stats.extra["pred_logp_gain"] = pred_gain

        # ---------------- 4. AdamW side ----------------------------------
        beta1, beta2 = adam_group["betas"]
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        wd_a = muon_group["weight_decay"]
        for p in adam_group["params"]:
            if p.grad is None:
                continue
            gr = p.grad
            st = self.state[p]
            if "exp_avg" not in st:
                st["exp_avg"] = torch.zeros_like(p)
                st["exp_avg_sq"] = torch.zeros_like(p)
            m, v = st["exp_avg"], st["exp_avg_sq"]
            m.mul_(beta1).add_(gr, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(gr, gr, value=1 - beta2)
            if wd_a > 0:
                p.mul_(1.0 - adam_group["adam_lr"] * wd_a)
            denom = (v / bc2).sqrt_().add_(adam_group["adam_eps"])
            p.addcdiv_(m / bc1, denom, value=-adam_group["adam_lr"])

        return loss

    # -- helpers -----------------------------------------------------------
    def _xo_norm(self, x_sub: torch.Tensor, D: torch.Tensor, n_rows: int) -> float:
        """||X O^T||_F estimated from a token subsample of X."""
        xo = (x_sub.to(D.dtype) @ D.t()).float()
        return float(xo.norm()) * math.sqrt(n_rows / max(1, x_sub.shape[0]))

    def _anchored(self, raw: List[Optional[float]], lr: float) -> List[Optional[float]]:
        """Trust ratio / alpha, clipped and re-anchored at the running median."""
        vals = [v for v in raw if v is not None]
        if not vals:
            return [None] * len(raw)
        med = torch.tensor(vals).median().item()
        lo, hi = self.orstyle_clip
        out = []
        for v in raw:
            if v is None:
                out.append(None)
            else:
                r = min(max(v / (med + self.eps), lo), hi)
                out.append(lr * r)
        return out

    def _clamp_alpha(self, raw: List[Optional[float]]) -> List[Optional[float]]:
        """Guard the intentional layer step against a near-zero denominator."""
        vals = [v for v in raw if v is not None]
        if not vals:
            return [None] * len(raw)
        med = torch.tensor(vals).median().item()
        cap = self.alpha_clip_ratio * med
        out = [None if v is None else min(v, cap) for v in raw]
        n_clip = sum(1 for v in raw if v is not None and v > cap)
        self.stats.extra["frac_clipped"] = n_clip / max(1, len(vals))
        return out


def build_optimizer(model: nn.Module, cfg, force_mode: str = None) -> MuonIntentional:
    """Split params and build the optimizer from a config object."""
    muon, adam = split_muon_params(model.named_parameters())
    if not muon:
        raise ValueError("no Muon-eligible params")
    opt = MuonIntentional(
        muon, adam,
        mode=force_mode or MODE_ALIAS.get(cfg.opt_mode, cfg.opt_mode), lr=cfg.lr,
        momentum=cfg.momentum, nesterov=cfg.nesterov,
        adam_lr=cfg.adam_lr, weight_decay=cfg.weight_decay,
        alpha_clip_ratio=cfg.alpha_clip_ratio,
    )
    names = {}
    for n, p in model.named_parameters():
        names[id(p)] = n
    opt.register_names(names)
    return opt


class PlainAdamW(torch.optim.Optimizer):
    """Reference AdamW arm (same interface: exposes .stats and .set_layer_inputs)."""

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, step=0))
        self.stats = IntentionalStats()

    def register_names(self, mapping): self._names = mapping
    def set_layer_inputs(self, x=None): self._layer_inputs = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            group["step"] += 1
            b1, b2 = group["betas"]
            bc1 = 1 - b1 ** group["step"]
            bc2 = 1 - b2 ** group["step"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                gr = p.grad
                st = self.state[p]
                if "exp_avg" not in st:
                    st["exp_avg"] = torch.zeros_like(p)
                    st["exp_avg_sq"] = torch.zeros_like(p)
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(gr, alpha=1 - b1)
                v.mul_(b2).addcmul_(gr, gr, value=1 - b2)
                if group["weight_decay"] > 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                denom = (v / bc2).sqrt_().add_(group["eps"])
                p.addcdiv_(m / bc1, denom, value=-group["lr"])
        return loss
