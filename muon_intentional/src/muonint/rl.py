"""LLM post-training RL (RLVR / GRPO-lite) for the same optimizers.

Why this setting matters for the proposal: a language model's cross-entropy is
a log-likelihood policy gradient, so "choose the step so that the policy's
log-probability moves by a prescribed amount" is exactly what Intentional
Policy Gradient does. In RL post-training the parameter-space learning rate is
also unusually fragile (entropy collapse, log-prob drift, clip-fraction blowups),
which is the failure mode an intentional step is supposed to remove.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .optim import MuonIntentional, PlainAdamW, split_muon_params
from .common import autocast

MODEL_PATH = "/mnt/oss/models/Qwen/Qwen3-0.6B"
SYS = ("You are a math solver. Reason briefly, then end with a final line of the "
       "exact form:\nAnswer: <number>")


@dataclass
class RLConfig:
    name: str = "muon"
    opt_mode: str = "fixed"       # fixed | orscale | intent_l | intent_t | adamw
    lr: float = 0.02
    adam_lr: float = 1e-5         # AdamW for the non-Muon params (or for everything)
    momentum: float = 0.95
    weight_decay: float = 0.0
    alpha_clip_ratio: float = 20.0
    prompts_per_step: int = 8
    group_size: int = 6
    max_new_tokens: int = 192
    steps: int = 200
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    n_train_prompts: int = 2000
    n_eval: int = 200
    eval_every: int = 50
    log_every: int = 5
    n_sub: int = 512
    measure_every: int = 25
    lr_schedule: str = "const"
    warmup_frac: float = 0.0
    out_dir: str = "/home/15Cu2Plus/muon_intentional/results/rl"
    data_dir: str = "/home/15Cu2Plus/muon_intentional/data"
    model_path: str = MODEL_PATH

    @property
    def tag(self) -> str:
        keys = ("name", "opt_mode", "lr", "adam_lr", "seed", "steps",
                "prompts_per_step", "group_size", "max_new_tokens")
        return "_".join(f"{k}={getattr(self, k)}" for k in keys)


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
def extract_number(text: str) -> Optional[str]:
    """Last plausible answer in the completion."""
    t = text.strip()
    m = re.findall(r"[Aa]nswer\s*[:：]\s*\**\s*(-?[\d,]+(?:\.\d+)?)", t)
    if m:
        return _norm(m[-1])
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", t)
    if m:
        return _norm(m[-1])
    m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", t)
    return _norm(m[-1]) if m else None


def _norm(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if abs(f - round(f)) < 1e-6 else f"{f:g}"
    except Exception:
        return s


def gold_number(answer: str) -> Optional[str]:
    if "####" not in answer:
        return None
    return _norm(answer.split("####")[-1])


# --------------------------------------------------------------------------
# model / data
# --------------------------------------------------------------------------
def load_model(device="cuda", model_path: str = MODEL_PATH):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
        local_files_only=True, attn_implementation="sdpa",
    ).to(device)
    return model, tok


def load_prompts(cfg: RLConfig):
    rows = []
    with open(os.path.join(cfg.data_dir, "gsm8k_train.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            rows.append((d["question"], gold_number(d["answer"])))
    test = []
    with open(os.path.join(cfg.data_dir, "gsm8k_test.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            test.append((d["question"], gold_number(d["answer"])))
    return rows[: cfg.n_train_prompts], test[: cfg.n_eval]


def chat_ids(tok, question: str) -> List[int]:
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _lr_at(step, total, base, warmup_frac, schedule):
    import math
    w = max(1, int(total * warmup_frac))
    if step < w:
        return base * (step + 1) / w
    if schedule == "const":
        return base
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def generate_group(model, tok, prompts, cfg: RLConfig, gen: torch.Generator):
    """Sample `group_size` completions per prompt in one batched call."""
    enc = [chat_ids(tok, q) for q, _ in prompts]
    flat, owner = [], []
    for i, ids in enumerate(enc):
        for _ in range(cfg.group_size):
            flat.append(ids)
            owner.append(i)
    maxlen = max(len(x) for x in flat)
    input_ids = torch.full((len(flat), maxlen), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros_like(input_ids)
    for j, ids in enumerate(flat):
        input_ids[j, maxlen - len(ids):] = torch.tensor(ids)
        attn[j, maxlen - len(ids):] = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids, attn = input_ids.to(device), attn.to(device)
    out = model.generate(
        input_ids=input_ids, attention_mask=attn,
        max_new_tokens=cfg.max_new_tokens, do_sample=True,
        temperature=cfg.temperature, top_p=cfg.top_p,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    comp = out[:, maxlen:]
    return comp, owner, maxlen


def completion_logprobs(model, input_ids, attn, comp_ids, comp_mask):
    """Per-token log-prob of the sampled completion tokens, masked to them only."""
    P = input_ids.shape[1]
    full = torch.cat([input_ids, comp_ids], dim=1)
    full_mask = torch.cat([attn, comp_mask], dim=1)
    with autocast(full.device):
        logits = model(input_ids=full, attention_mask=full_mask, use_cache=False).logits
    lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = full[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    pm = torch.zeros_like(tgt)
    pm[:, P - 1:] = comp_mask                       # only completion tokens count
    return tok_lp, pm


class LinearCapture:
    """Record (subsampled) inputs and ||output||_F of every Muon-eligible Linear.

    Needed because a pretrained HF model does not hand us its activations.
    """

    def __init__(self, model, allowed_ids, n_sub=512):
        self.allowed = set(allowed_ids)
        self.n_sub = n_sub
        self.records: Dict[int, tuple] = {}
        self.active = True
        self._handles = []
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear) and id(mod.weight) in self.allowed:
                self._handles.append(mod.register_forward_hook(self._hook))

    def _hook(self, mod, inp, out):
        if not self.active:
            return
        x = inp[0]
        xf = x.reshape(-1, x.shape[-1])
        n = xf.shape[0]
        k = min(self.n_sub, n)
        idx = torch.randperm(n, device=xf.device)[:k]
        xs = xf.index_select(0, idx).detach().to(torch.bfloat16)
        self.records[id(mod.weight)] = (xs, n, float(out.detach().float().norm()))

    def clear(self):
        self.records = {}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def run(cfg: RLConfig, device="cuda", verbose=True, reward_fn=None) -> Dict:
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, cfg.tag + ".json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "done":
            if verbose:
                print(f"[skip] {cfg.tag}")
            return prev

    model, tok = load_model(device, cfg.model_path)
    verify = reward_fn or (lambda text, gold: float(
        gold is not None and extract_number(text) == gold))
    model._n_sub = cfg.n_sub
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    train, test = load_prompts(cfg)
    rng = torch.Generator().manual_seed(cfg.seed)

    cap = None
    if cfg.opt_mode == "adamw":
        opt = PlainAdamW(model.parameters(), lr=cfg.adam_lr, weight_decay=cfg.weight_decay)
    else:
        muon, adam = split_muon_params(model.named_parameters())
        opt = MuonIntentional(muon, adam, mode=cfg.opt_mode, lr=cfg.lr,
                              momentum=cfg.momentum, adam_lr=cfg.adam_lr,
                              weight_decay=cfg.weight_decay,
                              alpha_clip_ratio=cfg.alpha_clip_ratio)
        names = {id(p): n for n, p in model.named_parameters()}
        opt.register_names(names)
        if cfg.opt_mode == "intent_l":
            cap = LinearCapture(model, [id(p) for p in muon], n_sub=cfg.n_sub)
    model.train()

    hist, t0 = [], time.time()
    capture = cfg.opt_mode == "intent_l"
    for step in range(cfg.steps):
        cur_lr = _lr_at(step, cfg.steps, cfg.lr, cfg.warmup_frac, cfg.lr_schedule)
        if isinstance(opt, MuonIntentional):
            opt.param_groups[0]["lr"] = cur_lr
        else:
            opt.param_groups[0]["lr"] = cur_lr

        idx = torch.randint(0, len(train), (cfg.prompts_per_step,), generator=rng).tolist()
        prompts = [train[i] for i in idx]
        golds = [g for _, g in prompts]

        if cap is not None:
            cap.active = False           # don't pay hook overhead inside generate()
        comp, owner, pre = generate_group(model, tok, prompts, cfg, gen)
        if cap is not None:
            cap.active = True
        comp_mask = (comp != tok.pad_token_id).long()
        comp_texts = tok.batch_decode(comp, skip_special_tokens=True)
        rewards = torch.tensor([verify(comp_texts[j], golds[owner[j]])
                                for j in range(len(comp_texts))],
                               dtype=torch.float32, device=device)

        # group-normalised advantage
        R = rewards.view(cfg.prompts_per_step, cfg.group_size)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)
        adv = adv.view(-1)

        enc = [chat_ids(tok, q) for q, _ in prompts]
        flat_ids = []
        for i, ids in enumerate(enc):
            flat_ids += [ids] * cfg.group_size
        maxlen = max(len(x) for x in flat_ids)
        input_ids = torch.full((len(flat_ids), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for j, ids in enumerate(flat_ids):
            input_ids[j, maxlen - len(ids):] = torch.tensor(ids)
            attn[j, maxlen - len(ids):] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)
        comp_ids = comp[:, :].to(device)

        opt.zero_grad(set_to_none=True)
        if cap is not None:
            cap.clear()
        tok_lp, pm = completion_logprobs(model, input_ids, attn, comp_ids, comp_mask)
        # per-sequence mean over completion tokens, then batch mean
        seq_lp = (tok_lp * pm).sum(-1) / pm.sum(-1).clamp(min=1)
        loss = -(adv * seq_lp).mean()
        lp_before = float(seq_lp.mean())
        loss.backward()
        if cap is not None:
            opt.set_layer_inputs(cap.records)
        opt.step()

        rec = dict(step=step, loss=float(loss.item()), acc=float(rewards.mean()),
                   reward_std=float(R.std()), lr=cur_lr,
                   mean_len=float(pm.sum(-1).float().mean()),
                   n_zero_adv_groups=int(((R.std(1) == 0)).sum()),
                   elapsed=time.time() - t0)
        if getattr(opt, "stats", None) and opt.stats.rel_delta_h:
            v = torch.tensor(list(opt.stats.rel_delta_h.values()))
            rec["rel_delta_h_mean"] = float(v.mean())
            rec["rel_delta_h_med"] = float(v.median())
            rec["rel_delta_h_spread"] = float(v.max() / (v.min() + 1e-12))
            rec["frac_clipped"] = float(opt.stats.extra.get("frac_clipped", 0.0))
        if getattr(opt, "stats", None) and opt.stats.alpha:
            a = torch.tensor(list(opt.stats.alpha.values()))
            rec["alpha_med"] = float(a.median())
        if getattr(opt, "stats", None):
            rec["pred_gain"] = float(opt.stats.extra.get("pred_logp_gain", 0.0))
        if cfg.measure_every and step % cfg.measure_every == 0:
            with torch.no_grad():
                lp2, pm2 = completion_logprobs(model, input_ids, attn, comp_ids, comp_mask)
                lp_after = float(((lp2 * pm2).sum(-1) / pm2.sum(-1).clamp(min=1)).mean())
            rec["realized_gain"] = lp_after - lp_before
            rec["entropy"] = float(
                -(lp2.exp() * lp2 * pm2).sum(-1).sum().item() / max(1.0, pm2.sum().item()))
        hist.append(rec)
        if verbose and step % cfg.log_every == 0:
            print(f"  {cfg.tag} step {step}/{cfg.steps} loss {rec['loss']:.4f} "
                  f"acc {rec['acc']:.2f} len {rec['mean_len']:.0f}")

        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            acc = evaluate(model, tok, test, cfg)
            hist.append(dict(step=step, eval_acc=acc, elapsed=time.time() - t0))
            if verbose:
                print(f"  [eval] step {step+1} gsm8k_subset_acc {acc:.3f}")

    wall = time.time() - t0
    result = dict(status="done", cfg=asdict(cfg), wall_sec=wall, history=hist)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[done] {cfg.tag} {wall:.0f}s")
    return result


@torch.no_grad()
def evaluate(model, tok, test, cfg: RLConfig, batch: int = 32) -> float:
    model.eval()
    correct, total = 0, 0
    for i in range(0, len(test), batch):
        chunk = test[i:i + batch]
        enc = [chat_ids(tok, q) for q, _ in chunk]
        maxlen = max(len(x) for x in enc)
        input_ids = torch.full((len(enc), maxlen), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)
        for j, ids in enumerate(enc):
            input_ids[j, maxlen - len(ids):] = torch.tensor(ids)
            attn[j, maxlen - len(ids):] = 1
        dev = next(model.parameters()).device
        out = model.generate(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                             max_new_tokens=cfg.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        texts = tok.batch_decode(out[:, maxlen:], skip_special_tokens=True)
        for (q, g), t in zip(chunk, texts):
            correct += int(g is not None and extract_number(t) == g)
            total += 1
    model.train()
    return correct / max(1, total)
