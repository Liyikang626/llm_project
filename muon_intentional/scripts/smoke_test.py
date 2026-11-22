"""CPU smoke test: every arm must run end-to-end before anything hits a GPU."""

import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/15Cu2Plus/muon_intentional/src")

from muonint.diagnostics import run_layer_heterogeneity, test_exactness
from muonint.model import GPT, GPTConfig
from muonint.optim import MuonIntentional, PlainAdamW, split_muon_params
from muonint.pretrain import RunConfig, run

DEV = "cpu"


def make_tokens(path, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, 200, size=n).astype(np.uint16)   # tiny vocab subset
    toks.tofile(path)


def main():
    print("=== A1 exactness ===")
    ex = test_exactness()
    print(ex)
    assert ex["max_rel_err"] < 1e-2, ex

    # 中间产物只写在项目目录内，不碰 /tmp
    tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    dpath = os.path.join(tmp, "toy.u16")
    make_tokens(dpath)

    print("\n=== optimizer arms (tiny model) ===")
    cfg = GPTConfig(vocab_size=50257, n_layer=2, n_head=2, d_model=64, seq_len=64)
    for mode in ["fixed", "orscale", "intent_l", "intent_t"]:
        torch.manual_seed(0)
        model = GPT(cfg)
        model._n_sub = 32
        muon, adam = split_muon_params(model.named_parameters())
        opt = MuonIntentional(muon, adam, mode=mode, lr=0.02)
        opt.register_names({id(p): n for n, p in model.named_parameters()})
        x = torch.randint(0, 200, (2, 64))
        y = torch.randint(0, 200, (2, 64))
        for _ in range(3):
            opt.zero_grad()
            _, loss = model(x, y, capture=(mode == "intent_l"))
            loss.backward()
            if mode == "intent_l":
                opt.set_layer_inputs(model.layer_inputs)
            opt.step()
        print(f"  {mode:9s} loss={loss.item():.4f} n_muon={len(muon)} "
              f"n_adam={len(adam)} alpha_med={float(torch.tensor(list(opt.stats.alpha.values())).median()):.5f} "
              f"rel_dH={list(opt.stats.rel_delta_h.values())[:2]}")
        if mode == "intent_l":
            r = torch.tensor(list(opt.stats.rel_delta_h.values()))
            assert (r - 0.02).abs().max() < 2e-2, f"intentional layer step not honoured: {r}"
            print("     intent_l delivered rel dH ~= eta  OK")

    print("\n=== adamw arm ===")
    model = GPT(cfg)
    opt = PlainAdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 200, (2, 64)); y = torch.randint(0, 200, (2, 64))
    opt.zero_grad(); _, loss = model(x, y); loss.backward(); opt.step()
    print("  adamw loss", round(loss.item(), 4))

    print("\n=== pretrain.run (5 steps, all modes) ===")
    for mode in ["fixed", "orscale", "intent_l", "intent_t", "adamw"]:
        c = RunConfig(name="smoke", opt_mode=mode, lr=(3e-3 if mode == "adamw" else 0.02),
                      n_layer=2, n_head=2, d_model=64, seq_len=64, batch_size=2,
                      tokens=5 * 2 * 64, log_every=2, eval_every=5, n_sub=32,
                      data_path=dpath, out_dir=os.path.join(tmp, "res"))
        r = run(c, device=DEV, verbose=False)
        print(f"  {mode:9s} val_loss={r['final_val_loss']:.4f}")

    print("\n=== diagnostics (tiny) ===")
    d = run_layer_heterogeneity(dpath, batch_sizes=(2, 4), steps=6, seq_len=64,
                                d_model=64, n_layer=2, n_head=2, device=DEV,
                                out_path=os.path.join(tmp, "diag.json"))
    for k, v in d["results"].items():
        print(f"  {k}: across_layer_cv={v['across_layer_cv']:.3f} "
              f"ratio_mean={v['ratio_mean']:.5f} temporal_cv={v['temporal_cv_mean']:.3f}")

    print("\nSMOKE OK")


def rl_smoke():
    """End-to-end GRPO plumbing on a tiny randomly-initialised Qwen."""
    import json
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from muonint.rl import RLConfig, run

    tmp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch", "rl_smoke")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    src = "/mnt/oss/models/Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(src, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp)
    cfgm = AutoConfig.from_pretrained(src, local_files_only=True)
    cfgm.vocab_size = len(tok)
    cfgm.hidden_size, cfgm.intermediate_size = 64, 128
    cfgm.num_hidden_layers, cfgm.num_attention_heads, cfgm.num_key_value_heads = 2, 4, 2
    cfgm.head_dim, cfgm.max_position_embeddings = 16, 1024
    cfgm.layer_types = ["full_attention"] * cfgm.num_hidden_layers
    cfgm.tie_word_embeddings = True
    m = AutoModelForCausalLM.from_config(cfgm)
    m.save_pretrained(tmp)
    print("  tiny qwen params", sum(p.numel() for p in m.parameters()))

    import random
    rnd = random.Random(0)
    rfn = lambda text, gold: float(rnd.random() < 0.5)

    for mode in ["fixed", "orscale", "intent_l", "intent_t", "adamw"]:
        c = RLConfig(name="rl_smoke", opt_mode=mode, lr=(1e-4 if mode == "adamw" else 0.02),
                     adam_lr=1e-4, prompts_per_step=2, group_size=3, max_new_tokens=12,
                     steps=3, n_eval=2, eval_every=3, log_every=1, model_path=tmp,
                     out_dir=os.path.join(tmp, "rl"), n_sub=32)
        r = run(c, device="cpu", verbose=True, reward_fn=rfn)
        print(f"  {mode:9s} ok, wall={r['wall_sec']:.1f}s")


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "rl":
        rl_smoke()
    else:
        main()
