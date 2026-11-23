"""A/B direction probes on a single, unmodified B1 Muon trajectory."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.func import functional_call

from muonint.common import autocast
from muonint.data import TokenBatcher
from muonint.optim import build_optimizer, zeropower_via_newtonschulz5
from muonint.pretrain import RunConfig, build_model, evaluate, lr_at


@torch.no_grad()
def probe(model, opt, fast_buffers, fast_beta, names, batches):
    # Construct both hypothetical matrix updates before the real optimizer step.
    # Other weights (including tied embeddings) are identical in both forwards.
    group = opt.param_groups[0]
    slow_weights, fast_weights = {}, {}
    for p, fast in zip(group["params"], fast_buffers):
        grad = p.grad
        slow_beta = group["momentum"]
        slow = opt.state[p]["momentum_buffer"].mul(slow_beta).add_(grad)
        def direction(buf, beta):
            momentum = grad.add(buf, alpha=beta) if opt.nesterov else buf
            return zeropower_via_newtonschulz5(momentum, steps=opt.ns_steps)
        ds = direction(slow, slow_beta)
        df = direction(fast, fast_beta)
        df = df * (ds.norm() / df.norm().clamp_min(1e-30))
        scale = -group["lr"] * math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
        slow_weights[names[p]] = p.add(ds, alpha=scale)
        fast_weights[names[p]] = p.add(df, alpha=scale)

    # FP32 scoring makes small differences visible near the end of cosine decay.
    losses = {}
    for label, weights in (("slow", slow_weights), ("fast", fast_weights)):
        for sample, (x, y) in batches.items():
            _, loss = functional_call(model, weights, (x, y))
            losses[f"{label}_{sample}"] = loss.item()
    if not all(map(math.isfinite, losses.values())):
        raise FloatingPointError(f"nonfinite candidate loss: {losses}")
    return losses


def summarize(rows, tolerance):
    # Positive margins mean fast had lower loss. Tied A defaults to baseline slow.
    a = np.array([r["slow_a"] - r["fast_a"] for r in rows])
    b = np.array([r["slow_b"] - r["fast_b"] for r in rows])
    decisive = (abs(a) > tolerance) & (abs(b) > tolerance)
    wins_a, wins_b = a > tolerance, b > tolerance
    agreement = (wins_a == wins_b)[decisive]
    gain_vs_slow = np.mean(np.where(wins_a, b, 0))
    return dict(probes=len(rows), decisive=int(decisive.sum()),
                fast_wins_a=int(wins_a.sum()), fast_wins_b=int(wins_b.sum()),
                ties_a=int((abs(a) <= tolerance).sum()), ties_b=int((abs(b) <= tolerance).sum()),
                agreement=float(agreement.mean()) if agreement.size else None,
                gain_vs_always_slow=float(gain_vs_slow),
                gain_vs_best_fixed=float(gain_vs_slow - max(0.0, b.mean())))


def train(cfg, spec, steps, observe, emit, *, momentum_schedule=None):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg.seed)
    model = build_model(cfg, "cuda")
    opt = build_optimizer(model, cfg)
    batcher = TokenBatcher(cfg.data_path, cfg.batch_size, cfg.seq_len, cfg.seed,
                           spec["data"]["validation_fraction"])
    group = opt.param_groups[0]
    if observe:
        pc = spec["probe"]
        fast = [torch.zeros_like(p) for p in group["params"]]
        names = {p: name for name, p in model.named_parameters()}
        probe_streams = {
            label: TokenBatcher(cfg.data_path, pc["batch_size"], cfg.seq_len, pc[f"seed_{label}"],
                                spec["data"]["validation_fraction"]) for label in ("a", "b")
        }
    rows = []
    start = time.monotonic()
    for step in range(1, steps + 1):
        group["lr"] = lr_at(step - 1, cfg.steps, cfg.lr, cfg.warmup_frac, cfg.lr_schedule)
        x, y = batcher.train_batch("cuda")
        opt.zero_grad(set_to_none=True)
        with autocast("cuda"):
            _, loss = model(x, y)
        loss.backward()
        grad_norm = torch.linalg.vector_norm(torch.stack(
            [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]))
        if not math.isfinite(grad_norm.item()) or not math.isfinite(loss.item()):
            raise FloatingPointError(f"nonfinite training at step {step}")
        record = {}
        if observe:
            with torch.no_grad():
                for buf, p in zip(fast, group["params"]):
                    buf.mul_(pc["fast_beta"]).add_(p.grad)
            if step % pc["every"] == 0:
                batches = {name: stream.train_batch("cuda") for name, stream in probe_streams.items()}
                result = probe(model, opt, fast, pc["fast_beta"], names, batches)
                rows.append(result)
                record.update(result, margin_a=result["slow_a"] - result["fast_a"],
                              margin_b=result["slow_b"] - result["fast_b"])
        if momentum_schedule is not None:
            boundary = momentum_schedule["initial_steps"]
            group["momentum"] = (momentum_schedule["initial_beta"] if step <= boundary
                                 else momentum_schedule["final_beta"])
            if step in (1, boundary, boundary + 1):
                record["momentum"] = group["momentum"]
        opt.step()
        if step % cfg.eval_every == 0 or step == steps:
            record.update(evaluate(model, batcher))
        if step % cfg.log_every == 0 or record:
            tokens = step * cfg.batch_size * cfg.seq_len
            record.update(train_loss=loss.item(), grad_norm=grad_norm.item(), lr=group["lr"],
                          tokens=tokens, tokens_per_sec=tokens / (time.monotonic() - start))
            if momentum_schedule is not None:
                record["momentum"] = group["momentum"]
            emit(step, record)
    return model, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dev-steps", type=int, help="local logs and baseline trajectory replay")
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    job = os.environ["SLURM_JOB_ID"]
    dev = args.dev_steps is not None
    directory = Path(spec["output"]).expanduser() / ("dev" if dev else "formal") / job
    directory.mkdir(parents=True, exist_ok=False)
    cache = Path(spec["data"]["path"]).expanduser()
    if cache.stat().st_size < 2 * spec["data"]["cache_tokens"]:
        raise ValueError("incomplete ClimbMix cache")
    cfg = RunConfig(name=spec["name"], **spec["training"], data_path=str(cache), out_dir=str(directory))
    steps = args.dev_steps if dev else cfg.steps
    if not spec["probe"]["every"] <= steps <= cfg.steps or spec["probe"]["every"] < 2:
        raise ValueError("run must include at least one probe after momentum initialization")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repo = "https://github.com/Liyikang626/llm_project"
    spec.update(job_id=job, git_sha=commit, steps=steps, gpu=torch.cuda.get_device_name(),
                model="from-scratch GPT, RoPE, RMSNorm, SwiGLU, tied embeddings", vocab_size=50257,
                branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                config_url=f"{repo}/blob/{commit}/muon_intentional/{args.config}",
                launch_script=f"{repo}/blob/{commit}/muon_intentional/slurm/momentum_diagnostic.sbatch",
                launch_command="sbatch slurm/momentum_diagnostic.sbatch" + (f" --dev-steps {steps}" if dev else ""),
                source_mode="dev working copy" if dev else "committed source",
                deterministic_algorithms=True, num_parameters=30805248,
                optimizer_detail="Nesterov Muon, 5 Newton-Schulz iterations; AdamW side betas 0.9/0.95, eps 1e-8")
    (directory / "config.json").write_text(json.dumps(spec, indent=2))
    # Compute jobs write a portable offline run. Only approved formal logs are synced.
    import swanlab
    run = swanlab.init(project="llm_from_scratch", workspace="liyikang24",
                       experiment_name=f"{cfg.name}-{job}", config=spec, mode="offline",
                       logdir=str(directory / "swanlog"))
    print(f"output={directory} steps={steps} dev={dev}", flush=True)
    vals = []
    started = time.monotonic()
    with (directory / "metrics.jsonl").open("x") as out:
        def emit(step, record):
            print(json.dumps({"step":step, **record}, allow_nan=False), file=out, flush=True)
            if "val_loss" in record:
                vals.append(record["val_loss"])
            run.log(record, step=step)
        try:
            model, rows = train(cfg, spec, steps, True, emit)
            summary = summarize(rows, spec["probe"]["tie_tolerance"])
            summary.update(final_val_loss=vals[-1], best_val_loss=min(vals), steps=steps,
                           tokens=steps*cfg.batch_size*cfg.seq_len, wall_sec=time.monotonic()-started)
            if dev:
                reference, _ = train(cfg, spec, steps, False, lambda *_: None)
                for name, p in model.named_parameters():
                    if not torch.equal(p, reference.get_parameter(name)):
                        raise AssertionError(f"trajectory changed: {name}")
                summary["baseline_replay_exact"] = True
            run.log({f"summary/{k}": v for k, v in summary.items() if v is not None}, step=steps+1)
            summary["status"] = "completed"
        except BaseException as error:
            (directory / "failure.txt").write_text(repr(error))
            run.finish(state="crashed", error=repr(error))
            raise
        else:
            run.finish()
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(summary)


if __name__ == "__main__":
    main()
