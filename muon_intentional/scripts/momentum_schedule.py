"""Compare fixed and switched Muon momentum on paired B1 training trajectories."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from momentum_diagnostic import train
from muonint.pretrain import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("arm", choices=("slow", "fast", "switch40"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--dev-steps", type=int)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    if args.seed not in spec["seeds"]:
        parser.error("seed must be one of the configured paired seeds")
    schedule = spec["schedules"][args.arm]
    dev = args.dev_steps is not None
    job = os.environ["SLURM_JOB_ID"]
    name = f'{spec["name"]}-{args.arm}-seed{args.seed}'
    out = Path(spec["output"]).expanduser() / ("dev" if dev else "formal") / job / name
    cache = Path(spec["data"]["path"]).expanduser()
    if cache.stat().st_size < 2 * spec["data"]["cache_tokens"]:
        raise ValueError("incomplete ClimbMix cache")
    cfg = RunConfig(name=name, seed=args.seed, momentum=schedule["initial_beta"],
                    data_path=str(cache), out_dir=str(out), **spec["training"])
    steps = args.dev_steps if dev else cfg.steps
    if not schedule["initial_steps"] < steps <= cfg.steps:
        parser.error("requested steps must cross the switch and fit the B1 budget")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    repo = f"https://github.com/Liyikang626/llm_project/blob/{sha}/muon_intentional"
    script = "slurm/momentum_schedule.sbatch"
    spec.update(arm=args.arm, seed=args.seed, schedule=schedule, git_sha=sha, job_id=job,
                gpu=torch.cuda.get_device_name(), precision="BF16 autocast, FP32 parameters",
                model="GPT, RoPE, RMSNorm, SwiGLU, tied embeddings, random initialization",
                vocab_size=50257, data_mix="100% ClimbMix part_0; tail 1% validation",
                optimizer="Nesterov Muon NS5; AdamW side betas 0.9/0.95, eps 1e-8",
                validation="8 fixed batches, crop RNG seed 12345",
                config_url=f"{repo}/{args.config}", launch_script=f"{repo}/{script}",
                launch_command=("development smoke test" if dev else
                                f"sbatch --exclude=llm-node04 {script} {args.arm} {args.seed}"),
                branch="lyk-dev", steps=steps)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(spec, indent=2))
    run = None
    if not dev:
        import swanlab
        run = swanlab.init(project="llm_from_scratch", workspace="liyikang24",
                           experiment_name=f"{name}-{job}", config=spec, mode="offline",
                           logdir=str(out / "swanlog"))
    vals = []
    start = time.monotonic()
    last_step = 0
    try:
        with (out / "metrics.jsonl").open("x") as file:
            def emit(step, record):
                nonlocal last_step
                last_step = step
                print(json.dumps({"step": step, **record}, allow_nan=False), file=file, flush=True)
                if "val_loss" in record:
                    vals.append(record["val_loss"])
                if run is not None:
                    run.log(record, step=step)

            train(cfg, spec, steps, False, emit, momentum_schedule=schedule)
        result = dict(final_val_loss=vals[-1], best_val_loss=min(vals), steps=steps,
                      tokens=steps * cfg.seq_len * cfg.batch_size, wall_sec=time.monotonic()-start)
        if run is not None:
            run.log({f"summary/{k}": v for k, v in result.items()}, step=steps+1)
            run.finish()
        result["status"] = "completed"
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        print(name, result)
    except BaseException as exc:
        (out / "failure.txt").write_text(f"last logged step {last_step}: {exc!r}")
        if run is not None:
            run.finish(state="crashed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
