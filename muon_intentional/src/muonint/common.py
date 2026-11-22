"""Shared helpers: device-aware autocast, seeding, result bookkeeping."""

from __future__ import annotations

import contextlib
import json
import os
import random

import numpy as np
import torch


def autocast(device):
    """bf16 autocast on CUDA; a no-op elsewhere (so CPU smoke tests work)."""
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def gpu_hours(seconds: float, ngpu: int = 1) -> float:
    return seconds * ngpu / 3600.0
