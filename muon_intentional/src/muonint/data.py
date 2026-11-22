"""Data: ClimbMix token shards for pretraining, GSM8K for RLVR."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

CLIMBMIX = "/mnt/oss/datasets/climbmix/part_0.tokenized.jsonl"
EOS = 50256  # GPT-2 <|endoftext|>; ClimbMix stores documents already tokenised


def build_token_cache(path: str, n_tokens: int, src: str = CLIMBMIX, verbose: bool = True) -> str:
    """Stream documents from `src` until `n_tokens` ids are collected.

    Documents are separated by EOS so that the cache can be treated as one
    long stream. Saved as uint16 (GPT-2 ids < 50257), ~2 bytes/token.
    """
    if os.path.exists(path) and os.path.getsize(path) >= 2 * n_tokens * 0.98:
        if verbose:
            print(f"[cache] {path} exists ({os.path.getsize(path)/1e6:.0f} MB), reusing")
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf = np.empty(n_tokens + 4096, dtype=np.uint16)
    i = 0
    with open(src, "r") as f:
        for line in f:
            toks = json.loads(line)["tokens"]
            if i + len(toks) + 1 >= len(buf):
                toks = toks[: len(buf) - i - 1]
            buf[i:i + len(toks)] = np.asarray(toks, dtype=np.uint16)
            i += len(toks)
            buf[i] = EOS
            i += 1
            if i >= n_tokens:
                break
    arr = np.array(buf[:i], dtype=np.uint16)
    tmp = path + ".tmp"
    arr.tofile(tmp)
    os.replace(tmp, path)
    if verbose:
        print(f"[cache] wrote {i/1e6:.1f}M tokens -> {path}")
    return path


def load_tokens(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


class TokenBatcher:
    """Random contiguous crops. The last `val_frac` of the stream is held out."""

    def __init__(self, path: str, batch_size: int, seq_len: int, seed: int = 0, val_frac: float = 0.01):
        self.tokens = load_tokens(path)
        n = len(self.tokens)
        self.batch_size, self.seq_len = batch_size, seq_len
        self.n_train = int(n * (1 - val_frac))
        self.n_val = n - self.n_train - 1
        self.rng = np.random.default_rng(seed)

    def train_batch(self, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, self.n_train - self.seq_len - 1, size=self.batch_size)
        idx = np.stack([np.arange(s, s + self.seq_len + 1) for s in starts])
        chunk = self.tokens[idx].astype(np.int64)
        x = torch.from_numpy(chunk[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(chunk[:, 1:]).to(device, non_blocking=True)
        return x, y

    def val_batches(self, n_batches: int = 8, batch_size: Optional[int] = None,
                    seq_len: Optional[int] = None):
        """Deterministic validation crops; same seeds on every call."""
        T = seq_len or self.seq_len
        bs = batch_size or self.batch_size
        rng = np.random.default_rng(12345)
        hi = self.n_train + max(1, self.n_val - T - 2)
        starts = rng.integers(self.n_train, hi, size=(n_batches, bs))
        out = []
        for row in starts:
            idx = np.stack([np.arange(s, s + T + 1) for s in row])
            chunk = self.tokens[idx].astype(np.int64)
            out.append((torch.from_numpy(chunk[:, :-1]), torch.from_numpy(chunk[:, 1:])))
        return out


# --------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------
def load_gsm8k(path: str, split: str = "train") -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({"question": d["question"], "answer": d["answer"]})
    return rows


def gold_number(answer: str) -> Optional[str]:
    if "####" not in answer:
        return None
    tail = answer.split("####")[-1].strip().replace(",", "")
    try:
        return str(int(float(tail)))
    except Exception:
        return None
