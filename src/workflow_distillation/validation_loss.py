"""Common teacher-forced metrics for retained FP16 LoRA checkpoints.

Kept outside lora_fp16: adding evaluation must not change that trainer's
hash-pinned source identity or break continuation of existing experiments.
"""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np

GROUPS = ("masked", "reasoning", "content", "action", "submit")
TOKEN_WEIGHTS = np.array([0, 1, 0.75, 1.5, 2], dtype=np.float64)
GROUP_WEIGHTS = np.array([0, 0.15, 0.05, 1, 0.5], dtype=np.float64)


def summarize_groups(record):
    counts = np.asarray(record["counts"], dtype=np.int64)
    sums = np.asarray(record["ce_sums"], dtype=np.float64)
    if (
        counts.shape != (5,)
        or sums.shape != (5,)
        or counts[0]
        or sums[0]
        or (counts < 0).any()
        or not np.isfinite(sums).all()
        or (sums < 0).any()
        or counts.sum() == 0
        or np.any((counts == 0) & (sums != 0))
    ):
        raise ValueError("Invalid supervised group statistics")
    coefficient = np.zeros(5)
    present = counts > 0
    coefficient[present] = GROUP_WEIGHTS[present] / np.sqrt(counts[present])
    return {
        "ce": float(sums.sum() / counts.sum()),
        "token_weighted_loss": float(sums @ TOKEN_WEIGHTS / (counts @ TOKEN_WEIGHTS)),
        "group_weighted_loss": float(sums @ coefficient / (counts @ coefficient)),
        **{
            f"{name}_ce": float(sums[g] / counts[g]) if counts[g] else None
            for g, name in enumerate(GROUPS)
            if g
        },
    }


def aggregate(records):
    records = list(records)
    if not records:
        raise ValueError("Empty validation")
    metrics = [summarize_groups(r) for r in records]
    counts = np.asarray([r["counts"] for r in records]).sum(0)
    sums = np.asarray([r["ce_sums"] for r in records]).sum(0)
    micro = summarize_groups(dict(counts=counts, ce_sums=sums))
    return {
        "trajectories": len(records),
        "supervised_tokens": int(counts.sum()),
        "ce_macro": float(np.mean([m["ce"] for m in metrics])),
        "ce_micro": micro["ce"],
        **{
            key: float(np.mean([m[key] for m in metrics]))
            for key in ("token_weighted_loss", "group_weighted_loss")
        },
        **{f"{name}_ce": micro[f"{name}_ce"] for name in GROUPS[1:]},
        **{f"{name}_tokens": int(counts[g]) for g, name in enumerate(GROUPS) if g},
    }


def evaluate_record(state, record, device, chunk_size=128):
    """Full trajectory forward; chunk ONLY vocabulary projection, never context.

    Target token at position t uses hidden state t-1. Masked tool observations
    remain in context but never enter the CE or its denominator.
    """
    import torch
    import torch.nn.functional as functional

    ids = torch.as_tensor(np.array(record["input_ids"], dtype=np.int64), device=device)[None]
    groups = torch.as_tensor(np.array(record["groups"], dtype=np.int64), device=device)
    if groups.shape != ids.shape[1:] or groups.min() < 0 or groups.max() > 4 or chunk_size < 1:
        raise ValueError("Invalid aligned trajectory groups or chunk size")
    positions = torch.where(groups[1:] > 0)[0] + 1
    counts = torch.zeros(5, dtype=torch.int64, device=device)
    sums = torch.zeros(5, dtype=torch.float64, device=device)
    context = (
        torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
    )
    with torch.inference_mode(), context:
        hidden = state.decoder(input_ids=ids, use_cache=False).last_hidden_state
        for indices in positions.split(chunk_size):
            logits = state.head(hidden[0, indices - 1].to(state.head.weight.device)).float()
            ce = functional.cross_entropy(
                logits, ids[0, indices].to(logits.device), reduction="none"
            )
            if not bool(torch.isfinite(ce).all()):
                raise FloatingPointError("Nonfinite validation CE")
            selected = groups[indices]
            counts.scatter_add_(0, selected, torch.ones_like(selected))
            sums.scatter_add_(0, selected, ce.to(device=device, dtype=torch.float64))
    result = dict(counts=counts.cpu().tolist(), ce_sums=sums.cpu().tolist())
    summarize_groups(result)
    return result


def load_adapter(model, path, manifest):
    """Load only tensor weights; never deserialize optimizer/RNG pickle files."""
    import torch

    from workflow_distillation.epoch_data import sha256_file

    if sha256_file(path) != manifest["files"]["adapter.pt"]:
        raise ValueError("Adapter checksum mismatch")
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    current = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if tensors.keys() != current.keys() or any(
        tensors[n].shape != p.shape
        or tensors[n].dtype != p.dtype
        or not bool(torch.isfinite(tensors[n]).all())
        for n, p in current.items()
    ):
        raise ValueError("Adapter keys/shapes/dtypes/nonfinite weights differ")
    with torch.no_grad():
        for name, parameter in current.items():
            parameter.copy_(tensors[name])


def load_results(path):
    """Resume complete trajectory rows; repair only an interrupted final write."""
    path = Path(path)
    rows = {}
    if not path.exists():
        return rows
    with path.open("rb+") as stream:
        good = 0
        for line in stream:
            if not line.endswith(b"\n"):
                stream.truncate(good)
                break
            row = json.loads(line)
            summarize_groups(row)
            key = row["key"], row["trajectory_id"]
            if key in rows and rows[key] != row:
                raise ValueError("Conflicting resumed trajectory results")
            if not math.isfinite(row.get("seconds", 0)):
                raise ValueError("Invalid elapsed time")
            rows[key] = row
            good += len(line)
    return rows
