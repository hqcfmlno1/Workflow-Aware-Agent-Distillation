"""CPU-only safety and result contracts for the single-batch DDP experiment."""

import math

from workflow_distillation.optimized_qlora import device_plan, validate_records


def rank_record(records, rank):
    """Accept exactly one complete, distinct training sample per rank."""
    if len(records) != 2:
        raise ValueError("This probe requires exactly two trajectories")
    if rank not in (0, 1):
        raise ValueError("Expected rank 0 or 1")
    validate_records(records)
    if len({r.get("task_key") for r in records}) != 2:
        raise ValueError("Samples must belong to distinct tasks")
    for record in records:
        if record.get("partition") != "train_pool" or record.get("resolved") != 1:
            raise ValueError("Only successful train_pool records are allowed")
        if record["full_token_count"] >= 30000:
            raise ValueError("Each trajectory must contain fewer than 30000 tokens")
    return records[rank]


def replica_device_plan(rank):
    """Keep all text weights on the rank's GPU, with frozen vision on CPU."""
    if rank not in (0, 1):
        raise ValueError("Expected rank 0 or 1")
    return {key: rank if value == 0 else value for key, value in device_plan(32).items()}


def verify_replicas(evidence):
    """Require matching complete gradients and updated adapter parameters."""
    if len(evidence) != 2:
        raise ValueError("Expected evidence from both ranks")
    for field in ("gradient_sha256", "adapter_sha256"):
        if not evidence[0].get(field) or evidence[0][field] != evidence[1].get(field):
            raise ValueError(f"DDP replicas disagree: {field}")
    if any(not math.isfinite(r["adapter_max_delta"]) or r["adapter_max_delta"] <= 0
           for r in evidence):
        raise ValueError("Both ranks must perform a finite, nonzero adapter update")
