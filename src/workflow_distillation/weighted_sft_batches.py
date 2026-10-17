"""Preserve equal trajectory weighting across distributed microbatches."""

from __future__ import annotations


def distributed_slots(indices, world_size):
    """Return per-rank (index, coefficient) slots, with zero-loss padding.

    Each rank sums its microsteps. DDP's rank average then produces the
    arithmetic mean over real trajectories, including an uneven final batch.
    """
    if not indices or world_size < 1:
        raise ValueError("A nonempty batch and positive world size are required")
    slots = [[] for _ in range(world_size)]
    total = ((len(indices) + world_size - 1) // world_size) * world_size
    for position in range(total):
        real = position < len(indices)
        index = indices[position] if real else indices[0]
        coefficient = world_size / len(indices) if real else 0.0
        slots[position % world_size].append((index, coefficient))
    return slots


def verify_resume(saved, current):
    """Reject continuation with changed data, objective, or execution settings."""
    for key in sorted(saved.keys() | current.keys()):
        if key not in saved or key not in current or saved[key] != current[key]:
            raise ValueError(f"Resume configuration differs: {key}")


def filter_full_history(records, max_length):
    """Filter complete trajectories without truncation, windows, or replacement."""
    if max_length < 2:
        raise ValueError("Context length must allow a target and its predecessor")
    kept, excluded = [], []
    for record in records:
        if record["length"] <= max_length:
            kept.append(record)
        else:
            excluded.append({
                "trajectory_id": record["trajectory_id"],
                "length": record["length"],
                "reason": "context_overflow",
            })
    return kept, excluded
