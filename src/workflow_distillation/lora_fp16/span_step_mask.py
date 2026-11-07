"""Project frozen span/reference arrays onto a strict subset of original targets."""

from __future__ import annotations

import numpy as np

from workflow_distillation.lora_fp16.span_cache import NAMES, validate_arrays


def project_cached_record(original, masked, arrays):
    """Preserve all context and align arrays by absolute token position, not rank."""
    ids, new_ids = np.asarray(original["input_ids"]), np.asarray(masked["input_ids"])
    groups, new_groups = np.asarray(original["groups"]), np.asarray(masked["groups"])
    if (
        original["trajectory_id"] != masked["trajectory_id"]
        or ids.ndim != 1
        or not np.array_equal(ids, new_ids)
        or groups.shape != ids.shape
        or new_groups.shape != groups.shape
        or np.any((new_groups != groups) & (new_groups != 0))
    ):
        raise ValueError("Step masks must only remove targets; context, roles and order are fixed")
    old_arrays = validate_arrays(groups, *(arrays[n] for n in NAMES))
    positions = np.flatnonzero(groups > 0)
    positions = positions[positions > 0]
    keep = new_groups[positions] > 0
    selected = [a[keep].copy() for a in old_arrays]
    validate_arrays(new_groups, *selected)
    return dict(zip(NAMES, selected, strict=True))
