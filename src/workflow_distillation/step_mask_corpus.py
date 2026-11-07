"""Apply whole-assistant-step loss masks without changing conditioning tokens."""

from __future__ import annotations

import numpy as np


def apply_step_mask(record, message_indices):
    ids = np.asarray(record["input_ids"])
    groups = np.asarray(record["groups"], dtype=np.uint8).copy()
    owners = np.asarray(record["token_message_indices"])
    if len(ids) != len(groups) or owners.shape != groups.shape:
        raise ValueError("Token ownership length mismatch")
    if len(set(message_indices)) != len(message_indices) or any(
        type(i) is not int or i not in record["assistant_message_indices"] for i in message_indices
    ):
        raise ValueError("Mask must reference unique assistant message indices")
    before = groups.copy()
    selected = np.isin(owners, message_indices)
    groups[selected] = 0
    counts = np.bincount(groups, minlength=5).tolist()
    denominator = int(np.sum(groups[1:] > 0))
    if not denominator:
        raise ValueError("Mask removes all supervised targets")
    result = {
        k: v
        for k, v in record.items()
        if k not in {"token_message_indices", "assistant_message_indices"}
    }
    result.update(
        groups=groups,
        denominator=denominator,
        group_counts=counts,
        mask_message_indices=list(message_indices),
        masked_target_tokens=int(np.sum((before > 0) & selected)),
    )
    return result
