"""Trajectory-wide token-group normalization, independent of stored legacy CE.

Group IDs: masked=0, reasoning=1, content=2, ordinary action/protocol=3,
terminal submit/protocol=4. All turns of a trajectory share these four groups.
"""

from __future__ import annotations

import numpy as np


def group_loss_parameters(groups, group_weights, length_alpha):
    """Return per-token coefficients and ONE shared trajectory denominator.

    sum_g w_g N_g**(1-a) mean_CE_g / sum_g w_g N_g**(1-a)
    = sum_tokens w_g N_g**(-a) CE_token / the same denominator.
    Position zero has no predecessor and must never enter a target count.
    """
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 1 or len(groups) < 2 or np.any((groups < 0) | (groups > 4)):
        raise ValueError("Expected complete trajectory groups in [0, 4]")
    counts = np.bincount(groups[1:], minlength=5)
    counts[0] = 0
    present = counts > 0
    coefficients = np.zeros(5, dtype=np.float64)
    coefficients[present] = np.asarray(group_weights)[present] * (
        counts[present].astype(np.float64) ** (-length_alpha)
    )
    denominator = float(np.dot(coefficients, counts))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Trajectory has no positively weighted supervised targets")
    return tuple(coefficients.tolist()), denominator


def chunked_hidden_loss(
    hidden, head, labels, groups, denominator, scale, chunk_size=128, *, group_weights=None
):
    """Checkpoint vocabulary chunks with configurable trajectory coefficients.

    Kept separate from weighted_sft_loss.py: that historical implementation is
    hash-pinned by old-run resume compatibility. No full sequence/vocab tensor.
    """
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    if hidden.shape[0] != 1 or labels.shape != groups.shape:
        raise ValueError("Chunked hidden loss requires one sequence and aligned groups")
    if denominator <= 0 or chunk_size < 1:
        raise ValueError("Positive denominator and chunk size required")
    labels = labels.to(hidden.device)
    groups = groups.to(hidden.device)
    head_device = head.weight.device
    positions = torch.where((groups[0] > 0) & (labels[0] != -100))[0]
    positions = positions[positions > 0]
    if not positions.numel():
        raise ValueError("No causally predictable target tokens")
    weight_table = torch.tensor(
        [0.0, 1.0, 0.75, 1.5, 2.0] if group_weights is None else group_weights,
        dtype=torch.float32,
        device=hidden.device,
    )

    def contribution(states, targets, weights):
        logits = head(states.to(head_device)).float()
        ce = functional.cross_entropy(logits, targets.to(head_device), reduction="none")
        return (ce * weights.to(head_device)).sum()

    total = torch.zeros((), dtype=torch.float32, device=head_device)
    for indices in positions.split(chunk_size):
        states = hidden[0, indices - 1]
        targets = labels[0, indices]
        weights = weight_table[groups[0, indices]]
        total = total + checkpoint(contribution, states, targets, weights, use_reentrant=False)
    return total * (scale / denominator)
