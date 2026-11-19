"""Adaptive token CE plus cached-reference, role-asymmetric target-token anchor."""

from __future__ import annotations

import math

import numpy as np


def chunked_span_loss(
    hidden, head, labels, groups, weights, ref_logp, anchor_weights, beta=0.03, chunk_size=128
):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    if (
        hidden.shape[0] != 1
        or labels.shape != groups.shape
        or labels.shape[1] != hidden.shape[1]
        or chunk_size < 1
        or not math.isfinite(beta)
        or beta < 0
    ):
        raise ValueError("Expected one aligned trajectory and valid beta/chunk size")
    groups, labels = groups.to(hidden.device), labels.to(hidden.device)
    positions = torch.where((groups[0] > 0) & (labels[0] != -100))[0]
    positions = positions[positions > 0]
    count = len(positions)
    arrays = [np.asarray(v, dtype="float32") for v in (weights, ref_logp, anchor_weights)]
    if (
        not count
        or any(v.shape != (count,) or not np.isfinite(v).all() for v in arrays)
        or np.any(arrays[0] < 0)
        or arrays[0].sum() <= 0
        or np.any(arrays[2] < 0)
    ):
        raise ValueError("Invalid aligned token weights/reference/anchor arrays")
    w, q, gamma = [torch.tensor(a, device=hidden.device, dtype=torch.float32) for a in arrays]
    head_device = head.weight.device

    def contribution(states, targets, local_w, local_q, local_gamma, local_groups):
        logits = head(states.to(head_device)).float()
        ce = F.cross_entropy(logits, targets.to(head_device), reduction="none")
        local_w, local_q, local_gamma, local_groups = [
            a.to(head_device) for a in (local_w, local_q, local_gamma, local_groups)
        ]
        # Paper implementation's numerical clamp; independent of gradient norm clipping.
        d = (local_q + ce).clamp(-20.0, 8.0)
        k3 = torch.expm1(d) - d
        return torch.stack(
            [
                (local_w * ce).sum(),
                ce.detach().sum(),
                (local_gamma * k3).sum(),
                *[(ce.detach() * (local_groups == g)).sum() for g in range(1, 5)],
            ]
        )

    sums = torch.zeros(7, device=head_device, dtype=torch.float32)
    for start in range(0, count, chunk_size):
        local = positions[start : start + chunk_size]
        sums = sums + checkpoint(
            contribution,
            hidden[0, local - 1],
            labels[0, local],
            w[start : start + chunk_size],
            q[start : start + chunk_size],
            gamma[start : start + chunk_size],
            groups[0, local],
            use_reentrant=False,
        )
    denominator = w.sum().to(head_device)
    gamma_sum = gamma.sum().to(head_device)
    ce_loss = sums[0] / denominator
    anchor = sums[2] / gamma_sum if float(gamma_sum.detach()) > 0 else sums[2] * 0
    loss = ce_loss + beta * anchor
    stats = sums.detach().cpu().tolist()
    target_groups = groups[0, positions].detach().cpu().numpy()
    counts = np.bincount(target_groups, minlength=5).tolist()
    weight_mass = [0.0] + [
        float(arrays[0][target_groups == g].sum(dtype=np.float64)) for g in range(1, 5)
    ]
    metrics = dict(
        weighted_ce=float(ce_loss.detach()),
        plain_ce=stats[1] / count,
        anchor_loss=float(anchor.detach()),
        anchor_scaled=beta * float(anchor.detach()),
        total_loss=float(loss.detach()),
        group_ce_sums=[0.0] + stats[3:],
        group_counts=counts,
        group_weight_sums=weight_mass,
        ce_weight_sum=float(denominator.detach()),
        anchor_weight_sum=float(gamma_sum.detach()),
    )
    return loss, metrics
