"""Weighted teacher-token CE; imports PyTorch only in the remote runtime."""

from __future__ import annotations


def weighted_logits_loss(logits, labels, groups):
    """Causal, weighted-token CE per trajectory followed by trajectory mean."""
    import torch
    import torch.nn.functional as functional

    target = labels[:, 1:]
    group = groups[:, 1:]
    weights = torch.tensor([0.0, 1.0, 0.75, 1.5, 2.0], device=logits.device)
    valid = (target != -100) & (group > 0)
    coefficients = weights[group] * valid
    denominators = coefficients.sum(dim=-1)
    if not (denominators > 0).all():
        raise ValueError("Every trajectory requires supervised targets")
    ce = functional.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        target,
        reduction="none",
        ignore_index=-100,
    )
    return ((ce * coefficients).sum(dim=-1) / denominators).mean()


def chunked_hidden_loss(hidden, head, labels, groups, denominator, scale, chunk_size=128):
    """Avoid a full sequence-by-vocabulary tensor without changing target CE.

    The external denominator is trajectory-wide, including when input is a
    bounded window. The caller supplies the sample-probability reduction scale.
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
    weight_table = torch.tensor([0.0, 1.0, 0.75, 1.5, 2.0], device=hidden.device)

    def contribution(states, targets, weights):
        # Copy only a vocabulary-projection chunk, preserving autograd across
        # devices when the final decoder layer and tied head are separated.
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
