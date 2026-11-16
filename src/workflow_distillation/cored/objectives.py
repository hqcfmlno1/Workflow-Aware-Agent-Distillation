"""Exact full-vocabulary losses; only chunk-size x vocabulary logits are live."""

import torch
from torch.utils.checkpoint import checkpoint


def _project(head, hidden):
    # Frozen output head; keep matmul in its configured dtype, reductions in FP32.
    return head(hidden.to(device=head.weight.device, dtype=head.weight.dtype)).float()


def segment_means(record, values):
    index = {p: i for i, p in enumerate(record.positions)}
    output = []
    for s in record.segments:
        rows = torch.tensor([index[p] for p in s.positions], device=values.device)
        fractions = values.new_tensor(s.fractions)
        output.append((values[rows] * fractions).sum() / fractions.sum())
    return torch.stack(output)


def token_coefficients(record, *, phase, keep=None, device=None):
    index = {p: i for i, p in enumerate(record.positions)}
    coeff = torch.zeros(len(index), device=device, dtype=torch.float32)
    denominator = 0.0
    for s in record.segments:
        w = s.webkd_weight
        fractions = coeff.new_tensor(s.fractions)
        if phase == 1:
            if s.kind == "reasoning":
                if keep is None or s.dropout_group not in keep:
                    raise ValueError("Explicit reasoning-step dropout masks required")
                w *= bool(keep[s.dropout_group])
            denominator += w * sum(s.fractions)
        elif phase == 2:
            if s.kind == "reasoning":
                if s.key not in record.council_weights:
                    raise ValueError("Missing frozen Phase 2 council weight")
                w *= record.council_weights[s.key]
            denominator += w
            fractions = fractions / fractions.sum()
        else:
            raise ValueError("Expected Phase 1 or 2")
        rows = torch.tensor([index[p] for p in s.positions], device=device)
        coeff.index_add_(0, rows, w * fractions)
    if denominator <= 0:
        raise ValueError("No supervised mass after dropout")
    return coeff / denominator


def _nll_kl(hidden, head, record, coefficients, reference, kl_coefficient, chunk_size):
    positions = record.positions
    if reference is not None and (
        reference.requires_grad or reference.shape != (len(positions), hidden.shape[-1])
    ):
        raise ValueError("Reference must be detached and aligned to supervised rows")
    if head.weight.requires_grad:
        raise ValueError("Output head must remain frozen")
    sums = hidden.new_zeros(2, dtype=torch.float32, device=head.weight.device)
    for start in range(0, len(positions), chunk_size):
        ps = positions[start : start + chunk_size]
        states = hidden[torch.tensor(ps, device=hidden.device) - 1]
        targets = torch.tensor([record.input_ids[p] for p in ps], device=head.weight.device)
        weights = coefficients[start : start + len(ps)].to(head.weight.device)
        ref = reference[start : start + len(ps)] if reference is not None else states.detach()

        def contribution(h, r, y, w):
            logp = _project(head, h).log_softmax(-1)
            nll = -(logp.gather(1, y[:, None]).squeeze(1) * w).sum()
            if reference is None:
                kl = nll * 0
            else:
                with torch.no_grad():
                    logq = _project(head, r).log_softmax(-1)
                kl = ((logq.exp() * (logq - logp)).sum(-1) * w).sum()
            return torch.stack((nll, kl))

        sums = sums + checkpoint(contribution, states, ref, targets, weights, use_reentrant=False)
    total = sums[0] + kl_coefficient * sums[1]
    return total, dict(nll=float(sums[0].detach()), kl=float(sums[1].detach()))


def functional_repulsion(own, peers, head, *, gamma=1.0, chunk_size=16):
    """One directed expert's contribution; distance is a SUM, not a token mean."""
    if gamma <= 0 or chunk_size < 1:
        raise ValueError("Positive gamma/chunk required")
    if not peers or own.shape[0] == 0:
        return own.sum() * 0
    if any(p.shape != own.shape for p in peers):
        raise ValueError("Aligned key-position hidden states required")
    distances = torch.zeros(len(peers), device=head.weight.device, dtype=torch.float32)
    for start in range(0, len(own), chunk_size):
        h = own[start : start + chunk_size]
        # Snapshot inputs, not late-bound loop variables: checkpoint recomputes later.
        other = tuple(p[start : start + chunk_size].detach() for p in peers)

        def distance(states, *references):
            logp = _project(head, states).log_softmax(-1)
            terms = []
            for r in references:
                with torch.no_grad():
                    logq = _project(head, r).log_softmax(-1)
                terms.append((logp - logq).square().sum())
            return torch.stack(terms)

        distances = distances + checkpoint(distance, h, *other, use_reentrant=False)
    return (-gamma * distances).exp().sum() / len(peers)


def phase1_loss(hidden, head, record, keep, peers, config):
    coefficients = token_coefficients(record, phase=1, keep=keep, device=hidden.device)
    loss, metrics = _nll_kl(hidden, head, record, coefficients, None, 0, config.projection_chunk)
    key = torch.tensor(record.key_positions, device=hidden.device, dtype=torch.long)
    rep = functional_repulsion(
        hidden[key - 1],
        peers,
        head,
        gamma=config.repulsion_gamma,
        chunk_size=config.projection_chunk,
    )
    metrics["repulsion"] = float(rep.detach())
    return loss + config.repulsion_coefficient * rep, metrics


def phase2_loss(hidden, head, record, reference, config):
    coefficients = token_coefficients(record, phase=2, device=hidden.device)
    return _nll_kl(
        hidden,
        head,
        record,
        coefficients,
        reference,
        config.kl_coefficient,
        config.projection_chunk,
    )


def orthogonality(experts, epsilon=1e-8):
    if len(experts) < 2 or any(set(e) != set(experts[0]) for e in experts):
        raise ValueError("Aligned LoRA layers from >=2 experts required")
    first = next(iter(experts[0].values()))[0]
    total = first.new_zeros((), dtype=torch.float32)
    for m, left in enumerate(experts):
        for n, right in enumerate(experts):
            if m == n:
                continue
            for key in left:
                a, b = [v.float() for v in left[key]]
                c, d = [v.float() for v in right[key]]
                total = total + (a @ c.T).square().sum() / (
                    a.square().sum().clamp_min(epsilon) * c.square().sum().clamp_min(epsilon)
                )
                total = total + (b.T @ d).square().sum() / (
                    b.square().sum().clamp_min(epsilon) * d.square().sum().clamp_min(epsilon)
                )
    return total


@torch.no_grad()
def council_statistics(states, head, chunk_size=16):
    if not states or chunk_size < 1 or any(h.shape != states[0].shape for h in states):
        raise ValueError("Aligned council hidden states and positive chunk required")
    uncertainties, disagreements = [], []
    for start in range(0, len(states[0]), chunk_size):
        mean_p = None
        entropy_sum = None
        for hidden in states:
            logp = _project(head, hidden[start : start + chunk_size]).log_softmax(-1)
            p = logp.exp()
            entropy = -(p * logp).sum(-1)
            mean_p = p if mean_p is None else mean_p + p
            entropy_sum = entropy if entropy_sum is None else entropy_sum + entropy
        mean_p = mean_p / len(states)
        u = entropy_sum / len(states)
        mixture_entropy = -torch.special.xlogy(mean_p, mean_p).sum(-1)
        v = mixture_entropy - u
        if bool((v < -1e-5).any()) or not bool(torch.isfinite(u).all()):
            raise FloatingPointError("Invalid full-vocabulary council statistics")
        uncertainties.append(u.cpu())
        disagreements.append(v.clamp_min(0).cpu())
    if not uncertainties:
        return torch.empty(0), torch.empty(0)
    return torch.cat(uncertainties), torch.cat(disagreements)
