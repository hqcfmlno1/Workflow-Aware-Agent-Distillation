"""Immutable harmonic key-span selection and corpus-wide council weighting."""

import math


def percentile_rank(values):
    """Ascending average ranks mapped to [0,1]; singleton rank is 1."""
    values = [float(v) for v in values]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Ranks require finite values")
    n = len(values)
    if n <= 1:
        return [1.0] * n
    ordered = sorted(range(n), key=values.__getitem__)
    result, i = [0.0] * n, 0
    while i < n:
        j = i + 1
        while j < n and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for index in ordered[i:j]:
            result[index] = ((i + j - 1) / 2) / (n - 1)
        i = j
    return result


def select_repulsion(webkd_weights, entropies, fraction=0.1):
    if len(webkd_weights) != len(entropies) or not 0 < fraction <= 1:
        raise ValueError("Aligned spans and a fraction in (0,1] required")
    if any(not 0.1 - 1e-7 <= v <= 1 for v in webkd_weights):
        raise ValueError("Expected WebKD span weights in [0.1,1]")
    if any(v < 0 for v in entropies):
        raise ValueError("Entropy cannot be negative")
    a, e = percentile_rank(webkd_weights), percentile_rank(entropies)
    scores = [2 * x * y / (x + y) if x + y else 0.0 for x, y in zip(a, e, strict=True)]
    count = math.ceil(fraction * len(scores))
    chosen = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count]
    return dict(action_rank=a, entropy_rank=e, score=scores, selected=sorted(chosen))


def council_weights(
    uncertainties,
    disagreements,
    *,
    lambda_u=0.5,
    lambda_d=0.5,
    epsilon=1e-3,
    standardization_epsilon=1e-8,
):
    """Fit population moments on ALL training reasoning spans, not per record."""
    if not uncertainties or len(uncertainties) != len(disagreements):
        raise ValueError("Nonempty aligned council statistics required")
    parameters = (lambda_u, lambda_d, epsilon, standardization_epsilon)
    if (
        any(not math.isfinite(v) for v in parameters)
        or min(lambda_u, lambda_d) < 0
        or lambda_u + lambda_d > 1
        or min(epsilon, standardization_epsilon) <= 0
    ):
        raise ValueError("Invalid council weighting coefficients")
    if any(not math.isfinite(v) or v < 0 for v in (*uncertainties, *disagreements)):
        raise ValueError("Invalid entropy/JS statistic")
    rho = [v / (u + epsilon) for u, v in zip(uncertainties, disagreements, strict=True)]

    def standardize(values):
        mean = math.fsum(values) / len(values)
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values))
        return [(v - mean) / (std + standardization_epsilon) for v in values], mean, std

    zu, mu, su = standardize(uncertainties)
    zr, mr, sr = standardize(rho)
    weights = [
        1 + lambda_u * math.tanh(u) + lambda_d * math.tanh(r) for u, r in zip(zu, zr, strict=True)
    ]
    return weights, dict(
        uncertainty_mean=mu,
        uncertainty_std=su,
        rho_mean=mr,
        rho_std=sr,
        reasoning_spans=len(rho),
        lambda_u=lambda_u,
        lambda_d=lambda_d,
        epsilon=epsilon,
        standardization_epsilon=standardization_epsilon,
    )
