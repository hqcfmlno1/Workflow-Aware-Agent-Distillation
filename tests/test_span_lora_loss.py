import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F


def api():
    from workflow_distillation.lora_fp16 import span_loss

    return span_loss


def fixture():
    torch.manual_seed(11)
    hidden = torch.randn(1, 8, 4, requires_grad=True)
    head = torch.nn.Linear(4, 11, bias=False)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    groups = torch.tensor([[0, 0, 1, 1, 2, 3, 4, 4]])
    labels = ids.masked_fill(groups == 0, -100)
    return hidden, head, ids, groups, labels


@pytest.mark.parametrize("beta", [0.0, 0.03])
def test_chunked_value_gradients_and_metrics_match_dense_objective(beta):
    m = api()
    hidden, head, ids, groups, labels = fixture()
    weights = np.array([0.1, 0.7, 1, 1, 1, 1], dtype="float32")
    q = np.array([-2.0, -3.0, -1.0, -4.0, -5.0, -1.0], dtype="float32")
    gamma = np.array([1, 1, 1e-6, 1e-6, 1e-6, 1], dtype="float32")
    actual, metrics = m.chunked_span_loss(hidden, head, labels, groups, weights, q, gamma, beta, 2)
    ce = F.cross_entropy(head(hidden)[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")[
        0, groups[0, 1:] > 0
    ]
    w, g = torch.tensor(weights), torch.tensor(gamma)
    d = (torch.tensor(q) + ce).clamp(-20, 8)
    expected = (ce * w).sum() / w.sum() + beta * ((torch.expm1(d) - d) * g).sum() / g.sum()
    torch.testing.assert_close(actual, expected)
    assert metrics["plain_ce"] == pytest.approx(float(ce.detach().mean()))
    assert metrics["group_counts"] == [0, 2, 1, 1, 2]
    ga = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    ge = torch.autograd.grad(expected, (hidden, head.weight))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e)


def test_reference_identity_and_no_reasoning_envelope_anchor():
    m = api()
    hidden, head, ids, groups, labels = fixture()
    ce = F.cross_entropy(head(hidden)[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")[
        0, groups[0, 1:] > 0
    ]
    q = -ce.detach().numpy()
    actual, metrics = m.chunked_span_loss(
        hidden, head, labels, groups, np.ones(6), q, [1, 1, 1e-6, 1e-6, 1e-6, 1], 0.03, 2
    )
    assert metrics["anchor_loss"] == pytest.approx(0, abs=1e-12)
    torch.testing.assert_close(actual, ce.mean())
    grad = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    expected = torch.autograd.grad(ce.mean(), (hidden, head.weight))
    for a, e in zip(grad, expected, strict=True):
        torch.testing.assert_close(a, e)
    groups = torch.where(groups == 1, 3, groups)
    _, metrics = m.chunked_span_loss(
        hidden, head, labels, groups, np.ones(6), q - 1, [1e-6] * 5 + [1], 0.03, 2
    )
    assert metrics["anchor_loss"] > 0
    _, metrics = m.chunked_span_loss(
        hidden, head, labels, groups, np.ones(6), q, np.zeros(6), 0.03, 2
    )
    assert metrics["anchor_loss"] == 0


def test_new_config_engine_accumulation_and_short_batch_keep_trajectory_mean():
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.lora_fp16.engine import train_update, trajectory_loss

    config = RunConfig(
        loss_normalization="offline_span_anchor",
        base_dtype="bfloat16",
        group_weights=(0, 1, 1, 1, 1),
        anchor_beta=0.03,
    )
    records = []
    for n in [6, 4, 3]:
        g = [0, 1, 1, 2, 3, 4][:n]
        records.append(
            dict(
                input_ids=list(range(1, n + 1)),
                groups=g,
                denominator=999,
                trajectory_id=str(n),
                weights=np.ones(n - 1),
                ref_logp=np.full(n - 1, -2.0),
                anchor_weights=np.array([1 if i == 1 else 1e-6 for i in g[1:]]),
            )
        )
    state = tiny_state()
    expected = copy.deepcopy(state)
    loss = sum(trajectory_loss(expected, r, config, torch.device("cpu")) for r in records) / len(
        records
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters, config.max_grad_norm)
    expected.optimizer.step()
    metrics = train_update(state, records, config, torch.device("cpu"))
    assert metrics["loss"] == pytest.approx(float(loss.detach()))
    assert "anchor_loss" in metrics and "group_nll" in metrics
    for a, b in zip(state.parameters, expected.parameters, strict=True):
        torch.testing.assert_close(a, b)


def test_extreme_reference_differences_are_finite():
    m = api()
    hidden, head, ids, groups, labels = fixture()
    value, metrics = m.chunked_span_loss(
        hidden, head, labels, groups, np.ones(6), [-1e8, 0, -1e8, 0, -1e8, 0], np.ones(6), 0.03, 2
    )
    assert torch.isfinite(value)
    value.backward()
    assert torch.isfinite(hidden.grad).all()
