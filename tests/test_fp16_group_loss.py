import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.engine import trajectory_loss

GROUP_WEIGHTS = (0, 0.15, 0.05, 1.0, 0.5)


def dense_reference(logits, ids, groups, weights, alpha):
    losses, masses = [], []
    ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")
    for group in range(1, 5):
        mask = groups[:, 1:] == group
        count = int(mask.sum())
        if count and weights[group]:
            mass = weights[group] * count ** (1 - alpha)
            losses.append(ce[mask].mean() * mass)
            masses.append(mass)
    return sum(losses) / sum(masses)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("weights", [GROUP_WEIGHTS, (0, 0, 0, 1, 0.5)])
def test_chunked_group_loss_matches_dense_value_and_gradients(alpha, weights):
    from workflow_distillation.lora_fp16.loss import chunked_hidden_loss, group_loss_parameters

    torch.manual_seed(17)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 2]])
    groups = torch.tensor([[4, 0, 1, 1, 1, 2, 3, 3, 4]])
    hidden = torch.randn(1, 9, 5, requires_grad=True)
    head = torch.nn.Linear(5, 11, bias=False)
    labels = ids.masked_fill(groups == 0, -100)
    coefficients, denominator = group_loss_parameters(groups[0].numpy(), weights, alpha)
    actual = chunked_hidden_loss(
        hidden, head, labels, groups, denominator, 1.0,
        chunk_size=2, group_weights=coefficients,
    )
    expected = dense_reference(head(hidden).float(), ids, groups, weights, alpha)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, head.weight))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b)


def test_group_normalization_uses_one_shared_denominator_and_omits_empty_groups():
    from workflow_distillation.lora_fp16.loss import group_loss_parameters

    weights, denominator = group_loss_parameters([0, 1, 1, 1, 1, 3, 4], GROUP_WEIGHTS, 0.5)
    assert denominator == pytest.approx(0.15 * 2 + 1 + 0.5)
    assert weights == pytest.approx([0, 0.15 / 2, 0, 1, 0.5])
    assert weights[4] != weights[3]
    with pytest.raises(ValueError, match="supervised"):
        group_loss_parameters([0, 0], GROUP_WEIGHTS, 0.5)


def test_engine_group_loss_ignores_stored_legacy_denominator():
    from test_fp16_training import tiny_state

    config = RunConfig(
        loss_normalization="token_group", length_alpha=0.5, group_weights=GROUP_WEIGHTS
    )
    record = dict(
        input_ids=[1, 2, 3, 4, 5, 6], groups=[0, 1, 1, 2, 3, 4], denominator=999999,
    )
    state = tiny_state()
    ids = torch.tensor([record["input_ids"]])
    groups = torch.tensor([record["groups"]])
    expected = dense_reference(
        state.head(state.decoder(input_ids=ids).last_hidden_state),
        ids, groups, GROUP_WEIGHTS, 0.5,
    )
    actual = trajectory_loss(state, record, config, torch.device("cpu"))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in state.parameters)
    # Legacy data normalization must not affect either train or validation objective.
    changed = trajectory_loss(state, dict(record, denominator=1), config, torch.device("cpu"))
    torch.testing.assert_close(changed, actual)


@pytest.mark.parametrize("kwargs", [
    {"length_alpha": -0.1}, {"length_alpha": 1.1}, {"length_alpha": float("nan")},
    {"group_weights": [0, 0.15, 0.05, 1]},
    {"group_weights": [0, -1, 0.05, 1, 0.5]},
    {"group_weights": [0, float("inf"), 0.05, 1, 0.5]},
    {"loss_normalization": "segment"},
])
def test_rejects_invalid_group_objective(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


def test_approved_profile_and_checkpoint_identity_are_explicit():
    path = Path("configs/training/gpus/rtx5090_32gb/group_loss.json")
    config = RunConfig(**json.loads(path.read_text()))
    assert config.loss_normalization == "token_group"
    assert config.length_alpha == 0.5
    assert config.group_weights == GROUP_WEIGHTS
    assert config.to_dict() != replace(config, length_alpha=1).to_dict()
    assert config.to_dict() != replace(config, group_weights=(0, 0.2, 0.05, 1, 0.5)).to_dict()


def test_objective_identity_survives_pilot_json_roundtrip():
    from workflow_distillation.lora_fp16.cli import validate_pilot

    config = RunConfig(
        loss_normalization="token_group", length_alpha=0.5, group_weights=GROUP_WEIGHTS
    )
    identity = {"settings": {"config": config.to_dict()}}
    hardware = {"name": "RTX5090"}
    report = json.loads(json.dumps(dict(passed=True, identity=identity, device=hardware)))
    validate_pilot(report, identity, hardware)


def test_accumulated_group_objective_is_uniform_trajectory_mean():
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.engine import train_update

    config = RunConfig(
        loss_normalization="token_group", length_alpha=0.5, group_weights=GROUP_WEIGHTS
    )
    records = [
        dict(input_ids=[1, 2, 3, 4, 5, 6][:n], groups=[0, 1, 1, 2, 3, 4][:n],
             denominator=123, trajectory_id=str(n))
        for n in [6, 5, 3]
    ]
    actual = tiny_state()
    expected = copy.deepcopy(actual)
    losses = []
    for r in records:
        ids = torch.tensor([r["input_ids"]])
        groups = torch.tensor([r["groups"]])
        logits = expected.head(expected.decoder(input_ids=ids).last_hidden_state)
        losses.append(dense_reference(logits, ids, groups, GROUP_WEIGHTS, 0.5))
    mean = sum(losses) / len(losses)
    mean.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters, config.max_grad_norm)
    expected.optimizer.step()
    result = train_update(actual, records, config, torch.device("cpu"))
    assert result["loss"] == pytest.approx(mean.item(), rel=1e-6)
    for a, b in zip(actual.parameters, expected.parameters, strict=True):
        torch.testing.assert_close(a, b)
