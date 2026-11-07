import copy

import numpy as np
import pytest


def fixture():
    original = dict(
        trajectory_id="a", input_ids=[5, 6, 7, 8, 9, 10, 11], groups=[0, 1, 1, 0, 2, 3, 4]
    )
    masked = dict(original, groups=[0, 0, 0, 0, 2, 3, 4])
    arrays = dict(
        ref_logp=np.array([-1, -2, -3, -4, -5], dtype="float32"),
        weights=np.array([0.1, 0.8, 1, 1, 1], dtype="float32"),
        anchor_weights=np.array([1, 1, 1e-6, 1e-6, 1], dtype="float32"),
    )
    return original, masked, arrays


def test_projection_keeps_absolute_target_alignment_and_context():
    from workflow_distillation.lora_fp16.span_step_mask import project_cached_record

    original, masked, arrays = fixture()
    before = copy.deepcopy((original, masked, arrays))
    projected = project_cached_record(original, masked, arrays)
    for name in arrays:
        np.testing.assert_array_equal(projected[name], arrays[name][2:])
        np.testing.assert_array_equal(arrays[name], before[2][name])
    assert original == before[0] and masked == before[1]


@pytest.mark.parametrize("change", ["ids", "group", "unmask", "identity", "empty"])
def test_projection_rejects_any_change_beyond_removing_target_groups(change):
    from workflow_distillation.lora_fp16.span_step_mask import project_cached_record

    original, masked, arrays = fixture()
    if change == "ids":
        masked["input_ids"] = [5, 6, 7, 0, 9, 10, 11]
    elif change == "group":
        masked["groups"] = [0, 0, 0, 0, 1, 3, 4]
    elif change == "unmask":
        masked["groups"] = [0, 0, 0, 1, 2, 3, 4]
    elif change == "identity":
        masked["trajectory_id"] = "b"
    else:
        masked["groups"] = [0] * 7
    with pytest.raises(ValueError):
        project_cached_record(original, masked, arrays)


def test_masked_span_objective_excludes_both_terms_and_their_denominators():
    import torch
    import torch.nn.functional as F

    from workflow_distillation.lora_fp16.span_loss import chunked_span_loss
    from workflow_distillation.lora_fp16.span_step_mask import project_cached_record

    original, masked, arrays = fixture()
    projected = project_cached_record(original, masked, arrays)
    torch.manual_seed(8)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    head = torch.nn.Linear(4, 13, bias=False)
    ids = torch.tensor([masked["input_ids"]])
    groups = torch.tensor([masked["groups"]])
    labels = ids.masked_fill(groups == 0, -100)
    actual, metrics = chunked_span_loss(
        hidden, head, labels, groups, projected["weights"], projected["ref_logp"],
        projected["anchor_weights"], chunk_size=2,
    )
    ce = F.cross_entropy(head(hidden[0, [3, 4, 5]]), ids[0, [4, 5, 6]], reduction="none")
    w = torch.tensor([1., 1., 1.])
    gamma = torch.tensor([1e-6, 1e-6, 1.])
    d = (torch.tensor([-3., -4., -5.]) + ce).clamp(-20, 8)
    expected = (w * ce).sum() / w.sum()
    expected += 0.03 * (gamma * (torch.expm1(d) - d)).sum() / gamma.sum()
    torch.testing.assert_close(actual, expected)
    assert metrics["ce_weight_sum"] == pytest.approx(3)
    assert metrics["anchor_weight_sum"] == pytest.approx(1.000002)
    actual_grad = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (hidden, head.weight))
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("skip", [True, False])
def test_explicit_pilot_skip_reaches_model_load_without_creating_fake_report(
    tmp_path, monkeypatch, skip
):
    from types import SimpleNamespace

    import torch

    from workflow_distillation.lora_fp16 import cli, model
    from workflow_distillation.lora_fp16.config import RunConfig

    class ModelLoadReached(Exception):
        pass

    def stop_at_model(config):
        raise ModelLoadReached

    monkeypatch.setattr(cli, "gpu_audit", lambda c: {"device": "test"})
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(model, "build_model", stop_at_model)
    args = SimpleNamespace(
        output=tmp_path, mode="train", pilot_report=None, skip_pilot=skip,
        resume=None, max_seconds=28800, hub_repo=None,
    )
    error = ModelLoadReached if skip else FileNotFoundError
    with pytest.raises(error):
        cli.execute(args, RunConfig(), None, [], [], [], {}, [], {})
    assert not (tmp_path / "pilot.json").exists()
