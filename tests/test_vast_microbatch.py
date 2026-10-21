"""Batch-size migration preserves consumed data and equal trajectory weighting."""

import copy

import pytest


def test_schedule_keeps_consumed_prefix_and_only_batches_remaining_samples_by_four():
    from workflow_distillation.vast_microbatch import batch_schedule

    schedule = batch_schedule(
        4000, {"global_step": 700, "next_batch": 700, "samples_consumed": 1400}
    )
    assert len(schedule) == 1350
    assert schedule[699] == [1398, 1399]
    assert schedule[700] == [1400, 1401, 1402, 1403]
    assert schedule[-1] == [3996, 3997, 3998, 3999]
    assert [i for batch in schedule for i in batch] == list(range(4000))
    tail = batch_schedule(14, {"global_step": 2, "next_batch": 2, "samples_consumed": 4})
    assert tail == [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13]]


@pytest.mark.parametrize(
    "progress",
    [
        {"global_step": 2, "next_batch": 1, "samples_consumed": 4},
        {"global_step": 2, "next_batch": 2, "samples_consumed": 5},
        {"global_step": 9, "next_batch": 9, "samples_consumed": 18},
    ],
)
def test_schedule_rejects_invalid_parent_cursor(progress):
    from workflow_distillation.vast_microbatch import batch_schedule

    with pytest.raises(ValueError):
        batch_schedule(14, progress)


def test_padded_batch_loss_matches_equal_trajectory_mean_and_gradients():
    import torch

    from workflow_distillation.vast_microbatch import batch_hidden_loss
    from workflow_distillation.weighted_sft_loss import weighted_logits_loss

    torch.manual_seed(32)
    hidden = torch.randn(2, 7, 3, requires_grad=True)
    head = torch.nn.Linear(3, 9, bias=False)
    labels = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 7, 8, 1, -100, -100, -100]])
    groups = torch.tensor([[0, 1, 2, 3, 4, 1, 3], [0, 2, 1, 4, 0, 0, 0]])
    weights = torch.tensor([0.0, 1.0, 0.75, 1.5, 2.0])
    denominators = [float(weights[g].sum()) for g in groups]
    actual = batch_hidden_loss(hidden, head, labels, groups, denominators, [0.5, 0.5])
    expected = weighted_logits_loss(head(hidden), labels, groups)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    b = torch.autograd.grad(expected, (hidden, head.weight))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)


def test_transition_identity_changes_batch_size_and_pins_parent(tmp_path):
    from workflow_distillation.vast_microbatch import transition_identity

    parent = {
        "complete": True,
        "world_size": 2,
        "identity": {
            "script_sha256": "old",
            "settings": {
                "config": {
                    "per_gpu_batch_size": 1,
                    "global_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "rank": 32,
                }
            },
        },
        "progress": {"global_step": 2, "next_batch": 2, "samples_consumed": 4},
    }
    expected = copy.deepcopy(parent["identity"])
    script = tmp_path / "launch.py"
    script.write_text("# launcher\n")
    child = transition_identity(parent, expected, "pinned-sha", script)
    assert parent["identity"] == expected
    assert child["settings"]["config"]["global_batch_size"] == 4
    assert child["settings"]["config"]["per_gpu_batch_size"] == 2
    assert child["settings"]["config"]["rank"] == 32
    assert child["settings"]["microbatch_transition"]["parent_manifest_sha256"] == "pinned-sha"
    parent["identity"]["settings"]["config"]["rank"] = 16
    with pytest.raises(ValueError, match="identity"):
        transition_identity(parent, expected, "pinned-sha", script)


def test_exact_parent_restores_state_and_child_resume_is_strict(tmp_path):
    import torch
    from test_filtered_continuation import make_training, update

    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.training_checkpoints import CheckpointManager, _manifest
    from workflow_distillation.vast_microbatch import migration_manager

    old = {"stage": "batch-two"}
    new = {"stage": "batch-four"}
    source = make_training()
    update(source, [2, 3])
    p = CheckpointManager(tmp_path / "parent", old).save(
        model=source[0],
        optimizer=source[1],
        scaler=source[2],
        scheduler=None,
        progress={"global_step": 2, "next_batch": 2, "samples_consumed": 4},
    )
    binding = {
        "parent": {
            "manifest_sha256": sha256_file(p / "manifest.json"),
            **{k: _manifest(p)[k] for k in ("identity", "progress", "world_size")},
        }
    }
    cls = migration_manager(binding)
    restored = make_training()
    before_rng = torch.load(p / "rank-00000.pt", weights_only=False)["rng"]["torch"]
    mgr = cls(tmp_path / "child", new)
    mgr.load(p, model=restored[0], optimizer=restored[1], scaler=restored[2], scheduler=None)
    assert mgr.identity == new
    assert torch.equal(torch.get_rng_state(), before_rng)
    for a, b in zip(source[0].parameters(), restored[0].parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert source[2].state_dict() == restored[2].state_dict()
    for key, state in source[1].state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, restored[1].state_dict()["state"][key][field], rtol=0, atol=0
            )
    update(restored, [4, 5])
    saved = mgr.save(
        model=restored[0],
        optimizer=restored[1],
        scaler=restored[2],
        scheduler=None,
        progress={"global_step": 3, "next_batch": 3, "samples_consumed": 8},
    )
    mgr.load(saved, model=restored[0], optimizer=restored[1], scaler=restored[2], scheduler=None)
    wrong = CheckpointManager(tmp_path / "wrong", {"stage": "other"}).save(
        model=restored[0],
        optimizer=restored[1],
        scaler=restored[2],
        scheduler=None,
        progress={"global_step": 4, "next_batch": 4, "samples_consumed": 10},
    )
    with pytest.raises(ValueError, match="identity"):
        mgr.load(
            wrong, model=restored[0], optimizer=restored[1], scaler=restored[2], scheduler=None
        )
