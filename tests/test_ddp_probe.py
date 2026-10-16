"""Prevent an accidental extra batch, truncated sample, or split-model DDP probe."""

import pytest

from workflow_distillation import ddp_probe as probe


def records():
    return [
        {"trajectory_id": "a", "task_key": "repo#1", "partition": "train_pool",
         "resolved": 1, "full_token_count": 4, "input_ids": [1, 2, 3, 4],
         "groups": [0, 1, 3, 4]},
        {"trajectory_id": "b", "task_key": "repo#2", "partition": "train_pool",
         "resolved": 1, "full_token_count": 3, "input_ids": [1, 5, 6],
         "groups": [0, 3, 4]},
    ]


def test_one_distinct_complete_training_record_is_assigned_to_each_rank():
    batch = records()
    assert probe.rank_record(batch, 0) is batch[0]
    assert probe.rank_record(batch, 1) is batch[1]
    for invalid in (batch[:1], batch + [batch[0]]):
        with pytest.raises(ValueError, match="exactly two"):
            probe.rank_record(invalid, 0)
    with pytest.raises(ValueError, match="rank"):
        probe.rank_record(batch, 2)


@pytest.mark.parametrize("change", [
    {"partition": "validation"}, {"resolved": 0}, {"task_key": "repo#1"},
    {"full_token_count": 30000}, {"groups": [0, 3]},
])
def test_invalid_or_heldout_or_truncated_samples_abort_before_gpu_loading(change):
    batch = records()
    batch[1].update(change)
    with pytest.raises(ValueError):
        probe.rank_record(batch, 0)


def test_cap_is_strictly_below_30000_and_replicas_stay_on_their_own_gpu():
    batch = records()
    batch[1].update(input_ids=[7] * 30000, groups=[0] + [3] * 29999,
                    full_token_count=30000)
    with pytest.raises(ValueError, match="30000"):
        probe.rank_record(batch, 1)
    for rank in (0, 1):
        mapping = probe.replica_device_plan(rank)
        assert mapping["model.visual"] == "cpu"
        assert mapping["lm_head"] == mapping["model.language_model.embed_tokens"] == rank
        assert all(mapping[f"model.language_model.layers.{i}"] == rank for i in range(32))


def test_different_gradients_or_adapter_updates_cannot_be_reported_as_ddp_success():
    evidence = [
        {"gradient_sha256": "g", "adapter_sha256": "p", "adapter_max_delta": 0.1},
        {"gradient_sha256": "g", "adapter_sha256": "p", "adapter_max_delta": 0.1},
    ]
    probe.verify_replicas(evidence)
    for field, value in (("gradient_sha256", "wrong"), ("adapter_sha256", "wrong"),
                         ("adapter_max_delta", 0.0), ("adapter_max_delta", float("nan"))):
        broken = [dict(r) for r in evidence]
        broken[1][field] = value
        with pytest.raises(ValueError):
            probe.verify_replicas(broken)
