"""Full-history and placement contracts for the remote QLoRA probe."""

import pytest

from workflow_distillation import optimized_qlora as probe


def test_two_gpu_plan_covers_every_layer_and_keeps_tied_weights_together():
    mapping = probe.device_plan(num_layers=32, split_at=12)
    assert mapping["model.visual"] == "cpu"
    assert mapping["model.language_model.embed_tokens"] == mapping["lm_head"] == 0
    assert [mapping[f"model.language_model.layers.{i}"] for i in range(32)] == [0]*12 + [1]*20
    assert mapping["model.language_model.norm"] == 1


def test_single_gpu_plan_still_offloads_vision():
    mapping = probe.device_plan(num_layers=32, split_at=None)
    assert mapping["model.visual"] == "cpu"
    assert set(v for k, v in mapping.items() if k != "model.visual") == {0}


def test_probe_rejects_truncated_or_misaligned_full_trajectories():
    record = {"trajectory_id": "a", "input_ids": [7, 8, 9, 10],
              "groups": [0, 1, 3, 4], "full_token_count": 4}
    assert probe.validate_records([record]) is None
    with pytest.raises(ValueError, match="full"):
        probe.validate_records([{**record, "full_token_count": 5}])
    with pytest.raises(ValueError, match="align"):
        probe.validate_records([{**record, "groups": [0, 1, 3]}])
    with pytest.raises(ValueError, match="predecessor"):
        probe.validate_records([{**record, "groups": [1, 1, 3, 4]}])


def test_invalid_model_partition_is_rejected():
    for split in (0, 32, 33, -1):
        with pytest.raises(ValueError, match="split"):
            probe.device_plan(num_layers=32, split_at=split)
