"""Verify that distributed final batches neither repeat nor drop training targets."""

import pytest

from workflow_distillation import weighted_sft_batches as batches


def test_uneven_final_batch_has_zero_weight_padding_and_exact_global_mean():
    slots = batches.distributed_slots([10, 11, 12, 13, 14], world_size=2)
    assert slots == [
        [(10, .4), (12, .4), (14, .4)],
        [(11, .4), (13, .4), (10, 0.)],
    ]
    # DDP averages ranks after summing their microstep contributions.
    values = {10: 1., 11: 2., 12: 3., 13: 4., 14: 5.}
    reduced = sum(sum(values[i]*scale for i, scale in rank) for rank in slots)/2
    assert reduced == pytest.approx(3.)


def test_final_batch_smaller_than_gpu_count_still_supervises_once():
    assert batches.distributed_slots([7], world_size=2) == [[(7, 2.)], [(7, 0.)]]


def test_resume_refuses_changed_data_or_loss_configuration():
    baseline = {"source_sha256": "abc", "weights": [1., .75, 1.5, 2.], "world_size": 2}
    batches.verify_resume(baseline, baseline.copy())
    changes = (("source_sha256", "different"), ("weights", [1., 1., 1.5, 2.]), ("world_size", 1))
    for key, changed in changes:
        candidate = {**baseline, key: changed}
        with pytest.raises(ValueError, match=key):
            batches.verify_resume(baseline, candidate)


def test_full_history_filter_retains_exact_records_and_reports_exclusions():
    records = [
        {"trajectory_id": "a", "length": 7, "offset": 0},
        {"trajectory_id": "b", "length": 8, "offset": 7},
        {"trajectory_id": "c", "length": 9, "offset": 15},
    ]
    kept, excluded = batches.filter_full_history(records, max_length=8)
    assert kept == records[:2]
    assert kept[0] is records[0]
    assert kept[1] is records[1]
    assert excluded == [{"trajectory_id": "c", "length": 9, "reason": "context_overflow"}]
    assert records[2]["length"] == 9
