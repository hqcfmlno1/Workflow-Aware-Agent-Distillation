import copy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_profiles_are_complete_and_keep_experiment_math_separate():
    from workflow_distillation.training.config import Experiment, load_profile

    for name, batch, mode in [
        ("t4_16gb", 1, "cpu"),
        ("rtx3090_24gb", 2, "cpu"),
        ("rtx4080_32gb", 2, "hybrid"),
    ]:
        profile = load_profile(ROOT / f"configs/training/gpus/{name}/profile.json")
        experiment = Experiment()
        config = experiment.effective_config(profile)
        assert config["per_gpu_batch_size"] == batch
        assert config["global_batch_size"] == 2 * batch
        assert config["rank"] == 32
        assert profile.activation_storage == mode
        assert profile.attention == "t4_repeated_kv"
    assert (
        load_profile(ROOT / "configs/training/gpus/rtx4080_32gb/profile.json").gpu_budget_bytes
        == 4 * 2**30
    )


def test_profiles_fail_closed_on_unsupported_or_ambiguous_settings():
    from dataclasses import replace

    from workflow_distillation.training.config import GPUProfile

    profile = GPUProfile("test", "NVIDIA GeForce RTX 3090", 22 * 2**30, 2, "cpu", 0)
    for changes in (
        {"attention": "flash_repeated_kv"},
        {"gpu_budget_bytes": -1},
        {"per_gpu_batch_size": 3},
        {"activation_storage": "automatic"},
        {"activation_storage": "cpu", "gpu_budget_bytes": 4096},
        {"per_gpu_batch_size": True},
    ):
        with pytest.raises(ValueError):
            replace(profile, **changes)
    with pytest.raises(ValueError, match="GPU"):
        profile.validate_devices([{"name": profile.device_name, "total_memory": 16 * 2**30}] * 2)


def test_batch_schedule_matches_legacy_including_partial_tail():
    from workflow_distillation.training.data import batch_schedule
    from workflow_distillation.vast_microbatch import batch_schedule as legacy

    transition = {"global_step": 710, "next_batch": 710, "samples_consumed": 1420}
    actual = batch_schedule(9966, 2, transition)
    assert actual == legacy(9966, transition)
    assert len(actual) == 2847
    assert actual[-1] == [9964, 9965]
    assert batch_schedule(5, 1) == [[0, 1], [2, 3], [4]]
    with pytest.raises(ValueError):
        batch_schedule(9966, 1, transition)


def test_approved_pool_is_rebuilt_and_tampering_is_rejected(tmp_path):
    import json

    from test_diverse_continuation import source_files

    from workflow_distillation.diverse_continuation import build_plan
    from workflow_distillation.training.data import read_pool_plan
    from workflow_distillation.vast_pool_continuation import build_pool_plan

    index, source = source_files(tmp_path)
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(build_plan(index, source, resume_step=1, target_step=3)))
    pool = tmp_path / "pool.json"
    plan = build_pool_plan(index, source, cohort)
    pool.write_text(json.dumps(plan))
    assert read_pool_plan(index, source, cohort, pool) == plan
    tampered = copy.deepcopy(plan)
    tampered["ordered_indices"][-2:] = tampered["ordered_indices"][-2:][::-1]
    pool.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="Pool plan"):
        read_pool_plan(index, source, cohort, pool)
