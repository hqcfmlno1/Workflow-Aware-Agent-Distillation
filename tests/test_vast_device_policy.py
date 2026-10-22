import pytest


def test_accepts_two_4080_cards_with_actual_32gb_memory():
    from workflow_distillation.vast_device_policy import validate_4080_pair

    devices = [{"name": "NVIDIA GeForce RTX 4080", "total_memory": 32760 * 2**20}] * 2
    assert validate_4080_pair(devices) == devices


@pytest.mark.parametrize("devices", [
    [{"name": "NVIDIA GeForce RTX 4080", "total_memory": 16 * 2**30}] * 2,
    [{"name": "NVIDIA GeForce RTX 4080", "total_memory": 32 * 2**30}],
    [{"name": "NVIDIA GeForce RTX 3090", "total_memory": 32 * 2**30}] * 2,
])
def test_rejects_insufficient_memory_wrong_model_or_wrong_rank_count(devices):
    from workflow_distillation.vast_device_policy import validate_4080_pair

    with pytest.raises(ValueError):
        validate_4080_pair(devices)


def test_boundary_logger_stops_and_saves_after_the_requested_completed_update(tmp_path):
    from workflow_distillation.epoch_training import run_epoch_loop
    from workflow_distillation.vast_device_policy import boundary_logger

    events, saved = [], []
    stop_file = tmp_path / "STOP"
    result = run_epoch_loop(
        [[i] for i in range(10)],
        progress={"global_step": 3, "next_batch": 3, "samples_consumed": 3},
        step=lambda batch, number: {"loss": 1.0},
        save=lambda progress: saved.append(dict(progress)),
        stop=stop_file.exists,
        emit=boundary_logger(events.append, stop_file, 7),
        save_every=10,
    )
    assert [e["global_step"] for e in events] == [4, 5, 6, 7]
    assert saved == [{"global_step": 7, "next_batch": 7, "samples_consumed": 7}]
    assert result["status"] == "stopped_checkpointed"


def test_cgroup_v1_memory_is_read_without_disabling_the_worker_safety_check(tmp_path):
    from workflow_distillation.vast_device_policy import compatible_cgroup_memory

    root = tmp_path / "memory"
    root.mkdir()
    (root / "memory.limit_in_bytes").write_text(str(64 * 2**30))
    (root / "memory.usage_in_bytes").write_text(str(12 * 2**30))
    (root / "memory.max_usage_in_bytes").write_text(str(15 * 2**30))
    (root / "memory.stat").write_text("total_rss 100\ntotal_cache 200\ntotal_shmem 20\n")
    (root / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 3\n")
    value = compatible_cgroup_memory(tmp_path, original=lambda root: {})
    assert value["current_bytes"] == 12 * 2**30
    assert value["limit_bytes"] == 64 * 2**30
    assert value["peak_bytes"] == 15 * 2**30
    assert value["stat"]["anon"] == 100
    assert value["stat"]["file"] == 200
    assert value["stat"]["shmem"] == 20
    assert value["events"]["oom_kill"] == 3


def test_cgroup_v2_measurements_are_preserved(tmp_path):
    from workflow_distillation.vast_device_policy import compatible_cgroup_memory

    expected = {"current_bytes": 123, "limit_bytes": 456}
    assert compatible_cgroup_memory(tmp_path, original=lambda root: expected) == expected
