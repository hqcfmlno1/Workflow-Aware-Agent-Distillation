import json
from pathlib import Path

import pytest
import torch


def test_model_contract_matches_frozen_runtime_and_target_regex():
    from workflow_distillation.qlora_runtime import CONFIG
    from workflow_distillation.training.config import LORA_TARGETS, Experiment, load_profile
    from workflow_distillation.training.model import model_kwargs

    profile = load_profile(Path("configs/training/gpus/t4_16gb/profile.json"))
    actual = Experiment().effective_config(profile)
    assert {**actual, "saved_inputs": "CPU pinned"} == CONFIG
    args = model_kwargs(1)
    assert args["device_map"]["model.language_model.layers.31"] == 1
    assert args["device_map"]["model.visual"] == "cpu"
    import re

    assert re.fullmatch(LORA_TARGETS, "model.language_model.layers.31.mlp.down_proj")
    assert re.fullmatch(LORA_TARGETS, "model.language_model.layers.0.linear_attn.in_proj_qkv")
    assert not re.fullmatch(LORA_TARGETS, "model.visual.down_proj")


@pytest.mark.parametrize("mode", ["cpu", "hybrid"])
def test_explicit_activation_context_keeps_gradients_without_global_patch(mode):
    from workflow_distillation.training.activation_storage import activation_context

    original = torch.autograd.graph.save_on_cpu
    torch.manual_seed(42)
    x = torch.randn(5, requires_grad=True)
    with activation_context(mode, 4096 if mode == "hybrid" else 0):
        y = (x * x).sum()
    y.backward()
    torch.testing.assert_close(x.grad, 2 * x)
    assert torch.autograd.graph.save_on_cpu is original


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_distributed_loss_equals_uniform_trajectory_mean_for_uneven_batches(size):
    from workflow_distillation.training.loss import batch_loss
    from workflow_distillation.weighted_sft_batches import distributed_slots
    from workflow_distillation.weighted_sft_loss import weighted_logits_loss

    torch.manual_seed(31)
    hidden = torch.randn(size, 6, 3, requires_grad=True)
    head = torch.nn.Linear(3, 11, bias=False)
    labels = torch.arange(6).repeat(size, 1)
    groups = torch.tensor([0, 1, 2, 3, 4, 1]).repeat(size, 1)
    if size > 1:
        labels[1, -2:] = -100
        groups[1, -2:] = 0
    denominator = torch.tensor([0, 1, 0.75, 1.5, 2])[groups].sum(-1).tolist()
    local = []
    for rank_slots in distributed_slots(list(range(size)), 2):
        indices = [i for i, c in rank_slots]
        local.append(
            batch_loss(
                hidden[indices],
                head,
                labels[indices],
                groups[indices],
                [denominator[i] for i in indices],
                [c for _, c in rank_slots],
            )
        )
    actual = sum(local) / 2
    expected = weighted_logits_loss(head(hidden), labels, groups)
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, (hidden, head.weight), retain_graph=True)
    b = torch.autograd.grad(expected, (hidden, head.weight))
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y)


def _gradient_worker(rank, rendezvous, result):
    import torch.distributed as dist

    from workflow_distillation.training.distributed import average_gradients, initialize_replica

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(rank + 1.0)
        initialize_replica(model)
        assert model.weight.detach().tolist() == [[1.0, 1.0]]
        model(torch.tensor([[rank + 1.0, rank + 2.0]])).sum().backward()
        average_gradients(list(model.parameters()))
        Path(f"{result}-{rank}.json").write_text(json.dumps(model.weight.grad.tolist()))
    finally:
        dist.destroy_process_group()


def test_real_two_rank_sync_broadcasts_adapters_and_averages_dense_gradients(tmp_path):
    from torch.multiprocessing import spawn

    rendezvous = f"file://{tmp_path}/rendezvous"
    spawn(_gradient_worker, args=(rendezvous, str(tmp_path / "result")), nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f"result-{rank}.json").read_text()) == [[1.5, 2.5]]


def test_host_admission_conservatively_rejects_or_serializes_before_forward():
    from workflow_distillation.training.memory import execution_mode

    memory = {"current_bytes": 10 * 2**30, "limit_bytes": 30 * 2**30}
    assert execution_mode([3 * 2**30, 3 * 2**30], memory, allow_serial=False) == "parallel"
    assert execution_mode([9 * 2**30, 9 * 2**30], memory, allow_serial=True) == "serial"
    with pytest.raises(MemoryError):
        execution_mode([9 * 2**30, 9 * 2**30], memory, allow_serial=False)
    with pytest.raises(MemoryError):
        execution_mode([29 * 2**30] * 2, memory, allow_serial=True)
    with pytest.raises(MemoryError):
        execution_mode([0, 0], {}, allow_serial=False)


def test_t4_admission_matches_legacy_observed_pinned_memory_and_reclaimable_cache():
    import importlib.util

    from workflow_distillation.training.memory import HostMemoryPolicy

    path = Path("cloud/kaggle/recovery/adaptive_memory.py")
    spec = importlib.util.spec_from_file_location("frozen_memory_for_test", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    current, original = HostMemoryPolicy(), legacy.MemoryPolicy()
    for token_count, allocated in ((27999, 9 * 2**30), (12000, 8 * 2**30)):
        current.observe(token_count, allocated)
        original.observe(token_count, allocated)
        for cache in (0, 18 * 2**30):
            memory = {
                "current_bytes": 24 * 2**30,
                "limit_bytes": 30 * 2**30,
                "stat": {
                    "file": cache,
                    "shmem": 1 * 2**30,
                    "file_dirty": 1 * 2**30,
                    "file_writeback": 1 * 2**30,
                },
            }
            assert current.decide([27999, 12000], memory) == original.decide([27999, 12000], memory)
    assert current.multiplier > 1.15


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_reader_preserves_cache_accounting_for_memory_policy(tmp_path, version):
    from workflow_distillation.training.memory import cgroup_memory

    if version == 2:
        (tmp_path / "memory.current").write_text("24000")
        (tmp_path / "memory.max").write_text("30000")
        (tmp_path / "memory.stat").write_text(
            "file 18000\nshmem 1000\nfile_dirty 2000\nfile_writeback 3000\n"
        )
    else:
        memory = tmp_path / "memory"
        memory.mkdir()
        (memory / "memory.usage_in_bytes").write_text("24000")
        (memory / "memory.limit_in_bytes").write_text("30000")
        (memory / "memory.stat").write_text(
            "total_cache 18000\ntotal_shmem 1000\ntotal_dirty 2000\ntotal_writeback 3000\n"
        )
    actual = cgroup_memory(tmp_path)
    assert actual["current_bytes"] == 24000
    assert actual["limit_bytes"] == 30000
    assert actual["stat"] == {
        "file": 18000,
        "shmem": 1000,
        "file_dirty": 2000,
        "file_writeback": 3000,
    }
