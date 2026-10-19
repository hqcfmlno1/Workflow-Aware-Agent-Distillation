"""Adaptive execution must preserve the batch objective and avoid known host peaks."""

import importlib
import json
from datetime import timedelta
from pathlib import Path

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "cloud/kaggle/recovery"
GIB = 2**30


@pytest.fixture
def adaptive(monkeypatch):
    monkeypatch.syspath_prepend(str(HOOKS))
    return importlib.import_module("adaptive_memory")


def snapshot(used=12 * GIB, *, limit=30 * GIB, cache=0):
    return {"current_bytes": used + cache, "limit_bytes": limit,
            "stat": {"file": cache, "shmem": 0, "file_dirty": 0, "file_writeback": 0}}


def test_policy_handles_allocator_size_jump_and_known_step371(adaptive):
    policy = adaptive.MemoryPolicy()
    assert policy.activation_bytes(26215) > policy.activation_bytes(26214) * 1.9
    short = policy.decide([18000, 20000], snapshot())
    assert short["mode"] == "parallel"
    failed_batch = policy.decide([29671, 26737], snapshot())
    assert failed_batch["mode"] == "serial"
    assert failed_batch["predicted_parallel_bytes"] + failed_batch["reserve_bytes"] > 30 * GIB
    assert failed_batch["predicted_serial_bytes"] + failed_batch["reserve_bytes"] < 30 * GIB


def test_policy_never_calls_a_single_unsafe_sample_safe(adaptive):
    decision = adaptive.MemoryPolicy().decide([29997, 29997], snapshot(24 * GIB))
    assert decision["mode"] == "stop"
    assert adaptive.MemoryPolicy().decide([5000, 5000], {})["mode"] == "stop"


def test_reclaimable_cache_excludes_shmem_and_dirty_pages(adaptive):
    policy = adaptive.MemoryPolicy()
    memory = snapshot(12 * GIB, cache=7 * GIB)
    memory["stat"].update(shmem=2 * GIB, file_dirty=GIB, file_writeback=GIB)
    decision = policy.decide([5000, 5000], memory)
    assert decision["resident_baseline_bytes"] == 16 * GIB


def test_missing_stats_is_conservative_and_observations_only_raise_estimates(adaptive):
    policy = adaptive.MemoryPolicy()
    before = policy.activation_bytes(20000)
    policy.observe(20000, before * 2)
    assert policy.activation_bytes(20000) >= before * 2
    increased = policy.activation_bytes(20000)
    policy.observe(20000, 1)
    assert policy.activation_bytes(20000) == increased
    decision = policy.decide([5000, 5000], {"current_bytes": 29 * GIB,
                                          "limit_bytes": 30 * GIB})
    assert decision["mode"] == "stop"
    partial = {"current_bytes": 29 * GIB, "limit_bytes": 30 * GIB,
               "stat": {"file": 25 * GIB}}
    assert policy.decide([5000, 5000], partial)["mode"] == "stop"


def _distributed_worker(rank, directory, serial_first):
    import sys
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    sys.path.insert(0, str(HOOKS))
    from adaptive_memory import install_adaptive_execution

    from workflow_distillation.training_checkpoints import CheckpointManager
    from workflow_distillation.weighted_sft_batches import distributed_slots

    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=25))

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(17, 4)
            self.dropout = torch.nn.Dropout(0.2)
            self.output = torch.nn.Linear(4, 1)
            self.config = SimpleNamespace(hidden_size=2560)

        def forward(self, input_ids, use_cache=False):
            from torch.utils.checkpoint import checkpoint

            return checkpoint(lambda ids: self.output(self.dropout(self.embedding(ids))),
                              input_ids, use_reentrant=False)

    batches = [[0, 1], [2, 3], [4, 5], [6]]
    lengths = [8000, 10000, 29671, 26737, 18000, 20000, 28000]
    expected_modes = ["parallel", "serial", "parallel", "serial"]
    if serial_first:
        lengths = [29671, 26737, 8000, 10000, 28000, 27000, 12000]
        expected_modes = ["serial", "parallel", "serial", "parallel"]
    try:
        expected, expected_rng, expected_losses = None, None, None
        expected_optimizer, expected_scaler = None, None
        for adapted in (False, True):
            modes, losses = [], []

            def cleanup():
                # A serial rank must release its activation resources before its peer starts.
                (root / f"active-{rank}").unlink(missing_ok=True)

            def event(value, modes=modes):
                if value["event"] == "batch_execution_mode":
                    modes.append(value["mode"])

            def build(local_rank):
                # Exercise the same locally imported constructor as qlora_runtime.
                from torch.nn.parallel import DistributedDataParallel as Replica

                torch.manual_seed(27 + local_rank)
                full = Decoder()
                # Accelerate/PEFT may import DDP as a type during model loading.
                # Constructor interception must keep those isinstance checks valid.
                assert not isinstance(full, Replica)
                replica = Replica(full, broadcast_buffers=False, find_unused_parameters=False,
                                  gradient_as_bucket_view=True, bucket_cap_mb=25)
                optimizer = torch.optim.AdamW(full.parameters(), lr=0.001)
                scaler = torch.amp.GradScaler("cpu", init_scale=8)
                return full, replica, None, list(full.parameters()), optimizer, scaler

            from workflow_distillation import qlora_runtime

            runtime = SimpleNamespace(CONFIG=qlora_runtime.CONFIG.copy(), build_replica=build)

            class Manager(CheckpointManager):
                pass

            import host_memory_recovery as host

            original_memory, original_cleanup = host.cgroup_memory, host.clear_host_cache
            original_stats = torch.cuda.memory.host_memory_stats
            original_ddp = torch.nn.parallel.DistributedDataParallel

            class ForbiddenDDP(DistributedDataParallel):
                def __init__(self, *args, **kwargs):
                    raise AssertionError("Adaptive execution must not allocate a DDP reducer")

            memory_used = [12 * GIB]
            host.cgroup_memory = lambda used=memory_used: snapshot(used[0])
            host.clear_host_cache = lambda *args: cleanup()
            torch.cuda.memory.host_memory_stats = dict
            if adapted:
                torch.nn.parallel.DistributedDataParallel = ForbiddenDDP
                install_adaptive_execution(torch, runtime, Manager, event)
            full, model, _, parameters, optimizer, scaler = runtime.build_replica(rank)
            if adapted:
                assert torch.nn.parallel.DistributedDataParallel is ForbiddenDDP
                assert "__new__" not in ForbiddenDDP.__dict__
                assert not isinstance(model, DistributedDataParallel)
                assert not any(isinstance(m, DistributedDataParallel) for m in full.modules())
                assert not hasattr(model, "reducer")
            torch.manual_seed(123 + rank)
            manager = Manager(root / f"checkpoints-{adapted}", {}, keep_last=1)
            for position, batch in enumerate(batches):
                index, coefficient = distributed_slots(batch, 2)[rank][0]
                ids = torch.arange(lengths[index]).remainder(17).unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                with torch.autograd.graph.save_on_cpu(pin_memory=False):
                    value = model(input_ids=ids, use_cache=False)
                if adapted and modes[-1] == "serial":
                    assert not (root / f"active-{1-rank}").exists()
                    (root / f"active-{rank}").touch()
                loss = value.square().mean() * coefficient
                losses.append(float(loss.detach()))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if position == 1:
                    manager.save(model=full, optimizer=optimizer, scheduler=None, scaler=scaler,
                                 progress={"global_step": 2, "next_batch": 2,
                                           "samples_consumed": 4})
                    # Load a checkpoint written by ordinary DDP into the new executor.
                    full, model, _, parameters, optimizer, scaler = runtime.build_replica(rank)
                    checkpoint = root / "checkpoints-False/checkpoint-00000002"
                    restored = manager.load(checkpoint, model=full, optimizer=optimizer,
                                            scheduler=None, scaler=scaler)
                    assert restored["global_step"] == 2
            if not adapted:
                expected = {n: p.detach().clone() for n, p in full.named_parameters()}
                expected_rng, expected_losses = torch.get_rng_state().clone(), losses
                expected_optimizer, expected_scaler = optimizer.state_dict(), scaler.state_dict()
            else:
                assert modes == expected_modes
                assert losses == pytest.approx(expected_losses, abs=1e-7, rel=1e-6)
                for name, parameter in full.named_parameters():
                    torch.testing.assert_close(parameter, expected[name], atol=1e-7, rtol=1e-6)
                assert scaler.state_dict() == expected_scaler
                for key, state in optimizer.state_dict()["state"].items():
                    for field, value in state.items():
                        torch.testing.assert_close(value, expected_optimizer["state"][key][field],
                                                   atol=1e-7, rtol=1e-6)
                assert torch.equal(torch.get_rng_state(), expected_rng)
                # Both ranks must reject an unsafe batch before running decoder forward.
                memory_used[0] = 29 * GIB
                with pytest.raises(MemoryError, match="precheck rejected"):
                    model(input_ids=torch.zeros((1, 8000), dtype=torch.long))
                (root / f"result-{rank}.json").write_text(json.dumps({"modes": modes}))
                # One malformed local gradient must reject on BOTH ranks, without
                # stranding a peer inside a per-parameter collective.
                memory_used[0] = 12 * GIB
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=torch.zeros((1, 8000), dtype=torch.long)).square().mean()
                scaler.scale(loss).backward()
                if rank == 0:
                    parameters[0].grad = None
                with pytest.raises(RuntimeError, match="dense gradients on both ranks"):
                    scaler.unscale_(optimizer)

                from adaptive_memory import manual_replica

                with pytest.raises(ValueError, match="parameter signatures differ"):
                    manual_replica(torch, torch.nn.Linear(4, 2 + rank))
            host.cgroup_memory, host.clear_host_cache = original_memory, original_cleanup
            torch.cuda.memory.host_memory_stats = original_stats
            torch.nn.parallel.DistributedDataParallel = original_ddp
            if adapted:
                # A later regular DDP construction must still work after restoration.
                probe = DistributedDataParallel(Decoder())
                assert hasattr(probe, "reducer")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("serial_first", [False, True])
def test_two_rank_switching_and_resume_preserve_updates_and_rng(tmp_path, serial_first):
    torch = pytest.importorskip("torch")
    torch.multiprocessing.spawn(_distributed_worker, args=(str(tmp_path), serial_first),
                                nprocs=2, join=True)
    assert json.loads((tmp_path / "result-0.json").read_text())["modes"][-1] == "stop"


def test_constructor_override_is_restored_if_model_loading_fails(adaptive):
    from types import SimpleNamespace

    from workflow_distillation.qlora_runtime import CONFIG

    class Original:
        pass

    original = Original
    torch = SimpleNamespace(nn=SimpleNamespace(
        parallel=SimpleNamespace(DistributedDataParallel=original)))

    def build(rank):
        raise RuntimeError("model load failed")

    runtime = SimpleNamespace(CONFIG=CONFIG.copy(), build_replica=build)

    class Manager:
        def load(self):
            pass

    adaptive.install_adaptive_execution(torch, runtime, Manager, lambda event: None)
    with pytest.raises(RuntimeError, match="model load failed"):
        runtime.build_replica(0)
    assert torch.nn.parallel.DistributedDataParallel is original
    assert "__new__" not in original.__dict__
