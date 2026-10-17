"""Real two-rank training must match a trajectory-mean reference on an odd epoch."""

from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _worker(rank, root):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from workflow_distillation.epoch_data import epoch_batches
    from workflow_distillation.epoch_training import initial_progress, run_epoch_loop
    from workflow_distillation.training_checkpoints import CheckpointManager
    from workflow_distillation.weighted_sft_batches import distributed_slots

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(17)
        reference = torch.nn.Linear(3, 1)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
        batches = epoch_batches(5)
        xs = [torch.arange((i + 1) * 3, dtype=torch.float32).reshape(-1, 3) / 10
              for i in range(5)]
        initial = {name: value.clone() for name, value in reference.state_dict().items()}
        for batch in batches:
            reference_optimizer.zero_grad(set_to_none=True)
            loss = sum(reference(xs[index]).square().mean() for index in batch) / len(batch)
            loss.backward()
            reference_optimizer.step()

        model = torch.nn.Linear(3, 1)
        model.load_state_dict(initial)
        model = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8)
        manager = CheckpointManager(Path(root) / "checkpoints", {}, keep_last=1)
        consumed = []
        updates = []

        def update(batch, number):
            index, coefficient = distributed_slots(batch, 2)[rank][0]
            if coefficient:
                consumed.append(index)
            optimizer.zero_grad(set_to_none=True)
            loss = model(xs[index]).square().mean() * coefficient
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updates.append(number)
            return {}

        def save(progress):
            if manager.due(progress["global_step"], force=True):
                manager.save(model=model, optimizer=optimizer, scheduler=None,
                             scaler=scaler, progress=progress)

        result = run_epoch_loop(
            batches, progress=initial_progress(), step=update, save=save,
            stop=lambda: len(updates) == 1, emit=lambda _value: None, save_every=5,
        )
        assert result["status"] == "stopped_checkpointed"
        model = DistributedDataParallel(torch.nn.Linear(3, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8)
        progress = manager.load("latest", model=model, optimizer=optimizer,
                                scheduler=None, scaler=scaler)
        result = run_epoch_loop(
            batches, progress=progress, step=update, save=save, stop=lambda: False,
            emit=lambda _value: None, save_every=5,
        )
        assert result["status"] == "completed"
        assert result["progress"]["samples_consumed"] == 5
        assert updates == [1, 2, 3]
        for name, value in model.module.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], atol=1e-7, rtol=1e-6)
        all_consumed = [None, None]
        dist.all_gather_object(all_consumed, consumed)
        assert sorted(all_consumed[0] + all_consumed[1]) == list(range(5))
        assert len(list((Path(root) / "checkpoints").glob("checkpoint-*"))) == 1
    finally:
        dist.destroy_process_group()


def test_two_rank_odd_epoch_resume_matches_reference_without_replayed_samples(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
