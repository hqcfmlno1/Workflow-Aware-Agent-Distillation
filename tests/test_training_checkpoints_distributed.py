"""Two real CPU DDP processes must save/resume together and propagate write failures."""

import json
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _worker(rank, directory):
    import torch.distributed as dist
    from test_training_checkpoints import load, save, setup_training, step
    from torch.nn.parallel import DistributedDataParallel

    from workflow_distillation.training_checkpoints import CheckpointManager, latest_checkpoint

    root = Path(directory)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        training = list(setup_training())
        training[0] = DistributedDataParallel(training[0])
        random.seed(400 + rank)
        np.random.seed(400 + rank)
        torch.manual_seed(400 + rank)
        manager = CheckpointManager(root / "checkpoints", {"data": "same", "world_size": 2})
        step(training)
        save(manager, training, 1)
        expected_loss = step(training)
        expected_weights = training[0].module.adapter.weight.detach().clone()
        # Only rank one fails a local write precondition. Both ranks must exit
        # the checkpoint collective, leaving checkpoint one as the recovery point.
        if rank == 1:
            training[0].module.adapter.weight.grad = torch.ones_like(expected_weights)
        try:
            save(manager, training, 2)
        except RuntimeError as exc:
            assert "Clear gradients" in str(exc)
        else:
            raise AssertionError("A failed rank was silently committed")
        assert latest_checkpoint(root / "checkpoints").name == "checkpoint-00000001"
        resumed = list(setup_training())
        resumed[0] = DistributedDataParallel(resumed[0])
        progress = load(manager, resumed)
        actual_loss = step(resumed)
        torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
        torch.testing.assert_close(resumed[0].module.adapter.weight, expected_weights,
                                   atol=0, rtol=0)
        replicas = [torch.empty_like(expected_weights) for _ in range(2)]
        dist.all_gather(replicas, resumed[0].module.adapter.weight.detach())
        torch.testing.assert_close(replicas[0], replicas[1], atol=0, rtol=0)
        (root / f"result-{rank}.json").write_text(json.dumps({
            "rank": rank, "progress": progress, "resume_matches": True,
            "write_failure_propagated": True,
        }))
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpoint_resume_and_one_rank_failure_are_consistent(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in (0, 1):
        result = json.loads((tmp_path / f"result-{rank}.json").read_text())
        assert result["progress"] == {"global_step": 1, "next_batch": 1}
        assert result["resume_matches"] and result["write_failure_propagated"]
    committed = tmp_path / "checkpoints/checkpoint-00000001"
    assert sorted(p.name for p in committed.iterdir()) == [
        "adapter.pt", "manifest.json", "rank-00000.pt", "rank-00001.pt",
    ]
