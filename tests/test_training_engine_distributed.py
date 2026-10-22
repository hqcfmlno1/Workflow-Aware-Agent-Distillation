"""Execute the real Step on two CPU ranks, against an independent full-logit reference."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 3).requires_grad_(False)
        self.adapter = torch.nn.Linear(3, 3, bias=False)

    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.adapter(self.embedding(input_ids)))


class TinyDataset:
    def __init__(self):
        import numpy as np

        self.records = []
        for i, n in enumerate((5, 7, 4, 6)):
            groups = np.array([0] + [1, 2, 3, 4, 1, 3][: n - 1])
            weights = np.array([0, 1, 0.75, 1.5, 2])
            self.records.append(
                {
                    "input_ids": np.arange(n),
                    "groups": groups,
                    "trajectory_id": str(i),
                    "denominator": weights[groups].sum(),
                }
            )

    def record(self, index):
        return self.records[index]


def _step_worker(rank, root):
    import torch.distributed as dist

    from workflow_distillation.training.config import GPUProfile
    from workflow_distillation.training.device import DeviceRuntime
    from workflow_distillation.training.distributed import initialize_replica
    from workflow_distillation.training.engine import Step
    from workflow_distillation.training.model import ModelState
    from workflow_distillation.weighted_sft_loss import weighted_logits_loss

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(87)
        decoder = TinyDecoder()
        head = torch.nn.Linear(3, 11, bias=False).requires_grad_(False)
        initialize_replica(decoder)
        parameters = [p for p in decoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
        state = ModelState(decoder, decoder, head, parameters, optimizer, scaler)
        reference = copy.deepcopy(decoder)
        ref_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=0.01
        )
        dataset = TinyDataset()
        profile = GPUProfile("test", "CPU", 2**30, 2, "cpu", 0)
        prepared = SimpleNamespace(dataset=dataset, profile=profile, batches=[[0, 1, 2, 3]])
        events = []
        update = Step(
            state,
            prepared,
            SimpleNamespace(expected_step_seconds=0),
            events.append,
            torch.device("cpu"),
            hardware=DeviceRuntime(torch.device("cpu")),
            memory_reader=lambda: {"current_bytes": 0, "limit_bytes": 120 * 2**30},
        )
        for number, indices in enumerate(([0, 1, 2, 3], [0, 1, 2], [0], [0, 1]), start=1):
            if number == 4:
                prepared.profile = GPUProfile(
                    "t4-test", "CPU", 2**30, 1, "cpu", 0, allow_serial=True
                )
                update.memory_reader = lambda: {
                    "current_bytes": int(0.9 * 2**30),
                    "limit_bytes": 4 * 2**30,
                }
            losses = []
            for i in indices:
                record = dataset.record(i)
                ids = torch.tensor(record["input_ids"]).unsqueeze(0)
                groups = torch.tensor(record["groups"]).unsqueeze(0)
                labels = ids.masked_fill(groups == 0, -100)
                losses.append(
                    weighted_logits_loss(head(reference(ids).last_hidden_state), labels, groups)
                )
            loss = sum(losses) / len(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in reference.parameters() if p.requires_grad], 1.0, foreach=False
            )
            ref_optimizer.step()
            ref_optimizer.zero_grad(set_to_none=True)
            metric = update(indices, number)
            torch.testing.assert_close(
                torch.tensor(metric["loss"]), loss.detach(), rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                decoder.adapter.weight, reference.adapter.weight, rtol=1e-5, atol=1e-6
            )
            assert all(p.grad is None for p in parameters)
        assert [e["event"] for e in events].count("optimizer_started") == 4
        assert [e for e in events if e["event"] == "batch_execution_mode"][-1]["mode"] == "serial"
        Path(root, f"result-{rank}.json").write_text(json.dumps({"passed": True}))
    finally:
        dist.destroy_process_group()


def test_real_engine_updates_match_reference_for_full_uneven_and_singleton_batches(tmp_path):
    torch.multiprocessing.spawn(_step_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"result-{r}.json").read_text())["passed"] for r in (0, 1))
