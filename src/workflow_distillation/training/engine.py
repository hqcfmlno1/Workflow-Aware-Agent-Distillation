"""One explicit training loop shared across GPU profiles."""

from __future__ import annotations

import os
import random
import time
import traceback
from datetime import timedelta

import numpy as np

from workflow_distillation.epoch_training import run_epoch_loop
from workflow_distillation.training.activation_storage import activation_context
from workflow_distillation.training.checkpoints import MigrationManager, MigrationPublisher
from workflow_distillation.training.device import DeviceRuntime
from workflow_distillation.training.distributed import average_gradients
from workflow_distillation.training.events import EventLog
from workflow_distillation.training.loss import batch_loss
from workflow_distillation.training.memory import (
    HostMemoryPolicy,
    activation_bytes,
    cgroup_memory,
    clear_inactive_cache,
    execution_mode,
)
from workflow_distillation.training.model import build_model
from workflow_distillation.training_checkpoints import StopBudget
from workflow_distillation.weighted_sft_batches import distributed_slots


def execute_loop(
    batches,
    *,
    progress,
    step,
    save,
    should_stop,
    emit,
    save_every,
    stop_at_step=None,
    first_save_step=None,
):
    completed = progress["global_step"]

    def update(batch, number):
        nonlocal completed
        result = step(batch, number)
        completed = number
        return result

    def logged(event):
        emit(event)
        if event["global_step"] == first_save_step:
            save({k: event[k] for k in ("global_step", "next_batch", "samples_consumed")})

    return run_epoch_loop(
        batches,
        progress=progress,
        step=update,
        save=save,
        emit=logged,
        stop=lambda: should_stop() or (stop_at_step is not None and completed >= stop_at_step),
        save_every=save_every,
    )


def make_batch(dataset, indices, rank, device):
    import torch

    slots = distributed_slots(indices, 2)[rank]
    records = [dataset.record(i) for i, _ in slots]
    lengths = [len(r["input_ids"]) for r in records]
    if any(n >= 28000 for n in lengths):
        raise ValueError("Untrained trajectories must stay strictly below 28000 tokens")
    width = max(lengths)
    ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
    groups = torch.zeros_like(ids)
    for i, record in enumerate(records):
        ids[i, : lengths[i]] = torch.tensor(
            np.array(record["input_ids"], dtype=np.int64), device=device
        )
        groups[i, : lengths[i]] = torch.tensor(
            np.array(record["groups"], dtype=np.int64), device=device
        )
    return ids, groups, records, slots, lengths


class Step:
    def __init__(
        self, state, prepared, budget, emit, device, *, hardware=None, memory_reader=cgroup_memory
    ):
        self.state, self.prepared, self.budget = state, prepared, budget
        self.emit, self.device = emit, device
        self.hardware = hardware or DeviceRuntime(device)
        self.memory_reader = memory_reader
        self.memory_policy = HostMemoryPolicy()
        self.previous_observation = None

    def admission(self, records, lengths):
        import torch.distributed as dist

        profile = self.prepared.profile
        if profile.allow_serial:
            if len(records) != 1:
                raise ValueError("Adaptive T4 memory policy requires one trajectory per rank")
            gathered = [None] * profile.world_size
            dist.all_gather_object(
                gathered, {"tokens": lengths[0], "previous": self.previous_observation}
            )
            decision = [None]
            if dist.get_rank() == 0:
                for value in gathered:
                    if value["previous"] is not None:
                        self.memory_policy.observe(*value["previous"])
                decision[0] = self.memory_policy.decide(
                    [v["tokens"] for v in gathered], self.memory_reader()
                )
            dist.broadcast_object_list(decision, src=0)
            return decision[0]
        predicted = [None] * profile.world_size
        dist.all_gather_object(predicted, len(records) * activation_bytes(max(lengths)))
        modes = [None] * profile.world_size
        try:
            local_mode = execution_mode(predicted, self.memory_reader(), allow_serial=False)
        except MemoryError:
            local_mode = "stop"
        dist.all_gather_object(modes, local_mode)
        return {
            "mode": "stop" if "stop" in modes else "parallel",
            "activation_estimates_bytes": predicted,
        }

    def __call__(self, indices, number):
        import torch
        import torch.distributed as dist

        state, profile = self.state, self.prepared.profile
        rank = dist.get_rank()
        ids, groups, records, slots, lengths = make_batch(
            self.prepared.dataset,
            indices,
            rank,
            self.device,
        )
        self.hardware.clear_cache()
        decision = self.admission(records, lengths)
        mode = decision["mode"]
        if mode == "stop":
            raise MemoryError("Host RAM precheck rejected the batch before forward")
        self.emit(
            {
                "event": "batch_execution_mode",
                "global_step": number,
                **decision,
                "ddp_bucket_bytes": 0,
            }
        )
        labels = ids.masked_fill(groups == 0, -100)
        state.optimizer.zero_grad(set_to_none=True)
        self.hardware.prepare_step()
        tick = time.monotonic()
        if mode == "serial" and rank == 1:
            dist.barrier()
        self.emit(
            {
                "event": "forward_started",
                "global_step": number,
                "tokens": sum(lengths),
                "trajectory_ids": [r["trajectory_id"] for r in records],
                "trajectory_lengths": lengths,
                "input_shape": list(ids.shape),
            }
        )
        with self.hardware.autocast():
            with activation_context(profile.activation_storage, profile.gpu_budget_bytes):
                hidden = state.decoder(input_ids=ids, use_cache=False).last_hidden_state
            loss = batch_loss(
                hidden,
                state.head,
                labels,
                groups,
                [r["denominator"] for r in records],
                [c for _, c in slots],
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
        self.emit({"event": "backward_started", "global_step": number})
        state.scaler.scale(loss).backward()
        if profile.allow_serial:
            self.previous_observation = (lengths[0], self.hardware.pinned_bytes())
        if mode == "serial":
            self.hardware.clear_cache()
            if rank == 0:
                dist.barrier()
            dist.barrier()
        average_gradients(state.parameters)
        state.scaler.unscale_(state.optimizer)
        if any(not bool(torch.isfinite(p.grad).all()) for p in state.parameters):
            raise FloatingPointError("Nonfinite adapter gradients")
        norm = torch.nn.utils.clip_grad_norm_(state.parameters, 1.0, foreach=False)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite gradient norm")
        self.emit({"event": "optimizer_started", "global_step": number})
        old_scale = state.scaler.get_scale()
        state.scaler.step(state.optimizer)
        state.scaler.update()
        if state.scaler.get_scale() < old_scale or any(
            not bool(torch.isfinite(p).all()) for p in state.parameters
        ):
            raise FloatingPointError("Skipped update or nonfinite updated adapters")
        metric = torch.tensor(
            [float(loss.detach()), sum(n for n, (_, c) in zip(lengths, slots, strict=True) if c)],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(metric)
        state.optimizer.zero_grad(set_to_none=True)
        del hidden, loss, ids, groups, labels
        self.hardware.synchronize()
        seconds = time.monotonic() - tick
        self.budget.expected_step_seconds = max(self.budget.expected_step_seconds, seconds)
        return {
            "loss": float(metric[0]) / profile.world_size,
            "grad_norm": float(norm),
            "scale": state.scaler.get_scale(),
            "seconds": seconds,
            "global_tokens": int(metric[1]),
            "global_tokens_per_second": float(metric[1]) / seconds,
            "epoch_fraction": number / len(self.prepared.batches),
            **self.hardware.memory_metrics(),
        }


def worker(args, prepared):
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != prepared.profile.world_size:
        raise ValueError("Unexpected worker world size")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    emit = EventLog(args.output / f"rank{rank}.jsonl", args.run_name, rank)
    report = {"run_name": args.run_name, "rank": rank, "status": "running"}
    started = time.monotonic()
    budget = StopBudget(
        max_seconds=args.max_train_seconds,
        started=started,
        margin_seconds=args.stop_margin_seconds,
        expected_step_seconds=args.expected_step_seconds,
    )
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
        emit({"event": "initialize", "config": prepared.identity["settings"]["config"]})
        emit(
            {
                "event": "model_load_started",
                "total_samples": len(prepared.dataset),
                "total_optimizer_steps": len(prepared.batches),
            }
        )
        state = build_model(rank)
        emit(
            {
                "event": "model_loaded",
                "trainable_parameters": sum(p.numel() for p in state.parameters),
                "load_seconds": time.monotonic() - started,
                "ddp_bucket_bytes": 0,
            }
        )
        publisher = (
            MigrationPublisher(
                args.checkpoint_hub_repo,
                prefix=args.checkpoint_hub_prefix,
                keep_last=args.checkpoint_hub_keep_last,
                repo_type=args.checkpoint_hub_repo_type,
                binding=prepared.binding,
                child_identity=prepared.identity,
            )
            if args.checkpoint_hub_repo
            else None
        )
        manager = MigrationManager(
            args.checkpoint_dir or args.output / "checkpoints",
            prepared.identity,
            binding=prepared.binding,
            save_every=args.save_every_steps,
            keep_last=args.keep_checkpoints,
            publisher=publisher,
        )
        progress = manager.load(
            args.resume_from,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=None,
            scaler=state.scaler,
        )
        manager.republish(model=state.model)
        emit({"event": "resumed", **progress, "checkpoint": str(manager.last_checkpoint)})

        def stop():
            value = torch.tensor(
                int(budget.should_stop() or (args.output / "STOP").exists()), device=device
            )
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value)

        def save(position):
            if manager.due(position["global_step"], force=True):
                clear_inactive_cache(torch)
                tick = time.monotonic()
                emit({"event": "checkpoint_started", **position})
                manager.save(
                    model=state.model,
                    optimizer=state.optimizer,
                    scheduler=None,
                    scaler=state.scaler,
                    progress=position,
                )
                emit(
                    {
                        "event": "checkpoint_complete",
                        **position,
                        "path": str(manager.last_checkpoint),
                        "seconds": time.monotonic() - tick,
                    }
                )

        with budget:
            result = execute_loop(
                prepared.batches,
                progress=progress,
                step=Step(state, prepared, budget, emit, device),
                save=save,
                should_stop=stop,
                emit=emit,
                save_every=args.save_every_steps,
                stop_at_step=args.stop_at_step,
                first_save_step=prepared.binding["progress"]["global_step"] + 1,
            )
        report.update(
            result,
            checkpoint=str(manager.last_checkpoint),
            total_samples=len(prepared.dataset),
            total_optimizer_steps=len(prepared.batches),
        )
        emit({"event": "session_finished", **report})
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        emit({"event": "failed", **report})
        raise
    finally:
        import json

        report["seconds"] = time.monotonic() - started
        (args.output / f"rank{rank}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()
