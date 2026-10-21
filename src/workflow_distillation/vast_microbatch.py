"""Explicit, pinned transition from two to four trajectories per optimizer update."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.epoch_training import validate_progress
from workflow_distillation.training_checkpoints import CheckpointManager, latest_checkpoint


def batch_schedule(count, parent_progress):
    step = parent_progress.get("global_step")
    if (
        type(step) is not int
        or step < 0
        or parent_progress.get("next_batch") != step
        or parent_progress.get("samples_consumed") != 2 * step
        or 2 * step > count
    ):
        raise ValueError("Batch-two parent cursor differs")
    cursor = 2 * step
    batches = [list(range(i, i + 2)) for i in range(0, cursor, 2)]
    batches += [list(range(i, min(i + 4, count))) for i in range(cursor, count, 4)]
    validate_progress(batches, parent_progress)
    return batches


def transition_identity(parent, expected_identity, parent_sha, entrypoint):
    if (
        parent.get("complete") is not True
        or parent.get("world_size") != 2
        or parent["identity"] != expected_identity
    ):
        raise ValueError("Batch-two parent identity differs")
    result = copy.deepcopy(expected_identity)
    config = result["settings"]["config"]
    if (
        config["per_gpu_batch_size"] != 1
        or config["global_batch_size"] != 2
        or config["gradient_accumulation_steps"] != 1
    ):
        raise ValueError("Expected the verified batch-two config")
    config.update(per_gpu_batch_size=2, global_batch_size=4)
    result["script_sha256"] = sha256_file(Path(entrypoint))
    result["settings"]["microbatch_transition"] = {
        "parent_manifest_sha256": parent_sha,
        "parent_progress": parent["progress"],
        "implementation_sha256": sha256_file(Path(__file__)),
        "padding": "right; causal forward; zero loss on padding",
        "reduction": "equal trajectory mean; manual gradient SUM/world_size",
        "execution": "two concurrent bucket-free replicas; microbatch=2",
    }
    return result


def migration_manager(binding):
    class MicrobatchManager(CheckpointManager):
        def load(self, checkpoint, **kwargs):
            path = (
                latest_checkpoint(self.directory)
                if str(checkpoint) == "latest"
                else Path(checkpoint)
            )
            child_identity = self.identity
            try:
                if sha256_file(path / "manifest.json") == binding["parent"]["manifest_sha256"]:
                    self.identity = binding["parent"]["identity"]
                return super().load(path, **kwargs)
            finally:
                self.identity = child_identity

    return MicrobatchManager


def batch_hidden_loss(hidden, head, labels, groups, denominators, coefficients):
    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    size = hidden.shape[0]
    if not (
        1 <= size <= 2
        and size == len(denominators) == len(coefficients)
        and labels.shape == groups.shape == hidden.shape[:2]
    ):
        raise ValueError("Expected one or two aligned trajectories")
    return sum(
        chunked_hidden_loss(
            hidden[i : i + 1],
            head,
            labels[i : i + 1],
            groups[i : i + 1],
            denominators[i],
            coefficient,
            chunk_size=64,
        )
        for i, coefficient in enumerate(coefficients)
    )


def build_replica(rank):
    import torch
    from adaptive_memory import manual_replica

    from workflow_distillation.qlora_runtime import build_replica as original_build

    cls = torch.nn.parallel.DistributedDataParallel
    had_new, original_new = "__new__" in cls.__dict__, cls.__dict__.get("__new__")
    constructions = []

    def construct(kind, module, **kwargs):
        if kind is not cls or constructions:
            raise ValueError("Unexpected extra replica construction")
        constructions.append(True)
        return manual_replica(torch, module, **kwargs)

    cls.__new__ = staticmethod(construct)
    try:
        result = original_build(rank)
    finally:
        if had_new:
            cls.__new__ = original_new
        else:
            del cls.__new__
    if len(constructions) != 1:
        raise ValueError("Expected one bucket-free local replica")
    return result


def worker(trainer, args, binding, identity):
    import os
    import random
    import time
    import traceback
    from datetime import timedelta

    import numpy as np
    import torch
    import torch.distributed as dist
    from adaptive_memory import MemoryPolicy
    from host_memory_recovery import (
        cgroup_memory,
        install_cache_cleanup,
        install_checkpoint_cleanup,
        start_monitor,
    )

    from workflow_distillation.epoch_training import run_epoch_loop
    from workflow_distillation.training_checkpoints import StopBudget
    from workflow_distillation.weighted_sft_batches import distributed_slots

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    emit = trainer.EventLog(args.output / f"rank{rank}.jsonl", args.run_name, rank)
    started = time.monotonic()
    report = {"run_name": args.run_name, "rank": rank, "status": "running"}
    budget = StopBudget(
        max_seconds=args.max_train_seconds,
        started=started,
        margin_seconds=args.stop_margin_seconds,
        expected_step_seconds=args.expected_step_seconds,
    )
    try:
        if int(os.environ["WORLD_SIZE"]) != 2:
            raise ValueError("Expected two ranks")
        emit({"event": "initialize", "config": trainer.CONFIG})
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
        install_cache_cleanup(torch)
        install_checkpoint_cleanup(torch, trainer.CheckpointManager)
        start_monitor(torch, progress_dir=args.output)
        dataset = trainer.EpochDataset(args.index, verify=False)
        trainer.check_dataset(dataset)
        batches = trainer.epoch_batches(len(dataset), args.seed)
        emit(
            {
                "event": "model_load_started",
                "total_samples": len(dataset),
                "total_optimizer_steps": len(batches),
            }
        )
        model, replica, head, parameters, optimizer, scaler = build_replica(rank)
        emit(
            {
                "event": "model_loaded",
                "trainable_parameters": sum(p.numel() for p in parameters),
                "load_seconds": time.monotonic() - started,
                "ddp_bucket_bytes": 0,
            }
        )
        manager = trainer.CheckpointManager(
            args.checkpoint_dir,
            identity,
            save_every=args.save_every_steps,
            keep_last=args.keep_checkpoints,
            publisher=trainer.HuggingFacePublisher(
                args.checkpoint_hub_repo,
                prefix=args.checkpoint_hub_prefix,
                keep_last=args.checkpoint_hub_keep_last,
                repo_type=args.checkpoint_hub_repo_type,
            )
            if args.checkpoint_hub_repo
            else None,
        )
        progress = manager.load(
            args.resume_from, model=model, optimizer=optimizer, scaler=scaler, scheduler=None
        )
        validate_progress(batches, progress)
        manager.republish(model=model)
        emit({"event": "resumed", **progress, "checkpoint": str(manager.last_checkpoint)})

        def stop():
            value = torch.tensor(
                int(budget.should_stop() or (args.output / "STOP").exists()), device=device
            )
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value)

        def save(position):
            if manager.due(position["global_step"], force=True):
                tick = time.monotonic()
                emit({"event": "checkpoint_started", **position})
                manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=scaler,
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

        def update(batch, number):
            slots = distributed_slots(batch, 2)[rank]
            records = [dataset.record(index) for index, _ in slots]
            lengths = [len(r["input_ids"]) for r in records]
            if any(n >= 28000 for n in lengths):
                raise ValueError(
                    "Batch-four continuation requires untrained trajectories below 28k"
                )
            width = max(lengths)
            gathered = [None, None]
            dist.all_gather_object(gathered, len(records) * MemoryPolicy().activation_bytes(width))
            memory = cgroup_memory()
            # Conservatively include all page cache; on this 120 GiB cgroup this
            # leaves room even for two longest padded batches and checkpoint IO.
            current, limit = memory.get("current_bytes"), memory.get("limit_bytes")
            safe = torch.tensor(
                int(
                    current is not None
                    and limit is not None
                    and current + sum(gathered) + max(3 * 2**30, 0.1 * limit) < limit
                ),
                device=device,
            )
            dist.all_reduce(safe, op=dist.ReduceOp.MIN)
            if not bool(safe):
                raise MemoryError("Host RAM precheck rejected the batch before forward")
            ids = torch.zeros((len(records), width), dtype=torch.long, device=device)
            groups = torch.zeros_like(ids)
            for i, record in enumerate(records):
                ids[i, : lengths[i]] = torch.tensor(
                    np.array(record["input_ids"], dtype=np.int64), device=device
                )
                groups[i, : lengths[i]] = torch.tensor(
                    np.array(record["groups"], dtype=np.int64), device=device
                )
            labels = ids.masked_fill(groups == 0, -100)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            emit(
                {
                    "event": "forward_started",
                    "global_step": number,
                    "tokens": sum(lengths),
                    "trajectory_ids": [r["trajectory_id"] for r in records],
                    "trajectory_lengths": lengths,
                    "input_shape": list(ids.shape),
                }
            )
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    hidden = replica(input_ids=ids, use_cache=False).last_hidden_state
                loss = batch_hidden_loss(
                    hidden,
                    head,
                    labels,
                    groups,
                    [r["denominator"] for r in records],
                    [coefficient for _, coefficient in slots],
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            emit({"event": "backward_started", "global_step": number})
            scaler.scale(loss).backward()
            invalid = torch.tensor(
                int(any(p.grad is None or p.grad.is_sparse for p in parameters)), device=device
            )
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            if bool(invalid):
                raise RuntimeError("Missing or sparse gradients")
            for p in parameters:
                dist.all_reduce(p.grad)
                p.grad.div_(2)
            scaler.unscale_(optimizer)
            if any(not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Nonfinite gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite norm")
            emit({"event": "optimizer_started", "global_step": number})
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale or any(
                not bool(torch.isfinite(p).all()) for p in parameters
            ):
                raise FloatingPointError("Skipped update or nonfinite parameters")
            metric = torch.tensor(
                [
                    float(loss.detach()),
                    sum(
                        n for n, (_, coefficient) in zip(lengths, slots, strict=True) if coefficient
                    ),
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(metric)
            optimizer.zero_grad(set_to_none=True)
            del hidden, loss, ids, groups, labels
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - tick
            free, total = torch.cuda.mem_get_info(device)
            budget.expected_step_seconds = max(budget.expected_step_seconds, seconds)
            return {
                "loss": float(metric[0]) / 2,
                "grad_norm": float(norm),
                "scale": scaler.get_scale(),
                "seconds": seconds,
                "global_tokens": int(metric[1]),
                "global_tokens_per_second": float(metric[1]) / seconds,
                "epoch_fraction": number / len(batches),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
                "cuda_free": free,
                "cuda_capacity": total,
            }

        def logged(event):
            emit(event)
            # Establish a full recovery point immediately after the first batch
            # under the new config, then use the normal cadence.
            if (
                event.get("event") == "step"
                and event["global_step"] == binding["parent"]["progress"]["global_step"] + 1
            ):
                save({k: event[k] for k in ("global_step", "next_batch", "samples_consumed")})

        with budget:
            result = run_epoch_loop(
                batches,
                progress=progress,
                step=update,
                save=save,
                stop=stop,
                emit=logged,
                save_every=args.save_every_steps,
            )
        report.update(
            result,
            checkpoint=str(manager.last_checkpoint),
            total_samples=len(dataset),
            total_optimizer_steps=len(batches),
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
        report["seconds"] = time.monotonic() - started
        (args.output / f"rank{rank}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if report["status"] != "failed" and dist.is_initialized():
            dist.destroy_process_group()
