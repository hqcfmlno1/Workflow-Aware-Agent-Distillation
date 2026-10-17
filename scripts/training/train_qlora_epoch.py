"""One full shuffled epoch of successful <30k trajectories, remote DDP only."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from workflow_distillation.epoch_data import EpochDataset, epoch_batches, sha256_file  # noqa: E402
from workflow_distillation.epoch_training import (  # noqa: E402
    initial_progress,
    run_epoch_loop,
    run_logged_process,
    validate_progress,
)
from workflow_distillation.qlora_runtime import CONFIG, MODEL, REVISION, build_replica  # noqa: E402
from workflow_distillation.training_checkpoints import (  # noqa: E402
    CheckpointManager,
    HuggingFacePublisher,
    StopBudget,
    add_checkpoint_arguments,
    checkpoint_arguments,
    training_identity,
    validate_checkpoint_arguments,
)
from workflow_distillation.weighted_sft_batches import distributed_slots  # noqa: E402


class EventLog:
    def __init__(self, path, run_name, rank=None):
        self.path, self.run_name, self.rank = Path(path), run_name, rank

    def __call__(self, event):
        value = {"time": datetime.now(UTC).isoformat(),
                 "run_name": self.run_name, "rank": self.rank, **event}
        line = json.dumps(value, ensure_ascii=False)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)


def check_dataset(dataset):
    if (dataset.manifest.get("selection") != "all train_pool resolved=1 token_count<30000"
            or dataset.manifest["weights"] != CONFIG["loss_weights"]):
        raise ValueError("Epoch data selection/weights differ from the approved experiment")
    for row in dataset.records:
        if (not 2 <= row["length"] < 30000 or row["length"] != row["full_token_count"]
                or row["denominator"] <= 0):
            raise ValueError("Dataset must retain complete positive-loss trajectories below30k")


def worker(args):
    import numpy as np
    import psutil
    import torch
    import torch.distributed as dist

    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("Expected two DDP ranks")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    emit = EventLog(args.output / f"rank{rank}.jsonl", args.run_name, rank)
    started = time.monotonic()
    budget = StopBudget(max_seconds=args.max_train_seconds, started=started,
                        margin_seconds=args.stop_margin_seconds,
                        expected_step_seconds=args.expected_step_seconds)
    report = {"run_name": args.run_name, "rank": rank, "status": "running"}
    try:
        emit({"event": "initialize", "config": CONFIG})
        dist.init_process_group("nccl", timeout=timedelta(seconds=1800), device_id=device)
        dataset = EpochDataset(args.index, verify=False)  # Parent verified file hashes.
        check_dataset(dataset)
        batches = epoch_batches(len(dataset), args.seed)
        emit({"event": "model_load_started", "total_samples": len(dataset),
              "total_optimizer_steps": len(batches)})
        model, ddp, head, parameters, optimizer, scaler = build_replica(rank)
        emit({"event": "model_loaded", "trainable_parameters": sum(p.numel() for p in parameters),
              "load_seconds": time.monotonic() - started})
        source = ROOT / "src/workflow_distillation"
        identity = training_identity(
            __file__, model=MODEL, revision=REVISION, records_sha256=sha256_file(args.index),
            settings={"config": CONFIG, "seed": args.seed,
                      "runtime_sources": {
                          name: sha256_file(source / name) for name in
                          ("epoch_data.py", "epoch_training.py", "qlora_runtime.py",
                           "weighted_sft_batches.py", "training_checkpoints.py")}},
        )
        manager = CheckpointManager(
            args.checkpoint_dir or args.output / "checkpoints", identity,
            save_every=args.save_every_steps, keep_last=args.keep_checkpoints,
            publisher=(HuggingFacePublisher(args.checkpoint_hub_repo,
                                           prefix=args.checkpoint_hub_prefix,
                                           keep_last=args.checkpoint_hub_keep_last,
                                           repo_type=args.checkpoint_hub_repo_type)
                       if args.checkpoint_hub_repo else None),
        )
        progress = initial_progress()
        if args.resume_from:
            progress = manager.load(args.resume_from, model=model, optimizer=optimizer,
                                    scheduler=None, scaler=scaler)
            validate_progress(batches, progress)
            manager.republish(model=model)
            emit({"event": "resumed", **progress, "checkpoint": str(manager.last_checkpoint)})

        def stop():
            value = torch.tensor(int(budget.should_stop()), device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value)

        def save(position):
            if manager.due(position["global_step"], force=True):
                tick = time.monotonic()
                emit({"event": "checkpoint_started", **position})
                manager.save(model=model, optimizer=optimizer, scheduler=None,
                             scaler=scaler, progress=position)
                emit({"event": "checkpoint_complete", **position,
                      "path": str(manager.last_checkpoint),
                      "seconds": time.monotonic() - tick})

        def update(batch, number):
            slot, coefficient = distributed_slots(batch, 2)[rank][0]
            record = dataset.record(slot)
            ids = torch.tensor(np.asarray(record["input_ids"], dtype=np.int64),
                               device=device).unsqueeze(0)
            groups = torch.tensor(np.asarray(record["groups"], dtype=np.int64),
                                  device=device).unsqueeze(0)
            labels = ids.masked_fill(groups == 0, -100)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            tick = time.monotonic()
            emit({"event": "forward_started", "global_step": number,
                  "trajectory_id": record["trajectory_id"], "tokens": ids.numel(),
                  "padding": coefficient == 0})
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    hidden = ddp(input_ids=ids, use_cache=False).last_hidden_state
                loss = chunked_hidden_loss(hidden, head, labels, groups, record["denominator"],
                                           coefficient, chunk_size=64)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss")
            emit({"event": "backward_started", "global_step": number})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError("Missing/nonfinite adapter gradients")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
            emit({"event": "optimizer_started", "global_step": number})
            scaler.step(optimizer)
            scaler.update()
            if any(not bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("Nonfinite updated adapters")
            # DDP averaged the gradients; reduce metrics with the same coefficients.
            metrics = torch.tensor([float(loss.detach()),
                                    record["full_token_count"] if coefficient else 0.0],
                                   dtype=torch.float64, device=device)
            dist.all_reduce(metrics)
            optimizer.zero_grad(set_to_none=True)
            del hidden, loss, ids, groups, labels
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - tick
            budget.expected_step_seconds = max(budget.expected_step_seconds, seconds)
            free, total = torch.cuda.mem_get_info(device)
            return {"loss": float(metrics[0]) / 2, "grad_norm": float(norm),
                    "scale": scaler.get_scale(), "seconds": seconds,
                    "global_tokens": int(metrics[1]),
                    "global_tokens_per_second": float(metrics[1]) / seconds,
                    "epoch_fraction": number / len(batches),
                    "peak_allocated": torch.cuda.max_memory_allocated(device),
                    "peak_reserved": torch.cuda.max_memory_reserved(device),
                    "cuda_free": free, "cuda_capacity": total,
                    "host_available": psutil.virtual_memory().available}

        with budget:
            result = run_epoch_loop(batches, progress=progress, step=update, save=save,
                                    stop=stop, emit=emit, save_every=args.save_every_steps)
        report.update(result, checkpoint=str(manager.last_checkpoint),
                      total_samples=len(dataset), total_optimizer_steps=len(batches))
        emit({"event": "session_finished", **report})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        emit({"event": "failed", **report})
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        (args.output / f"rank{rank}-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training/first run")
    parser.add_argument("--run-name", default="first run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-seconds", type=int, default=32400)
    add_checkpoint_arguments(parser)
    parser.add_argument("--checkpoint-hub-repo-type", choices=("model", "dataset"),
                        default="model", help="Hub repository type; dataset URLs need dataset")
    parser.add_argument(
        "--checkpoint-hub-keep-last", type=int, default=None,
        help="DESTRUCTIVE opt-in: keep >=2 snapshots in a dedicated Hub repo and permanently "
             "squash its main history after each upload; single writer only. "
             "Default: preserve all.",
    )
    parser.set_defaults(keep_checkpoints=1, checkpoint_hub_prefix="first-run",
                        stop_margin_seconds=600, expected_step_seconds=240)
    args = parser.parse_args()
    validate_checkpoint_arguments(args)
    if args.max_train_seconds <= 0:
        parser.error("max-train-seconds must be positive")
    if not Path("/kaggle/working").exists():
        raise SystemExit("Model loading/training is permitted only on remote Kaggle GPUs")
    args.output.mkdir(parents=True, exist_ok=True)
    if "LOCAL_RANK" in os.environ:
        worker(args)
        return
    emit = EventLog(args.output / "launcher.jsonl", args.run_name)
    summary = {"run_name": args.run_name, "status": "preparing", "config": CONFIG,
               "checkpoint_policy": {"save_every_steps": args.save_every_steps,
                                     "keep_checkpoints": args.keep_checkpoints,
                                     "hub_repo": args.checkpoint_hub_repo,
                                     "hub_repo_type": args.checkpoint_hub_repo_type,
                                     "hub_prefix": args.checkpoint_hub_prefix,
                                     "hub_keep_last": args.checkpoint_hub_keep_last}}
    try:
        dataset = EpochDataset(args.index)
        check_dataset(dataset)
        summary.update(samples=len(dataset), tokens=dataset.manifest["tokens"],
                       total_optimizer_steps=math.ceil(len(dataset) / 2),
                       index_sha256=sha256_file(args.index))
        emit({"event": "dataset_verified", **summary})
        from packaging.version import Version

        try:
            if Version(version("torchao")) < Version("0.16"):
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                               check=True)
        except PackageNotFoundError:
            pass
        import torch
        from huggingface_hub import snapshot_download

        if torch.cuda.device_count() != 2:
            raise RuntimeError("Expected exactly two T4 GPUs")
        if args.checkpoint_hub_repo:
            HuggingFacePublisher(args.checkpoint_hub_repo,
                                 prefix=args.checkpoint_hub_prefix,
                                 keep_last=args.checkpoint_hub_keep_last,
                                 repo_type=args.checkpoint_hub_repo_type).validate()
        snapshot_download(MODEL, revision=REVISION, allow_patterns=[
            "*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
        ])
        env = {**os.environ, "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
               "HF_HUB_OFFLINE": "0" if args.checkpoint_hub_repo else "1"}
        run_logged_process([
            sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=2", "--max_restarts=0", str(Path(__file__).resolve()),
            "--index", str(args.index.resolve()), "--output", str(args.output.resolve()),
            "--run-name", args.run_name, "--seed", str(args.seed),
            "--max-train-seconds", str(args.max_train_seconds), *checkpoint_arguments(args),
        ], args.output / "first run.log", timeout=args.max_train_seconds + 900, env=env)
        ranks = [json.loads((args.output / f"rank{r}-summary.json").read_text()) for r in (0, 1)]
        if ranks[0]["status"] != ranks[1]["status"] or ranks[0]["progress"] != ranks[1]["progress"]:
            raise RuntimeError("DDP session results disagree")
        summary.update(status=ranks[0]["status"], progress=ranks[0]["progress"], ranks=ranks)
    except Exception as exc:
        summary.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        emit({"event": "launcher_finished", **summary})


if __name__ == "__main__":
    main()
