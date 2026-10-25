"""Preflight, isolated CUDA pilot, and fresh multi-epoch 16-bit LoRA on one 5090."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.checkpoints import CheckpointManager, HuggingFacePublisher
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.data import load_bundle
from workflow_distillation.lora_fp16.engine import (
    evaluate,
    run_loop,
    schedule,
    train_update,
    validate_cursor,
)
from workflow_distillation.lora_fp16.loss import group_loss_parameters
from workflow_distillation.training.events import EventLog
from workflow_distillation.training_checkpoints import (
    StopBudget,
    download_latest_checkpoint,
)

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "configs/training/gpus/rtx5090_32gb"


def validate_pilot(report, identity, device):
    if (
        report.get("passed") is not True
        or report.get("identity") != identity
        or report.get("device") != device
    ):
        raise ValueError("A passing pilot for this exact config/data/software/GPU is required")


def invalidate_pilot(path, identity, device):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"passed": False, "status": "running", "identity": identity, "device": device}, indent=2
        )
        + "\n"
    )
    temporary.replace(path)


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def prepare(config, index, selection_path=None):
    selection = json.loads((selection_path or PROFILE / "selection.json").read_text())
    if sha256_file(index) != selection["index_sha256"]:
        raise ValueError("Dataset differs from the frozen experiment selection")
    if selection_path is not None:
        from workflow_distillation.lora_fp16.metacurate import load_corpus

        if config.loss_normalization != "vanilla":
            raise ValueError("The complete MetaCurate profile requires vanilla SFT")
        dataset, split = load_corpus(index, selection)
    else:
        dataset, split = load_bundle(index)
        if split != selection["split"]:
            raise ValueError("Holdout selection differs")
    by_id = {r["trajectory_id"]: i for i, r in enumerate(dataset.records)}
    train = [by_id[t] for t in split["train_ids"]]
    validation = [by_id[t] for t in split["validation_ids"]]
    batches = schedule(
        len(train), epochs=config.epochs, accumulation=config.accumulation, seed=config.seed
    )
    source = ROOT / "src/workflow_distillation"
    paths = sorted((source / "lora_fp16").glob("*.py")) + [
        source / "weighted_sft_loss.py",
        source / "optimized_qlora.py",
        source / "training_checkpoints.py",
        source / "epoch_data.py",
        source / "training/activation_storage.py",
        source / "training/config.py",
        source / "training/model.py",
    ]
    identity = {
        "model": config.model,
        "revision": config.revision,
        "records_sha256": sha256_file(index),
        "settings": {
            "config": config.to_dict(),
            "loss_weights": list(config.group_weights),
            "packages": {
                p: package_version(p)
                for p in (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "numpy",
                    "fla-core",
                    "flash-linear-attention",
                    "triton",
                    "causal-conv1d",
                )
            },
            "sources": {str(p.relative_to(source)): sha256_file(p) for p in paths},
        },
    }
    return dataset, train, validation, batches, identity


def gpu_audit(config):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one RTX5090 with CUDA_VISIBLE_DEVICES")
    props = torch.cuda.get_device_properties(0)
    if "5090" not in props.name or props.total_memory < config.minimum_vram_bytes:
        raise ValueError("Expected RTX5090 with at least 30 GiB actual VRAM")
    power = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit,power.max_limit",
            "--format=csv",
        ],
        text=True,
    )
    # Fail before loading 8GB of weights if this torch build cannot execute on
    # Blackwell. Architecture strings alone need not cover PTX-compatible builds.
    if config.base_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU/runtime does not support BF16")
    probe = torch.ones((32, 32), device="cuda", dtype=getattr(torch, config.base_dtype))
    torch.testing.assert_close(probe @ probe, torch.full_like(probe, 32))
    torch.cuda.synchronize()
    return {
        "name": props.name,
        "total_memory": props.total_memory,
        "capability": list(torch.cuda.get_device_capability()),
        "power": power,
    }


def make_scheduler(optimizer, config, total):
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total * config.warmup_ratio), num_training_steps=total
    )


def pilot(state, dataset, train, config, device, identity, hardware, emit, destination):
    # Four longest trajectories test one complete accumulated optimizer update.
    longest = sorted(train, key=lambda i: dataset.records[i]["length"], reverse=True)
    by_length = sorted(train, key=lambda i: dataset.records[i]["length"])
    rng = np.random.default_rng(config.seed)
    typical = rng.choice(by_length, size=min(16, len(train)), replace=False).tolist()
    trials = [("longest", longest[: config.accumulation])]
    if config.loss_normalization == "vanilla":
        # Repeat with initialized Adam state and reversed long-sequence order.
        trials += [("longest_repeat", list(reversed(longest[: config.accumulation])))]
    trials += [
        ("representative", typical[i : i + config.accumulation])
        for i in range(0, len(typical), config.accumulation)
    ]
    results = []
    for label, indices in trials:
        result = train_update(state, [dataset.record(i) for i in indices], config, device, emit)
        result.update(
            kind=label, trajectory_lengths=[dataset.records[i]["length"] for i in indices]
        )
        results.append(result)
        emit({"event": "pilot_update", **result})
    representative = [r for r in results if r["kind"] == "representative"]
    throughput = sum(r["tokens"] for r in representative) / sum(
        r["seconds"] for r in representative
    )
    peak = max(r["peak_reserved"] for r in results)
    # Account for allocations made by other CUDA users at the measured boundary.
    headroom = min(
        r["cuda_capacity"]
        - r["peak_reserved"]
        - max(0, r["cuda_capacity"] - r["cuda_free"] - r["peak_reserved"])
        for r in results
    )
    report = {
        "identity": identity,
        "device": hardware,
        "updates": results,
        "passed": headroom >= config.minimum_headroom_bytes,
        "minimum_headroom_bytes": headroom,
        "peak_reserved": peak,
        "representative_tokens_per_second": throughput,
        "compute_hours_estimate": sum(dataset.records[i]["length"] for i in train)
        * config.epochs
        / throughput
        / 3600,
        "estimate_excludes": "validation, checkpoint/network IO and setup",
        "weights_discarded": True,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    emit(
        {
            "event": "pilot_complete",
            **{k: v for k, v in report.items() if k not in ("identity", "updates")},
        }
    )
    if not report["passed"]:
        raise MemoryError(
            "Pilot lacks required headroom; explicitly change activation policy and repeat"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "pilot", "train"), default="preflight")
    parser.add_argument("--config", type=Path, default=PROFILE / "run.json")
    parser.add_argument("--selection", type=Path, help="Pinned complete MetaCurate corpus receipt")
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data/processed/tqs1140_holdout20_fp16/index.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/training/rtx5090_32gb/tqs060-fp16"
    )
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--resume", help="latest (local), hub, or a trusted checkpoint directory")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-seconds", type=int, default=28800)
    parser.add_argument("--hub-repo", help="Dedicated private model repo for this fresh run")
    parser.add_argument("--hub-prefix", default="fp16-tqs060")
    parser.add_argument(
        "--prune-hub-history",
        action="store_true",
        help="Squash dedicated repo history; keep epochs + configured recovery count",
    )
    args = parser.parse_args(argv)
    if args.max_seconds < 1 or (args.prune_hub_history and not args.hub_repo):
        parser.error("Positive time budget required; history pruning requires a dedicated Hub repo")
    config = RunConfig(**json.loads(args.config.read_text()))
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    dataset, train, validation, batches, identity = prepare(config, args.index, args.selection)
    steps_per_epoch = math.ceil(len(train) / config.accumulation)
    pinned = list(range(steps_per_epoch, len(batches) + 1, steps_per_epoch))
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "event": "preflight_verified",
        "train_trajectories": len(train),
        "validation_trajectories": len(validation),
        "steps_per_epoch": steps_per_epoch,
        "total_updates": len(batches),
        "epoch_checkpoint_steps": pinned,
        "tokens_per_epoch": sum(dataset.records[i]["length"] for i in train),
        "max_tokens": max(dataset.records[i]["length"] for i in train),
        "config": config.to_dict(),
        "identity": identity,
        "cuda_pilot_verified": False,
    }
    if config.loss_normalization == "token_group":
        shares = []
        for i in train:
            groups = dataset.record(i)["groups"]
            coefficients, denominator = group_loss_parameters(
                groups, config.group_weights, config.length_alpha
            )
            counts = np.bincount(np.asarray(groups[1:], dtype=np.int64), minlength=5)
            shares.append(np.asarray(coefficients) * counts / denominator)
        report["mean_loss_coefficient_share"] = dict(
            zip(("masked", "reasoning", "content", "action", "submit"),
                np.mean(shares, axis=0).tolist(), strict=True)
        )
    if args.mode == "preflight":
        (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "identity"}, indent=2))
        return
    # A kernel-enforced lock prevents double writers, including across SSH sessions.
    import fcntl

    with (args.output / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute(args, config, dataset, train, validation, batches, identity, pinned, report)
        except Exception as exc:
            failure = {"event": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            EventLog(args.output / "failures.jsonl", args.output.name)(failure)
            raise


def execute(args, config, dataset, train, validation, batches, identity, pinned, report):
    import torch

    from workflow_distillation.lora_fp16.model import build_model

    emit = EventLog(
        args.output / ("pilot.jsonl" if args.mode == "pilot" else "events.jsonl"),
        args.output.name,
    )
    pilot_path = args.pilot_report or args.output / "pilot.json"
    if args.mode == "pilot":
        invalidate_pilot(pilot_path, identity, None)
    hardware = gpu_audit(config)
    (args.output / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    if args.mode == "train":
        validate_pilot(json.loads(pilot_path.read_text()), identity, hardware)
        if (args.output / "checkpoints/latest.json").exists() and not args.resume:
            raise ValueError("Existing checkpoints: pass --resume latest or choose a new output")
        if not args.resume and (args.output / "events.jsonl").exists():
            raise ValueError("Existing run events: choose a fresh output for a new run")
        if (args.output / "STOP").exists():
            raise ValueError("Remove STOP only when ready to resume")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    emit({**report, "event": "initializing", "hardware": hardware})
    manager = None
    started = time.monotonic()
    budget = StopBudget(
        max_seconds=args.max_seconds, margin_seconds=300, expected_step_seconds=180, started=started
    )
    with budget:
        publisher = None
        if args.mode == "train" and args.hub_repo:
            publisher = HuggingFacePublisher(
                args.hub_repo,
                prefix=args.hub_prefix,
                keep_last=config.keep_recovery if args.prune_hub_history else None,
                pinned_steps=pinned,
            )
            info = publisher.validate()
            from huggingface_hub import HfApi

            files = HfApi().list_repo_files(args.hub_repo, revision=info.sha)
            if f"{args.hub_prefix}/latest.json" in files and not args.resume:
                raise ValueError("Hub already contains this run: use --resume hub or a fresh repo")
        state = build_model(config)
        emit(
            {
                "event": "model_loaded",
                "trainable_parameters": sum(p.numel() for p in state.parameters),
                "base_dtype": config.base_dtype,
                "quantization": "none",
                "load_seconds": time.monotonic() - started,
            }
        )
        if args.mode == "pilot":
            pilot(state, dataset, train, config, device, identity, hardware, emit, pilot_path)
            return
        scheduler = make_scheduler(state.optimizer, config, len(batches))
        manager = CheckpointManager(
            args.output / "checkpoints",
            identity,
            save_every=config.save_every,
            keep_last=config.keep_recovery,
            pinned_steps=pinned,
            publisher=publisher,
        )
        progress = dict(global_step=0, next_batch=0, samples_consumed=0)
        if args.resume:
            resume = args.resume
            if resume == "hub":
                if not args.hub_repo:
                    raise ValueError("--resume hub requires --hub-repo")
                resume = download_latest_checkpoint(
                    args.hub_repo, args.output / "hub-download", prefix=args.hub_prefix
                )
            progress = manager.load(
                resume,
                model=state.model,
                optimizer=state.optimizer,
                scheduler=scheduler,
                scaler=state.scaler,
            )
            validate_cursor(batches, progress)
            manager.republish(model=state.model)
            emit({"event": "resumed", **progress})

        def should_stop():
            return budget.should_stop() or (args.output / "STOP").exists()

        def update(indices, number):
            metrics = train_update(
                state,
                [dataset.record(train[i]) for i in indices],
                config,
                device,
                lambda event: emit({**event, "global_step": number}),
            )
            used_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            budget.expected_step_seconds = max(budget.expected_step_seconds, metrics["seconds"])
            return {**metrics, "learning_rate": used_lr}

        def validate(epoch):
            if not validation:
                emit({"event": "epoch_boundary", "epoch": epoch,
                      "validation": "disabled; all selected trajectories used for training"})
                # Atomic checkpoints already contain adapter and optimizer state.
                # Also provide standard PEFT exports ready for inference.
                if epoch > 0:
                    export = args.output / "adapters" / f"epoch-{epoch:03d}"
                    state.model.save_pretrained(export, safe_serialization=True)
                return
            emit({"event": "validation_started", "epoch": epoch})
            metrics = evaluate(state, (dataset.record(i) for i in validation), config, device, emit)
            emit({"event": "validation_complete", "epoch": epoch, **metrics})
            if epoch > 0:
                export = args.output / "adapters" / f"epoch-{epoch:03d}"
                state.model.save_pretrained(export, safe_serialization=True)
                (export / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")

        def save(position):
            if manager.due(position["global_step"], force=True):
                emit({"event": "checkpoint_started", **position})
                manager.save(
                    model=state.model,
                    optimizer=state.optimizer,
                    scheduler=scheduler,
                    scaler=state.scaler,
                    progress=position,
                )
                emit(
                    {
                        "event": "checkpoint_complete",
                        **position,
                        "path": str(manager.last_checkpoint),
                    }
                )

        result = run_loop(
            batches,
            steps_per_epoch=len(batches) // config.epochs,
            progress=progress,
            update=update,
            evaluate=validate,
            save=save,
            emit=emit,
            save_every=config.save_every,
            should_stop=should_stop,
        )
        status = "complete" if result["global_step"] == len(batches) else "stopped"
        final = {"status": status, "progress": result, "seconds": time.monotonic() - started}
        (args.output / "summary.json").write_text(json.dumps(final, indent=2) + "\n")
        emit({"event": "session_finished", **final})
