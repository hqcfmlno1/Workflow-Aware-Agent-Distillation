"""Single training entrypoint: validate, prepare, launch, or execute one worker."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.epoch_training import validate_progress
from workflow_distillation.training.compatibility import validate_parent
from workflow_distillation.training.config import load_profile
from workflow_distillation.training.data import TrainingDataset, batch_schedule, read_pool_plan
from workflow_distillation.training.events import EventLog
from workflow_distillation.training.identity import build_identity, read_binding
from workflow_distillation.training_checkpoints import (
    HuggingFacePublisher,
    _manifest,
    validate_checkpoint_arguments,
)

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class PreparedRun:
    profile: object
    plan: dict
    dataset: TrainingDataset
    batches: list
    binding: dict
    identity: dict
    progress: dict


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "index",
        "source-plan",
        "cohort",
        "pool-plan",
        "gpu-profile",
        "output",
        "resume-from",
    ):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--migration-parent-manifest", type=Path)
    result.add_argument("--migration-parent-manifest-sha256")
    result.add_argument("--run-name", default="first run")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--max-train-seconds", type=int, default=32400)
    result.add_argument("--stop-at-step", type=int)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--save-every-steps", type=int, default=10)
    result.add_argument("--keep-checkpoints", type=int, default=1)
    result.add_argument("--checkpoint-dir", type=Path)
    result.add_argument("--checkpoint-hub-repo")
    result.add_argument("--checkpoint-hub-prefix", default="first-run")
    result.add_argument("--stop-margin-seconds", type=int, default=300)
    result.add_argument("--expected-step-seconds", type=int, default=180)
    result.add_argument(
        "--checkpoint-hub-repo-type", choices=("model", "dataset"), default="dataset"
    )
    result.add_argument("--checkpoint-hub-keep-last", type=int)
    result.set_defaults(
        save_every_steps=10,
        keep_checkpoints=1,
        checkpoint_hub_prefix="first-run",
        stop_margin_seconds=300,
        expected_step_seconds=180,
    )
    return result


def validate_args(args):
    validate_checkpoint_arguments(args)
    if args.max_train_seconds <= 0 or (args.stop_at_step is not None and args.stop_at_step < 1):
        raise ValueError("Positive time budget and stop step required")
    if bool(args.migration_parent_manifest) != bool(args.migration_parent_manifest_sha256):
        raise ValueError("Migration requires both parent manifest and SHA256")


def prepare(args):
    validate_args(args)
    profile = load_profile(args.gpu_profile)
    plan = read_pool_plan(args.index, args.source_plan, args.cohort, args.pool_plan)
    resumed = _manifest(args.resume_from)
    binding = read_binding(
        resumed, args.migration_parent_manifest, args.migration_parent_manifest_sha256
    )
    parent = binding["manifest"]
    validate_parent(parent, profile, plan, args.seed)
    identity = build_identity(binding, profile, plan, args.seed)
    exact_parent = sha256_file(args.resume_from / "manifest.json") == binding["manifest_sha256"]
    if not exact_parent and resumed["identity"] != identity:
        raise ValueError("Resume is neither the exact migration parent nor this exact child")
    if resumed["world_size"] != profile.world_size:
        raise ValueError("Resume world size differs")
    transition = (
        parent["identity"]["settings"].get("microbatch_transition", {}).get("parent_progress")
    )
    batches = batch_schedule(plan["total_records"], profile.per_gpu_batch_size, transition)
    validate_progress(batches, parent["progress"])
    validate_progress(batches, resumed["progress"])
    if args.stop_at_step is not None and not resumed["progress"][
        "global_step"
    ] < args.stop_at_step <= len(batches):
        raise ValueError("Stop step must be after resume and within the data schedule")
    dataset = TrainingDataset(args.index, plan, verify="LOCAL_RANK" not in os.environ)
    if any(
        row["length"] >= plan["max_tokens"]
        for row in dataset.records[resumed["progress"]["samples_consumed"] :]
    ):
        raise ValueError("Untrained dataset suffix violates the token limit")
    return PreparedRun(profile, plan, dataset, batches, binding, identity, resumed["progress"])


def ensure_fresh_output(output):
    if (
        any(output.glob("rank*.jsonl"))
        or (output / "report.json").exists()
        or (output / "STOP").exists()
    ):
        raise ValueError("Choose a fresh output directory; existing run artifacts found")


def worker_command(args, arguments):
    return [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={2}",
        "--max_restarts=0",
        "--log-dir",
        str(args.output / "torchrun"),
        "--tee",
        "3",
        str(ROOT / "scripts/train.py"),
        *arguments,
    ]


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    prepared = prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "event": "preflight_verified",
        "samples": len(prepared.dataset),
        "total_optimizer_steps": len(prepared.batches),
        "resume_progress": prepared.progress,
        "identity": prepared.identity,
        "cuda_pilot_verified": False,
    }
    if "LOCAL_RANK" in os.environ:
        from workflow_distillation.training.engine import worker

        worker(args, prepared)
        return
    ensure_fresh_output(args.output)
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "launch.json").write_text(json.dumps(arguments, indent=2) + "\n")
    emit = EventLog(args.output / "launcher.jsonl", args.run_name)
    emit(report)
    if args.preflight_only:
        return
    import subprocess

    import torch
    from huggingface_hub import snapshot_download

    devices = [
        {
            "name": torch.cuda.get_device_name(i),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ]
    prepared.profile.validate_devices(devices)
    power = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit,power.max_limit",
            "--format=csv",
        ],
        text=True,
    )
    (args.output / "hardware-audit.json").write_text(
        json.dumps({"devices": devices, "power": power}, indent=2) + "\n"
    )
    if args.checkpoint_hub_repo:
        HuggingFacePublisher(
            args.checkpoint_hub_repo,
            prefix=args.checkpoint_hub_prefix,
            keep_last=args.checkpoint_hub_keep_last,
            repo_type=args.checkpoint_hub_repo_type,
        ).validate()
    snapshot_download(
        prepared.identity["model"],
        revision=prepared.identity["revision"],
        local_files_only=True,
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "model*.safetensors.index.json",
        ],
    )
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "2",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    from workflow_distillation.training.monitor import RunMonitor
    from workflow_distillation.training.process import run_logged_process

    try:
        with RunMonitor(args.output) as monitor:
            run_logged_process(
                worker_command(args, arguments),
                args.output / "first run.log",
                timeout=args.max_train_seconds + 600,
                env=env,
                monitor=monitor,
            )
        ranks = [
            json.loads((args.output / f"rank{rank}-summary.json").read_text()) for rank in (0, 1)
        ]
        if ranks[0]["status"] != ranks[1]["status"] or ranks[0]["progress"] != ranks[1]["progress"]:
            raise RuntimeError("Rank results disagree")
        (args.output / "report.json").write_text(json.dumps(ranks, indent=2) + "\n")
    except Exception as exc:
        (args.output / "report.json").write_text(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}, indent=2
            )
            + "\n"
        )
        raise
