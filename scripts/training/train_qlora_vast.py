"""Resume a pinned filtered checkpoint on Vast with the checked diversity schedule."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud/kaggle/recovery"))

from workflow_distillation.epoch_data import sha256_file  # noqa: E402
from workflow_distillation.epoch_training import run_logged_process  # noqa: E402
from workflow_distillation.training_checkpoints import (  # noqa: E402
    _manifest,
    add_checkpoint_arguments,
    checkpoint_arguments,
    validate_checkpoint_arguments,
)
from workflow_distillation.vast_continuation import bind_parent, configure_trainer  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--source-plan", required=True, type=Path)
    p.add_argument("--cohort", required=True, type=Path)
    p.add_argument("--parent-checkpoint", required=True, type=Path)
    p.add_argument("--parent-manifest-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--run-name", default="first run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-seconds", type=int, default=32400)
    p.add_argument(
        "--stop-at-step", type=int, help="Pilot stops and saves at this optimizer boundary"
    )
    p.add_argument("--preflight-only", action="store_true")
    add_checkpoint_arguments(p)
    p.add_argument("--checkpoint-hub-repo-type", choices=["model", "dataset"], default="dataset")
    p.add_argument("--checkpoint-hub-keep-last", type=int)
    p.set_defaults(
        keep_checkpoints=1,
        checkpoint_hub_prefix="first-run",
        stop_margin_seconds=300,
        expected_step_seconds=180,
    )
    return p


def prepare(args):
    # This preflight runs before any model weights are allocated on a GPU.
    import workflow_distillation.qlora_runtime as runtime
    from workflow_distillation.training_checkpoints import CheckpointManager

    if "LOCAL_RANK" in os.environ:
        import torch
        from adaptive_memory import install_adaptive_execution
        from host_memory_recovery import emit, install_cache_cleanup, install_checkpoint_cleanup

        install_cache_cleanup(torch)
        install_checkpoint_cleanup(torch, CheckpointManager)
        install_adaptive_execution(torch, runtime, CheckpointManager, emit)
    spec = importlib.util.spec_from_file_location(
        "vast_original_epoch", ROOT / "scripts/training/train_qlora_epoch.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    cohort = json.loads(args.cohort.read_text())
    binding = bind_parent(
        args.index,
        args.source_plan,
        cohort,
        args.parent_checkpoint / "manifest.json",
        expected_sha256=args.parent_manifest_sha256,
        filtered_entrypoint=ROOT / "scripts/training/train_qlora_filtered.py",
    )
    configure_trainer(
        trainer,
        cohort,
        binding,
        entrypoint=__file__,
        stop_at_step=args.stop_at_step,
        stop_file=args.output / "STOP",
    )
    source = ROOT / "src/workflow_distillation"
    identity = trainer.training_identity(
        __file__,
        model=trainer.MODEL,
        revision=trainer.REVISION,
        records_sha256=sha256_file(args.index),
        settings={
            "config": trainer.CONFIG,
            "seed": args.seed,
            "runtime_sources": {
                name: sha256_file(source / name)
                for name in (
                    "epoch_data.py",
                    "epoch_training.py",
                    "qlora_runtime.py",
                    "weighted_sft_batches.py",
                    "training_checkpoints.py",
                )
            },
        },
    )
    return trainer, cohort, binding, identity


def main():
    args = parser().parse_args()
    validate_checkpoint_arguments(args)
    if args.max_train_seconds <= 0:
        raise ValueError("Positive training budget required")
    args.resume_from = args.resume_from or str(args.parent_checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FIRST_RUN_RECOVERY_LOG_DIR"] = str(args.output / "telemetry")
    trainer, cohort, binding, identity = prepare(args)
    if "LOCAL_RANK" in os.environ:
        import torch
        from host_memory_recovery import start_monitor

        start_monitor(torch, progress_dir=args.output)
        trainer.worker(args)
        return
    parent = _manifest(args.parent_checkpoint)
    if parent["world_size"] != 2:
        raise ValueError("Vast continuation requires exactly two saved ranks")
    dataset = trainer.EpochDataset(args.index)
    trainer.check_dataset(dataset)
    resume = _manifest(Path(args.resume_from))
    is_parent = (
        sha256_file(Path(args.resume_from) / "manifest.json")
        == binding["parent"]["manifest_sha256"]
    )
    if not is_parent and resume["identity"] != identity:
        raise ValueError("Resume checkpoint is neither the pinned parent nor this Vast run")
    from workflow_distillation.epoch_training import validate_progress

    validate_progress(trainer.epoch_batches(len(dataset), args.seed), resume["progress"])
    report = {
        "event": "vast_preflight_verified",
        "parent_progress": parent["progress"],
        "resume_progress": resume["progress"],
        "samples": len(dataset),
        "total_optimizer_steps": cohort["target_optimizer_steps"],
        "identity": identity,
    }
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    trainer.EventLog(args.output / "launcher.jsonl", args.run_name)(report)
    if args.preflight_only:
        return
    import torch

    if torch.cuda.device_count() != 2 or any(
        "3090" not in torch.cuda.get_device_name(i) for i in range(2)
    ):
        raise ValueError("This launch policy was selected for exactly two RTX3090 GPUs")
    if args.checkpoint_hub_repo:
        trainer.HuggingFacePublisher(
            args.checkpoint_hub_repo,
            prefix=args.checkpoint_hub_prefix,
            keep_last=args.checkpoint_hub_keep_last,
            repo_type=args.checkpoint_hub_repo_type,
        ).validate()
    # Require the pinned snapshot prepared during setup.
    # The training binaries already contain tokens, so no tokenizer download is needed.
    from huggingface_hub import snapshot_download

    snapshot_download(
        trainer.MODEL,
        revision=trainer.REVISION,
        local_files_only=True,
        allow_patterns=["config.json", "model*.safetensors", "model*.safetensors.index.json"],
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
    from host_memory_recovery import start_monitor

    os.environ["FIRST_RUN_MONITOR_OWNER_PID"] = str(os.getpid())
    env["FIRST_RUN_MONITOR_OWNER_PID"] = str(os.getpid())
    start_monitor(progress_dir=args.output)
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "--max_restarts=0",
        "--log-dir",
        str(args.output / "torchrun"),
        "--tee",
        "3",
        str(Path(__file__).resolve()),
    ]
    for flag, value in (
        ("--index", args.index),
        ("--source-plan", args.source_plan),
        ("--cohort", args.cohort),
        ("--parent-checkpoint", args.parent_checkpoint),
        ("--parent-manifest-sha256", args.parent_manifest_sha256),
        ("--output", args.output),
        ("--run-name", args.run_name),
        ("--seed", args.seed),
        ("--max-train-seconds", args.max_train_seconds),
        ("--stop-at-step", args.stop_at_step),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    command.extend(checkpoint_arguments(args))
    run_logged_process(
        command, args.output / "first run.log", timeout=args.max_train_seconds + 600, env=env
    )
    ranks = [json.loads((args.output / f"rank{r}-summary.json").read_text()) for r in (0, 1)]
    if ranks[0]["status"] != ranks[1]["status"] or ranks[0]["progress"] != ranks[1]["progress"]:
        raise RuntimeError("Rank results disagree")
    (args.output / "report.json").write_text(json.dumps(ranks, indent=2) + "\n")


if __name__ == "__main__":
    main()
