"""Continue the exact Vast cohort with a real microbatch of two per RTX3090."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud/kaggle/recovery"))

from workflow_distillation.epoch_data import sha256_file  # noqa: E402
from workflow_distillation.epoch_training import run_logged_process, validate_progress  # noqa: E402
from workflow_distillation.training_checkpoints import (  # noqa: E402
    HuggingFacePublisher,
    _manifest,
    checkpoint_arguments,
    validate_checkpoint_arguments,
)
from workflow_distillation.vast_microbatch import (  # noqa: E402
    batch_schedule,
    migration_manager,
    transition_identity,
    worker,
)

spec = importlib.util.spec_from_file_location(
    "vast_batch_two_baseline", Path(__file__).with_name("train_qlora_vast.py")
)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)


def parser():
    p = baseline.parser()
    p.description = __doc__
    p.add_argument("--batch-parent", required=True, type=Path)
    p.add_argument("--batch-parent-manifest-sha256", required=True)
    return p


def prepare(args):
    from filtered_continuation import publish_continuation

    # Rebuild and verify the original immutable baseline, without installing
    # its single-input-only adaptive execution hooks in a batch-two worker.
    local_rank = os.environ.pop("LOCAL_RANK", None)
    try:
        trainer, cohort, _, old_identity = baseline.prepare(args)
    finally:
        if local_rank is not None:
            os.environ["LOCAL_RANK"] = local_rank
    path = args.batch_parent
    checksum = sha256_file(path / "manifest.json")
    if checksum != args.batch_parent_manifest_sha256:
        raise ValueError("Pinned batch-two parent checksum differs")
    parent = _manifest(path)
    identity = transition_identity(parent, old_identity, checksum, __file__)
    batches = batch_schedule(cohort["total_records"], parent["progress"])
    binding = {
        "parent": {
            "manifest_sha256": checksum,
            **{k: parent[k] for k in ("identity", "progress", "world_size")},
        }
    }
    dataset = trainer.EpochDataset(args.index, verify=False)
    if any(r["length"] >= 28000 for r in dataset.records[parent["progress"]["samples_consumed"] :]):
        raise ValueError("Untrained cohort must remain below 28000 tokens")

    def schedule(count, seed=42):
        if count != cohort["total_records"] or seed != cohort["seed"]:
            raise ValueError("Cohort count or seed differs")
        return batches

    original_publish = HuggingFacePublisher.__call__

    class Publisher(HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, binding, identity, original_publish)

    trainer.CONFIG = dict(identity["settings"]["config"])
    trainer.CheckpointManager = migration_manager(binding)
    trainer.HuggingFacePublisher = Publisher
    trainer.epoch_batches = schedule
    cohort = {**cohort, "target_optimizer_steps": len(batches)}
    return trainer, cohort, binding, identity


def main():
    args = parser().parse_args()
    validate_checkpoint_arguments(args)
    if args.max_train_seconds <= 0 or args.stop_at_step is not None:
        raise ValueError("Positive budget required; use STOP for a safe early stop")
    args.resume_from = args.resume_from or str(args.batch_parent)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FIRST_RUN_RECOVERY_LOG_DIR"] = str(args.output / "telemetry")
    trainer, cohort, binding, identity = prepare(args)
    if "LOCAL_RANK" in os.environ:
        worker(trainer, args, binding, identity)
        return
    dataset = trainer.EpochDataset(args.index)
    trainer.check_dataset(dataset)
    resumed = _manifest(Path(args.resume_from))
    if (
        sha256_file(Path(args.resume_from) / "manifest.json")
        != binding["parent"]["manifest_sha256"]
        and resumed["identity"] != identity
    ):
        raise ValueError("Resume identity differs from exact parent or batch-four child")
    validate_progress(trainer.epoch_batches(len(dataset), args.seed), resumed["progress"])
    report = {
        "event": "vast_batch4_preflight_verified",
        "samples": len(dataset),
        "total_optimizer_steps": cohort["target_optimizer_steps"],
        "resume_progress": resumed["progress"],
        "identity": identity,
    }
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    trainer.EventLog(args.output / "launcher.jsonl", args.run_name)(report)
    if args.preflight_only:
        return
    import torch

    if torch.cuda.device_count() != 2 or any(
        "3090" not in torch.cuda.get_device_name(i) for i in (0, 1)
    ):
        raise ValueError("Expected exactly two RTX3090 GPUs")
    if args.checkpoint_hub_repo:
        trainer.HuggingFacePublisher(
            args.checkpoint_hub_repo,
            prefix=args.checkpoint_hub_prefix,
            keep_last=args.checkpoint_hub_keep_last,
            repo_type=args.checkpoint_hub_repo_type,
        ).validate()
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
        "FIRST_RUN_MONITOR_OWNER_PID": str(os.getpid()),
    }
    os.environ["FIRST_RUN_MONITOR_OWNER_PID"] = str(os.getpid())
    from host_memory_recovery import start_monitor

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
    for name, value in (
        ("--index", args.index),
        ("--source-plan", args.source_plan),
        ("--cohort", args.cohort),
        ("--parent-checkpoint", args.parent_checkpoint),
        ("--parent-manifest-sha256", args.parent_manifest_sha256),
        ("--batch-parent", args.batch_parent),
        ("--batch-parent-manifest-sha256", args.batch_parent_manifest_sha256),
        ("--output", args.output),
        ("--run-name", args.run_name),
        ("--seed", args.seed),
        ("--max-train-seconds", args.max_train_seconds),
    ):
        command.extend([name, str(value)])
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
