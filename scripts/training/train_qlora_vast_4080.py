"""Run the unchanged pinned pool worker on an admitted 32GB RTX4080 pair.

The existing checkpoint numerical identity remains exact: this launcher changes
only hardware admission, recording the actual device separately in an audit.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud/kaggle/recovery"))

from workflow_distillation.epoch_data import sha256_file  # noqa: E402
from workflow_distillation.epoch_training import run_logged_process, validate_progress  # noqa: E402
from workflow_distillation.training_checkpoints import (  # noqa: E402
    _manifest,
    validate_checkpoint_arguments,
)
from workflow_distillation.vast_device_policy import (  # noqa: E402
    boundary_logger,
    install_cgroup_compatibility,
    validate_4080_pair,
)

spec = importlib.util.spec_from_file_location(
    "original_pool_launcher", Path(__file__).with_name("train_qlora_vast_pool.py")
)
pool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pool
spec.loader.exec_module(pool)


def main():
    args = pool.parser().parse_args()
    validate_checkpoint_arguments(args)
    if args.max_train_seconds <= 0:
        raise ValueError("Positive budget required")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["FIRST_RUN_RECOVERY_LOG_DIR"] = str(args.output / "telemetry")
    install_cgroup_compatibility()
    stop_at_step, args.stop_at_step = args.stop_at_step, None
    trainer, cohort, binding, identity = pool.prepare(args)
    args.stop_at_step = stop_at_step
    if stop_at_step is not None:
        if stop_at_step <= _manifest(Path(args.resume_from))["progress"]["global_step"]:
            raise ValueError("Pilot stop must be after the restored checkpoint")
        original_log = trainer.EventLog

        def event_log(*positional, **keywords):
            return boundary_logger(original_log(*positional, **keywords),
                                   args.output / "STOP", stop_at_step)

        trainer.EventLog = event_log
    if "LOCAL_RANK" in os.environ:
        pool.batch4.worker(trainer, args, binding, identity)
        return
    dataset = trainer.EpochDataset(args.index)
    trainer.check_dataset(dataset)
    resumed = _manifest(Path(args.resume_from))
    if (sha256_file(Path(args.resume_from) / "manifest.json")
            != binding["parent"]["manifest_sha256"] and resumed["identity"] != identity):
        raise ValueError("Resume identity differs from exact parent or pool child")
    validate_progress(trainer.epoch_batches(len(dataset), args.seed), resumed["progress"])
    report = {"event": "vast_4080_preflight_verified", "samples": len(dataset),
              "total_optimizer_steps": cohort["target_optimizer_steps"],
              "resume_progress": resumed["progress"], "identity": identity}
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    trainer.EventLog(args.output / "launcher.jsonl", args.run_name)(report)
    if args.preflight_only:
        return
    import torch
    devices = validate_4080_pair([
        {"name": torch.cuda.get_device_name(i),
         "total_memory": torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())
    ])
    power = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,power.limit,power.max_limit",
        "--format=csv",
    ], text=True)
    (args.output / "hardware-audit.json").write_text(json.dumps({
        "devices": devices, "nvidia_smi": power,
        "numerical_identity_changed": resumed["identity"] != identity,
        "worker_body_changed": False,
        "launcher_sha256": sha256_file(Path(__file__)),
    }, indent=2) + "\n")
    if args.checkpoint_hub_repo:
        trainer.HuggingFacePublisher(
            args.checkpoint_hub_repo, prefix=args.checkpoint_hub_prefix,
            keep_last=args.checkpoint_hub_keep_last,
            repo_type=args.checkpoint_hub_repo_type,
        ).validate()
    from huggingface_hub import snapshot_download
    snapshot_download(trainer.MODEL, revision=trainer.REVISION, local_files_only=True,
                      allow_patterns=["config.json", "model*.safetensors",
                                      "model*.safetensors.index.json"])
    env = {**os.environ, "OMP_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1",
           "TOKENIZERS_PARALLELISM": "false", "PYTORCH_ALLOC_CONF": "expandable_segments:True",
           "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
           "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
           "FIRST_RUN_MONITOR_OWNER_PID": str(os.getpid())}
    os.environ["FIRST_RUN_MONITOR_OWNER_PID"] = str(os.getpid())
    from host_memory_recovery import start_monitor
    start_monitor(progress_dir=args.output)
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", "--max_restarts=0", "--log-dir",
               str(args.output / "torchrun"), "--tee", "3", str(Path(__file__).resolve()),
               *sys.argv[1:]]
    run_logged_process(command, args.output / "first run.log",
                       timeout=args.max_train_seconds + 600, env=env)
    ranks = [json.loads((args.output / f"rank{r}-summary.json").read_text()) for r in (0, 1)]
    if ranks[0]["status"] != ranks[1]["status"] or ranks[0]["progress"] != ranks[1]["progress"]:
        raise RuntimeError("Rank results disagree")
    (args.output / "report.json").write_text(json.dumps(ranks, indent=2) + "\n")


if __name__ == "__main__":
    main()
