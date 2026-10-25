"""Resume with epoch-only checkpoint IO, leaving frozen training identity intact.

This entrypoint changes only the save/upload callbacks. Model, data, optimizer,
scheduler, scaler, RNG restoration and original trainer sources remain unchanged,
so an existing run can resume without rewriting its checkpoint manifests.
The original config's save_every is historical; runtime-checkpoint-policy.json
and the checkpoint_policy event record the effective storage schedule.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from workflow_distillation.lora_fp16 import cli
from workflow_distillation.lora_fp16.checkpoints import HuggingFacePublisher
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.engine import run_loop


def publish_preserving_rng(publisher, path):
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            publisher(path)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def epoch_loop(publisher, output):
    def run(*args, **kwargs):
        interval = kwargs["steps_per_epoch"]
        save_local = kwargs["save"]
        emit = kwargs["emit"]
        uploaded = set()

        def save(position):
            save_local(position)
            step = position["global_step"]
            if step > 0 and step % interval == 0 and step not in uploaded:
                path = Path(output) / "checkpoints" / f"checkpoint-{step:08d}"
                emit({"event": "epoch_upload_started", **position})
                publish_preserving_rng(publisher, path)
                uploaded.add(step)
                emit({"event": "epoch_upload_complete", **position})

        emit({"event": "checkpoint_policy", "save_every": interval,
              "upload": "epoch_end_only", "graceful_stop_checkpoint": "local_only"})
        kwargs["save_every"] = interval
        kwargs["save"] = save
        position = kwargs["progress"]
        if position["global_step"] > 0 and position["global_step"] % interval == 0:
            # A network failure may have occurred after the local epoch commit.
            # Retry it before consuming another batch; Hub publication is idempotent.
            save(position)
        return run_loop(*args, **kwargs)

    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "index", "selection", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--hub-repo", required=True)
    parser.add_argument("--hub-prefix", required=True)
    parser.add_argument("--resume", default="latest",
                        help="latest or a trusted local checkpoint directory")
    parser.add_argument("--max-seconds", type=int, default=28800)
    parser.add_argument("--prune-hub-history", action="store_true")
    args = parser.parse_args()
    if args.resume == "hub":
        parser.error("Use latest or a trusted local checkpoint directory for this entrypoint")
    config = RunConfig(**json.loads(args.config.read_text()))
    _, train, _, batches, _ = cli.prepare(config, args.index, args.selection)
    interval = (len(train) + config.accumulation - 1) // config.accumulation
    pins = list(range(interval, len(batches) + 1, interval))
    publisher = HuggingFacePublisher(
        args.hub_repo, prefix=args.hub_prefix, pinned_steps=pins,
        keep_last=config.keep_recovery if args.prune_hub_history else None,
    )
    # Hub connectivity is needed only at an epoch upload. The base and resume
    # state are local: do not make an idle paid GPU wait for a startup API call.
    policy = {"save_every": interval, "epoch_checkpoint_steps": pins,
              "upload": "epoch_end_only", "graceful_stop_checkpoint": "local_only",
              "original_identity_preserved": True}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runtime-checkpoint-policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    cli.run_loop = epoch_loop(publisher, args.output)
    # Keep network publication outside the legacy manager: its stop checkpoint
    # remains local; only the two epoch-boundary callbacks above touch the Hub.
    cli.main([
        "--mode", "train", "--config", str(args.config), "--index", str(args.index),
        "--selection", str(args.selection), "--output", str(args.output),
        "--resume", args.resume, "--max-seconds", str(args.max_seconds),
    ])


if __name__ == "__main__":
    main()
