"""Detach a training session from SSH; keep an exclusive launch lock and PID."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/training/rtx5090_32gb/tqs060-fp16"
    )
    parser.add_argument("--hub-repo", default="hqcfmlno1/QwenStacy-4b-fp16-tqs060-run1")
    parser.add_argument("--no-hub", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)
    output = args.output.resolve()
    command = [
        sys.executable,
        "-u",
        "-m",
        "workflow_distillation.lora_fp16",
        "--mode",
        "train",
        "--output",
        str(output),
        *extra,
    ]
    if not args.no_hub:
        command += ["--hub-repo", args.hub_repo]
    logfile = output / "train.log"
    print(shlex.join(command), flush=True)
    print(f"Log: tail -n 30 -F {shlex.quote(str(logfile))}", flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_lock = (output / ".run.lock").open("a")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            run_lock.close()
        if not args.no_hub:
            from huggingface_hub import HfApi

            api = HfApi()
            # Existing public repos are never repurposed as private state stores.
            from huggingface_hub.errors import RepositoryNotFoundError

            try:
                if not api.repo_info(args.hub_repo).private:
                    raise ValueError("Checkpoint repo must be private and dedicated")
            except RepositoryNotFoundError:
                api.create_repo(args.hub_repo, private=True)
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
        with logfile.open("ab", buffering=0) as log, open(os.devnull, "rb") as null:
            child = subprocess.Popen(
                command,
                stdin=null,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=env,
                cwd=ROOT,
            )
        (output / "launcher.json").write_text(
            json.dumps({"pid": child.pid, "command": command}, indent=2) + "\n"
        )
        # Give admission/argument failures a chance to surface before reporting a PID.
        time.sleep(2)
        code = child.poll()
        if code is not None and code != 0:
            raise RuntimeError(f"Training exited with code {code}; inspect {logfile}")
        print(
            f"Process started: PID {child.pid}. SSH may disconnect. "
            "Startup is not a completed update.",
            flush=True,
        )


if __name__ == "__main__":
    main()
