"""Optimizer-boundary orchestration for exactly one deterministic epoch."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from threading import Thread


def initial_progress():
    return {"global_step": 0, "next_batch": 0, "samples_consumed": 0}


def validate_progress(batches, progress):
    position = progress.get("next_batch")
    if (type(position) is not int or not 0 <= position <= len(batches)
            or progress.get("global_step") != position
            or progress.get("samples_consumed") != sum(map(len, batches[:position]))):
        raise ValueError("Checkpoint epoch cursor differs from completed batches")


def run_epoch_loop(batches, *, progress, step, save, stop, emit, save_every=5):
    """Callbacks perform real updates/collective IO; no data work is replayed."""
    validate_progress(batches, progress)
    progress = dict(progress)
    for position in range(progress["next_batch"], len(batches)):
        if stop():
            save(progress)
            return {"status": "stopped_checkpointed", "progress": progress}
        batch = batches[position]
        metrics = step(batch, position + 1)
        progress = {"global_step": position + 1, "next_batch": position + 1,
                    "samples_consumed": progress["samples_consumed"] + len(batch)}
        emit({"event": "step", **progress, **metrics})
        stopping = stop()
        if (position == 0 or (position + 1) % save_every == 0
                or position + 1 == len(batches) or stopping):
            save(progress)
        if stopping and position + 1 < len(batches):
            return {"status": "stopped_checkpointed", "progress": progress}
    return {"status": "completed", "progress": progress}


def run_logged_process(command, log_path, *, timeout, env=None):
    """Tee both streams without buffering training logs; bound all descendants."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        failures = []

        def copy():
            try:
                for line in iter(process.stdout.readline, b""):
                    stream.write(line)
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            except Exception as exc:
                failures.append(exc)
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)

        pump = Thread(target=copy, daemon=True)
        pump.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        finally:
            pump.join(timeout=10)
            if pump.is_alive():
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                pump.join(timeout=5)
            process.stdout.close()
        if failures:
            raise failures[0]
        if pump.is_alive():
            raise RuntimeError("Training log stream did not terminate")
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
