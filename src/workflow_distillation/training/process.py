"""Owned torchrun subprocess and streaming logs, without patching subprocess APIs."""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from threading import Thread


def run_logged_process(command, log_path, *, timeout, monitor, env=None):
    """Tee both streams without buffering training logs; bound all descendants."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    process_token = uuid.uuid4().hex
    environment = {
        **(os.environ if env is None else env),
        "WORKFLOW_TRAINING_PROCESS_TOKEN": process_token,
    }
    with path.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        monitor.register_worker(process.pid, process_token=process_token)
        failures = []

        def copy():
            try:
                for line in iter(process.stdout.readline, b""):
                    stream.write(line)
                    sys.stdout.write(line.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            except Exception as exc:
                failures.append(exc)
                monitor.terminate_owned()

        pump = Thread(target=copy, daemon=True)
        pump.start()
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            monitor.terminate_owned()
            process.wait()
            raise
        finally:
            # A fast launcher exit can orphan detached ranks before the first
            # tree scan. The inherited run marker identifies those processes.
            monitor.terminate_owned()
            pump.join(timeout=10)
            if pump.is_alive():
                monitor.terminate_owned()
                pump.join(timeout=5)
            process.stdout.close()
        if failures:
            raise failures[0]
        if pump.is_alive():
            raise RuntimeError("Training log stream did not terminate")
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
