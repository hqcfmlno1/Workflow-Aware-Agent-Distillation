"""CPU-only progress and fail-fast supervision for the owned torchrun process."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from workflow_distillation.training.memory import cgroup_memory


def tail_events(path, limit=131072):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            raw = stream.read().decode("utf-8", errors="replace")
        events = []
        for line in raw.splitlines():
            with suppress(ValueError):
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
        return events
    except FileNotFoundError:
        return []


def progress_summary(output):
    state = {
        "global_step": 0,
        "samples_consumed": 0,
        "checkpoint_step": 0,
        "phases": {},
        "loss": None,
        "total_steps": None,
        "total_samples": None,
    }
    with suppress(OSError, ValueError, KeyError):
        preflight = json.loads((Path(output) / "preflight.json").read_text())
        state.update(
            total_steps=preflight["total_optimizer_steps"], total_samples=preflight["samples"]
        )
    for rank in (0, 1):
        events = tail_events(Path(output) / f"rank{rank}.jsonl")
        if events:
            state["phases"][str(rank)] = events[-1].get("event", "unknown")
        for event in events:
            if event.get("event") == "model_load_started":
                state["total_steps"] = event["total_optimizer_steps"]
                state["total_samples"] = event["total_samples"]
            if (
                event.get("event") in ("step", "resumed", "checkpoint_complete")
                and event["global_step"] >= state["global_step"]
            ):
                state.update(
                    global_step=event["global_step"], samples_consumed=event["samples_consumed"]
                )
                if "loss" in event:
                    state["loss"] = event["loss"]
            if event.get("event") in ("resumed", "checkpoint_complete"):
                state["checkpoint_step"] = max(state["checkpoint_step"], event["global_step"])
    return state


def child_processes(pid=None):
    pid = os.getpid() if pid is None else pid
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return {int(pid) for pid in path.read_text().split()}
    except OSError:
        return set()


def process_identity(pid):
    """Linux start time prevents a stale PID from identifying an unrelated job."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


class RunMonitor:
    def __init__(self, output, *, interval=2, status_interval=60):
        self.output = Path(output)
        self.folder = self.output / "telemetry"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.interval, self.status_interval = interval, status_interval
        self.finished = threading.Event()
        self.failure = None
        self.root_pid = None
        self.process_token = None
        self.owned_processes = {}
        self.process_lock = threading.RLock()

    def __enter__(self):
        self.oom_baseline = cgroup_memory().get("events", {})
        self.thread = threading.Thread(target=self.watch, daemon=True, name="training-monitor")
        self.thread.start()
        return self

    def stop_owned_worker(self, failure):
        self.failure = failure
        (self.folder / "failure.json").write_text(json.dumps(failure) + "\n")
        self.terminate_owned()

    def register_worker(self, pid, *, process_token=None):
        with self.process_lock:
            if self.root_pid is not None:
                raise ValueError("Monitor already owns a worker")
            self.root_pid = pid
            self.process_token = process_token
            identity = process_identity(pid)
            if identity is not None:
                self.owned_processes[pid] = identity
            if self.failure is not None:
                self.terminate_owned()

    def terminate_owned(self):
        with self.process_lock:
            self.track_owned_processes()
            for pid, identity in reversed(list(self.owned_processes.items())):
                with suppress(ProcessLookupError):
                    if process_identity(pid) == identity:
                        # Torchrun can give each worker a separate session.
                        os.kill(pid, signal.SIGKILL)

    def track_owned_processes(self):
        with self.process_lock:
            if self.process_token is not None:
                marker = f"WORKFLOW_TRAINING_PROCESS_TOKEN={self.process_token}".encode()
                # The canonical runner supplies an inherited per-launch marker.
                # It also finds ranks orphaned before a parent-tree scan. Never
                # infer ownership from new unrelated children or session IDs.
                for path in Path("/proc").glob("[0-9]*/environ"):
                    with suppress(OSError):
                        if marker in path.read_bytes().split(b"\0"):
                            pid = int(path.parent.name)
                            identity = process_identity(pid)
                            if identity is not None:
                                self.owned_processes[pid] = identity
            pending = [
                pid
                for pid, identity in self.owned_processes.items()
                if process_identity(pid) == identity
            ]
            visited = set()
            while pending:
                pid = pending.pop()
                if pid in visited:
                    continue
                visited.add(pid)
                identity = process_identity(pid)
                if identity is not None:
                    self.owned_processes[pid] = identity
                    pending.extend(child_processes(pid))

    def watch(self):
        last_status = -float("inf")
        while not self.finished.is_set():
            try:
                self.track_owned_processes()
                memory = cgroup_memory()
                events = memory.get("events", {})
                if any(
                    events.get(k, 0) > self.oom_baseline.get(k, 0)
                    for k in ("oom_kill", "oom_group_kill")
                ):
                    self.stop_owned_worker({"event": "host_oom_detected", "cgroup": memory})
                    return
                for rank in (0, 1):
                    failed = next(
                        (
                            e
                            for e in reversed(tail_events(self.output / f"rank{rank}.jsonl"))
                            if e.get("event") == "failed"
                        ),
                        None,
                    )
                    if failed:
                        self.stop_owned_worker(
                            {
                                "event": "rank_failure_detected",
                                "rank": rank,
                                "error_type": failed.get("error_type"),
                            }
                        )
                        return
                if time.monotonic() - last_status >= self.status_interval:
                    state = progress_summary(self.output)
                    state.update(time=datetime.now(UTC).isoformat(), cgroup=memory)
                    temporary = self.folder / "status.json.tmp"
                    temporary.write_text(json.dumps(state) + "\n")
                    temporary.replace(self.folder / "status.json")
                    line = (
                        f"[{state['time']}] step {state['global_step']}/{state['total_steps']} | "
                        f"trajectories {state['samples_consumed']}/{state['total_samples']} | "
                        f"checkpoint {state['checkpoint_step']} | loss {state['loss']} | "
                        f"{state['phases']}"
                    )
                    with (self.folder / "progress.log").open("a") as stream:
                        stream.write(line + "\n")
                    with (self.folder / "memory.jsonl").open("a") as stream:
                        stream.write(json.dumps(state) + "\n")
                    print(line, flush=True)
                    last_status = time.monotonic()
            except Exception as exc:
                print(
                    json.dumps({"event": "monitor_error", "error_type": type(exc).__name__}),
                    flush=True,
                )
            self.finished.wait(self.interval)

    def __exit__(self, *_args):
        self.finished.set()
        self.thread.join(timeout=max(3, self.interval + 1))
