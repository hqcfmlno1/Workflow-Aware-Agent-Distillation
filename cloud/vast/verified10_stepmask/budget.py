"""Linux wall-clock deadlines; campaign time is carried across clean resumes."""

from __future__ import annotations

import json
import math
import signal
import time
from contextlib import contextmanager
from pathlib import Path


class TaskBudgetExceeded(TimeoutError):
    kind = "task_budget_exhausted"


class RunBudgetExceeded(KeyboardInterrupt):
    """Pause the campaign like Ctrl-C, retaining its unfinished step."""


class RunBudget:
    def __init__(self, path, limit_seconds):
        self.path = Path(path)
        allowance = self.path.with_name("execution-budget.json")
        if allowance.exists() and limit_seconds is not None:
            extra = json.loads(allowance.read_text())
            if extra["original_limit_seconds"] != limit_seconds:
                raise ValueError("Execution allowance does not match protocol budget")
            limit_seconds = max(limit_seconds, extra["limit_seconds"])
        self.limit_seconds = limit_seconds
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        if saved and saved["limit_seconds"] != limit_seconds:
            raise ValueError("Campaign budget changed; use a new output")
        self.spent = saved.get("elapsed_seconds", 0.0)
        self.started = None
        self.task_end = None
        self.expired = False
        self.old_handler = None

    def elapsed(self):
        return self.spent + (time.monotonic() - self.started if self.started is not None else 0.0)

    def remaining(self):
        if self.limit_seconds is None:
            return math.inf
        return self.limit_seconds - self.elapsed()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "limit_seconds": self.limit_seconds,
                    "elapsed_seconds": self.elapsed(),
                    "updated": time.time(),
                }
            )
            + "\n"
        )
        temporary.replace(self.path)

    def arm(self):
        if self.expired:
            signal.setitimer(signal.ITIMER_REAL, 0)
            return
        remaining = self.remaining()
        if self.task_end is not None:
            remaining = min(remaining, self.task_end - time.monotonic())
        signal.setitimer(
            signal.ITIMER_REAL, max(0.000001, remaining) if math.isfinite(remaining) else 0
        )

    def alarm(self, signum, frame):
        # Disable before unwinding so cancellation, journaling and cleanup can finish.
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.remaining() <= 0.001:
            self.expired = True
            raise RunBudgetExceeded("Campaign wall-clock budget exhausted; paused for resume")
        self.task_end = None
        raise TaskBudgetExceeded("Per-agent task wall-clock budget exhausted")

    def start(self):
        if self.remaining() <= 0:
            raise RunBudgetExceeded("Saved campaign budget is already exhausted")
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise RuntimeError("Another wall-clock timer is already installed")
        self.started = time.monotonic()
        self.old_handler = signal.signal(signal.SIGALRM, self.alarm)
        self.arm()

    @contextmanager
    def task(self, remaining_seconds):
        if self.remaining() <= 0:
            self.expired = True
            raise RunBudgetExceeded("Campaign wall-clock budget exhausted")
        if remaining_seconds <= 0:
            raise TaskBudgetExceeded("Saved task budget is already exhausted")
        self.task_end = time.monotonic() + remaining_seconds
        self.arm()
        try:
            yield
        finally:
            self.task_end = None
            self.arm()

    def close(self):
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)
        self.save()
