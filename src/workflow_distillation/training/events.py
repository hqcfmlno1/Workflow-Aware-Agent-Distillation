"""Structured per-rank events; no training globals or launch-time imports."""

import json
from datetime import UTC, datetime
from pathlib import Path


class EventLog:
    def __init__(self, path, run_name, rank=None):
        self.path, self.run_name, self.rank = Path(path), run_name, rank

    def __call__(self, event):
        value = {
            "time": datetime.now(UTC).isoformat(),
            "run_name": self.run_name,
            "rank": self.rank,
            **event,
        }
        line = json.dumps(value, ensure_ascii=False)
        with self.path.open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
        print(line, flush=True)
