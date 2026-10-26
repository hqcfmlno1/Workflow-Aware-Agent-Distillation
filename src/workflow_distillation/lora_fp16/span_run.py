"""Epoch-only publication and cursor-consistent exportable training events."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from workflow_distillation.lora_fp16.checkpoints import CheckpointManager


class EpochPublisher:
    def __init__(self, publisher, pinned_steps, output):
        self.publisher, self.pinned_steps, self.output = (
            publisher,
            frozenset(pinned_steps),
            Path(output),
        )
        self.model = None

    def validate(self):
        return self.publisher.validate()

    def __call__(self, checkpoint):
        checkpoint = Path(checkpoint)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        step = manifest["progress"]["global_step"]
        if step not in self.pinned_steps:
            return
        # Snapshots are immutable on retry, including when Hub accepted a lost response.
        for name in (
            "events.jsonl",
            "train.log",
            "preflight.json",
            "hardware.json",
            "run-config.json",
            "span-sidecar.json",
            "scoring-summary.json",
        ):
            source = self.output / name
            if source.is_file() and not (checkpoint / name).exists():
                shutil.copyfile(source, checkpoint / name)
        if self.model is not None and not (checkpoint / "adapter_config.json").exists():
            self.model.save_pretrained(checkpoint, safe_serialization=True)
        for attempt in range(3):
            try:
                self.publisher(checkpoint)
                return
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status in (409, 412, 429, 500, 502, 503, 504) or isinstance(
                    exc, (TimeoutError, ConnectionError)
                )
                if not retryable or attempt == 2:
                    raise
                print(
                    json.dumps(
                        dict(
                            event="epoch_upload_retry",
                            step=step,
                            attempt=attempt + 1,
                            status=status,
                        )
                    ),
                    flush=True,
                )
                time.sleep(2 ** (attempt + 1))


def recover_events(path, step):
    """Keep only committed update events; archive discarded work rather than losing it."""
    path = Path(path)
    if not path.exists():
        return
    original = path.read_text()
    lines = original.splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("Corrupt training events before final partial line") from None
            break
        cursor = event.get("global_step", event.get("progress", {}).get("global_step", 0))
        if cursor > step:
            break
        kept.append(line)
    text = "".join(line + "\n" for line in kept)
    if text != original:
        archive = path.with_name(f"events.before-resume-{time.time_ns()}.jsonl")
        shutil.copyfile(path, archive)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text(text)
        temporary.replace(path)


class SpanCheckpointManager(CheckpointManager):
    """Local recoveries are actually local: no publication callback or misleading flag."""

    def save(self, **kwargs):
        publisher = self.publisher
        if kwargs["progress"]["global_step"] not in self.pinned_steps:
            self.publisher = None
        try:
            return super().save(**kwargs)
        finally:
            self.publisher = publisher

    def republish(self, *, model):
        if self.last_step in self.pinned_steps:
            return super().republish(model=model)
