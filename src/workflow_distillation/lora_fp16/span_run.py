"""Epoch-only publication and cursor-consistent exportable training events."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.checkpoints import CheckpointManager
from workflow_distillation.lora_fp16.span_cache import atomic_json


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
        prepare_publication(checkpoint, self.output, self.model)
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


PUBLICATION_RECEIPT = "epoch-publication.json"


def validate_publication(checkpoint):
    checkpoint = Path(checkpoint)
    receipt = json.loads((checkpoint / PUBLICATION_RECEIPT).read_text())
    if receipt.get("complete") is not True:
        raise ValueError("Incomplete epoch publication snapshot")
    for name, expected in receipt["file_sha256"].items():
        if Path(name).name != name or sha256_file(checkpoint / name) != expected:
            raise ValueError("Corrupt epoch publication snapshot")
    return receipt


def prepare_publication(checkpoint, output, model=None):
    """Commit a complete immutable log/PEFT snapshot before any Hub upload."""
    checkpoint, output = Path(checkpoint), Path(output)
    if (checkpoint / PUBLICATION_RECEIPT).exists():
        validate_publication(checkpoint)
        return
    with tempfile.TemporaryDirectory(prefix=".epoch-export-", dir=checkpoint) as temporary:
        staging = Path(temporary)
        names = [
            "events.jsonl",
            "train.log",
            "preflight.json",
            "hardware.json",
            "run-config.json",
            "span-sidecar.json",
            "scoring-summary.json",
        ]
        names.extend(p.name for p in output.glob("train-restored-step-*.log"))
        for name in names:
            if (output / name).is_file():
                shutil.copyfile(output / name, staging / name)
        if model is not None:
            from safetensors import safe_open

            model.save_pretrained(staging, safe_serialization=True)
            json.loads((staging / "adapter_config.json").read_text())
            with safe_open(staging / "adapter_model.safetensors", framework="pt") as tensors:
                if not list(tensors.keys()):
                    raise ValueError("Empty standard PEFT adapter export")
        files = [p for p in staging.iterdir() if p.is_file()]
        receipt = {"complete": True, "file_sha256": {p.name: sha256_file(p) for p in files}}
        for path in files:
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            path.replace(checkpoint / path.name)
        fd = os.open(checkpoint, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_json(checkpoint / PUBLICATION_RECEIPT, receipt)


def restore_checkpoint_logs(checkpoint, output):
    """Restore historical events on a new server without replacing open stdout logs."""
    checkpoint, output = Path(checkpoint), Path(output)
    validate_publication(checkpoint)
    source = checkpoint / "events.jsonl"
    if not source.is_file():
        raise ValueError("Hub epoch checkpoint is missing its training event snapshot")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "events.jsonl"
    if target.exists():
        shutil.copyfile(target, output / f"events.before-hub-resume-{time.time_ns()}.jsonl")
    temporary = target.with_suffix(".jsonl.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(target)
    step = json.loads((checkpoint / "manifest.json").read_text())["progress"]["global_step"]
    # train.log is open by the detached launcher's stdout FD. Preserve that inode;
    # keep historical stdout separately and include it in the next epoch upload.
    for old_log in [checkpoint / "train.log", *checkpoint.glob("train-restored-step-*.log")]:
        if old_log.is_file():
            name = (
                f"train-restored-step-{step:08d}.log"
                if old_log.name == "train.log"
                else old_log.name
            )
            shutil.copyfile(old_log, output / name)
