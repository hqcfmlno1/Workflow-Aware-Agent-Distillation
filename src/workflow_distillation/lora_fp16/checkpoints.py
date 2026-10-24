"""Epoch-aware retention over the unchanged, hash-pinned checkpoint transport.

The legacy manager remains byte-for-byte intact for old run resume identities.
Publishing first commits complete state; retention then removes only obsolete
recovery points. A failure between phases leaves extra snapshots, never deletes
an epoch boundary, and can be retried through the original idempotent protocol.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path, PurePosixPath

from workflow_distillation.training_checkpoints import (
    CheckpointManager as AtomicManager,
)
from workflow_distillation.training_checkpoints import (
    HuggingFacePublisher as AtomicPublisher,
)


def validate_pins(values):
    values = tuple(values)
    if any(type(step) is not int or step < 1 for step in values):
        raise ValueError("Pinned checkpoint steps must be positive integers")
    return frozenset(values)


def retained_names(names, keep_last, pinned):
    recent = sorted(names, key=lambda name: int(name.split("-")[1]))[-keep_last:]
    return set(recent) | {name for name in names if int(name.split("-")[1]) in pinned}


class CheckpointManager(AtomicManager):
    def __init__(self, *args, keep_last=3, pinned_steps=(), **kwargs):
        if type(keep_last) is not int or keep_last < 1:
            raise ValueError("Positive recovery retention required")
        self.recovery_count = keep_last
        self.pinned_steps = validate_pins(pinned_steps)
        # Disable the legacy recent-only pruning, apply our policy after its
        # complete atomic save/publication and integrity checks have succeeded.
        super().__init__(*args, keep_last=sys.maxsize, **kwargs)

    def save(self, **kwargs):
        result = super().save(**kwargs)
        own = {}
        for candidate in self.directory.glob("checkpoint-*"):
            if candidate.is_symlink():
                continue
            try:
                manifest = json.loads((candidate / "manifest.json").read_text())
                if manifest["identity"] == self.identity:
                    own[candidate.name] = candidate
            except (OSError, ValueError, KeyError):
                continue
        retained = retained_names(own, self.recovery_count, self.pinned_steps)
        for name, path in own.items():
            if name not in retained:
                shutil.rmtree(path)
        return result


class HuggingFacePublisher(AtomicPublisher):
    def __init__(self, *args, pinned_steps=(), **kwargs):
        self.pinned_steps = validate_pins(pinned_steps)
        super().__init__(*args, **kwargs)

    def __call__(self, checkpoint):
        # Without the explicit destructive opt-in the original protocol retains
        # all files/history. With it, upload without recent-only pruning first.
        keep_last = self.keep_last
        try:
            if keep_last is not None:
                self.keep_last = sys.maxsize
            super().__call__(checkpoint)
        finally:
            self.keep_last = keep_last
        if keep_last is None:
            return
        from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download

        revision = self.validate().sha
        api = HfApi()
        pointer = hf_hub_download(
            self.repo_id, f"{self.prefix}/latest.json", repo_type=self.repo_type, revision=revision
        )
        if json.loads(Path(pointer).read_text())["checkpoint"] != Path(checkpoint).name:
            raise ValueError("Hub advanced during retention; stop this writer")
        files = api.list_repo_files(self.repo_id, repo_type=self.repo_type, revision=revision)
        self._dedicated_files(files)
        names = {
            PurePosixPath(name).parts[-2]
            for name in files
            if name.startswith(f"{self.prefix}/checkpoint-")
        }
        retained = retained_names(names, keep_last, self.pinned_steps)
        operations = [
            CommitOperationDelete(path_in_repo=name)
            for name in files
            if name.startswith(f"{self.prefix}/checkpoint-")
            and PurePosixPath(name).parts[-2] not in retained
        ]
        if operations:
            committed = api.create_commit(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
                operations=operations,
                parent_commit=revision,
                commit_message="Retain epoch checkpoints and recent recovery",
            )
            self._squash(api, committed.oid)
