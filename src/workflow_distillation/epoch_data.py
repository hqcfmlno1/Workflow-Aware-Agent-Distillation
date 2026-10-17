"""Read verified compact token/group arrays and schedule one shuffled epoch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EpochDataset:
    """Memory-map complete trajectories; verify binaries in the parent process."""

    def __init__(self, index_path: str | Path, *, verify: bool = True):
        index_path = Path(index_path)
        self.manifest = json.loads(index_path.read_text())
        self.records = self.manifest["data"]
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported epoch dataset version")
        token_count = self.manifest["tokens"]
        if token_count <= 0 or not self.records:
            raise ValueError("Empty epoch dataset")
        offset = 0
        for row in self.records:
            if row["offset"] != offset or row["length"] < 2:
                raise ValueError("Invalid record offsets")
            offset += row["length"]
        if offset != token_count or len(self.records) != self.manifest["records"]:
            raise ValueError("Record index does not cover binary arrays")
        paths = {name: index_path.parent / name for name in ("tokens.bin", "groups.bin")}
        for name, expected_bytes in (("tokens.bin", token_count * 4), ("groups.bin", token_count)):
            if paths[name].stat().st_size != expected_bytes:
                raise ValueError(f"Invalid {name} size")
            if verify and sha256_file(paths[name]) != self.manifest["file_sha256"][name]:
                raise ValueError(f"{name} checksum mismatch")
        self._tokens = np.memmap(paths["tokens.bin"], dtype="<u4", mode="r")
        self._groups = np.memmap(paths["groups.bin"], dtype="u1", mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> dict:
        row = self.records[index]
        start = row["offset"]
        end = start + row["length"]
        return {
            "input_ids": self._tokens[start:end],
            "groups": self._groups[start:end],
            "trajectory_id": row["trajectory_id"],
            "task_key": row["task_key"],
            "full_token_count": row["full_token_count"],
            "denominator": row["denominator"],
        }


def epoch_batches(count: int, seed: int = 42) -> list[list[int]]:
    """Return deterministic global batches; the last may contain one record."""
    if count < 0:
        raise ValueError("Record count cannot be negative")
    order = np.random.default_rng(seed).permutation(count).tolist()
    return [order[start : start + 2] for start in range(0, count, 2)]
