"""Complete successful MetaCurate token corpus, separate from the legacy holdout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import EpochDataset, sha256_file

PROFILE = "metacurate-success-le36k-vanilla-v1"
WEIGHTS = (0, 1, 1, 1, 1)


def validate_record(row, ids, groups):
    ids, groups = np.asarray(ids), np.asarray(groups)
    if (
        row.get("resolved") is not True
        or not row.get("trajectory_id") or not row.get("task_key")
        or ids.ndim != 1 or groups.ndim != 1
        or not 2 <= len(ids) <= 36000
        or len(ids) != len(groups) or row["full_token_count"] != len(ids)
        or groups[0] != 0 or np.any((groups < 0) | (groups > 4))
        or not np.any(groups[1:] > 0)
        or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0)
    ):
        raise ValueError("Require successful complete <=36k trajectory with valid targets")


def write_corpus(records, output, provenance):
    """Write intact arrays; one record per attempt, retaining sibling attempts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, offset, seen = [], 0, set()
    with (output / "tokens.bin").open("wb") as tokens, (output / "groups.bin").open("wb") as groups:
        for record in records:
            ids = np.asarray(record["input_ids"])
            labels = np.asarray(record["groups"])
            validate_record(record, ids, labels)
            tid = record["trajectory_id"]
            if tid in seen:
                raise ValueError("Duplicate attempt ID")
            seen.add(tid)
            counts = np.bincount(labels, minlength=5).tolist()
            row = {k: v for k, v in record.items() if k not in (
                "input_ids", "groups", "denominator", "group_counts",
            )}
            row.update(offset=offset, length=len(ids), denominator=sum(counts[1:]),
                       group_counts=counts)
            np.asarray(ids, dtype="<u4").tofile(tokens)
            np.asarray(labels, dtype="u1").tofile(groups)
            rows.append(row)
            offset += len(ids)
    if not rows:
        raise ValueError("Empty corpus")
    manifest = {
        "format_version": 1, "profile": PROFILE, "tokens": offset, "records": len(rows),
        "data": rows, "weights": list(WEIGHTS), "provenance": provenance,
        "split": {"train_ids": [r["trajectory_id"] for r in rows], "validation_ids": [],
                  "validation_task_keys": [], "excluded_sibling_ids": []},
        "group_counts": np.sum([r["group_counts"] for r in rows], axis=0).tolist(),
        "file_sha256": {n: sha256_file(output / n) for n in ("tokens.bin", "groups.bin")},
    }
    (output / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_corpus(index, receipt):
    if sha256_file(Path(index)) != receipt["index_sha256"]:
        raise ValueError("MetaCurate index differs from the pinned receipt")
    dataset = EpochDataset(index)
    manifest = dataset.manifest
    expected = receipt["selected_ids"]
    actual = [r["trajectory_id"] for r in dataset.records]
    if (
        manifest.get("profile") != PROFILE or manifest["weights"] != list(WEIGHTS)
        or actual != expected or len(set(actual)) != len(actual)
        or manifest["split"] != {
            "train_ids": expected, "validation_ids": [], "validation_task_keys": [],
            "excluded_sibling_ids": [],
        }
    ):
        raise ValueError("Expected the complete selected MetaCurate corpus without holdout")
    for i, row in enumerate(dataset.records):
        record = dataset.record(i)
        validate_record(row, record["input_ids"], record["groups"])
        counts = np.bincount(record["groups"], minlength=5).tolist()
        if row["denominator"] != sum(counts[1:]) or row["group_counts"] != counts:
            raise ValueError("Vanilla target counts differ")
    return dataset, manifest["split"]
