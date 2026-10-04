"""Add each run's teacher model to historical teacher records.

Usage: uv run python scripts/backfill_record_models.py

The manifest is the source of truth for records.jsonl. Per-run record.json
files use their adjacent metadata.json because some older paths were reused.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator


IDENTITY_FIELDS = ("task_id", "workflow", "language", "status")


def manifest_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank manifest line {line_number}")
            row = json.loads(line)
            if row.get("record_path"):
                yield row


def _validate_pair(record: dict[str, Any], metadata: dict[str, Any], label: str) -> str:
    mismatched = [
        field for field in IDENTITY_FIELDS if record.get(field) != metadata.get(field)
    ]
    runtime = record.get("runtime")
    metadata_runtime = metadata.get("runtime_seconds")
    if not isinstance(runtime, (int, float)) or not isinstance(metadata_runtime, (int, float)):
        mismatched.append("runtime")
    elif abs(runtime - metadata_runtime) > 1e-7:
        mismatched.append("runtime")
    if mismatched:
        raise ValueError(f"{label}: record/metadata mismatch: {mismatched}")
    model = metadata.get("teacher_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label}: missing teacher_model")
    if record.get("model", model) != model:
        raise ValueError(f"{label}: existing model disagrees with metadata")
    return model


def backfill_aggregate(records_path: Path, manifest_path: Path) -> int:
    temporary = records_path.with_name(records_path.name + ".model.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    count = 0
    try:
        with records_path.open(encoding="utf-8") as records, temporary.open(
            "x", encoding="utf-8", newline="\n"
        ) as output:
            for count, pair in enumerate(
                zip_longest(records, manifest_records(manifest_path)), 1
            ):
                line, metadata = pair
                if line is None or metadata is None:
                    raise ValueError("records.jsonl and manifest have different record counts")
                record = json.loads(line)
                model = _validate_pair(record, metadata, f"line {count}")
                record["model"] = model
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, records_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def backfill_per_run(manifest_path: Path, runs_root: Path) -> int:
    root = runs_root.resolve()
    paths = {Path(row["record_path"]) for row in manifest_records(manifest_path)}
    updated = 0
    for record_path in sorted(paths):
        if not record_path.resolve().is_relative_to(root):
            raise ValueError(f"Record path is outside {root}: {record_path}")
        metadata_path = record_path.with_name("metadata.json")
        if not record_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = _validate_pair(record, metadata, str(record_path))
        if record.get("model") == model:
            continue
        record["model"] = model
        temporary = record_path.with_name(record_path.name + ".model.tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(record, output, ensure_ascii=False)
            os.replace(temporary, record_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("teacher_dataset"))
    args = parser.parse_args()
    root = args.dataset_root
    aggregate = backfill_aggregate(root / "records.jsonl", root / "manifest.jsonl")
    per_run = backfill_per_run(root / "manifest.jsonl", root / "runs")
    print(f"aggregate_records={aggregate} per_run_records_updated={per_run}")


if __name__ == "__main__":
    main()
