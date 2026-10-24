"""Freeze the >=0.6 TQS selection and build a portable full-token bundle on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from workflow_distillation.epoch_data import EpochDataset, sha256_file
from workflow_distillation.lora_fp16.data import select_split, write_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    args = parser.parse_args()
    if args.selection_output.exists():
        raise FileExistsError(args.selection_output)
    rows = pd.read_parquet(args.scores).query("tqs_score >= 0.6").to_dict("records")
    if len(rows) != 1140:
        raise ValueError("This pinned experiment expects exactly 1140 selected trajectories")
    source = EpochDataset(args.source_index)
    by_id = {r["trajectory_id"]: r for r in source.records}
    for row in rows:
        old = by_id[row["trajectory_id"]]
        if old["task_key"] != row["task_key"] or old["length"] != row["total_tokens"]:
            raise ValueError("TQS catalog and token corpus differ")
    split = select_split(rows)
    manifest = write_bundle(source, split, args.output)
    columns = [
        "trajectory_id",
        "task_key",
        "repo",
        "language",
        "category",
        "tqs_score",
        "total_tokens",
    ]
    receipt = {
        "format_version": 1,
        "threshold": 0.6,
        "tqs_formula": "upstream unchanged",
        "source_index_sha256": sha256_file(args.source_index),
        "scores_sha256": sha256_file(args.scores),
        "index_sha256": sha256_file(args.output / "index.json"),
        "split": split,
        "candidates": [
            {k: r[k] for k in columns} for r in sorted(rows, key=lambda r: r["trajectory_id"])
        ],
    }
    args.selection_output.parent.mkdir(parents=True, exist_ok=True)
    args.selection_output.write_text(json.dumps(receipt, indent=2) + "\n")
    by_id = {r["trajectory_id"]: r for r in manifest["data"]}
    for name in ("train_ids", "validation_ids", "excluded_sibling_ids"):
        selected = [by_id[i] for i in split[name]]
        print(
            json.dumps(
                {
                    "split": name,
                    "trajectories": len(selected),
                    "tasks": len({r["task_key"] for r in selected}),
                    "tokens": sum(r["length"] for r in selected),
                    "max_tokens": max((r["length"] for r in selected), default=0),
                }
            )
        )


if __name__ == "__main__":
    main()
