"""Deterministic task holdout and portable, checksum-verified token bundles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import EpochDataset, sha256_file
from workflow_distillation.lora_fp16.config import WEIGHTS

QUOTAS = {"bug-fix": 10, "feature-request": 6, "enhancement": 2, "other": 2}


def select_split(rows, seed=42):
    rows = sorted(rows, key=lambda r: r["trajectory_id"])
    if len({r["trajectory_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trajectory IDs")
    tasks = defaultdict(list)
    for row in rows:
        if not row["task_key"] or not np.isfinite(row["tqs_score"]) or row["tqs_score"] < 0.6:
            raise ValueError("Require canonical tasks and TQS >=0.6")
        tasks[row["task_key"]].append(row)

    def tie(row):
        return hashlib.sha256(f"{seed}:{row['trajectory_id']}".encode()).hexdigest()

    # One seeded representative per task. Neither model loss nor evaluation
    # outcomes influence selection. All sibling attempts are excluded from train.
    representatives = [min(attempts, key=tie) for attempts in tasks.values()]
    languages = Counter(r["language"] for r in representatives)
    bins = Counter(min(r["total_tokens"] // 5000, 5) for r in representatives)
    chosen, repos, langs, lengths = [], set(), Counter(), Counter()
    for category, count in QUOTAS.items():
        available = [r for r in representatives if r["category"] == category]
        if len(available) < count:
            raise ValueError(f"Not enough distinct tasks in {category}")
        for _ in range(count):
            # Prefer different repos, then underrepresented languages/length
            # bins relative to the candidate distribution; seeded stable tie.
            def priority(r):
                language = r["language"]
                length = min(r["total_tokens"] // 5000, 5)
                return (
                    r["repo"] in repos,
                    langs[language] / languages[language],
                    lengths[length] / bins[length],
                    tie(r),
                )

            row = min(available, key=priority)
            available.remove(row)
            chosen.append(row)
            repos.add(row["repo"])
            langs[row["language"]] += 1
            lengths[min(row["total_tokens"] // 5000, 5)] += 1
    held = {r["task_key"] for r in chosen}
    validation = {r["trajectory_id"] for r in chosen}
    return {
        "seed": seed,
        "validation_task_keys": sorted(held),
        "validation_categories": dict(QUOTAS),
        "validation_ids": sorted(validation),
        "train_ids": [r["trajectory_id"] for r in rows if r["task_key"] not in held],
        "excluded_sibling_ids": [
            r["trajectory_id"]
            for r in rows
            if r["task_key"] in held and r["trajectory_id"] not in validation
        ],
    }


def validate_split(dataset, split):
    by_id = {r["trajectory_id"]: r for r in dataset.records}
    parts = [split[k] for k in ("train_ids", "validation_ids", "excluded_sibling_ids")]
    flattened = sum(parts, [])
    if (
        not parts[0]
        or not parts[1]
        or len(set(flattened)) != len(flattened)
        or set(flattened) != set(by_id)
    ):
        raise ValueError("Split IDs must partition the complete unique corpus")
    held = set(split["validation_task_keys"])
    if (
        len(held) != len(parts[1])
        or {by_id[t]["task_key"] for t in parts[1]} != held
        or any(by_id[t]["task_key"] in held for t in parts[0])
        or any(by_id[t]["task_key"] not in held for t in parts[2])
    ):
        raise ValueError("Task leakage or invalid validation representatives")
    if dataset.manifest["weights"] != list(WEIGHTS):
        raise ValueError("Loss weights differ from the source objective")
    for i, row in enumerate(dataset.records):
        record = dataset.record(i)
        groups = record["groups"]
        if (
            row["length"] != row["full_token_count"]
            or row["length"] >= 30000
            or groups[0] != 0
            or np.any(groups > 4)
        ):
            raise ValueError("Invalid complete trajectory/groups")
        denominator = float(np.asarray(WEIGHTS)[groups].sum())
        if denominator <= 0 or denominator != row["denominator"]:
            raise ValueError("Supervision denominator differs")


def write_bundle(source, split, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    by_id = {r["trajectory_id"]: i for i, r in enumerate(source.records)}
    ids = split["train_ids"] + split["validation_ids"] + split["excluded_sibling_ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError("Invalid selected trajectory IDs")
    output.mkdir(parents=True)
    rows, offset = [], 0
    with (output / "tokens.bin").open("wb") as tokens, (output / "groups.bin").open("wb") as groups:
        for tid in ids:
            index = by_id[tid]
            row = dict(source.records[index], offset=offset)
            record = source.record(index)
            np.asarray(record["input_ids"], dtype="<u4").tofile(tokens)
            np.asarray(record["groups"], dtype="u1").tofile(groups)
            rows.append(row)
            offset += row["length"]
    manifest = {
        **source.manifest,
        "data": rows,
        "records": len(rows),
        "tokens": offset,
        "selection": "TQS >=0.6, full trajectories, task-disjoint holdout",
        "split": split,
        "file_sha256": {name: sha256_file(output / name) for name in ("tokens.bin", "groups.bin")},
    }
    manifest["group_counts"] = np.asarray([r["group_counts"] for r in rows]).sum(0).tolist()
    (output / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    load_bundle(output / "index.json")
    return manifest


def load_bundle(index):
    dataset = EpochDataset(index)
    split = dataset.manifest["split"]
    validate_split(dataset, split)
    return dataset, split
