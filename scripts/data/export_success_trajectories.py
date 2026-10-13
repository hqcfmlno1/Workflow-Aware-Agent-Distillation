"""Export successful mini-SWE trajectories on CPU, preserving raw conversation payloads.

Run with python -m scripts.data.export_success_trajectories --offline.
No model, tokenizer, reference patch, or teacher command is loaded/executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.data.prepare_action_focused_sft import (
    ROOT,
    Source,
    benchmark_keys,
    task_key,
    write_json,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def success_rows(rows, teacher, benchmark, heldout):
    selected = []
    for row in rows:
        model = row["teacher_model"]
        if row["resolved"] != 1 or model["name"] != teacher or model["enable_thinking"] is not True:
            continue
        key = task_key(row["repo"], row["instance_id"])
        partition = "excluded_omnicode" if key in benchmark else heldout.get(key, "train_pool")
        selected.append({**row, "task_key": key, "partition": partition})
    return selected


def project_success(table, rows):
    import pyarrow as pa

    selected = table.take(pa.array([r["row_offset"] for r in rows], type=pa.int64()))
    if selected["trajectory_id"].to_pylist() != [r["trajectory_id"] for r in rows]:
        raise ValueError("Source row locator mismatch")
    metadata = pa.Table.from_pylist(rows)
    for name in metadata.column_names:
        if name not in selected.column_names:
            selected = selected.append_column(name, metadata[name])
    return selected


def run(config_path, output, offline=False):
    import pyarrow.parquet as pq

    config = json.loads(config_path.read_text())
    manifest_path = ROOT / config["local_paths"]["processed"] / "task_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"] != config:
        raise ValueError("Phase-1 manifest/config mismatch")
    heldout = {}
    for split in ("validation", "test"):
        for row in manifest["selection"].get(split, []):
            if row["task_key"] in heldout:
                raise ValueError("Validation/test task leakage")
            heldout[row["task_key"]] = split
    benchmark, benchmark_info = benchmark_keys(config, offline)
    source = Source(config, offline)
    rows = success_rows(source.index(), config["selection"]["teacher"], benchmark, heldout)
    if len({r["trajectory_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate trajectory IDs")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["shard"], row["row_group"])].append(row)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": config["dataset"],
        "selection": config["selection"],
        "phase1_manifest_sha256": sha256_file(manifest_path),
        "benchmark": benchmark_info,
        "export": "All resolved=1 Thinking trajectories; all categories and attempts retained",
    }
    provenance_path = output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Output belongs to another source/filter/split configuration")
    write_json(provenance_path, provenance)
    artifacts = {}
    for index, ((shard, group), candidates) in enumerate(sorted(groups.items())):
        path = output / f"part-{index:05d}.parquet"
        receipt_path = path.with_suffix(".receipt.json")
        ids = [r["trajectory_id"] for r in candidates]
        if path.exists() and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt["trajectory_ids"] != ids
                or receipt["shard"] != shard
                or receipt["row_group"] != group
                or receipt["sha256"] != sha256_file(path)
            ):
                raise ValueError(f"Existing export mismatch: {path}")
        else:
            table = project_success(source.group(shard, group), candidates)
            temporary = path.with_suffix(".partial")
            pq.write_table(table, temporary, compression="zstd")
            restored = pq.read_table(temporary)
            if not restored.equals(table):
                raise ValueError(f"Parquet round-trip changed payload: {path}")
            del restored, table
            temporary.replace(path)
            receipt = {
                "shard": shard,
                "row_group": group,
                "trajectory_ids": ids,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            write_json(receipt_path, receipt)
        artifacts[path.name] = {k: receipt[k] for k in ("sha256", "bytes")}
        print(f"Exported {index + 1}/{len(groups)} groups: {len(candidates)} successes", flush=True)
    summary = {
        **provenance,
        "trajectories": len(rows),
        "unique_tasks": len({r["task_key"] for r in rows}),
        "categories": dict(Counter(r["category"] for r in rows)),
        "partitions": dict(Counter(r["partition"] for r in rows)),
        "partition_rule": (
            "Only train_pool is eligible for later train preparation; "
            "exclude ALL attempts of validation/test/OmniCode tasks"
        ),
        "artifacts": artifacts,
        "total_bytes": sum(v["bytes"] for v in artifacts.values()),
        "payload_roundtrip_verified": True,
        "commands_executed": 0,
        "limitations": [
            "Raw source payloads, not SFT labels or Qwen-serialized text",
            "Successful outcome does not imply every action is valid",
            "Parquet row-group reads can transfer unsuccessful neighboring rows",
        ],
    }
    write_json(output / "report.json", summary)
    print(
        json.dumps({k: summary[k] for k in ("trajectories", "partitions", "total_bytes")}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/interim/nvidia_mini_swe_agent_v1_2_success"
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.offline)


if __name__ == "__main__":
    main()
