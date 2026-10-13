"""Render ALL exported successes with the pinned official Qwen3.5-4B tokenizer.

CPU only; no weights, command execution, truncation, packing or training labels.
Each source row has an output row, including explicit errors with original payloads.
Run: python -m scripts.data.convert_success_trajectories [--workers 3] [--verify]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.data.export_success_trajectories import sha256_file
from scripts.data.prepare_action_focused_sft import (
    ROOT,
    digest,
    normalize,
    serialize,
    verify_hashes,
    write_json,
)

_TOKENIZER = None


def convert_record(row, tokenizer):
    from jinja2 import TemplateError
    from jsonschema.exceptions import SchemaError

    metadata = {k: v for k, v in row.items() if k not in ("messages", "tools", "reference_patch")}
    try:
        messages, tools, checks = normalize(row)
        rendered = serialize(messages, tools, tokenizer)
        return {
            **metadata,
            "messages": messages,
            "tools": tools,
            "checks": checks,
            **rendered,
            "serialization_status": "ok",
            "source_messages_sha256": digest(json.dumps(row["messages"], ensure_ascii=False)),
            "command_availability": "unverified_no_harness_environment",
        }
    except (ValueError, TypeError, KeyError, IndexError, SchemaError, TemplateError) as error:
        return {
            **metadata,
            "raw_messages": row["messages"],
            "raw_tools": row["tools"],
            "serialization_status": "error",
            "error": f"{type(error).__name__}: {error}",
            "text": None,
            "token_count": None,
            "action_candidates": [],
        }


def initialize(tokenizer_path):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def convert_part(source, target, identity):
    import pyarrow.parquet as pq

    receipt_path = target.with_suffix(".receipt.json")
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["sha256"] != sha256_file(target):
            raise ValueError(f"Existing output mismatch: {target}")
        return receipt
    counts, partitions, lengths, errors = Counter(), Counter(), [], []
    temporary = target.with_suffix(".partial")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as stream:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=4):
            for row in batch.to_pylist():
                result = convert_record(row, _TOKENIZER)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["serialization_status"]] += 1
                partitions[result["partition"]] += 1
                if result["serialization_status"] == "ok":
                    lengths.append(result["token_count"])
                else:
                    errors.append(
                        {k: result[k] for k in ("trajectory_id", "task_key", "partition", "error")}
                    )
    # Verify compressed JSON round trip and membership before making the part visible.
    expected_ids = pq.read_table(source, columns=["trajectory_id"])["trajectory_id"].to_pylist()
    with gzip.open(temporary, "rt", encoding="utf-8") as stream:
        actual_ids = [json.loads(line)["trajectory_id"] for line in stream]
    if actual_ids != expected_ids:
        raise ValueError(f"Output membership changed: {target}")
    temporary.replace(target)
    receipt = {
        "identity": identity,
        "sha256": sha256_file(target),
        "rows": len(actual_ids),
        "status": dict(counts),
        "partitions": dict(partitions),
        "lengths": lengths,
        "errors": errors,
    }
    write_json(receipt_path, receipt)
    return receipt


def run(config_path, source, output, workers=3, verify=False):
    config = json.loads(config_path.read_text())
    source_report = json.loads((source / "report.json").read_text())
    manifest = json.loads(
        (ROOT / config["local_paths"]["processed"] / "task_selection_manifest.json").read_text()
    )
    tokenizer_info = manifest["tokenizer"]
    if (
        tokenizer_info["revision"] != config["student"]["revision"]
        or source_report["dataset"] != config["dataset"]
    ):
        raise ValueError("Pinned source/tokenizer configuration mismatch")
    tokenizer_path = ROOT / config["student"]["tokenizer_path"]
    verify_hashes(tokenizer_path, tokenizer_info["file_sha256"])
    initialize(tokenizer_path)
    if digest(_TOKENIZER.get_chat_template()) != tokenizer_info["template_sha256"]:
        raise ValueError("Official template checksum mismatch")
    provenance = {
        "format_version": 1,
        "source_report_sha256": sha256_file(source / "report.json"),
        "dataset": config["dataset"],
        "tokenizer": tokenizer_info,
        "add_generation_prompt": False,
        "truncation": False,
        "labels_created": False,
        "format": "Sharded UTF-8 JSONL gzip; partition retained on each row",
    }
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError("Output provenance differs; use a new directory")
    write_json(provenance_path, provenance)
    parts = []
    for name, artifact in sorted(source_report["artifacts"].items()):
        path = source / name
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        target = output / (path.stem + ".jsonl.gz")
        parts.append(
            (path, target, {"source_sha256": artifact["sha256"], "provenance": provenance})
        )
    if verify:
        return verify_outputs(parts, output, workers, tokenizer_path)
    receipts = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize, initargs=(tokenizer_path,)
    ) as executor:
        futures = {executor.submit(convert_part, *part): part[1].name for part in parts}
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append((futures[future], receipt))
            print(
                f"Converted {len(receipts)}/{len(parts)} parts: "
                f"{receipt['rows']} rows, status={receipt['status']}",
                flush=True,
            )
    counts, partitions, lengths, errors = Counter(), Counter(), [], []
    for _, receipt in receipts:
        counts.update(receipt["status"])
        partitions.update(receipt["partitions"])
        lengths.extend(receipt["lengths"])
        errors.extend(receipt["errors"])
    if (
        sum(counts.values()) != source_report["trajectories"]
        or dict(partitions) != source_report["partitions"]
    ):
        raise ValueError("Source/output counts or partitions differ")
    lengths.sort()
    report = {
        **provenance,
        "rows": sum(counts.values()),
        "status": dict(counts),
        "partitions": dict(partitions),
        "length_tokens": {
            "total": sum(lengths),
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            **{
                f"p{p}": lengths[min(len(lengths) - 1, int((len(lengths) - 1) * p / 100))]
                for p in (50, 90, 95, 99)
                if lengths
            },
        },
        "artifacts": {
            name: {"sha256": r["sha256"], "rows": r["rows"]} for name, r in sorted(receipts)
        },
        "commands_executed": 0,
        "limitations": [
            "No SFT labels or loss masks",
            "Template may omit historical reasoning; normalized messages retain it",
            "Successful trajectory does not imply all calls valid",
            "Error rows retain original payload; exclude them from training",
        ],
    }
    write_json(output / "errors.json", sorted(errors, key=lambda e: e["trajectory_id"]))
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("rows", "status", "partitions", "length_tokens")}, indent=2
        )
    )


def verify_part(source, target, identity):
    import pyarrow.parquet as pq

    receipt = json.loads(target.with_suffix(".receipt.json").read_text())
    if receipt["identity"] != identity or receipt["sha256"] != sha256_file(target):
        raise ValueError(f"Output checksum mismatch: {target}")
    counts, partitions, ids = Counter(), Counter(), []
    with gzip.open(target, "rt", encoding="utf-8") as stream:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=4):
            for raw in batch.to_pylist():
                record = json.loads(next(stream))
                ids.append(record["trajectory_id"])
                if record != convert_record(raw, _TOKENIZER):
                    raise ValueError(
                        f"Source/template round-trip mismatch: {record['trajectory_id']}"
                    )
                counts[record["serialization_status"]] += 1
                partitions[record["partition"]] += 1
        if next(stream, None) is not None:
            raise ValueError("Extra output record")
    return dict(counts), dict(partitions), ids


def verify_outputs(parts, output, workers, tokenizer_path):
    """Replay every source record through the official renderer/tokenizer."""
    counts, partitions, seen = Counter(), Counter(), set()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=initialize, initargs=(tokenizer_path,)
    ) as executor:
        futures = {executor.submit(verify_part, *part): part[1].name for part in parts}
        for future in as_completed(futures):
            part_counts, part_partitions, ids = future.result()
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Duplicate trajectory ID")
            seen.update(ids)
            counts.update(part_counts)
            partitions.update(part_partitions)
            print(f"Verified {len(seen)} rows ({futures[future]})", flush=True)
    report = json.loads((output / "report.json").read_text())
    if (
        report["rows"] != len(seen)
        or report["status"] != dict(counts)
        or report["partitions"] != dict(partitions)
    ):
        raise ValueError("Verification totals differ")
    write_json(
        output / "verification_report.json",
        {
            "rows": len(seen),
            "status": dict(counts),
            "partitions": dict(partitions),
            "all_source_payloads_and_official_renders_verified": True,
            "report_sha256": sha256_file(output / "report.json"),
            "commands_executed": 0,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json",
    )
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/interim/nvidia_mini_swe_agent_v1_2_success"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(args.config, args.source, args.output, args.workers, args.verify)


if __name__ == "__main__":
    main()
