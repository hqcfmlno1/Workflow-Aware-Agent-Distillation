"""Prepare every successful train_pool trajectory under 30,000 tokens on CPU."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.weighted_sft_data import TEMPLATE_SHA256, WEIGHTS, annotate_record


def eligible_record(record: dict) -> bool:
    return (
        record["partition"] == "train_pool"
        and record["resolved"] == 1
        and record["token_count"] < 30000
    )


def prepare(source: Path, output: Path, tokenizer_path: Path) -> dict:
    source, output, tokenizer_path = Path(source), Path(output), Path(tokenizer_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    report_bytes = (source / "report.json").read_bytes()
    source_report = json.loads(report_bytes)
    shards = sorted(source.glob("part-*.jsonl.gz"))
    if not shards or {p.name for p in shards} != set(source_report["artifacts"]):
        raise ValueError("Source shard list differs from source report")
    template_sha = sha256_file(tokenizer_path / "chat_template.jinja")
    if (
        template_sha != TEMPLATE_SHA256
        or source_report["tokenizer"]["template_sha256"] != template_sha
    ):
        raise ValueError("Unexpected tokenizer template checksum")
    for name, expected in source_report["tokenizer"].get("file_sha256", {}).items():
        if sha256_file(tokenizer_path / name) != expected:
            raise ValueError(f"Tokenizer file checksum mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    output.mkdir(parents=True)
    rows = []
    group_counts = Counter()
    total_tokens = 0
    eligible = 0
    seen = set()
    started = time.monotonic()
    with (
        (output / "tokens.bin").open("wb") as token_file,
        (output / "groups.bin").open("wb") as group_file,
    ):
        for shard in shards:
            receipt = source_report["artifacts"][shard.name]
            if sha256_file(shard) != receipt["sha256"]:
                raise ValueError(f"Source shard checksum mismatch: {shard.name}")
            shard_rows = 0
            with gzip.open(shard, "rt", encoding="utf-8") as stream:
                for line in stream:
                    shard_rows += 1
                    record = json.loads(line)
                    if not eligible_record(record):
                        continue
                    eligible += 1
                    tid = record["trajectory_id"]
                    if tid in seen:
                        raise ValueError(f"Duplicate trajectory: {tid}")
                    seen.add(tid)
                    annotated = annotate_record(record, tokenizer)
                    ids = annotated["input_ids"]
                    groups = annotated["groups"]
                    length = len(ids)
                    if length != record["token_count"] or ids != tokenizer.encode(
                        record["text"], add_special_tokens=False
                    ):
                        raise ValueError(f"Official token IDs/count changed: {tid}")
                    if length < 2 or len(groups) != length or groups[0] != 0:
                        raise ValueError(f"Invalid first-token mask/groups: {tid}")
                    if (
                        min(ids) < 0
                        or max(ids) >= 2**32
                        or any(group not in range(5) for group in groups)
                    ):
                        raise ValueError(f"Invalid token or group value: {tid}")
                    counts = [groups.count(i) for i in range(5)]
                    denominator = sum(WEIGHTS[i] * counts[i] for i in range(5))
                    if (
                        counts != annotated["group_counts"]
                        or denominator != annotated["denominator"]
                        or denominator <= 0
                    ):
                        raise ValueError(f"Invalid supervision accounting: {tid}")
                    np.asarray(ids, dtype="<u4").tofile(token_file)
                    np.asarray(groups, dtype="u1").tofile(group_file)
                    rows.append(
                        {
                            "trajectory_id": tid,
                            "task_key": record["task_key"],
                            "offset": total_tokens,
                            "length": length,
                            "full_token_count": record["token_count"],
                            "denominator": denominator,
                            "group_counts": counts,
                            "omitted_reasoning_messages": annotated["omitted_reasoning_messages"],
                            "boundary_masked_tokens": annotated["boundary_masked_tokens"],
                        }
                    )
                    total_tokens += length
                    group_counts.update(dict(enumerate(counts)))
            if shard_rows != receipt["rows"]:
                raise ValueError(f"Source shard row count mismatch: {shard.name}")
            print(
                json.dumps(
                    {
                        "shard": shard.name,
                        "records": len(rows),
                        "tokens": total_tokens,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if not rows or len(rows) != eligible:
        raise ValueError("No eligible records or selected record count mismatch")
    manifest = {
        "format_version": 1,
        "selection": "all train_pool resolved=1 token_count<30000",
        "records": len(rows),
        "tokens": total_tokens,
        "group_counts": [group_counts[i] for i in range(5)],
        "weights": list(WEIGHTS),
        "template_sha256": template_sha,
        "tokenizer": source_report["tokenizer"],
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "source_artifacts": source_report["artifacts"],
        "file_sha256": {name: sha256_file(output / name) for name in ("tokens.bin", "groups.bin")},
        "context_policy": "Full trajectory, no truncation or windowing",
        "commands_executed": 0,
        "model_loaded": False,
        "seconds": time.monotonic() - started,
        "data": rows,
    }
    staging = output / "index.json.tmp"
    staging.write_text(json.dumps(manifest, indent=2) + "\n")
    staging.replace(output / "index.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/first_run_epoch_sft"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/raw/qwen35_4b_tokenizer"))
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.tokenizer)
    print(json.dumps({k: v for k, v in result.items() if k != "data"}), flush=True)


if __name__ == "__main__":
    main()
