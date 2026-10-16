"""Create verified token/group arrays on CPU; never load model weights."""

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

from workflow_distillation.weighted_sft_data import TEMPLATE_SHA256, WEIGHTS, annotate_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("data/processed/mini_swe_agent_rule_baseline_4000")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/weighted_sft_tokens_4000")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("data/raw/qwen35_4b_tokenizer"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    template_hash = hashlib.sha256(
        (args.tokenizer / "chat_template.jinja").read_bytes()
    ).hexdigest()
    if template_hash != TEMPLATE_SHA256:
        raise SystemExit("Unexpected tokenizer template checksum")
    manifest_path = args.source / "task_selection_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    wanted = {row["trajectory_id"]: row for row in manifest["selection"]}
    if len(wanted) != 4000:
        raise SystemExit("This run requires exactly the approved 4,000 trajectories")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    args.output.mkdir(parents=True)
    index = []
    seen_tasks = set()
    counts = Counter()
    offset = 0
    started = time.monotonic()
    with (
        (args.output / "tokens.bin").open("wb") as token_file,
        (args.output / "groups.bin").open("wb") as group_file,
    ):
        for shard in sorted(args.source.glob("part-*.jsonl.gz")):
            with gzip.open(shard, "rt") as stream:
                for line in stream:
                    record = json.loads(line)
                    tid = record["trajectory_id"]
                    if tid not in wanted or record["partition"] != "train_pool":
                        raise ValueError(f"Unexpected training record: {tid}")
                    task = record["task_key"]
                    if task in seen_tasks:
                        raise ValueError(f"Duplicate task: {task}")
                    result = annotate_record(record, tokenizer)
                    size = len(result["input_ids"])
                    if size != wanted[tid]["token_count"]:
                        raise ValueError(f"Token count changed: {tid}")
                    np.asarray(result["input_ids"], dtype="<u4").tofile(token_file)
                    np.asarray(result["groups"], dtype="u1").tofile(group_file)
                    index.append(
                        {
                            "trajectory_id": tid,
                            "task_key": task,
                            "offset": offset,
                            "length": size,
                            "denominator": result["denominator"],
                            "group_counts": result["group_counts"],
                            "omitted_reasoning_messages": result["omitted_reasoning_messages"],
                            "boundary_masked_tokens": result["boundary_masked_tokens"],
                            "license": record.get("license"),
                        }
                    )
                    seen_tasks.add(task)
                    wanted.pop(tid)
                    offset += size
                    counts.update(dict(enumerate(result["group_counts"])))
            print(
                json.dumps(
                    {
                        "records": len(index),
                        "tokens": offset,
                        "seconds": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    if wanted:
        raise ValueError(f"Missing {len(wanted)} selected trajectories")
    report = {
        "format_version": 1,
        "records": len(index),
        "tokens": offset,
        "group_counts": dict(counts),
        "weights": list(WEIGHTS),
        "template_sha256": template_hash,
        "tokenizer": manifest["tokenizer"],
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "data": index,
        "context_policy": (
            "Not applied: full token arrays retained; windows are a separate explicit policy."
        ),
        "commands_executed": 0,
        "model_loaded": False,
        "seconds": time.monotonic() - started,
    }
    for name in ("tokens.bin", "groups.bin"):
        digest = hashlib.sha256()
        with (args.output / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        report.setdefault("file_sha256", {})[name] = digest.hexdigest()
    (args.output / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "data"}), flush=True)


if __name__ == "__main__":
    main()
