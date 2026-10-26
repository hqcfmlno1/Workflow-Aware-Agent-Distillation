"""CPU-only complete MetaCurate export; verify official render, never execute tools."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.metacurate import load_corpus, write_corpus

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "data/processed/metacurate_deepseek_v4pro_success_le36k_tqs"
DEST = ROOT / "data/processed/metacurate1180_vanilla"
PROFILE = ROOT / "configs/training/gpus/rtx5090_32gb"


def initialize(tokenizer_path, tools_path):
    from transformers import AutoTokenizer

    global TOKENIZER, TOOLS
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    TOOLS = json.loads(Path(tools_path).read_text())


def render(row):
    from scripts.data.prepare_action_focused_sft import normalize, serialize
    from workflow_distillation.weighted_sft_data import annotate_record

    messages, tools, checks = normalize({"messages": row["messages"], "tools": TOOLS})
    rendered = serialize(messages, tools, TOKENIZER)
    result = annotate_record({"messages": messages, "tools": tools, **rendered}, TOKENIZER)
    if len(result["input_ids"]) != row["total_tokens"]:
        raise ValueError(f"Official render length changed: {row['id']}")
    return {
        **result, "trajectory_id": row["id"], "task_key": row["instance_id"],
        "source_trajectory_id": row["trajectory_id"], "resolved": row["resolved"],
        "full_token_count": row["total_tokens"], "tqs_score": row["tqs_score"],
        "rank": row["rank"], "repo": row["repo"], "dataset": row["dataset"],
        "schema_invalid_calls": sum(c["status"] == "invalid" for c in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--receipt", type=Path, default=PROFILE / "metacurate_selection.json")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/raw/qwen35_4b_tokenizer")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    if args.workers < 1:
        parser.error("Positive worker count required")
    selection = json.loads((args.source / "selection.json").read_text())
    with gzip.open(args.source / "trajectories-ranked.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    ids = [r["id"] for r in rows]
    if len(ids) != 1180 or ids != selection["selected_ids_in_order"]:
        raise ValueError("Require the frozen complete 1180-attempt selection")
    expected_order = sorted(rows, key=lambda r: (-r["tqs_score"], r["id"]))
    if ids != [r["id"] for r in expected_order]:
        raise ValueError("Source order differs from descending TQS")
    provenance = {
        "source_dataset": selection["dataset"], "source_revision": selection["source_revision"],
        "source_sha256": sha256_file(args.source / "trajectories-ranked.jsonl.gz"),
        "source_selection_sha256": sha256_file(args.source / "selection.json"),
        "tokenizer": selection["tokenizer"], "tokenizer_revision": selection["tokenizer_revision"],
        "tokenizer_files": {p.name: sha256_file(p) for p in sorted(args.tokenizer.iterdir())
                            if p.is_file()},
        "renderer_sources": {
            n: sha256_file(ROOT / n) for n in (
                "scripts/data/prepare_action_focused_sft.py",
                "src/workflow_distillation/weighted_sft_data.py",
                "scripts/data/prepare_metacurate_vanilla.py",
            )
        },
        "selection": selection["selection"], "no_truncation": True,
        "holdout": False, "teacher_commands_executed": 0,
        "assistant_prefix": "Supplied generation prefix is masked, as in prior trainer",
    }
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=initialize,
        initargs=(args.tokenizer, args.source / "tools.json"),
    ) as pool:
        def records():
            for i, record in enumerate(pool.map(render, rows, chunksize=1), 1):
                if i == 1 or i % 50 == 0 or i == len(rows):
                    print(json.dumps({"event": "tokenized", "completed": i, "total": len(rows)}),
                          flush=True)
                yield record

        manifest = write_corpus(records(), args.output, provenance)
    receipt = {
        "index_sha256": sha256_file(args.output / "index.json"), "selected_ids": ids,
        "records": len(ids), "unique_tasks": len({r["instance_id"] for r in rows}),
        "total_tokens": manifest["tokens"], "maximum_tokens": max(r["total_tokens"] for r in rows),
        "source_dataset": provenance["source_dataset"],
        "source_revision": provenance["source_revision"], "holdout": False,
        "files": {"index.json": sha256_file(args.output / "index.json"),
                  **manifest["file_sha256"]},
    }
    load_corpus(args.output / "index.json", receipt)
    for name in ("LICENSE", "NOTICE.md", "tools.json"):
        shutil.copyfile(args.source / name, args.output / name)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    summary = {
        **{k: v for k, v in receipt.items() if k not in ("selected_ids", "files")},
        "status": "verified", "group_counts": manifest["group_counts"],
        "trajectories_with_template_omitted_reasoning": sum(
            bool(r["omitted_reasoning_messages"]) for r in manifest["data"]
        ),
        "schema_invalid_calls": sum(r["schema_invalid_calls"] for r in manifest["data"]),
        "all_token_counts_match_source": True,
        "mean_tokens": float(np.mean([r["length"] for r in manifest["data"]])),
    }
    (args.output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
