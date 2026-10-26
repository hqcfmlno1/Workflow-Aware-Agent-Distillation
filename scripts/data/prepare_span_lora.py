"""CPU-only provenance export for the immutable MetaCurate vanilla corpus."""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from workflow_distillation.epoch_data import EpochDataset, sha256_file
from workflow_distillation.lora_fp16.metacurate import load_corpus
from workflow_distillation.lora_fp16.span_data import SCORING_VERSION, UPSTREAM_REVISION

ROOT = Path(__file__).resolve().parents[2]


def initialize(index, tokenizer, tools, output):
    from transformers import AutoTokenizer

    global DATASET, TOKENIZER, TOOLS, OUTPUT
    DATASET = EpochDataset(index, verify=False)
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    TOOLS, OUTPUT = tools, Path(output)


def export(item):
    from scripts.data.prepare_action_focused_sft import normalize
    from workflow_distillation.lora_fp16.span_data import prepare_record

    i, row = item
    sample = DATASET.record(i)
    if row["id"] != sample["trajectory_id"]:
        raise ValueError("Source/corpus attempt order differs")
    messages, tools, _ = normalize(dict(messages=row["messages"], tools=TOOLS))
    record = prepare_record(
        messages, tools, TOKENIZER, sample["input_ids"].tolist(), sample["groups"].tolist()
    )
    record.update(trajectory_id=row["id"], length=len(sample["input_ids"]))
    name = f"{i:06d}.json.gz"
    target = OUTPUT / name
    with gzip.open(target.with_suffix(".tmp"), "wt", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False)
    target.with_suffix(".tmp").replace(target)
    eligible = [t for t in record["turns"] if t["status"] == "eligible"]
    return dict(
        trajectory_id=row["id"],
        file=name,
        sha256=sha256_file(target),
        length=record["length"],
        eligible_turns=len(eligible),
        scoring_variants=sum(1 + len(t["spans"]) for t in eligible),
        omitted_messages=len(record["omitted_reasoning_messages"]),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index", type=Path, default=ROOT / "data/processed/metacurate1180_vanilla/index.json"
    )
    p.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "configs/training/gpus/rtx5090_32gb/metacurate_selection.json",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/processed/metacurate_deepseek_v4pro_success_le36k_tqs",
    )
    p.add_argument("--tokenizer", type=Path, default=ROOT / "data/raw/qwen35_4b_tokenizer")
    p.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/metacurate1180_span_provenance"
    )
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args(argv)
    if args.workers < 1:
        p.error("Positive worker count required")
    corpus, _ = load_corpus(args.index, json.loads(args.selection.read_text()))
    with gzip.open(args.source / "trajectories-ranked.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != len(corpus):
        raise ValueError("Source/corpus count differs")
    args.output.mkdir(parents=True, exist_ok=False)
    tools = json.loads((args.source / "tools.json").read_text())
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.index, args.tokenizer, tools, args.output),
    ) as pool:
        records = []
        for row in pool.map(export, enumerate(rows), chunksize=1):
            records.append(row)
            if len(records) == 1 or len(records) % 50 == 0:
                print(json.dumps(dict(event="provenance_export", records=len(records))), flush=True)
    manifest = dict(
        format_version=1,
        profile=SCORING_VERSION,
        upstream_revision=UPSTREAM_REVISION,
        corpus_sha256=sha256_file(args.index),
        source_sha256=sha256_file(args.source / "trajectories-ranked.jsonl.gz"),
        template_sha256=sha256_file(args.tokenizer / "chat_template.jinja"),
        tokenizer_files={
            f.name: sha256_file(f) for f in sorted(args.tokenizer.iterdir()) if f.is_file()
        },
        data=records,
        eligible_turns=sum(r["eligible_turns"] for r in records),
        scoring_variants=sum(r["scoring_variants"] for r in records),
    )
    (args.output / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if k not in ("data", "tokenizer_files")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
