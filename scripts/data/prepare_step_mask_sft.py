"""Build a new whole-step masked vanilla SFT corpus; never alter source tokens."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.data.prepare_action_focused_sft import normalize, serialize
from workflow_distillation.epoch_data import EpochDataset, sha256_file
from workflow_distillation.lora_fp16.metacurate import load_corpus, write_corpus
from workflow_distillation.step_mask_corpus import apply_step_mask
from workflow_distillation.step_mask_token_provenance import annotate_record
from workflow_distillation.trajectory_step_masking import (
    atomic_json,
    fingerprint,
    prepare_record,
    read_completed,
    request_identity,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/metacurate_deepseek_v4pro_success_le36k_tqs"
BASELINE = ROOT / "data/processed/metacurate1180_vanilla"
OUTPUT = ROOT / "data/processed/metacurate1180_stepmask_v1"
AUDITS = ROOT / "outputs/data_audits"
REVIEWS = (
    (AUDITS / "deepseek1180_llm_step_mask_v1", ROOT / "configs/data/step_mask_policy_v1.txt"),
    (
        AUDITS / "deepseek1180_llm_step_mask_noquote_v2",
        ROOT / "configs/data/step_mask_policy_noquote_v2.txt",
    ),
)


def load_annotations(rows):
    loaded = []
    for directory, policy_path in REVIEWS:
        manifest = json.loads((directory / "manifest.json").read_text())
        policy = policy_path.read_text()
        if hashlib.sha256(policy.encode()).hexdigest() != manifest["policy_sha256"]:
            raise ValueError("Review policy changed")
        loaded.append((directory, manifest["config"], policy))
    annotations, used_files = {}, {}
    for row in rows:
        _, sources, steps = prepare_record(row)
        candidates = []
        for directory, config, policy in loaded:
            path = directory / "results" / f"{row['id']}.json"
            value = read_completed(
                path, request_identity(row, config, policy), row["id"], sources, steps
            )
            if value:
                if value["source_record_sha256"] != fingerprint(row):
                    raise ValueError("Annotation refers to a different source record")
                candidates.append((path, value))
        if len(candidates) != 1:
            raise ValueError(f"Require exactly one completed review for {row['id']}")
        path, result = candidates[0]
        annotations[row["id"]] = result
        used_files[str(path.relative_to(ROOT))] = sha256_file(path)
    return annotations, used_files


def initialize(tokenizer_path, tools_path, baseline_index):
    from transformers import AutoTokenizer

    global TOKENIZER, TOOLS, BASE
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    TOOLS = json.loads(Path(tools_path).read_text())
    BASE = EpochDataset(baseline_index, verify=False)


def render_masked(job):
    index, row, annotation = job
    baseline = BASE.record(index)
    if row["id"] != baseline["trajectory_id"] or row["total_tokens"] != len(baseline["input_ids"]):
        raise ValueError("Source and baseline order/length disagree")
    metadata = {
        k: v
        for k, v in BASE.records[index].items()
        if k not in {"offset", "length", "denominator", "group_counts"}
    }
    mask = annotation["mask_message_indices"]
    if mask:
        messages, tools, checks = normalize({"messages": row["messages"], "tools": TOOLS})
        if len(messages) != len(row["messages"]):
            raise ValueError("Normalization changed original message indices")
        rendered = serialize(messages, tools, TOKENIZER)
        annotated = annotate_record(
            {"messages": messages, "tools": tools, **rendered},
            TOKENIZER,
            include_message_indices=True,
        )
        np.testing.assert_array_equal(annotated["input_ids"], baseline["input_ids"])
        np.testing.assert_array_equal(annotated["groups"], baseline["groups"])
        masked = apply_step_mask(annotated, mask)
        groups = np.asarray(masked["groups"])
        changed = groups != baseline["groups"]
        if np.any(groups[changed] != 0):
            raise ValueError("Only masking targets is permitted")
        masked_tokens = masked["masked_target_tokens"]
    else:
        groups = np.array(baseline["groups"], copy=True)
        masked_tokens = 0
    return {
        **metadata,
        "input_ids": np.array(baseline["input_ids"], copy=True),
        "groups": groups,
        "mask_step_ids": annotation["mask_step_ids"],
        "mask_message_indices": mask,
        "masked_target_tokens": masked_tokens,
        "original_target_tokens": int(np.sum(baseline["groups"] > 0)),
        "annotation_schema": annotation["config"].get("schema_version", "quote_v1"),
        "annotation_request_sha256": annotation["request_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/raw/qwen35_4b_tokenizer")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Positive workers required")
    if args.output.exists():
        parser.error("Output already exists; never overwrite a prepared corpus")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    with gzip.open(SOURCE / "trajectories-ranked.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 1180:
        raise ValueError("Expected frozen 1,180 source records")
    baseline = EpochDataset(BASELINE / "index.json")
    if [r["id"] for r in rows] != [r["trajectory_id"] for r in baseline.records]:
        raise ValueError("Source and baseline IDs disagree")
    annotations, review_hashes = load_annotations(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_name(args.output.name + ".building")
    if staging.exists():
        parser.error("Staging exists; inspect previous attempt before retrying")
    provenance = {
        **baseline.manifest["provenance"],
        "experiment": "llm-whole-step-masked-vanilla-sft-v1",
        "baseline_index_sha256": sha256_file(BASELINE / "index.json"),
        "annotation_files_sha256": review_hashes,
        "review_manifests": {
            str(d.relative_to(ROOT)): sha256_file(d / "manifest.json") for d, _ in REVIEWS
        },
        "mask_policy": "whole assistant message; context tokens unchanged",
        "ce": "mean over retained assistant targets per trajectory; then mean over trajectories",
        "anchor": False,
        "source_commands_executed": 0,
        "builder_source_sha256": {
            n: sha256_file(ROOT / n)
            for n in (
                "scripts/data/prepare_step_mask_sft.py",
                "src/workflow_distillation/step_mask_corpus.py",
                "src/workflow_distillation/weighted_sft_data.py",
                "src/workflow_distillation/step_mask_token_provenance.py",
            )
        },
    }
    jobs = [(i, row, annotations[row["id"]]) for i, row in enumerate(rows)]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.tokenizer, SOURCE / "tools.json", BASELINE / "index.json"),
    ) as pool:

        def records():
            for count, record in enumerate(pool.map(render_masked, jobs, chunksize=1), 1):
                if count == 1 or count % 20 == 0 or count == len(rows):
                    print(
                        json.dumps(
                            {
                                "event": "corpus_prepared",
                                "completed": count,
                                "total": len(rows),
                                "trajectory_id": record["trajectory_id"],
                            }
                        ),
                        flush=True,
                    )
                yield record

        manifest = write_corpus(records(), staging, provenance)
    if manifest["file_sha256"]["tokens.bin"] != baseline.manifest["file_sha256"]["tokens.bin"]:
        raise ValueError("Input tokens changed")
    receipt = {
        "index_sha256": sha256_file(staging / "index.json"),
        "selected_ids": [r["id"] for r in rows],
        "records": len(rows),
        "files": {"index.json": sha256_file(staging / "index.json"), **manifest["file_sha256"]},
        "holdout": False,
        "experiment": "llm-whole-step-masked-vanilla-sft-v1",
    }
    corpus, _ = load_corpus(staging / "index.json", receipt)
    for i in range(len(corpus)):
        before, after = baseline.record(i), corpus.record(i)
        np.testing.assert_array_equal(before["input_ids"], after["input_ids"])
        changed = after["groups"] != before["groups"]
        if np.any(after["groups"][changed] != 0):
            raise ValueError("A label changed to a different supervised group")
    annotations_out = [
        {
            "trajectory_id": row["id"],
            "source_record_sha256": annotations[row["id"]]["source_record_sha256"],
            "annotation": annotations[row["id"]]["annotation"],
            "schema": annotations[row["id"]]["config"].get("schema_version", "quote_v1"),
        }
        for row in rows
    ]
    atomic_json(staging / "annotations.json", annotations_out)
    atomic_json(staging / "selection.json", receipt)
    summary = {
        "status": "verified",
        "records": len(rows),
        "tokens": manifest["tokens"],
        "input_tokens_identical_to_baseline": True,
        "masked_steps": sum(len(r["mask_step_ids"]) for r in manifest["data"]),
        "masked_target_tokens": sum(r["masked_target_tokens"] for r in manifest["data"]),
        "original_target_tokens": sum(r["original_target_tokens"] for r in manifest["data"]),
        "retained_target_tokens": sum(manifest["group_counts"][1:]),
        "trajectories_with_masks": sum(bool(r["mask_step_ids"]) for r in manifest["data"]),
        "annotation_schema_counts": {
            s: sum(r["annotation_schema"] == s for r in manifest["data"])
            for s in ("quote_v1", "no_quote_v2")
        },
        "masked_target_fraction": sum(r["masked_target_tokens"] for r in manifest["data"])
        / sum(r["original_target_tokens"] for r in manifest["data"]),
        "group_counts": manifest["group_counts"],
        "no_holdout": True,
        "no_anchor": True,
        "weights": manifest["weights"],
    }
    atomic_json(staging / "verification.json", summary)
    for name in ("LICENSE", "NOTICE.md", "tools.json"):
        shutil.copyfile(BASELINE / name, staging / name)
    staging.rename(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
