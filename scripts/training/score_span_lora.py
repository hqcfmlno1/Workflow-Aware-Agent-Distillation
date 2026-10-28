"""Resume offline span scoring and frozen-base target logps; estimate needs no GPU."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import time
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.metacurate import load_corpus
from workflow_distillation.lora_fp16.span_cache import ScoreStore, atomic_json
from workflow_distillation.lora_fp16.span_cached_scoring import (
    PrefixScorer,
    reference_identity,
    reference_source,
)
from workflow_distillation.lora_fp16.span_data import token_weights
from workflow_distillation.lora_fp16.span_scoring import (
    build_reference,
    cache_reference,
    score_turn,
)
from workflow_distillation.training.events import EventLog

ROOT = Path(__file__).resolve().parents[2]


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
        "--provenance",
        type=Path,
        default=ROOT / "data/processed/metacurate1180_span_provenance/index.json",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_bf16.json",
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/metacurate1180_span_scores"
    )
    p.add_argument("--estimate", action="store_true")
    p.add_argument("--reuse-reference-cache", type=Path)
    p.add_argument(
        "--limit-turns", type=int, help="Stop scoring after N new turns; partial cache cannot train"
    )
    p.add_argument("--max-seconds", type=int, default=28800)
    args = p.parse_args(argv)
    if args.max_seconds < 1 or (args.limit_turns is not None and args.limit_turns < 1):
        p.error("Positive limits required")
    config = RunConfig(**json.loads(args.config.read_text()))
    if args.reuse_reference_cache and config.span_scoring_backend != "prefix-cache":
        p.error("Reference-only migration requires prefix-cache backend")
    corpus, _ = load_corpus(args.index, json.loads(args.selection.read_text()))
    manifest = json.loads(args.provenance.read_text())
    if manifest["corpus_sha256"] != sha256_file(args.index):
        raise ValueError("Provenance/corpus differs")
    if [r["trajectory_id"] for r in manifest["data"]] != [
        r["trajectory_id"] for r in corpus.records
    ]:
        raise ValueError("Provenance attempt order differs")
    # Validate every exported source before loading paid GPU weights.
    for row in manifest["data"]:
        if sha256_file(args.provenance.parent / row["file"]) != row["sha256"]:
            raise ValueError("Corrupt provenance record")
    estimate = dict(
        records=len(corpus),
        eligible_turns=manifest["eligible_turns"],
        ablation_scoring_variants=manifest["scoring_variants"],
        reference_full_trajectory_forwards=len(corpus),
        note=(
            "No GPU loaded. Variants have different context lengths; "
            "measure seconds before projecting cost."
        ),
    )
    print(json.dumps(estimate, indent=2), flush=True)
    if args.estimate:
        return
    identity = reference_identity(config, sha256_file(args.provenance))
    store = ScoreStore(args.output, sha256_file(args.index), identity)
    original_references = (
        reference_source(
            args.reuse_reference_cache,
            config,
            sha256_file(args.index),
            sha256_file(args.provenance),
        )
        if args.reuse_reference_cache
        else None
    )
    emit = EventLog(args.output / "scoring.jsonl", "span-offline")
    with (args.output / ".score.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from transformers import AutoTokenizer

        from workflow_distillation.training_checkpoints import StopBudget

        tokenizer = AutoTokenizer.from_pretrained(
            config.model, revision=config.revision, local_files_only=True
        )
        if sha256_file(args.provenance) != identity["provenance_sha256"]:
            raise ValueError("Provenance changed during scoring")
        started = time.monotonic()
        with StopBudget(
            max_seconds=args.max_seconds, margin_seconds=30, expected_step_seconds=30
        ) as budget:
            state = build_reference(config)
            cached_scorer = (
                PrefixScorer(state, tokenizer, config.span_prefill_chunk_tokens)
                if config.span_scoring_backend == "prefix-cache"
                else None
            )
            turns_done = 0
            for i, row in enumerate(manifest["data"]):
                if budget.should_stop():
                    emit(dict(event="scoring_stopped", records_before=i))
                    return
                sample = corpus.record(i)
                ids, groups = sample["input_ids"], sample["groups"]
                if store.read_record(i, ids, groups) is not None:
                    continue
                if cached_scorer is not None:
                    cached_scorer.reset()
                with gzip.open(args.provenance.parent / row["file"], "rt") as stream:
                    source = json.load(stream)
                if (
                    source["length"] != len(ids)
                    or tokenizer.encode(source["text"], add_special_tokens=False) != ids.tolist()
                ):
                    raise ValueError("Scoring tokenizer differs from immutable corpus")
                q = store.read_reference(i, ids, groups)
                if q is None and original_references is not None:
                    q = original_references.read_reference(i, ids, groups)
                    if q is not None:
                        store.write_reference(i, ids, groups, q)
                        emit(
                            dict(
                                event="reference_reused",
                                record=i,
                                source=str(args.reuse_reference_cache),
                            )
                        )
                if q is None:
                    tick = time.monotonic()
                    q = cache_reference(state, ids, groups, config.loss_chunk_size)
                    store.write_reference(i, ids, groups, q)
                    emit(
                        dict(
                            event="reference_cached",
                            record=i,
                            seconds=time.monotonic() - tick,
                            tokens=len(ids),
                            supervised_tokens=len(q),
                        )
                    )
                weighted_spans, values = [], []
                for turn in source["turns"]:
                    if turn["status"] != "eligible":
                        continue
                    if budget.should_stop() or (
                        args.limit_turns and turns_done >= args.limit_turns
                    ):
                        emit(dict(event="scoring_stopped", record=i, new_turns=turns_done))
                        return
                    done = store.read_turn(i, turn["message_index"])
                    if done and "weights" in done:
                        result = done
                    else:
                        progress = dict((done or {}).get("variants", {}))

                        def checkpoint_variant(
                            index,
                            value,
                            *,
                            progress=progress,
                            record_index=i,
                            message_index=turn["message_index"],
                        ):
                            progress[str(index)] = value
                            store.write_turn(record_index, message_index, dict(variants=progress))
                            if budget.should_stop():
                                raise InterruptedError("Scoring stopped at saved variant boundary")

                        tick = time.monotonic()
                        try:
                            scorer = (
                                cached_scorer.score_turn
                                if cached_scorer is not None
                                else lambda turn, **kw: score_turn(state, tokenizer, turn, **kw)
                            )
                            result = scorer(
                                dict(turn, text=source["text"]),
                                floor=config.span_floor,
                                dead_zone=config.span_dead_zone,
                                chunk_size=config.loss_chunk_size,
                                completed=progress,
                                on_variant=checkpoint_variant,
                            )
                        except InterruptedError:
                            emit(dict(event="scoring_stopped", record=i))
                            return
                        store.write_turn(i, turn["message_index"], result)
                        turns_done += 1
                        emit(
                            dict(
                                event="turn_scored",
                                record=i,
                                message_index=turn["message_index"],
                                spans=len(turn["spans"]),
                                seconds=time.monotonic() - tick,
                                new_turns=turns_done,
                                weights=result["weights"],
                                cache_stats=dict(cached_scorer.stats) if cached_scorer else None,
                            )
                        )
                    weighted_spans.extend(turn["spans"])
                    values.extend(result["weights"])
                w = np.asarray(
                    token_weights(source["offsets"], weighted_spans, values), dtype="<f4"
                )
                # Existing groups pin field boundaries; format stays unit CE weight.
                w[np.asarray(groups) != 1] = 1.0
                positions = np.flatnonzero(groups > 0)
                positions = positions[positions > 0]
                gamma = np.asarray(source["anchor_weights"], dtype="<f4")[positions]
                store.write_record(i, ids, groups, q, w[positions], gamma)
                emit(dict(event="record_scored", record=i, trajectory_id=sample["trajectory_id"]))
            final = store.finalize(corpus.record(i) for i in range(len(corpus)))
            atomic_json(
                args.output / "summary.json",
                dict(
                    **estimate,
                    seconds=time.monotonic() - started,
                    supervised_tokens=final["tokens"],
                    complete=True,
                ),
            )
            emit(
                dict(
                    event="scoring_complete",
                    seconds=time.monotonic() - started,
                    tokens=final["tokens"],
                )
            )


if __name__ == "__main__":
    main()
