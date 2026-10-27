"""Compare cached/full action scores and measure representative GPU scoring work."""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.span_cache import atomic_json
from workflow_distillation.lora_fp16.span_cached_scoring import PrefixScorer
from workflow_distillation.lora_fp16.span_scoring import build_reference, score_turn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    from transformers import AutoTokenizer

    config = RunConfig(**json.loads(args.config.read_text()))
    cases = json.loads(args.cases.read_text())
    state = build_reference(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model, revision=config.revision, local_files_only=True
    )
    rows = []
    scorer = PrefixScorer(state, tokenizer, config.span_prefill_chunk_tokens)
    previous_file = None
    for case in cases:
        if case["file"] != previous_file:
            scorer.reset()
        previous_file = case["file"]
        with gzip.open(case["file"], "rt") as stream:
            source = json.load(stream)
        turn = next(t for t in source["turns"] if t["message_index"] == case["message_index"])
        turn = dict(turn, text=source["text"])
        torch.cuda.synchronize()
        tick = time.monotonic()
        full = score_turn(state, tokenizer, turn, chunk_size=config.loss_chunk_size)
        torch.cuda.synchronize()
        full_seconds = time.monotonic() - tick
        torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        cached = scorer.score_turn(turn, chunk_size=config.loss_chunk_size)
        torch.cuda.synchronize()
        cached_seconds = time.monotonic() - tick
        delta_error = float(np.max(np.abs(np.array(full["deltas"]) - cached["deltas"])))
        weight_error = float(np.max(np.abs(np.array(full["weights"]) - cached["weights"])))
        per_token_error = max(
            abs(full["full_logp"] - cached["full_logp"]),
            max(
                abs(a - b)
                for a, b in zip(full["ablated_logps"], cached["ablated_logps"], strict=True)
            ),
        )
        per_token_error /= len(turn["action_ids"])
        row = dict(
            **case,
            full_seconds=full_seconds,
            cached_seconds=cached_seconds,
            speedup=full_seconds / cached_seconds,
            delta_max_error=delta_error,
            action_mean_logp_max_error=per_token_error,
            weight_max_error=weight_error,
            full_weights=full["weights"],
            cached_weights=cached["weights"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            stats=dict(scorer.stats),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        atomic_json(args.output, dict(complete=False, cases=rows))
        # This is an explicit numerical admission, not a claim of bitwise equality.
        if per_token_error > 0.02 or weight_error > 0.1:
            raise RuntimeError("Cached/full score disagreement exceeds profile admission")
    atomic_json(
        args.output,
        dict(complete=True, cases=rows, numerical_limits=dict(mean_action_logp=0.02, weight=0.1)),
    )


if __name__ == "__main__":
    main()
