"""Teacher-forced action scoring with isolated hybrid-cache branches."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.span_data import span_weights
from workflow_distillation.lora_fp16.span_scoring import reference_identity as full_identity


def reference_identity(config, provenance_hash):
    identity = full_identity(config, provenance_hash)
    if config.span_scoring_backend == "prefix-cache":
        identity = dict(
            identity,
            scoring_backend="hybrid-prefix-cache-v1",
            prefill_chunk_tokens=config.span_prefill_chunk_tokens,
            cached_scorer_sha256=sha256_file(Path(__file__)),
        )
    return identity


def reference_source(path, config, corpus_hash, provenance_hash):
    from workflow_distillation.lora_fp16.span_cache import ScoreStore

    path = Path(path)
    if not (path / "identity.json").is_file():
        raise FileNotFoundError("Existing reference cache identity required")
    return ScoreStore(path, corpus_hash, full_identity(config, provenance_hash))


def variants(tokenizer, turn):
    """Retokenize each complete variant and verify identical teacher action targets."""
    text, spans = turn["text"], turn["spans"]
    left, right = turn["action_chars"]
    targets = turn["action_ids"]
    if not spans or not targets or not 0 <= left < right <= len(text):
        raise ValueError("Eligible reasoning/action pair required")
    out = []
    for span in [None, *spans]:
        prefix = text[:left]
        if span is not None:
            a, b = span
            if not 0 <= a < b <= left:
                raise ValueError("Reasoning ablation overlaps action")
            prefix = prefix[:a] + prefix[b:]
        ids = tokenizer.encode(prefix + text[left:right], add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        if not prefix_ids or ids != prefix_ids + targets:
            raise ValueError("Ablation changed action target tokenization")
        out.append((ids, len(prefix_ids)))
    return out


class PrefixScorer:
    """One original-history cache and one short-lived fork, never N live branches."""

    def __init__(self, state, tokenizer, prefill_chunk_tokens=2048):
        if prefill_chunk_tokens < 1:
            raise ValueError("Positive prefill chunk required")
        self.state, self.tokenizer = state, tokenizer
        self.chunk = prefill_chunk_tokens
        self.cache, self.prefix_ids = None, []
        self.stats = dict(prefix_resets=0, reused_prefix_tokens=0, forward_tokens=0)

    def reset(self):
        self.cache, self.prefix_ids = None, []

    def _forward(self, ids, cache):
        import torch

        device = next(self.state.decoder.parameters()).device
        x = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        output = self.state.decoder(input_ids=x, past_key_values=cache, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("Decoder did not return a hybrid cache")
        self.stats["forward_tokens"] += len(ids)
        return output

    def _prefix(self, ids):
        if self.cache is None or ids[: len(self.prefix_ids)] != self.prefix_ids:
            self.reset()
            self.stats["prefix_resets"] += 1
        self.stats["reused_prefix_tokens"] += len(self.prefix_ids)
        for start in range(len(self.prefix_ids), len(ids), self.chunk):
            output = self._forward(ids[start : start + self.chunk], self.cache)
            self.cache = output.past_key_values
            del output
        self.prefix_ids = ids.copy()

    def _action_logp(self, ids, target_start, prefix_length, projection_chunk):
        import torch
        import torch.nn.functional as F

        cache = deepcopy(self.cache)  # Includes mutable conv/recurrent state, not only K/V.
        total = np.float64(0)
        head_device = self.state.head.weight.device
        # Final target token need not be fed: position j predicts token j+1.
        for start in range(prefix_length, len(ids) - 1, self.chunk):
            end = min(start + self.chunk, len(ids) - 1)
            output = self._forward(ids[start:end], cache)
            cache = output.past_key_values
            first = max(start, target_start - 1)
            for j in range(first, end, projection_chunk):
                stop = min(j + projection_chunk, end)
                hidden = output.last_hidden_state[0, j - start : stop - start].to(head_device)
                logits = self.state.head(hidden).float()
                targets = torch.tensor(ids[j + 1 : stop + 1], device=head_device)
                logps = -F.cross_entropy(logits, targets, reduction="none")
                total += logps.double().sum().item()
            del output
        if not np.isfinite(total):
            raise FloatingPointError("Nonfinite action score")
        return float(total)

    def score_turn(
        self,
        turn,
        *,
        floor=0.1,
        dead_zone=0.05,
        chunk_size=128,
        completed=None,
        on_variant=lambda i, v: None,
    ):
        import torch

        if chunk_size < 1:
            raise ValueError("Positive projection chunk required")
        rows = variants(self.tokenizer, turn)
        if len(turn["spans"]) == 1:
            return dict(weights=[1.0], diagnostics="singleton_constant_weight_no_forwards")
        # Keep the hidden position predicting the first action token in every suffix.
        common = min(start - 1 for _, start in rows)
        original = rows[0][0]
        for ids, _ in rows[1:]:
            common = next((i for i in range(common) if ids[i] != original[i]), common)
        values, done = [], dict(completed or {})
        self.state.model.eval()
        with torch.inference_mode():
            self._prefix(original[:common])
            for index, (ids, start) in enumerate(rows):
                key = str(index)
                if key in done:
                    value = float(done[key])
                else:
                    value = self._action_logp(ids, start, common, chunk_size)
                    on_variant(index, value)
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite action score")
                values.append(value)
        deltas = [values[0] - value for value in values[1:]]
        return dict(
            full_logp=values[0],
            ablated_logps=values[1:],
            deltas=deltas,
            weights=span_weights(deltas, floor, dead_zone),
        )
