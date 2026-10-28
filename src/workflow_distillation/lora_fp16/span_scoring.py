"""Frozen-base teacher-forced scoring with chunked output projection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from workflow_distillation.lora_fp16.span_data import span_weights


def token_logps(state, input_ids, positions, chunk_size=128):
    import torch
    import torch.nn.functional as F

    if (
        chunk_size < 1
        or not len(positions)
        or min(positions) < 1
        or max(positions) >= len(input_ids)
    ):
        raise ValueError("Invalid teacher-forcing positions/chunk size")
    state.model.eval()
    device = next(state.decoder.parameters()).device
    head_device = state.head.weight.device
    ids = torch.as_tensor(np.array(input_ids, dtype=np.int64), device=device)[None]
    values = []
    with torch.inference_mode():
        hidden = state.decoder(input_ids=ids, use_cache=False).last_hidden_state
        for start in range(0, len(positions), chunk_size):
            ix = torch.as_tensor(positions[start : start + chunk_size], device=device)
            logits = state.head(hidden[0, ix - 1].to(head_device)).float()
            logp = -F.cross_entropy(logits, ids[0, ix].to(head_device), reduction="none")
            values.append(logp.cpu().numpy())
    out = np.concatenate(values).astype("<f4")
    if not np.isfinite(out).all():
        raise FloatingPointError("Nonfinite frozen-reference log-probabilities")
    return out


def cache_reference(state, input_ids, groups, chunk_size=128):
    if len(input_ids) != len(groups):
        raise ValueError("Reference groups/IDs mismatch")
    positions = np.flatnonzero(np.asarray(groups) > 0)
    positions = positions[positions > 0]
    return token_logps(state, input_ids, positions, chunk_size)


def score_turn(
    state,
    tokenizer,
    turn,
    *,
    floor=0.1,
    dead_zone=0.05,
    chunk_size=128,
    completed=None,
    on_variant=lambda i, v: None,
):
    """Score the identical joint action target, preserving all non-ablated context."""
    text, spans = turn["text"], turn["spans"]
    left, right = turn["action_chars"]
    target_ids = turn["action_ids"]
    if not spans or not target_ids:
        raise ValueError("Eligible reasoning/action pair required")
    cached = dict(completed or {})
    values = []
    for index, span in enumerate([None] + spans):
        key = str(index)
        if key in cached:
            value = float(cached[key])
        else:
            prefix = text[:left]
            if span is not None:
                a, b = span
                if not 0 <= a < b <= left:
                    raise ValueError("Reasoning ablation overlaps action")
                prefix = prefix[:a] + prefix[b:]
            full_ids = tokenizer.encode(prefix + text[left:right], add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            if full_ids != prefix_ids + target_ids or not prefix_ids:
                raise ValueError("Ablation changed action target tokenization")
            positions = list(range(len(prefix_ids), len(full_ids)))
            value = float(token_logps(state, full_ids, positions, chunk_size).sum(dtype=np.float64))
            on_variant(index, value)
        if not np.isfinite(value):
            raise FloatingPointError("Nonfinite action score")
        values.append(value)
    deltas = [values[0] - v for v in values[1:]]
    return dict(
        full_logp=values[0],
        ablated_logps=values[1:],
        deltas=deltas,
        weights=span_weights(deltas, floor, dead_zone),
    )


def build_reference(config):
    """One unquantized frozen base, same architecture/kernel as training, no adapter."""
    from functools import partial
    from inspect import unwrap

    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from workflow_distillation.lora_fp16.model import model_kwargs
    from workflow_distillation.optimized_qlora import blocked_delta_rule

    if config.gdn_backend == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    else:
        kernel = unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
        if not getattr(kernel, "_blocked_training_kernel", False):
            blocked = partial(blocked_delta_rule, kernel, block_size=1024)
            blocked._blocked_training_kernel = True
            modeling_qwen3_5.torch_chunk_gated_delta_rule = blocked
    model = Qwen3_5ForConditionalGeneration.from_pretrained(**model_kwargs(config))
    model.requires_grad_(False).eval()
    model.config.use_cache = False
    return SimpleNamespace(
        model=model, decoder=model.model.language_model, head=model.get_output_embeddings()
    )


def reference_identity(config, provenance_hash):
    """Bind offline probabilities to frozen weights, renderer, scoring code and runtime."""
    from importlib.metadata import PackageNotFoundError, version
    from pathlib import Path

    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.lora_fp16.span_data import SCORING_VERSION, UPSTREAM_REVISION

    packages = {}
    for name in ("torch", "transformers", "numpy", "fla-core", "flash-linear-attention", "triton"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).parent
    return dict(
        model=config.model,
        revision=config.revision,
        dtype=config.base_dtype,
        backend=config.gdn_backend,
        span_floor=config.span_floor,
        span_dead_zone=config.span_dead_zone,
        chunk_size=config.loss_chunk_size,
        provenance_sha256=provenance_hash,
        profile=SCORING_VERSION,
        upstream_revision=UPSTREAM_REVISION,
        packages=packages,
        sources={
            n: sha256_file(source / n) for n in ("span_scoring.py", "span_data.py", "span_cache.py")
        },
    )
