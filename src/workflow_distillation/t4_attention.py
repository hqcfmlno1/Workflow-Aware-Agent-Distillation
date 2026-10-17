"""Exact repeated-KV SDPA for T4, which lacks FlashAttention GQA support."""

from __future__ import annotations


def repeated_kv_sdpa(module, query, key, value, attention_mask, dropout=0.0,
                     scaling=None, is_causal=None, **kwargs):
    """Use explicit KV heads so SDPA can select its memory-efficient kernel.

    Native enable_gqa=True can force a quadratic math fallback on T4.
    Repeating KV heads is mathematically equivalent; verify outputs/gradients
    against native GQA. On CUDA, fail rather than fall back to math silently.
    """
    from contextlib import nullcontext

    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if kwargs.get("output_attentions") or kwargs.get("cache") is not None:
        raise ValueError("Training attention does not support attention outputs or caches")
    repeats = query.shape[1] // key.shape[1]
    if repeats * key.shape[1] != query.shape[1]:
        raise ValueError("Query heads must be divisible by KV heads")
    key = key.repeat_interleave(repeats, dim=1)
    value = value.repeat_interleave(repeats, dim=1)
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    causal = bool(causal and query.shape[2] > 1 and attention_mask is None)
    backend = (
        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        if query.device.type == "cuda" else nullcontext()
    )
    with backend:
        result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=dropout, is_causal=causal, scale=scaling,
        )
    return result.transpose(1, 2).contiguous(), None


def register_t4_sdpa():
    """Register the attention and official SDPA causal-mask builder together."""
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

    AttentionInterface.register("t4_repeated_kv", repeated_kv_sdpa)
    AttentionMaskInterface.register("t4_repeated_kv", sdpa_mask)
