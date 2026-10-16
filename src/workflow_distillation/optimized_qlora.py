"""CPU-safe contracts for full-history, remote-only QLoRA experiments."""

from __future__ import annotations


def blocked_delta_rule(kernel, query, key, value, g, beta, chunk_size=64,
                       initial_state=None, output_final_state=False,
                       use_qk_l2norm_in_kernel=False, block_size=1024, **kwargs):
    """Checkpoint the reference GDN scan without shortening its history.

    Block boundaries align with the kernel's existing chunks. Recurrent state
    stays attached to autograd across every boundary; no history is discarded.
    The reference kernel still owns all normalization and numerical operations.
    """
    if block_size < chunk_size or chunk_size < 1 or block_size % chunk_size:
        raise ValueError("GDN block size must be a positive multiple of chunk size")
    if kwargs.get("cu_seqlens") is not None:
        raise ValueError("Blocked GDN does not support packed variable-length inputs")
    import torch
    from torch.utils.checkpoint import checkpoint

    def run(q, k, v, decay, strength, state):
        return kernel(q, k, v, decay, strength, chunk_size=chunk_size,
                      initial_state=state, output_final_state=True,
                      use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

    outputs = []
    state = initial_state
    for start in range(0, query.shape[1], block_size):
        end = start + block_size
        block, state = checkpoint(
            run, query[:, start:end], key[:, start:end], value[:, start:end],
            g[:, start:end], beta[:, start:end], state,
            use_reentrant=False, preserve_rng_state=False,
        )
        outputs.append(block)
    return torch.cat(outputs, dim=1), state if output_final_state else None


def checkpoint_across_devices(function, *args):
    """Preserve dropout RNG even when dispatch moves inputs inside a layer.

    Ordinary checkpointing sees only devices already present in its arguments.
    Empty sentinels make destination devices visible without retaining data.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    sentinels = tuple(torch.empty(0, device=f"cuda:{i}")
                      for i in range(torch.cuda.device_count()))

    def invoke(*inputs):
        return function(*inputs[len(sentinels):])

    return checkpoint(invoke, *sentinels, *args, use_reentrant=False)


def device_plan(num_layers: int, split_at: int | None = None) -> dict:
    """Partition text layers while keeping frozen tied embeddings together."""
    if num_layers < 1:
        raise ValueError("At least one decoder layer is required")
    if split_at is not None and not 0 < split_at < num_layers:
        raise ValueError("split must be inside the decoder layer range")
    last = 0 if split_at is None else 1
    return {
        "model.visual": "cpu",
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        **{
            f"model.language_model.layers.{i}": 0 if split_at is None or i < split_at else 1
            for i in range(num_layers)
        },
        "model.language_model.norm": last,
        "lm_head": 0,
    }


def validate_records(records: list[dict]) -> None:
    """Reject windows, invalid supervision, and silently truncated inputs."""
    if not records:
        raise ValueError("At least one full trajectory is required")
    seen = set()
    for record in records:
        ids, groups = record["input_ids"], record["groups"]
        if len(ids) != record.get("full_token_count") or len(ids) < 2:
            raise ValueError("Input must retain the full trajectory token count")
        if len(groups) != len(ids):
            raise ValueError("Token IDs and groups must align")
        if groups[0] != 0:
            raise ValueError("First token has no causal predecessor and must be masked")
        if any(g not in range(5) for g in groups) or not any(groups):
            raise ValueError("Invalid or missing supervision groups")
        if record["trajectory_id"] in seen:
            raise ValueError("Duplicate trajectory")
        seen.add(record["trajectory_id"])
