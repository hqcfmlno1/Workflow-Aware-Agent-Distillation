"""TIES on effective dense updates, never on independently merged A/B factors."""

import math

import torch


def ties_merge(updates, density=0.2):
    if (
        not updates
        or not math.isfinite(density)
        or not 0 < density <= 1
        or any(u.shape != updates[0].shape for u in updates)
    ):
        raise ValueError("Aligned updates and density in (0,1] required")
    values = torch.stack([u.detach().float().flatten() for u in updates])
    if not bool(torch.isfinite(values).all()) or values.shape[1] == 0:
        raise ValueError("Empty/nonfinite update")
    k = max(1, math.ceil(values.shape[1] * density))
    # Stable ordering makes cutoff ties deterministic.
    order = values.abs().argsort(dim=1, descending=True, stable=True)[:, :k]
    mask = torch.zeros_like(values, dtype=torch.bool).scatter_(1, order, True)
    values = values * mask
    signs = values.sum(0).sign()
    agreeing = (values.sign() == signs) & (values != 0)
    merged = (values * agreeing).sum(0) / agreeing.sum(0).clamp_min(1)
    return merged.reshape(updates[0].shape)


@torch.no_grad()
def export_ties(runtime, tokenizer, output, config, identity, tokenizer_source=None):
    """Fold exact FP32 TIES into each frozen base layer, then persist a standalone base.

    Mutates this runtime. It must not be used again for council scoring/training.
    Frozen base dtype determines the final rounding (recorded in the manifest).
    """
    from pathlib import Path

    from .artifacts import finish_export

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty TIES export directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for _, module in runtime.model.named_modules():
        if not hasattr(module, "lora_A") or not all(n in module.lora_A for n in runtime.names):
            continue
        updates = [module.get_delta_weight(n).detach().float().cpu() for n in runtime.names]
        merged = ties_merge(updates, config.ties_density) * config.ties_scale
        weight = module.get_base_layer().weight
        weight.copy_((weight.float() + merged.to(weight.device)).to(weight.dtype))
    base = runtime.model.unload()
    base.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    if tokenizer_source is not None:
        import shutil

        # Preserve original rendering/tokenization bytes even if save_pretrained
        # normalizes tokenizer metadata. Model config.json is deliberately excluded.
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "special_tokens_map.json",
            "added_tokens.json",
            "tokenizer.model",
        ):
            source = Path(tokenizer_source) / filename
            if source.is_file():
                shutil.copyfile(source, output / filename)
    finish_export(
        output,
        dict(
            identity, operation="effective-delta-TIES", rounding=str(next(base.parameters()).dtype)
        ),
    )
