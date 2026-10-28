"""Single-GPU accumulation and teacher-forced validation with exact task cursors."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext

import numpy as np

from workflow_distillation.lora_fp16.loss import chunked_hidden_loss, group_loss_parameters
from workflow_distillation.training.activation_storage import activation_context


def schedule(count, *, epochs, accumulation, seed):
    if min(count, epochs, accumulation) < 1:
        raise ValueError("Positive dataset size, epochs and accumulation required")
    batches = []
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(count).tolist()
        batches.extend(order[i : i + accumulation] for i in range(0, count, accumulation))
    return batches


def validate_cursor(batches, progress):
    step = progress["global_step"]
    if (
        type(step) is not int
        or not 0 <= step <= len(batches)
        or progress["next_batch"] != step
        or progress["samples_consumed"] != sum(map(len, batches[:step]))
    ):
        raise ValueError("Checkpoint cursor differs from the deterministic epoch schedule")


def trajectory_loss(state, record, config, device, *, return_metrics=False):
    import torch

    coefficients = None
    denominator = record["denominator"]
    if config.loss_normalization in ("token_group", "vanilla"):
        coefficients, denominator = group_loss_parameters(
            record["groups"], config.group_weights, config.length_alpha
        )
    ids = torch.as_tensor(np.array(record["input_ids"], dtype=np.int64), device=device)[None]
    groups = torch.as_tensor(np.array(record["groups"], dtype=np.int64), device=device)[None]
    labels = ids.masked_fill(groups == 0, -100)
    context = (
        torch.autocast("cuda", dtype=getattr(torch, config.base_dtype))
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        with activation_context(config.activation_storage, config.gpu_budget_bytes):
            hidden = state.decoder(input_ids=ids, use_cache=False).last_hidden_state
        if config.loss_normalization == "offline_span_anchor":
            from workflow_distillation.lora_fp16.span_loss import chunked_span_loss

            loss, metrics = chunked_span_loss(
                hidden,
                state.head,
                labels,
                groups,
                record["weights"],
                record["ref_logp"],
                record["anchor_weights"],
                beta=config.anchor_beta,
                chunk_size=config.loss_chunk_size,
            )
            return (loss, metrics) if return_metrics else loss
        return chunked_hidden_loss(
            hidden,
            state.head,
            labels,
            groups,
            denominator,
            1.0,
            chunk_size=config.loss_chunk_size,
            group_weights=coefficients,
        )


def train_update(state, records, config, device, emit=lambda event: None):
    import torch

    if not records:
        raise ValueError("Empty update")
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    tick, total = time.monotonic(), 0.0
    span_metrics = []
    for index, record in enumerate(records):
        emit(
            {
                "event": "microbatch_started",
                "microbatch": index + 1,
                "trajectory_id": record["trajectory_id"],
                "tokens": len(record["input_ids"]),
            }
        )
        if config.loss_normalization == "offline_span_anchor":
            loss, detail = trajectory_loss(state, record, config, device, return_metrics=True)
            span_metrics.append(detail)
        else:
            loss = trajectory_loss(state, record, config, device)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
        total += float(loss.detach())
        # Actual group size, including the short final accumulation of each epoch.
        state.scaler.scale(loss / len(records)).backward()
        del loss
    state.scaler.unscale_(state.optimizer)
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in state.parameters):
        raise FloatingPointError("Missing/nonfinite LoRA gradients; update not committed")
    norm = torch.nn.utils.clip_grad_norm_(state.parameters, config.max_grad_norm, foreach=False)
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite gradient norm")
    previous_scale = state.scaler.get_scale()
    state.scaler.step(state.optimizer)
    state.scaler.update()
    if state.scaler.get_scale() < previous_scale or any(
        not bool(torch.isfinite(p).all()) for p in state.parameters
    ):
        raise FloatingPointError("Skipped/nonfinite update; cursor not advanced")
    state.optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.monotonic() - tick
    tokens = sum(len(r["input_ids"]) for r in records)
    result = dict(
        loss=total / len(records),
        grad_norm=float(norm),
        seconds=seconds,
        trajectories=len(records),
        tokens=tokens,
        tokens_per_second=tokens / seconds,
        seconds_per_trajectory=seconds / len(records),
        scale=state.scaler.get_scale(),
    )
    if span_metrics:
        for key in ("weighted_ce", "plain_ce", "anchor_loss", "anchor_scaled", "total_loss"):
            result[key] = sum(m[key] for m in span_metrics) / len(span_metrics)
        counts = np.sum([m["group_counts"] for m in span_metrics], axis=0)
        sums = np.sum([m["group_ce_sums"] for m in span_metrics], axis=0)
        masses = np.sum([m["group_weight_sums"] for m in span_metrics], axis=0)
        names = ("masked", "reasoning", "content", "action", "submit")
        result["group_counts"] = dict(zip(names, counts.tolist(), strict=True))
        result["group_nll"] = {
            names[g]: float(sums[g] / counts[g]) if counts[g] else None for g in range(1, 5)
        }
        result["group_weight_sums"] = dict(zip(names, masses.tolist(), strict=True))
        result["reasoning_weight_mass_fraction"] = float(masses[1] / masses.sum())
        result["anchor_weight_sum"] = sum(m["anchor_weight_sum"] for m in span_metrics)
        reasoning_weights = []
        for record in records:
            target_g = np.asarray(record["groups"])[1:]
            reasoning_weights.extend(np.asarray(record["weights"])[target_g[target_g > 0] == 1])
        result["reasoning_weight_quantiles"] = (
            np.quantile(reasoning_weights, [0, 0.25, 0.5, 0.75, 1]).tolist()
            if reasoning_weights
            else []
        )
    if device.type == "cuda":
        free, capacity = torch.cuda.mem_get_info(device)
        result.update(
            peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),
            cuda_free=free,
            cuda_capacity=capacity,
        )
    return result


def evaluate(state, records, config, device, emit=lambda event: None):
    import torch

    # Evaluation must not change dropout/data RNG cursors or optimizer state.
    modes = [(module, module.training) for module in state.model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    losses, tokens = [], 0
    tick = time.monotonic()
    try:
        state.model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            for record in records:
                value = float(trajectory_loss(state, record, config, device))
                if not math.isfinite(value):
                    raise FloatingPointError("Nonfinite validation loss")
                losses.append(value)
                tokens += len(record["input_ids"])
                emit(
                    {
                        "event": "validation_trajectory",
                        "trajectory_id": record["trajectory_id"],
                        "loss": value,
                        "tokens": len(record["input_ids"]),
                    }
                )
    finally:
        for module, mode in modes:
            module.training = mode
    if not losses:
        raise ValueError("Empty validation")
    return dict(
        loss=sum(losses) / len(losses),
        trajectories=len(losses),
        tokens=tokens,
        seconds=time.monotonic() - tick,
    )


def run_loop(
    batches, *, steps_per_epoch, progress, update, evaluate, save, emit, save_every, should_stop
):
    validate_cursor(batches, progress)
    progress = dict(progress)
    if progress["global_step"] % steps_per_epoch == 0 and not should_stop():
        evaluate(progress["global_step"] // steps_per_epoch)
    while progress["global_step"] < len(batches) and not should_stop():
        index = progress["next_batch"]
        metrics = update(batches[index], index + 1)
        progress = dict(
            global_step=index + 1,
            next_batch=index + 1,
            samples_consumed=progress["samples_consumed"] + len(batches[index]),
        )
        emit({"event": "step", **progress, "epoch": (index + 1) / steps_per_epoch, **metrics})
        boundary = (index + 1) % steps_per_epoch == 0
        if boundary or (index + 1) % save_every == 0:
            # Commit the completed update before a potentially long validation.
            save(progress)
        if boundary and not should_stop():
            evaluate((index + 1) // steps_per_epoch)
    if progress["global_step"] > 0:
        save(progress)
    return progress
