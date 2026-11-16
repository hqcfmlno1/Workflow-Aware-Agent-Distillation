"""Sequential directed backwards, joint optimizer updates, deterministic resume."""

import math
import random
import time
from pathlib import Path

import torch

from workflow_distillation.lora_fp16.engine import schedule

from .artifacts import atomic_json, load_checkpoint, save_checkpoint
from .objectives import orthogonality, phase1_loss, phase2_loss


def train_update(runtime, records, phase, config, rng):
    if not records or phase not in (1, 2):
        raise ValueError("Nonempty update and phase 1/2 required")
    runtime.optimizer.zero_grad(set_to_none=True)
    started, metrics = time.monotonic(), {}
    total = 0.0
    for record in records:
        if phase == 1:
            snapshots = [runtime.snapshot(record, n, record.key_positions) for n in runtime.names]
            for m, name in enumerate(runtime.names):
                with runtime.active(name):
                    runtime.model.train()
                    hidden = runtime.hidden(record)
                    keep = {
                        s.dropout_group: rng.random() >= config.supervision_dropout
                        for s in record.reasoning
                    }
                    peers = [s for i, s in enumerate(snapshots) if i != m]
                    loss, detail = phase1_loss(hidden, runtime.head, record, keep, peers, config)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite council loss")
                    (loss / len(records)).backward()  # MUST precede adapter switching.
                    total += float(loss.detach()) / len(records)
                    for k, v in detail.items():
                        metrics[k] = metrics.get(k, 0.0) + v / len(records)
                    del loss, hidden
        else:
            if len(runtime.names) != 1:
                raise ValueError("Phase 2 requires one student")
            reference = runtime.snapshot(record, None, record.positions)
            with runtime.active(runtime.names[0]):
                runtime.model.train()
                hidden = runtime.hidden(record)
                loss, detail = phase2_loss(hidden, runtime.head, record, reference, config)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite student loss")
                (loss / len(records)).backward()
                total += float(loss.detach()) / len(records)
                for k, v in detail.items():
                    metrics[k] = metrics.get(k, 0.0) + v / len(records)
                del loss, hidden
    runtime.all_trainable()
    if phase == 1 and config.ortho_coefficient:
        ortho = orthogonality(runtime.factors())
        (config.ortho_coefficient * ortho).backward()
        metrics["orthogonality"] = float(ortho.detach())
        total += config.ortho_coefficient * float(ortho.detach())
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in runtime.parameters):
        raise FloatingPointError("Missing/nonfinite expert gradients; no optimizer update")
    norm = torch.nn.utils.clip_grad_norm_(runtime.parameters, config.max_grad_norm, foreach=False)
    runtime.optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in runtime.parameters):
        raise FloatingPointError("Nonfinite expert parameters")
    runtime.optimizer.zero_grad(set_to_none=True)
    return dict(metrics, loss=total, grad_norm=float(norm), seconds=time.monotonic() - started)


def _rng_state(rng):
    return dict(
        dropout=rng.getstate(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def train(
    runtime,
    records,
    config,
    *,
    phase,
    output,
    identity,
    resume=None,
    max_updates=None,
    emit=lambda e: None,
):
    output = Path(output)
    batches = schedule(
        len(records), epochs=config.epochs, accumulation=config.accumulation, seed=config.seed
    )
    rng, cursor = random.Random(config.seed), 0
    if resume:
        state = load_checkpoint(resume, identity)
        cursor = state["cursor"]
        if not 0 <= cursor <= len(batches) or state["schedule"] != batches:
            raise ValueError("Checkpoint training schedule differs")
        runtime.restore_adapters(state["adapters"])
        runtime.optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["rng"]["dropout"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    elif (output / "latest.json").exists():
        raise FileExistsError("Run has checkpoints; specify --resume explicitly")
    warmup = math.ceil(config.warmup_ratio * len(batches))
    updates = 0
    while cursor < len(batches):
        # Warmup then linear decay, indexed by the update about to be committed.
        scale = (
            (cursor + 1) / warmup
            if cursor < warmup
            else (len(batches) - cursor) / max(1, len(batches) - warmup)
        )
        for group in runtime.optimizer.param_groups:
            group["lr"] = config.learning_rate * scale
        detail = train_update(runtime, [records[i] for i in batches[cursor]], phase, config, rng)
        cursor += 1
        updates += 1
        event = dict(
            event="update",
            phase=phase,
            step=cursor,
            total_steps=len(batches),
            samples_consumed=sum(map(len, batches[:cursor])),
            learning_rate=runtime.optimizer.param_groups[0]["lr"],
            **detail,
        )
        emit(event)
        stopping = max_updates is not None and updates >= max_updates
        if cursor % config.save_every == 0 or cursor == len(batches) or stopping:
            state = dict(
                cursor=cursor,
                schedule=batches,
                adapters=runtime.adapter_state(),
                optimizer=runtime.optimizer.state_dict(),
                rng=_rng_state(rng),
            )
            path = save_checkpoint(output, cursor, state, identity)
            emit(dict(event="checkpoint", step=cursor, path=str(path)))
        if stopping:
            break
    if cursor == len(batches):
        atomic_json(
            output / "completed.json",
            dict(
                identity=identity,
                step=cursor,
                samples=len(records) * config.epochs,
                checkpoint=f"checkpoint-{cursor:08d}",
            ),
        )
    return cursor
