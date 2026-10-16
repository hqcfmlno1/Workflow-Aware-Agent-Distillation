"""Remote-only QLoRA memory probe on complete trajectories, never a baseline run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUTPUT = ROOT / "outputs/training/optimized_qlora"

from workflow_distillation.training_checkpoints import (  # noqa: E402
    CheckpointManager,
    HuggingFacePublisher,
    StopBudget,
    add_checkpoint_arguments,
    checkpoint_arguments,
    run_worker_process,
    training_identity,
    validate_checkpoint_arguments,
)


def worker(args):
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    from workflow_distillation.optimized_qlora import (
        blocked_delta_rule,
        checkpoint_across_devices,
        device_plan,
        validate_records,
    )
    from workflow_distillation.t4_attention import register_t4_sdpa
    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    records = json.loads(Path(args.records).read_text())
    validate_records(records)
    output = OUTPUT / args.placement
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "baseline_training": False, "full_trajectories": True,
        "model": MODEL, "model_revision": REVISION,
        "placement": args.placement,
        "records_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
        "config": {
            "quantization": "NF4 double quantization", "compute_dtype": "float16",
            "adapter_dtype": "float32", "embedding_dtype": "float16",
            "vision": "CPU offload, unused for text", "batch_size": 1,
            "weights": [0, 1, 0.75, 1.5, 2], "optimizer": "PagedAdamW32bit",
            "optimizer_state_dtype": "float32",
            "learning_rate": 5e-5, "rank": 32, "alpha": 64, "dropout": 0.05,
            "attention": "t4_repeated_kv", "loss_chunk_size": 64,
            "gradient_checkpointing": "nonreentrant", "saved_inputs": "CPU pinned",
            "checkpoint_rng": "all CUDA devices",
            "delta_rule_block_size": args.delta_block_size,
            "use_cache": False, "steps_per_trajectory": args.steps,
        },
        "runs": [], "memory": [], "status": "running",
    }
    started = time.monotonic()

    def persist():
        report["seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def memory(stage):
        devices = []
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
            free, total = torch.cuda.mem_get_info(device)
            devices.append({
                "device": device, "name": torch.cuda.get_device_name(device),
                "capacity": total, "free": free,
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            })
        import psutil

        snapshot = {"stage": stage, "devices": devices,
                    "process_rss": psutil.Process().memory_info().rss,
                    "host_available": psutil.virtual_memory().available}
        report["memory"].append(snapshot)
        print(json.dumps({"event": "memory", **snapshot}), flush=True)
        persist()
        return devices

    try:
        torch.manual_seed(42)
        torch.set_num_threads(4)
        register_t4_sdpa()
        if args.delta_block_size:
            from transformers.models.qwen3_5 import modeling_qwen3_5

            modeling_qwen3_5.torch_chunk_gated_delta_rule = partial(
                blocked_delta_rule, modeling_qwen3_5.torch_chunk_gated_delta_rule,
                block_size=args.delta_block_size,
            )
        mapping = device_plan(32, split_at=12 if args.placement == "split" else None)
        report["device_map"] = mapping
        memory("before_load")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float16, device_map=mapping,
            attn_implementation="t4_repeated_kv",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=["lm_head", "model.visual"],
                llm_int8_enable_fp32_cpu_offload=True,
            ),
        )
        model.config.use_cache = False
        memory("quantized_load")
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        # PEFT upcasts all nonquantized weights. Restore only the frozen tied
        # embedding/head; sensitive norms and the official GDN math stay FP32.
        embedding = model.get_input_embeddings()
        head = model.get_output_embeddings()
        if embedding.weight is not head.weight or embedding.weight.requires_grad:
            raise RuntimeError("Expected frozen tied embedding/head")
        embedding.to(dtype=torch.float16)
        if head.weight.dtype != torch.float16 or embedding.weight is not head.weight:
            raise RuntimeError("Embedding/head precision or tying changed unexpectedly")
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=(r"model\.language_model\.layers\.\d+\..*"
                            r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|"
                            r"in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)"),
        ))
        model.train()
        decoder = model.get_base_model().model.language_model
        for layer in decoder.layers:
            layer._gradient_checkpointing_func = checkpoint_across_devices
        head = model.get_output_embeddings()
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters or any(p.dtype != torch.float32 for p in parameters):
            raise RuntimeError("Expected FP32 trainable adapters")
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        report["trainable_devices"] = sorted({str(p.device) for p in parameters})
        if args.placement == "split" and report["trainable_devices"] != ["cuda:0", "cuda:1"]:
            raise RuntimeError("Adapters were not partitioned over both GPUs")
        optimizer = bnb.optim.PagedAdamW32bit(parameters, lr=5e-5)
        scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
        manager = CheckpointManager(
            (args.checkpoint_dir / args.placement
             if args.checkpoint_dir else output / "checkpoints"),
            training_identity(__file__, model=MODEL, revision=REVISION,
                              records_sha256=report["records_sha256"],
                              settings={"config": report["config"], "placement": args.placement}),
            save_every=args.save_every_steps, keep_last=args.keep_checkpoints,
            publisher=(HuggingFacePublisher(args.checkpoint_hub_repo,
                       prefix=f"{args.checkpoint_hub_prefix}/{args.placement}")
                       if args.checkpoint_hub_repo else None),
        )
        progress = {"global_step": 0, "record_index": 0, "step_index": 0}
        if args.resume_from:
            progress = manager.load(args.resume_from, model=model, optimizer=optimizer,
                                    scheduler=None, scaler=scaler)
            if (not 0 <= progress["record_index"] <= len(records)
                    or not 0 <= progress["step_index"] < args.steps
                    or progress["global_step"] !=
                    progress["record_index"] * args.steps + progress["step_index"]
                    or (progress["record_index"] == len(records) and progress["step_index"] != 0)):
                raise ValueError("Invalid checkpoint data cursor")
            manager.republish(model=model)
            report["resumed_from"] = args.resume_from
        report["global_step"] = progress["global_step"]
        budget = StopBudget(max_seconds=args.max_train_seconds, started=started,
                            margin_seconds=args.stop_margin_seconds,
                            expected_step_seconds=args.expected_step_seconds)
        torch.cuda.empty_cache()
        memory("prepared_adapters")
        report["load_seconds"] = time.monotonic() - started
        budget.__enter__()
        stopped = False
        for record_index in range(progress["record_index"], len(records)):
            record = records[record_index]
            report["current_trajectory_id"] = record["trajectory_id"]
            report["current_length"] = len(record["input_ids"])
            run = {"trajectory_id": record["trajectory_id"],
                   "length": len(record["input_ids"]), "steps": []}
            report["runs"].append(run)
            ids = torch.tensor(record["input_ids"], device="cuda:0").unsqueeze(0)
            groups = torch.tensor(record["groups"], device="cuda:0").unsqueeze(0)
            labels = ids.masked_fill(groups == 0, -100)
            weights = torch.tensor([0., 1., .75, 1.5, 2.], device="cuda:0")
            denominator = float(weights[groups].sum())
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
            first_step = progress["step_index"] if record_index == progress["record_index"] else 0
            for step in range(first_step, args.steps):
                if budget.should_stop():
                    optimizer.zero_grad(set_to_none=True)
                    path = manager.last_checkpoint
                    if manager.due(progress["global_step"], force=True):
                        path = manager.save(model=model, optimizer=optimizer, scheduler=None,
                                            scaler=scaler, progress=progress)
                    report.update(status="stopped_checkpointed", checkpoint=str(path))
                    stopped = True
                    break
                optimizer.zero_grad(set_to_none=True)
                step_started = time.monotonic()
                report["current_step"] = step + 1
                report["current_stage"] = "forward"
                persist()
                # Offload saved checkpoint inputs, not the model or vocabulary
                # projection. Each individual layer still runs entirely on GPU.
                with torch.autocast("cuda", dtype=torch.float16):
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        hidden = decoder(input_ids=ids, use_cache=False).last_hidden_state
                    memory(f"{run['length']}/step{step + 1}/forward")
                    loss = chunked_hidden_loss(
                        hidden, head, labels, groups, denominator, 1.0, chunk_size=64,
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite forward loss")
                report["current_stage"] = "backward"
                persist()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                memory(f"{run['length']}/step{step + 1}/backward")
                if not any(p.grad is not None for p in parameters):
                    raise FloatingPointError("No adapter gradients")
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                       for p in parameters):
                    raise FloatingPointError("Missing or nonfinite adapter gradients")
                # Sample a trainable B matrix per GPU to verify a real update.
                checks = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and "lora_B" in name and str(param.device) not in checks:
                        checks[str(param.device)] = (param, param.detach().clone())
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("Nonfinite gradient norm")
                report["current_stage"] = "optimizer"
                persist()
                scaler.step(optimizer)
                scaler.update()
                changes = {device: float((param.detach() - before).abs().max())
                           for device, (param, before) in checks.items()}
                if not changes or any(not math.isfinite(delta) or delta <= 0
                                      for delta in changes.values()):
                    raise RuntimeError("No adapter update on at least one GPU")
                if any(not bool(torch.isfinite(p).all()) for p in parameters):
                    raise FloatingPointError("Nonfinite adapter parameters after optimizer update")
                metrics = {"loss": float(loss.detach()), "grad_norm": float(norm),
                           "adapter_max_delta": changes, "scale": scaler.get_scale()}
                memory(f"{run['length']}/step{step + 1}/optimizer")
                optimizer.zero_grad(set_to_none=True)
                del hidden, loss, checks
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                metrics["seconds"] = time.monotonic() - step_started
                metrics["tokens_per_second"] = run["length"] / metrics["seconds"]
                run["steps"].append(metrics)
                memory(f"{run['length']}/step{step + 1}/cleanup")
                print(json.dumps({"event": "step", "placement": args.placement,
                                  "length": run["length"], **metrics}), flush=True)
                progress = {"global_step": progress["global_step"] + 1,
                            "record_index": record_index + (step + 1 == args.steps),
                            "step_index": (step + 1) % args.steps}
                report["global_step"] = progress["global_step"]
                budget.expected_step_seconds = max(args.expected_step_seconds, metrics["seconds"])
                final_step = progress["record_index"] == len(records)
                stop_now = budget.should_stop()
                if manager.due(progress["global_step"], force=final_step or stop_now):
                    path = manager.save(model=model, optimizer=optimizer, scheduler=None,
                                        scaler=scaler, progress=progress)
                    report["checkpoint"] = str(path)
                if stop_now and not final_step:
                    report["status"] = "stopped_checkpointed"
                    stopped = True
                    break
            del ids, labels, groups
            if stopped:
                break
        else:
            report["status"] = "passed"
        # Never expose pilot updates as a baseline checkpoint.
        if any(run["steps"] for run in report["runs"]):
            model.save_pretrained(output / "probe_only_adapter")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if "budget" in locals():
            budget.__exit__()
        persist()
        print(json.dumps(report), flush=True)
    return report["status"] in ("passed", "stopped_checkpointed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--placement", choices=["single", "split"])
    parser.add_argument("--placements", nargs="+", choices=["single", "split"],
                        default=["single", "split"])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-train-seconds", type=int, default=900)
    parser.add_argument("--delta-block-size", type=int, default=1024,
                        help="Checkpoint reference GDN blocks; 0 keeps its original scan")
    add_checkpoint_arguments(parser)
    args = parser.parse_args()
    try:
        validate_checkpoint_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
    if not Path("/kaggle/working").exists():
        raise SystemExit("This pilot loads and trains models only on remote Kaggle GPUs.")
    if args.steps < 2 or args.max_train_seconds < 1:
        parser.error("Require at least two optimizer steps and a positive time budget")
    if args.delta_block_size < 0 or args.delta_block_size % 64:
        parser.error("GDN block size must be 0 or a positive multiple of 64")
    if args.placement:
        if not worker(args):
            raise SystemExit(1)
        return
    # Remove the optional incompatible image package; QLoRA uses bitsandbytes.
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        if Version(version("torchao")) < Version("0.16"):
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                           check=True)
    except PackageNotFoundError:
        pass
    subprocess.run([sys.executable, str(ROOT / "scripts/training/check_weighted_loss.py")],
                   check=True)
    import torch

    if torch.cuda.device_count() != 2:
        raise SystemExit("Expected two T4 GPUs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_hub_repo:
        HuggingFacePublisher(args.checkpoint_hub_repo).validate()
    failed = False
    for placement in args.placements:
        try:
            run_worker_process([
                sys.executable, str(Path(__file__).resolve()), "--placement", placement,
                "--records", args.records, "--steps", str(args.steps),
                "--max-train-seconds", str(args.max_train_seconds),
                "--delta-block-size", str(args.delta_block_size),
                *checkpoint_arguments(args),
            ], env={**os.environ, "TOKENIZERS_PARALLELISM": "false",
                    "PYTORCH_ALLOC_CONF": "expandable_segments:True"},
                timeout=args.max_train_seconds + 600, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            failed = True
            (OUTPUT / f"{placement}_worker_error.json").write_text(
                json.dumps({"error": str(exc), "baseline_training": False}) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
