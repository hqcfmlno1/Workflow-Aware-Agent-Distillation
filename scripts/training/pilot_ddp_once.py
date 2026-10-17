"""One disposable DDP batch: two complete samples, two T4s, one optimizer step.

Remote only. No warmup, retry, sweep, evaluation, or baseline checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from workflow_distillation.ddp_probe import (  # noqa: E402
    rank_record,
    replica_device_plan,
    verify_replicas,
)
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

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUTPUT = ROOT / "outputs/training/ddp_once"


def worker(records, args):
    import bitsandbytes as bnb
    import psutil
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.checkpoint import checkpoint
    from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from workflow_distillation.optimized_qlora import blocked_delta_rule
    from workflow_distillation.t4_attention import register_t4_sdpa
    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    rank = int(os.environ["LOCAL_RANK"])
    record = rank_record(records, rank)
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("This experiment requires exactly two ranks")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    path = OUTPUT / f"rank{rank}.json"
    started = time.monotonic()
    report = {
        "status": "running", "rank": rank, "world_size": 2,
        "baseline_training": False, "full_trajectories": True,
        "trajectory_id": record["trajectory_id"], "task_key": record["task_key"],
        "length": record["full_token_count"], "optimizer_steps": 0, "memory": [],
    }

    def persist(stage):
        report.update(stage=stage, seconds=time.monotonic() - started)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"rank": rank, "stage": stage,
                          "seconds": report["seconds"]}), flush=True)

    def memory(stage):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        report["memory"].append({
            "stage": stage, "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "free": free, "capacity": total,
            "process_rss": psutil.Process().memory_info().rss,
            "host_available": psutil.virtual_memory().available,
        })
        persist(stage)

    def fingerprint(named_parameters, gradients=False):
        digest = hashlib.sha256()
        for name, param in named_parameters:
            tensor = param.grad if gradients else param
            if tensor is None or not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"Missing or nonfinite tensor: {name}")
            digest.update(name.encode())
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    try:
        persist("initialize")
        torch.set_num_threads(2)
        torch.manual_seed(42)
        dist.init_process_group("nccl", timeout=timedelta(seconds=600), device_id=device)
        register_t4_sdpa()
        modeling_qwen3_5.torch_chunk_gated_delta_rule = partial(
            blocked_delta_rule, modeling_qwen3_5.torch_chunk_gated_delta_rule,
            block_size=1024,
        )
        persist("load_model")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float16,
            device_map=replica_device_plan(rank), attn_implementation="t4_repeated_kv",
            local_files_only=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=["lm_head", "model.visual"],
                llm_int8_enable_fp32_cpu_offload=True,
            ),
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
        if embedding.weight is not head.weight or embedding.weight.requires_grad:
            raise RuntimeError("Expected frozen tied embedding/head")
        embedding.to(dtype=torch.float16)
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
            target_modules=(r"model\.language_model\.layers\.\d+\..*"
                            r"(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|"
                            r"in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)"),
        ))
        model.train()
        decoder = model.get_base_model().model.language_model
        # A replica never crosses GPUs: standard checkpointing preserves its local RNG.
        for layer in decoder.layers:
            layer._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        named_parameters = [(n, p) for n, p in decoder.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named_parameters]
        if not parameters or any(p.dtype != torch.float32 or p.device != device
                                 for p in parameters):
            raise RuntimeError("All FP32 adapters must reside on the local GPU")
        ddp = DistributedDataParallel(
            decoder, device_ids=[rank], output_device=rank,
            broadcast_buffers=False, find_unused_parameters=False,
            gradient_as_bucket_view=True, bucket_cap_mb=25,
        )
        report["trainable_parameters"] = sum(p.numel() for p in parameters)
        optimizer = bnb.optim.PagedAdamW32bit(parameters, lr=5e-5)
        scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
        manager = CheckpointManager(
            args.checkpoint_dir or OUTPUT / "checkpoints",
            training_identity(__file__, model=MODEL, revision=REVISION,
                              records_sha256=hashlib.sha256(args.records.read_bytes()).hexdigest(),
                              settings={"world_size": 2, "experiment": "ddp_once"}),
            save_every=args.save_every_steps, keep_last=args.keep_checkpoints,
            publisher=(HuggingFacePublisher(args.checkpoint_hub_repo,
                       prefix=f"{args.checkpoint_hub_prefix}/ddp_once")
                       if args.checkpoint_hub_repo else None),
        )
        progress = {"global_step": 0, "next_batch": 0}
        if args.resume_from:
            progress = manager.load(args.resume_from, model=model, optimizer=optimizer,
                                    scheduler=None, scaler=scaler)
            if progress not in ({"global_step": 0, "next_batch": 0},
                                {"global_step": 1, "next_batch": 1}):
                raise ValueError("Invalid one-batch DDP checkpoint position")
            manager.republish(model=model)
            report["resumed_from"] = str(manager.last_checkpoint)
            if progress["global_step"] == 1:
                report.update(status="passed", optimizer_steps=1, resumed_completed=True,
                              optimizer_steps_this_process=0)
                persist("already_complete_no_replay")
                return
        budget = StopBudget(max_seconds=args.max_train_seconds, started=started,
                            margin_seconds=args.stop_margin_seconds,
                            expected_step_seconds=args.expected_step_seconds)
        budget.__enter__()
        # A stop request from either rank stops the whole global batch at a boundary.
        stop = torch.tensor(int(budget.should_stop()), device=device)
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if bool(stop):
            if manager.due(progress["global_step"], force=True):
                manager.save(model=model, optimizer=optimizer, scheduler=None,
                             scaler=scaler, progress=progress)
            report.update(status="stopped_checkpointed", checkpoint=str(manager.last_checkpoint))
            persist("stopped_checkpointed")
            return
        # Fingerprints cover every trainable tensor, not just a sampled matrix.
        initial_hash = fingerprint(named_parameters)
        initial_hashes = [None, None]
        dist.all_gather_object(initial_hashes, initial_hash)
        if initial_hashes[0] != initial_hashes[1]:
            raise RuntimeError("Initial replicas differ")
        check_param = next(p for n, p in named_parameters if "lora_B" in n)
        before = check_param.detach().clone()
        ids = torch.tensor(record["input_ids"], device=device).unsqueeze(0)
        groups = torch.tensor(record["groups"], device=device).unsqueeze(0)
        labels = ids.masked_fill(groups == 0, -100)
        denominator = float(torch.tensor([0., 1., .75, 1.5, 2.], device=device)[groups].sum())
        torch.cuda.empty_cache()
        memory("prepared_ddp")
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        step_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        persist("forward_started")
        with torch.autocast("cuda", dtype=torch.float16):
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                hidden = ddp(input_ids=ids, use_cache=False).last_hidden_state
            loss = chunked_hidden_loss(
                hidden, head, labels, groups, denominator, 1.0, chunk_size=64,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
        report["loss"] = float(loss.detach())
        memory("forward_finished")
        persist("backward_started")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        memory("backward_finished")
        gradient_hash = fingerprint(named_parameters, gradients=True)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, foreach=False)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite gradient norm")
        report["grad_norm"] = float(norm)
        persist("optimizer_started")
        scaler.step(optimizer)
        scaler.update()
        report["optimizer_steps"] = 1
        report["adapter_max_delta"] = float((check_param.detach() - before).abs().max())
        report["gradient_sha256"] = gradient_hash
        report["adapter_sha256"] = fingerprint(named_parameters)
        if report["adapter_sha256"] == initial_hash:
            raise RuntimeError("No adapter update")
        memory("optimizer_finished")
        evidence = [None, None]
        dist.all_gather_object(evidence, {
            k: report[k] for k in ("gradient_sha256", "adapter_sha256", "adapter_max_delta")
        })
        verify_replicas(evidence)
        dist.barrier()
        torch.cuda.synchronize(device)
        report.update(status="passed", replicas_identical=True,
                      step_seconds=time.monotonic() - step_started)
        # This remains a ONE-step probe: final save is forced even with interval five.
        optimizer.zero_grad(set_to_none=True)
        del hidden, loss, before, evidence
        path = manager.save(model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
                            progress={"global_step": 1, "next_batch": 1})
        report["checkpoint"] = str(path)
        persist("complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        persist("failed")
        raise
    finally:
        if "budget" in locals():
            budget.__exit__()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--max-train-seconds", type=int, default=900)
    add_checkpoint_arguments(parser)
    args = parser.parse_args()
    try:
        validate_checkpoint_arguments(args)
        if args.max_train_seconds < 1:
            raise ValueError("max-train-seconds must be positive")
    except ValueError as exc:
        parser.error(str(exc))
    records = json.loads(args.records.read_text())
    rank_record(records, 0)
    if not Path("/kaggle/working").exists():
        raise SystemExit("Model loading and training are allowed only on remote Kaggle GPUs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if "LOCAL_RANK" in os.environ:
        worker(records, args)
        return
    summary = {
        "status": "running", "experiment": "one_disposable_ddp_batch",
        "model": MODEL, "revision": REVISION, "world_size": 2,
        "per_gpu_batch_size": 1, "global_batch_size": 2,
        "gradient_accumulation_steps": 1, "requested_optimizer_steps": 1,
        "retries": 0, "baseline_training": False,
        "checkpoint_policy": {"save_every_steps": args.save_every_steps,
                              "keep_last": args.keep_checkpoints,
                              "hub_repo": args.checkpoint_hub_repo,
                              "resume_from": args.resume_from},
        "config": {
            "quantization": "NF4 double quantization", "compute_dtype": "float16",
            "adapter_dtype": "float32", "embedding_and_head_dtype": "float16",
            "rank": 32, "alpha": 64, "dropout": 0.05,
            "optimizer": "PagedAdamW32bit", "optimizer_state_dtype": "float32",
            "learning_rate": 5e-5,
            "max_grad_norm": 1.0, "initial_gradient_scale": 8.0,
            "attention": "t4_repeated_kv", "loss_chunk_size": 64,
            "gradient_checkpointing": "nonreentrant", "saved_inputs": "CPU pinned",
            "delta_rule_block_size": 1024, "use_cache": False,
            "loss_weights": [0, 1, 0.75, 1.5, 2], "vision": "CPU offload",
            "ddp_find_unused_parameters": False, "ddp_broadcast_buffers": False,
            "ddp_gradient_as_bucket_view": True, "ddp_bucket_cap_mb": 25,
        },
        "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "samples": [{k: r[k] for k in ("trajectory_id", "task_key", "full_token_count")}
                    for r in records],
    }
    summary_path = OUTPUT / "report.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    started = time.monotonic()
    try:
        from importlib.metadata import PackageNotFoundError, version

        from packaging.version import Version

        try:
            if Version(version("torchao")) < Version("0.16"):
                subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                               check=True)
        except PackageNotFoundError:
            pass
        import torch

        if torch.cuda.device_count() != 2:
            raise RuntimeError("Expected exactly two T4 GPUs")
        if args.checkpoint_hub_repo:
            HuggingFacePublisher(args.checkpoint_hub_repo).validate()
        # One download shared by both ranks; no model forward or warmup here.
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, revision=REVISION, allow_patterns=[
            "*.json", "*.safetensors", "*.jinja", "*.txt", "*.model",
        ])
        env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
               "PYTORCH_ALLOC_CONF": "expandable_segments:True",
               "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
               "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
               "HF_HUB_OFFLINE": "0" if args.checkpoint_hub_repo else "1"}
        result = run_worker_process([
            sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nnodes=1", "--nproc_per_node=2", "--max_restarts=0",
            str(Path(__file__).resolve()), "--records", str(args.records.resolve()),
            "--max-train-seconds", str(args.max_train_seconds), *checkpoint_arguments(args),
        ], env=env, timeout=args.max_train_seconds + 300, check=False)
        summary["worker_exit_code"] = result.returncode
        summary["ranks"] = [
            json.loads((OUTPUT / f"rank{rank}.json").read_text())
            if (OUTPUT / f"rank{rank}.json").exists() else {"rank": rank, "status": "missing"}
            for rank in (0, 1)
        ]
        if result.returncode == 0 and all(r["status"] == "stopped_checkpointed"
                                         for r in summary["ranks"]):
            summary["status"] = "stopped_checkpointed"
        else:
            summary["status"] = "passed" if result.returncode == 0 and all(
                r["status"] == "passed" and r["optimizer_steps"] == 1
                for r in summary["ranks"]
            ) else "failed"
    except Exception as exc:
        summary.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
    finally:
        summary["seconds"] = time.monotonic() - started
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    if summary["status"] not in ("passed", "stopped_checkpointed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
