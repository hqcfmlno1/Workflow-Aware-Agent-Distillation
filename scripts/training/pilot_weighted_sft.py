"""Remote-only numerical and precision pilot; not a baseline training run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def worker(method, lengths, steps, records_path=None, attention="sdpa"):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    from workflow_distillation.weighted_sft_loss import chunked_hidden_loss

    output = ROOT / "outputs/training/precision_pilot" / method
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "method": method,
        "synthetic_probe_only": records_path is None,
        "baseline_training": False,
        "attention": attention,
        "runs": [],
    }
    started = time.monotonic()
    try:
        if attention == "t4_repeated_kv":
            from workflow_distillation.t4_attention import register_t4_sdpa

            register_t4_sdpa()
        kwargs = {
            "revision": REVISION,
            "dtype": torch.float16 if method == "lora-fp16" else torch.float32,
            "device_map": {"": 0},
            "attn_implementation": attention,
        }
        if method == "qlora-fp32":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32,
                llm_int8_skip_modules=["lm_head", "model.visual"],
            )
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, **kwargs)
        model.config.use_cache = False
        if method == "qlora-fp32":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
        lora = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=r"model\.language_model\.layers\.\d+\..*(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.train()
        model.print_trainable_parameters()
        decoder = model.get_base_model().model.language_model
        head = model.get_output_embeddings()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=5e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=method == "lora-fp16", init_scale=32.0)
        seed_ids = tokenizer.encode(
            "Inspect the failing test, read the source, then run pytest.\n",
            add_special_tokens=False,
        )
        report["loaded_seconds"] = time.monotonic() - started
        report["model_revision"] = REVISION
        report["gpu_name"] = torch.cuda.get_device_name()
        print(
            json.dumps(
                {"event": "model_loaded", **{k: v for k, v in report.items() if k != "runs"}}
            ),
            flush=True,
        )
        records = json.loads(Path(records_path).read_text()) if records_path else [
            {"input_ids": (seed_ids * (length // len(seed_ids) + 1))[:length]}
            for length in lengths
        ]
        for record in records:
            ids = torch.tensor(record["input_ids"], device="cuda").unsqueeze(0)
            length = ids.shape[1]
            report["current_trajectory_id"] = record.get("trajectory_id")
            report["current_length"] = length
            if "groups" in record:
                groups = torch.tensor(record["groups"], device="cuda").unsqueeze(0)
            else:
                groups = torch.ones_like(ids)
                groups[:, : max(1, length // 4)] = 0
                groups[:, -max(1, length // 4) :] = 3
                groups[:, -4:] = 4
            labels = ids.clone()
            labels[groups == 0] = -100
            weights = torch.tensor([0.0, 1.0, 0.75, 1.5, 2.0], device="cuda")
            denominator = float(weights[groups].sum())
            times = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                before = time.monotonic()
                with torch.autocast("cuda", dtype=torch.float16, enabled=method == "lora-fp16"):
                    hidden = decoder(input_ids=ids, use_cache=False).last_hidden_state
                    loss = chunked_hidden_loss(
                        hidden, head, labels, groups, denominator, 1.0, chunk_size=64
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite forward loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients = [p.grad for p in parameters if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Missing or nonfinite adapter gradients")
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                seconds = time.monotonic() - before
                times.append(seconds)
                print(
                    json.dumps(
                        {
                            "method": method,
                            "length": length,
                            "step": step + 1,
                            "loss": float(loss.detach()),
                            "grad_norm": float(norm),
                            "seconds": seconds,
                        }
                    ),
                    flush=True,
                )
            report["runs"].append(
                {
                    "length": length,
                    "trajectory_id": record.get("trajectory_id"),
                    "steps": steps,
                    "step_seconds": times,
                    "tokens_per_second": length / (sum(times) / len(times)),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        model.save_pretrained(output / "synthetic_probe_adapter")
        torch.save(
            {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()},
            output / "synthetic_probe_state.pt",
        )
        report["status"] = "passed"
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(traceback.format_exc(), flush=True)
    finally:
        report["seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    return 0 if report["status"] == "passed" else 1


def main():
    if not Path("/kaggle/working").exists():
        raise SystemExit("Model loading and training are restricted to remote Kaggle jobs.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lora-fp16", "qlora-fp32"])
    parser.add_argument("--lengths", default="128,512,2048")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--attention", choices=["sdpa", "t4_repeated_kv"], default="sdpa")
    parser.add_argument(
        "--records", help="JSON array of complete, verified trajectory token records"
    )
    args = parser.parse_args()
    if args.method:
        raise SystemExit(worker(
            args.method, [int(v) for v in args.lengths.split(",")], args.steps, args.records,
            args.attention,
        ))
    # Kaggle's optional torchao 0.10 conflicts with this pinned PEFT version.
    # Neither LoRA nor bitsandbytes QLoRA uses torchao.
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        if Version(version("torchao")) < Version("0.16"):
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=True)
    except PackageNotFoundError:
        pass
    # Numerical tests run before loading any base weights.
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/training/check_weighted_loss.py")], check=True
    )
    import torch

    if torch.cuda.device_count() != 2:
        raise SystemExit("Expected two T4 GPUs; refuse an unprofiled hardware configuration.")
    processes = []
    for device, method in enumerate(("lora-fp16", "qlora-fp32")):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device), "TOKENIZERS_PARALLELISM": "false"}
        extra = ["--records", args.records] if args.records else []
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--method",
                    method,
                    "--lengths",
                    args.lengths,
                    "--steps",
                    str(args.steps),
                    "--attention",
                    args.attention,
                    *extra,
                ],
                env=env,
            )
        )
    codes = [p.wait() for p in processes]
    if all(codes):
        raise SystemExit("Both precision pilots failed; do not start baseline training.")


if __name__ == "__main__":
    main()
