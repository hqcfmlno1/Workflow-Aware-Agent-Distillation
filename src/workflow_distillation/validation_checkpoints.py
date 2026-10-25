"""Evaluate base FP16 and every available checkpoint on the frozen 20-task holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.validation_loss import (
    aggregate,
    evaluate_record,
    load_adapter,
    load_results,
    summarize_groups,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/training/gpus/rtx5090_32gb"
NAME = re.compile(r"checkpoint-\d{8,}$")
ARCHITECTURE = (
    "model",
    "revision",
    "base_dtype",
    "quantization",
    "rank",
    "alpha",
    "dropout",
    "gdn_backend",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_manifest(path, run, config, index_hash):
    manifest = json.loads(Path(path).read_text())
    step = manifest["progress"]["global_step"]
    identity = manifest["identity"]
    saved = RunConfig(**identity["settings"]["config"])
    if (
        manifest["format_version"] != 1
        or manifest["complete"] is not True
        or type(step) is not int
        or step < 1
        or manifest["world_size"] != 1
        or identity["records_sha256"] != index_hash
        or identity["model"] != config.model
        or identity["revision"] != config.revision
        or saved.to_dict() != config.to_dict()
        or not re.fullmatch("[0-9a-f]{64}", manifest["files"]["adapter.pt"])
    ):
        raise ValueError(f"Checkpoint identity/config/data mismatch: {path}")
    return dict(run=run, step=step, manifest=manifest)


def discover(spec, config, index_hash, local_only=False):
    """Local complete saves include checkpoints whose HF upload failed."""
    found = {}
    local = ROOT / spec["local"]
    for path in sorted(local.glob("checkpoint-*/manifest.json")):
        if path.parent.is_symlink() or not NAME.fullmatch(path.parent.name):
            continue
        item = read_manifest(path, spec["name"], config, index_hash)
        if item["step"] != int(path.parent.name.split("-")[1]):
            raise ValueError(f"Checkpoint name/progress mismatch: {path}")
        if not (path.parent / "adapter.pt").is_file():
            raise ValueError(f"Missing committed local adapter: {path.parent}")
        found[item["step"]] = dict(
            item, adapter=str(path.parent / "adapter.pt"), source=str(path.parent)
        )
    if not local_only and spec.get("repo"):
        from huggingface_hub import HfApi, hf_hub_download

        revision = (
            HfApi()
            .repo_info(spec["repo"], repo_type="model", revision=spec.get("revision", "main"))
            .sha
        )
        files = HfApi().list_repo_files(spec["repo"], repo_type="model", revision=revision)
        for name in sorted(files):
            parts = name.split("/")
            if (
                len(parts) != len(spec["prefix"].split("/")) + 2
                or not name.startswith(spec["prefix"] + "/")
                or parts[-1] != "manifest.json"
                or not NAME.fullmatch(parts[-2])
            ):
                continue
            path = hf_hub_download(spec["repo"], name, revision=revision)
            item = read_manifest(path, spec["name"], config, index_hash)
            if item["step"] != int(parts[-2].split("-")[1]):
                raise ValueError(f"Remote checkpoint name/progress mismatch: {name}")
            remote = dict(
                repo=spec["repo"],
                revision=revision,
                filename=name.removesuffix("manifest.json") + "adapter.pt",
            )
            if remote["filename"] not in files:
                raise ValueError(f"Missing committed remote adapter: {name}")
            if item["step"] in found:
                if found[item["step"]]["manifest"] != item["manifest"]:
                    raise ValueError(
                        f"Local/Hub checkpoint conflict: {spec['name']}/{item['step']}"
                    )
                continue
            found[item["step"]] = dict(
                item, remote=remote, source=f"{spec['repo']}@{revision}/{name}"
            )
    return [found[k] for k in sorted(found)]


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def export(output, items, specs, rows, validation_ids, steps_per_epoch):
    complete, trajectory_rows, measured = [], [], set()
    base = None
    for item in items:
        current = [rows[(item["key"], tid)] for tid in validation_ids if (item["key"], tid) in rows]
        for row in current:
            trajectory_rows.append(
                {k: v for k, v in row.items() if k not in ("counts", "ce_sums")}
                | summarize_groups(row)
                | {
                    f"{name}_tokens": row["counts"][g]
                    for g, name in enumerate(("masked", "reasoning", "content", "action", "submit"))
                    if g
                }
            )
        if len(current) != len(validation_ids):
            continue
        metrics = aggregate(current)
        if item["run"] == "base":
            base = metrics
        result = dict(
            run=item["run"],
            step=item["step"],
            epoch=item["step"] / steps_per_epoch,
            **metrics,
            seconds=sum(r["seconds"] for r in current),
            source=item["source"],
            key=item["key"],
        )
        if base:
            result.update(
                {
                    f"{k}_delta_vs_base": result[k] - base[k]
                    for k in ("ce_macro", "ce_micro", "token_weighted_loss", "group_weighted_loss")
                }
            )
        complete.append(result)
        measured.add((item["run"], item["step"]))
    available = {(i["run"], i["step"]) for i in items}
    coverage = []
    for spec in specs:
        last = max(
            [spec.get("known_last_step", 0)]
            + spec.get("available_steps", [])
            + [i["step"] for i in items if i["run"] == spec["name"]]
        )
        for step in range(last + 1):
            key = spec["name"], step
            status = (
                "shared_base_measured"
                if step == 0 and base
                else "measured"
                if key in measured
                else "available_pending"
                if key in available
                else "available_not_selected"
                if step in spec.get("available_steps", [])
                else "base_pending"
                if step == 0
                else "weights_unavailable"
            )
            coverage.append(dict(run=spec["name"], step=step, status=status))
    write_csv(output / "validation-loss.csv", complete)
    write_csv(output / "trajectory-loss.csv", trajectory_rows)
    write_csv(output / "checkpoint-coverage.csv", coverage)
    lines = [
        "# Validation loss",
        "",
        "Cùng 20 trajectory holdout, teacher forcing, base FP16, không cập nhật trọng số.",
        "Mọi checkpoint được tính lại với CẢ HAI objective cố định để so sánh công bằng.",
        "ce_macro: trung bình CE từng trajectory; ce_micro: CE trên toàn bộ token được học.",
        "Loss riêng reasoning/content/action/submit là CE micro của nhóm; nhóm rỗng để trống.",
        "token_weighted_loss: trọng số 1/.75/1.5/2, trung bình đều các trajectory.",
        "group_weighted_loss: alpha=.5, trọng số .15/.05/1/.5, trung bình đều trajectory.",
        "Delta âm: loss thấp hơn base theo metric đó; không chứng minh SWE-bench tốt hơn.",
        "Step không có weights không thể tính lại; không nội suy, không điền loss giả.",
        "",
        "| Run | Step | CE macro | CE micro | Token loss | Group loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in complete:
        lines.append(
            f"| {r['run']} | {r['step']} | {r['ce_macro']:.6f} | "
            f"{r['ce_micro']:.6f} | {r['token_weighted_loss']:.6f} | "
            f"{r['group_weighted_loss']:.6f} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n")


def evaluate_items(
    state,
    dataset,
    validation,
    items,
    specs,
    output,
    identity,
    steps_per_epoch,
    device,
    chunk_size=128,
):
    """Persist after every trajectory and skip only exactly matching fingerprints."""
    import torch

    rows = load_results(output / "events.jsonl")
    tids = [dataset.records[i]["trajectory_id"] for i in validation]
    for item in items:
        item["key"] = digest(
            dict(
                evaluation=identity,
                run=item["run"],
                step=item["step"],
                adapter_sha256=item.get("manifest", {}).get("files", {}).get("adapter.pt", "base"),
            )
        )
    export(output, items, specs, rows, tids, steps_per_epoch)
    state.model.eval()
    for item in items:
        pending = [
            i for i in validation if (item["key"], dataset.records[i]["trajectory_id"]) not in rows
        ]
        if not pending:
            print(f"Skip complete: {item['run']} step {item['step']}", flush=True)
            continue
        print(
            f"Validation: {item['run']} step {item['step']}; remaining {len(pending)}", flush=True
        )
        if item["run"] != "base":
            path = item.get("adapter")
            if path is None:
                from huggingface_hub import hf_hub_download

                remote = item["remote"]
                path = hf_hub_download(
                    remote["repo"], remote["filename"], revision=remote["revision"]
                )
            load_adapter(state.model, path, item["manifest"])
        context = state.model.disable_adapter() if item["run"] == "base" else nullcontext()
        with context, torch.inference_mode():
            for i in pending:
                record = dataset.record(i)
                tick = time.monotonic()
                metrics = evaluate_record(state, record, device, chunk_size)
                row = dict(
                    key=item["key"],
                    run=item["run"],
                    step=item["step"],
                    trajectory_id=record["trajectory_id"],
                    tokens=len(record["input_ids"]),
                    seconds=time.monotonic() - tick,
                    **metrics,
                )
                with (output / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                rows[(item["key"], record["trajectory_id"])] = row
                print(
                    json.dumps(
                        dict(
                            run=item["run"],
                            step=item["step"],
                            trajectory_id=record["trajectory_id"],
                            **summarize_groups(metrics),
                        )
                    ),
                    flush=True,
                )
        export(output, items, specs, rows, tids, steps_per_epoch)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs/evaluation/validation_fp16.json"
    )
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data/processed/tqs1140_holdout20_fp16/index.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation/checkpoint-loss")
    parser.add_argument("--local-only", action="store_true", help="No Hub checkpoint discovery")
    parser.add_argument(
        "--plan-only", action="store_true", help="List checkpoints without loading GPU"
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", help="Optional subset; base always included"
    )
    args = parser.parse_args(argv)
    if args.steps and min(args.steps) < 0:
        parser.error("Steps must be nonnegative")
    from workflow_distillation.lora_fp16.cli import prepare

    specs = json.loads(args.plan.read_text())["runs"]
    if (
        not specs
        or len({s["name"] for s in specs}) != len(specs)
        or any(s["name"] == "base" for s in specs)
    ):
        raise ValueError("Require unique non-base run names")
    configs = [RunConfig(**json.loads((ROOT / spec["config"]).read_text())) for spec in specs]
    config = configs[0]
    if any(
        any(getattr(c, k) != getattr(config, k) for k in ARCHITECTURE)
        or c.accumulation != config.accumulation
        for c in configs
    ):
        raise ValueError("Comparison requires identical architecture/backend/batch schedule")
    dataset, train, validation, _, _ = prepare(config, args.index)
    items = [dict(run="base", step=0, source=f"{config.model}@{config.revision}")]
    for spec, run_config in zip(specs, configs, strict=True):
        found = discover(spec, run_config, sha256_file(args.index), args.local_only)
        spec["available_steps"] = [i["step"] for i in found]
        print(
            json.dumps(dict(run=spec["name"], available_steps=[i["step"] for i in found])),
            flush=True,
        )
        items.extend(i for i in found if args.steps is None or i["step"] in args.steps)
    if len(items) == 1:
        raise ValueError("No selected checkpoints found; check plan paths/Hub access")
    if args.plan_only:
        print("Plan only: no GPU model loaded; no validation computed.", flush=True)
        return
    import fcntl

    import torch

    from workflow_distillation.lora_fp16.model import build_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose one CUDA GPU with CUDA_VISIBLE_DEVICES=0")
    device = torch.device("cuda:0")
    args.output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        schema=1,
        index_sha256=sha256_file(args.index),
        config={k: getattr(config, k) for k in ARCHITECTURE},
        source_sha256={
            name: sha256_file(Path(__file__).parent / name)
            for name in (
                "validation_loss.py",
                "validation_checkpoints.py",
                "lora_fp16/model.py",
                "optimized_qlora.py",
                "training/model.py",
                "training/config.py",
                "epoch_data.py",
                "lora_fp16/data.py",
            )
        },
        packages={
            p: version(p)
            for p in ("torch", "transformers", "peft", "flash-linear-attention", "fla-core")
        },
        gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        chunk_size=config.loss_chunk_size,
    )
    with (args.output / ".validation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / "evaluation.json").write_text(json.dumps(identity, indent=2) + "\n")
        (args.output / "checkpoint-plan.json").write_text(
            json.dumps(dict(runs=specs, checkpoints=items), indent=2) + "\n"
        )
        torch.set_num_threads(4)
        torch.manual_seed(config.seed)
        state = build_model(config)
        # No optimizer step, backward, scheduler, training resume or HF write.
        state.model.gradient_checkpointing_disable()
        state.optimizer = state.scaler = None
        evaluate_items(
            state,
            dataset,
            validation,
            items,
            specs,
            args.output,
            identity,
            math.ceil(len(train) / config.accumulation),
            device,
            chunk_size=config.loss_chunk_size,
        )


if __name__ == "__main__":
    main()
