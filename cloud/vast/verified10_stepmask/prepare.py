"""Stage Verified10 offline; finalize only when the new masked epoch-2 exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/evaluation/verified10-span-stepmask-epoch2"
REFERENCE = ROOT / "outputs/evaluation/verified10-bf16-triplet"
RUNTIME_SOURCE = HERE / "runtime"
TRAIN_INDEX = ROOT / "data/processed/metacurate1180_stepmask_v1/index.json"
SIDECAR = ROOT / "data/processed/metacurate1180_span_stepmask_scores_v1/index.json"
SOURCE = {
    "repo": "hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-span-stepmask-run1",
    "repo_type": "model",
    "prefix": "bf16-metacurate1180-span-stepmask",
    "revision": None,
}
VARIANT = "span-stepmask"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def frozen_inputs():
    original = json.loads((ROOT / "configs/evaluation/verified10_bf16_triplet.json").read_text())
    cached_rows = REFERENCE / "grading-dataset.json"
    if cached_rows.exists():
        rows = json.loads(cached_rows.read_text())
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            "princeton-nlp/SWE-bench_Verified", split="test", revision=original["dataset_revision"]
        )
        by_id = {row["instance_id"]: dict(row) for row in dataset}
        rows = [by_id[iid] for iid in original["instance_ids"]]
    if (
        fingerprint(rows) != original["rows_sha256"]
        or [r["instance_id"] for r in rows] != original["instance_ids"]
        or len(rows) != 10
    ):
        raise ValueError("Original frozen ten-task dataset changed")
    training = json.loads(TRAIN_INDEX.read_text())
    if len(training["data"]) != 1180:
        raise ValueError("Expected the complete 1180 masked training trajectories")
    excluded = {row["task_key"] for row in training["data"]}
    overlap = sorted(set(original["instance_ids"]) & excluded)
    if overlap:
        raise ValueError("Benchmark overlaps masked training: " + str(overlap))
    config = {**original["config"], "model_name": VARIANT, "request_timeout_seconds": 1200}
    protocol = {
        k: original[k]
        for k in (
            "dataset_revision",
            "selection_rule",
            "seed",
            "instance_ids",
            "rows_sha256",
            "selection_uses",
            "base_model",
            "base_revision",
            "patch_policy",
            "diagnostic_only",
        )
    }
    protocol.update(
        name="verified10-span-stepmask-epoch2",
        config=config,
        training_index_sha256=digest(TRAIN_INDEX),
        span_sidecar_sha256=digest(SIDECAR),
        training_unique_tasks=len(excluded),
        training_overlap=overlap,
        sources={VARIANT: dict(SOURCE)},
        checkpoint="checkpoint-00000590",
        variant_order=[VARIANT],
        tool_call_limit=200,
        turn_timeout_seconds=1200,
        task_budget_seconds=1800,
        campaign_budget_seconds=None,
        execution_order="Original ten IDs in order; new masked span-anchor only.",
        rollout_policy="Fresh rollout per task; resume only this campaign's interrupted steps.",
    )
    return rows, protocol


def validate_training(manifest, protocol):
    identity = manifest.get("identity", {})
    settings = identity.get("settings", {})
    config = settings.get("config", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("progress", {}).get("global_step") != 590
        or identity.get("model") != protocol["base_model"]
        or identity.get("revision") != protocol["base_revision"]
        or identity.get("records_sha256") != protocol["training_index_sha256"]
        or settings.get("span_sidecar_sha256") != protocol["span_sidecar_sha256"]
        or config.get("base_dtype") != "bfloat16"
        or config.get("loss_normalization") != "offline_span_anchor"
        or config.get("anchor_beta") != 0.03
    ):
        raise ValueError("Checkpoint is not the complete new masked training epoch 2")


def pack(output, *, ready):
    server = output / "server"
    inventory = {
        str(p.relative_to(server)): digest(p)
        for p in sorted(server.rglob("*"))
        if p.is_file() and p.name != "bundle-manifest.json" and "__pycache__" not in p.parts
    }
    write_json(server / "bundle-manifest.json", inventory)
    archive = output / "verified10-stepmask-server.tar.gz"
    with tarfile.open(archive.with_suffix(".tmp"), "w:gz", compresslevel=1) as target:
        for name in [*inventory, "bundle-manifest.json"]:
            target.add(server / name, arcname="verified10-stepmask/" + name)
    archive.with_suffix(".tmp").replace(archive)
    protocol = json.loads((output / "protocol.json").read_text())
    write_json(
        output / "prepared.json",
        {
            "adapters_ready": ready,
            "checkpoint": protocol["checkpoint"],
            "source": protocol["sources"][VARIANT],
            "tasks": protocol["instance_ids"],
            "archive": str(archive),
            "archive_sha256": digest(archive),
            "archive_bytes": archive.stat().st_size,
            "paid_inference_started": False,
            "next_action": "deploy after training finishes"
            if ready
            else "finalize after checkpoint-00000590 is uploaded; no fallback",
        },
    )


def stage(output):
    if (output / "results").exists():
        raise ValueError("Results exist; preparation must never overwrite campaign state")
    if (output / "prepared.json").exists() and json.loads(
        (output / "prepared.json").read_text()
    ).get("adapters_ready"):
        raise ValueError("Already finalized; retain pinned bundle or choose a new output")
    rows, protocol = frozen_inputs()
    server = output / "server"
    runtime = server / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    sources = {}
    for source in sorted(RUNTIME_SOURCE.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(RUNTIME_SOURCE)
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text()
        if source.name == "export_adapter.py":
            # This isolated BF16 export fixes legacy FP16 provenance labels.
            text = text.replace('"base_dtype": "float16"', '"base_dtype": "bfloat16"')
            text = text.replace(
                'info.get("base_dtype") != "float16"', 'info.get("base_dtype") != "bfloat16"'
            )
            text = text.replace("Invalid FP16 benchmark export", "Invalid BF16 benchmark export")
        target.write_text(text)
        sources[str(relative)] = {
            "path": str(source.relative_to(ROOT)),
            "original_sha256": digest(source),
            "sha256": digest(target),
        }
    if not sources:
        raise FileNotFoundError("Operational runtime snapshot is missing")
    write_json(server / "runtime-sources.json", sources)
    for name in ("host.py", "setup.sh"):
        shutil.copyfile(HERE / name, server / name)
    for name in ("run.py", "budget.py"):
        shutil.copyfile(HERE / name, output / name)
    shutil.copyfile(
        HERE / "requirements-server.txt",
        server / "requirements-server.txt",
    )
    write_json(output / "grading-dataset.json", rows)
    allowed = ("instance_id", "repo", "base_commit", "version", "problem_statement")
    write_json(
        output / "tasks-agent-visible.json",
        [{k: row[k] for k in allowed if k in row} for row in rows],
    )
    (output / "instance-ids.txt").write_text("\n".join(protocol["instance_ids"]) + "\n")
    write_json(output / "protocol.json", protocol)
    write_json(server / "protocol.json", protocol)
    pack(output, ready=False)


def fetch_checkpoint(output, protocol):
    # Pin before downloading either manifest or adapter. No "latest" fallback.
    from benchmarks.swe_bench_verified.export_adapter import download_checkpoint

    source = protocol["sources"][VARIANT]
    return download_checkpoint(
        source["repo"],
        source["revision"] or "main",
        protocol["checkpoint"],
        output / "download",
        source["prefix"],
        repo_type="model",
    )


def finalize(output):
    if (output / "results").exists():
        raise ValueError("Results exist; cannot change a campaign checkpoint")
    protocol = json.loads((output / "protocol.json").read_text())
    _, expected = frozen_inputs()
    candidate = json.loads(json.dumps(protocol))
    candidate["sources"][VARIANT]["revision"] = None
    if candidate != expected:
        raise ValueError("Staged protocol or masked training identity changed")
    sys.path[:0] = [str(output / "server/runtime")]
    checkpoint, source = fetch_checkpoint(output, protocol)
    if (
        not re.fullmatch(r"[0-9a-f]{40}", source.get("revision", ""))
        or {**source, "revision": None} != SOURCE
    ):
        raise ValueError("Expected an immutable commit from the new stepmask model repo")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    validate_training(manifest, protocol)
    from benchmarks.swe_bench_verified.export_adapter import export_adapter, verify_export

    adapter = output / "server/adapters" / VARIANT
    info = (
        verify_export(adapter)
        if adapter.exists()
        else export_adapter(checkpoint, adapter, source=source)
    )
    if info["source"] != source or info["checkpoint"] != protocol["checkpoint"]:
        raise ValueError("Existing adapter is from a different checkpoint/revision")
    protocol["sources"][VARIANT] = source
    write_json(output / "protocol.json", protocol)
    write_json(output / "server/protocol.json", protocol)
    pack(output, ready=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "finalize"], nargs="?", default="stage")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    (stage if args.command == "stage" else finalize)(output)
    print((output / "prepared.json").read_text())


if __name__ == "__main__":
    main()
