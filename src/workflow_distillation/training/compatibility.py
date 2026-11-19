"""Read-only legacy audit boundary. Historical launchers are never executed here."""

import json
from pathlib import Path

from workflow_distillation.diverse_continuation import fingerprint
from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.training.config import MODEL, REVISION, Experiment

ROOT = Path(__file__).resolve().parents[3]
RETIRED_4080_RUNTIME = "src/workflow_distillation/gpu_profiles/rtx4080_32gb/runtime.py"
RETIRED_4080_SOURCES = {
    "scripts/training/gpus/rtx4080_32gb/train.py",
    "scripts/training/train_qlora_vast_4080.py",
    "src/workflow_distillation/gpu_profiles/__init__.py",
    "src/workflow_distillation/gpu_profiles/rtx4080_32gb/__init__.py",
    RETIRED_4080_RUNTIME,
    "src/workflow_distillation/vast_device_policy.py",
}


def source_hashes():
    expected = json.loads((ROOT / "legacy/first_run/source-hashes.json").read_text())
    for name, digest in expected.items():
        try:
            actual = sha256_file(ROOT / name)
        except FileNotFoundError:
            # Retired files are only needed as identity records for migration.
            if name in RETIRED_4080_SOURCES:
                continue
            raise
        if actual != digest:
            raise ValueError(f"Historical source checksum differs: {name}")
    return expected


def validate_parent(parent, profile, plan, seed):
    from workflow_distillation.training.identity import package_versions

    if (
        parent.get("complete") is not True
        or parent.get("format_version") != 1
        or parent.get("world_size") != profile.world_size
    ):
        raise ValueError("Unsupported parent checkpoint format or world size")
    identity = parent["identity"]
    settings = identity["settings"]
    if (
        identity["model"] != MODEL
        or identity["revision"] != REVISION
        or identity["records_sha256"] != plan["source_index_sha256"]
        or identity["packages"] != package_versions()
        or settings["seed"] != seed
        or settings.get("pool_continuation", {}).get("plan_sha256") != fingerprint(plan)
    ):
        raise ValueError("Parent model, dataset, pool, seed or package identity differs")
    actual = dict(settings["config"])
    expected = Experiment().effective_config(profile)
    # Placement is an explicit migration choice. Objective and batch geometry
    # must remain identical; batch changes need a separately reviewed transition.
    actual.pop("saved_inputs", None)
    expected.pop("saved_inputs")
    if actual != expected:
        raise ValueError("Parent objective, optimizer or batch configuration differs")
    hashes = source_hashes()
    math_names = ("optimized_qlora.py", "weighted_sft_loss.py", "t4_attention.py", "ddp_probe.py")
    if identity["math_sources"] != {
        n: hashes["src/workflow_distillation/" + n] for n in math_names
    }:
        raise ValueError("Parent mathematical source identity differs")
    scripts = {v for k, v in hashes.items() if k.startswith("scripts/training/")}
    if identity["script_sha256"] not in scripts:
        raise ValueError("Unknown historical launcher checksum")
    for name, digest in settings.get("runtime_sources", {}).items():
        if hashes.get("src/workflow_distillation/" + name) != digest:
            raise ValueError(f"Parent runtime checksum differs: {name}")
    for name, digest in settings.get("vast_continuation", {}).get("sources", {}).items():
        if hashes.get(name) != digest:
            raise ValueError(f"Parent continuation checksum differs: {name}")
    execution = settings.get("gpu_execution")
    if execution and execution["runtime_sha256"] != hashes[RETIRED_4080_RUNTIME]:
        raise ValueError("Parent execution adapter checksum differs")
