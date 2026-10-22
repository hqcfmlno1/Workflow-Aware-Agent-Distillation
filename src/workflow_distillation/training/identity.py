"""New-run identity binds canonical implementation, data plan and pinned parent."""

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from workflow_distillation.diverse_continuation import fingerprint
from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.training.config import MODEL, REVISION, Experiment

ROOT = Path(__file__).resolve().parents[3]


def package_versions():
    result = {}
    for name in ("torch", "numpy", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def implementation_hashes():
    package = Path(__file__).parent
    # Include the pure shared numerical/data/checkpoint primitives, not just the CLI.
    files = list(package.glob("*.py")) + [ROOT / "scripts/train.py"]
    files += [
        ROOT / "src/workflow_distillation" / name
        for name in (
            "epoch_data.py",
            "epoch_training.py",
            "weighted_sft_loss.py",
            "weighted_sft_batches.py",
            "t4_attention.py",
            "optimized_qlora.py",
            "ddp_probe.py",
            "training_checkpoints.py",
            "diverse_continuation.py",
            "vast_pool_continuation.py",
            "vast_microbatch.py",
        )
    ]
    files.append(ROOT / "legacy/first_run/source-hashes.json")
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)}


def build_identity(binding, profile, plan, seed):
    return {
        "model": MODEL,
        "revision": REVISION,
        "packages": package_versions(),
        "records_sha256": plan["source_index_sha256"],
        "settings": {
            "config": Experiment().effective_config(profile),
            "seed": seed,
            "data": {"plan_sha256": fingerprint(plan)},
            "execution": {
                "profile": profile.to_dict(),
                "ddp_bucket_bytes": 0,
                "gradient_sync": "scaled SUM/world_size before unscale",
            },
            "training_refactor": {
                "version": 1,
                "parent": binding,
                "bitwise_equivalent": False,
                "implementation_sha256": implementation_hashes(),
            },
        },
    }


def read_binding(resumed, parent_manifest=None, expected_sha256=None):
    from workflow_distillation.training.checkpoints import parent_binding

    if parent_manifest is not None:
        if not expected_sha256:
            raise ValueError("Explicit migration requires the parent manifest SHA256")
        return parent_binding(parent_manifest, expected_sha256)
    if expected_sha256 is not None:
        raise ValueError("Parent checksum supplied without a parent manifest")
    try:
        binding = resumed["identity"]["settings"]["training_refactor"]["parent"]
        # Our manifests are committed with this canonical serialization. Verify
        # embedded lineage before reconstructing the exact child identity.
        data = json.dumps(binding["manifest"], indent=2, sort_keys=True) + "\n"
        import hashlib

        if hashlib.sha256(data.encode()).hexdigest() != binding["manifest_sha256"]:
            raise ValueError("Embedded parent manifest checksum differs")
        return binding
    except KeyError as exc:
        raise ValueError("Legacy resume requires an explicit SHA-pinned parent manifest") from exc
