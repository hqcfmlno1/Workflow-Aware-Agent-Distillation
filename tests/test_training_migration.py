import copy
import json
from pathlib import Path

import pytest
import torch
from test_training_checkpoints import fake_hub, setup_training, step  # noqa: F401

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.training_checkpoints import (
    CheckpointManager,
    HuggingFacePublisher,
)


def make_parent(tmp_path, publisher=None):
    source = setup_training()
    step(source)
    manager = CheckpointManager(tmp_path / "parent", {"legacy": "exact"}, publisher=publisher)
    path = manager.save(
        model=source[0],
        optimizer=source[1],
        scheduler=source[2],
        scaler=source[3],
        progress={"global_step": 1, "next_batch": 1, "samples_consumed": 2},
    )
    return source, path


def test_parent_to_child_migration_preserves_next_update_optimizer_scaler_rng(tmp_path):
    from workflow_distillation.training.checkpoints import MigrationManager, parent_binding

    source, parent = make_parent(tmp_path)
    binding = parent_binding(parent / "manifest.json", sha256_file(parent / "manifest.json"))
    expected_loss = step(source)
    expected_weights = source[0].adapter.weight.detach().clone()
    restored = setup_training()
    manager = MigrationManager(tmp_path / "child", {"new": "identity"}, binding=binding)
    progress = manager.load(
        parent, model=restored[0], optimizer=restored[1], scheduler=restored[2], scaler=restored[3]
    )
    assert progress["samples_consumed"] == 2
    assert manager.identity == {"new": "identity"}
    assert step(restored) == expected_loss
    torch.testing.assert_close(restored[0].adapter.weight, expected_weights, rtol=0, atol=0)
    child = manager.save(
        model=restored[0],
        optimizer=restored[1],
        scheduler=restored[2],
        scaler=restored[3],
        progress={"global_step": 2, "next_batch": 2, "samples_consumed": 6},
    )
    again = setup_training()
    manager.load(child, model=again[0], optimizer=again[1], scheduler=again[2], scaler=again[3])
    with pytest.raises(ValueError, match="checksum"):
        parent_binding(parent / "manifest.json", "f" * 64)
    (child / "adapter.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        manager.load(child, model=again[0], optimizer=again[1], scheduler=again[2], scaler=again[3])


def test_migration_publisher_is_atomic_idempotent_retained_and_never_rolls_back(tmp_path, fake_hub):  # noqa: F811
    from workflow_distillation.training.checkpoints import (
        MigrationManager,
        MigrationPublisher,
        parent_binding,
    )

    source, parent = make_parent(tmp_path, HuggingFacePublisher("owner/run", prefix="first-run"))
    binding = parent_binding(parent / "manifest.json", sha256_file(parent / "manifest.json"))
    publisher = MigrationPublisher(
        "owner/run",
        prefix="first-run",
        keep_last=3,
        binding=binding,
        child_identity={"new": "identity"},
    )
    manager = MigrationManager(
        tmp_path / "child", {"new": "identity"}, binding=binding, publisher=publisher
    )
    manager.load(
        parent, model=source[0], optimizer=source[1], scheduler=source[2], scaler=source[3]
    )
    manager.republish(model=source[0])
    for number in range(2, 6):
        step(source)
        child = manager.save(
            model=source[0],
            optimizer=source[1],
            scheduler=source[2],
            scaler=source[3],
            progress={
                "global_step": number,
                "next_batch": number,
                "samples_consumed": 2 + 4 * (number - 1),
            },
        )
        publisher(child)
    tree = fake_hub.revisions[fake_hub.head[0]]
    assert json.loads(tree["first-run/latest.json"]) == {"checkpoint": "checkpoint-00000005"}
    assert len({p.split("/")[1] for p in tree if "/checkpoint-" in p}) == 3
    with pytest.raises(ValueError):
        publisher(parent)


def test_manifest_with_same_identity_but_different_cursor_cannot_be_parent(tmp_path, fake_hub):  # noqa: F811
    from workflow_distillation.training.checkpoints import (
        MigrationManager,
        MigrationPublisher,
        parent_binding,
    )

    source, parent = make_parent(tmp_path, HuggingFacePublisher("owner/run", prefix="first-run"))
    binding = parent_binding(parent / "manifest.json", sha256_file(parent / "manifest.json"))
    tree = fake_hub.revisions[fake_hub.head[0]]
    key = "first-run/checkpoint-00000001/manifest.json"
    tampered = json.loads(tree[key])
    tampered["progress"]["samples_consumed"] = 1
    tree[key] = json.dumps(tampered).encode()
    publisher = MigrationPublisher(
        "owner/run", prefix="first-run", binding=binding, child_identity={"new": "identity"}
    )
    manager = MigrationManager(
        tmp_path / "child", {"new": "identity"}, binding=binding, publisher=publisher
    )
    step(source)
    with pytest.raises(RuntimeError, match="identity"):
        manager.save(
            model=source[0],
            optimizer=source[1],
            scheduler=source[2],
            scaler=source[3],
            progress={"global_step": 2, "next_batch": 2, "samples_consumed": 6},
        )
    assert (tmp_path / "child/checkpoint-00000002/manifest.json").exists()
    assert len(fake_hub.commits) == 1


def test_parent_validation_rejects_objective_data_and_package_drift(tmp_path):
    from workflow_distillation.diverse_continuation import fingerprint
    from workflow_distillation.training.compatibility import validate_parent
    from workflow_distillation.training.config import Experiment, load_profile
    from workflow_distillation.training.identity import package_versions

    profile = load_profile(Path("configs/training/gpus/rtx4080_32gb/profile.json"))
    plan = {"ordered_indices": [0, 1, 2, 3], "source_index_sha256": "data"}
    # Use real historical source hashes and installed versions for a CPU-safe audit.
    from workflow_distillation.training.compatibility import source_hashes

    hashes = source_hashes()
    identity = {
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "records_sha256": "data",
        "packages": package_versions(),
        "math_sources": {
            n: hashes["src/workflow_distillation/" + n]
            for n in (
                "optimized_qlora.py",
                "weighted_sft_loss.py",
                "t4_attention.py",
                "ddp_probe.py",
            )
        },
        "script_sha256": hashes["scripts/training/gpus/rtx4080_32gb/train.py"],
        "settings": {
            "seed": 42,
            "config": Experiment().effective_config(profile),
            "pool_continuation": {"plan_sha256": fingerprint(plan)},
            "gpu_execution": {
                "runtime_sha256": hashes[
                    "src/workflow_distillation/gpu_profiles/rtx4080_32gb/runtime.py"
                ],
                "profile": {
                    "attention": "t4_repeated_kv",
                    "activation_storage": "hybrid",
                    "gpu_budget_bytes": 4 * 2**30,
                },
            },
        },
    }
    parent = {
        "format_version": 1,
        "complete": True,
        "world_size": 2,
        "identity": identity,
        "progress": {"global_step": 1, "next_batch": 1, "samples_consumed": 4},
    }
    validate_parent(parent, profile, plan, 42)
    for target, key, value in [
        ("config", "rank", 16),
        ("config", "learning_rate", 1e-3),
        ("identity", "records_sha256", "other"),
        ("packages", "numpy", "other"),
    ]:
        wrong = copy.deepcopy(parent)
        location = (
            wrong["identity"]["settings"]["config"]
            if target == "config"
            else (wrong["identity"]["packages"] if target == "packages" else wrong["identity"])
        )
        location[key] = value
        with pytest.raises(ValueError):
            validate_parent(wrong, profile, plan, 42)
