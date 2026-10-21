"""Vast migration must restore training state and preserve the approved prefix."""

import copy
import json
from pathlib import Path

import pytest
from test_filtered_continuation import make_training, parent_files, trainer_module, update
from test_training_checkpoints import fake_hub  # noqa: F401

from workflow_distillation import diverse_continuation as diversity
from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.training_checkpoints import CheckpointManager, training_identity


def setup_migration(tmp_path, monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "cloud/kaggle/recovery"))
    filtered = importlib.import_module("filtered_continuation")
    index, script, kwargs, manifest_path = parent_files(tmp_path)
    data = json.loads(index.read_text())
    for i, row in enumerate(data["data"]):
        row["task_key"] = f"owner/repo{i % 3}#{i}"
    index.write_text(json.dumps(data))
    kwargs["records_sha256"] = sha256_file(index)
    identity = training_identity(script, **kwargs)
    parent = json.loads(manifest_path.read_text())
    parent["identity"] = identity
    manifest_path.write_text(json.dumps(parent))
    source = filtered.build_plan(index, manifest_path)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    filtered_script = tmp_path / "filtered.py"
    filtered_script.write_text("# filtered\n")
    trainer = trainer_module(script)
    filtered.configure_trainer(trainer, source, entrypoint=filtered_script)
    filtered_identity = trainer.training_identity(filtered_script, **kwargs)
    training = make_training()
    update(training, [3, 0])
    update(training, [7, 2])
    checkpoint = CheckpointManager(tmp_path / "parent", filtered_identity).save(
        model=training[0],
        optimizer=training[1],
        scaler=training[2],
        scheduler=None,
        progress=source["parent"]["progress"],
    )
    plan = diversity.build_plan(index, source_path, resume_step=2, target_step=3)
    return index, source_path, plan, script, filtered_script, kwargs, checkpoint, training


def test_vast_parent_migration_preserves_optimizer_rng_and_uses_new_order(tmp_path, monkeypatch):
    import torch

    from workflow_distillation import vast_continuation as vast

    index, source, cohort, script, filtered_script, kwargs, checkpoint, training = setup_migration(
        tmp_path, monkeypatch
    )
    parent_sha = sha256_file(checkpoint / "manifest.json")
    binding = vast.bind_parent(
        index,
        source,
        cohort,
        checkpoint / "manifest.json",
        expected_sha256=parent_sha,
        filtered_entrypoint=filtered_script,
    )
    child_script = tmp_path / "vast.py"
    child_script.write_text("# portable\n")
    trainer = trainer_module(script)
    vast.configure_trainer(trainer, cohort, binding, entrypoint=child_script)
    child_identity = trainer.training_identity(child_script, **kwargs)
    expected_rng = torch.get_rng_state().clone()
    expected_weights = copy.deepcopy(training[0].state_dict())
    expected_optimizer = copy.deepcopy(training[1].state_dict())
    expected_scaler = training[2].state_dict()
    restored = make_training()
    manager = trainer.CheckpointManager(tmp_path / "vast-checkpoints", child_identity)
    progress = manager.load(
        checkpoint, model=restored[0], optimizer=restored[1], scaler=restored[2], scheduler=None
    )
    assert progress == cohort["progress"]
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert restored[2].state_dict() == expected_scaler
    for name, value in restored[0].state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    for key, state in restored[1].state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(
                value, expected_optimizer["state"][key][field], rtol=0, atol=0
            )
    dataset = trainer.EpochDataset(index)
    trainer.check_dataset(dataset)
    assert [r["trajectory_id"] for r in dataset.records] == [
        json.loads(index.read_text())["data"][i]["trajectory_id"] for i in cohort["ordered_indices"]
    ]
    assert trainer.epoch_batches(len(dataset), 42) == [[0, 1], [2, 3], [4, 5]]
    next_ids = cohort["ordered_indices"][4:6]
    expected_loss = update(training, next_ids)
    torch.set_rng_state(expected_rng)
    assert update(restored, next_ids) == expected_loss
    saved = manager.save(
        model=restored[0],
        optimizer=restored[1],
        scaler=restored[2],
        scheduler=None,
        progress={"global_step": 3, "next_batch": 3, "samples_consumed": 6},
    )
    manager.load(
        saved, model=restored[0], optimizer=restored[1], scaler=restored[2], scheduler=None
    )
    assert manager.identity == child_identity
    damaged = copy.deepcopy(kwargs)
    damaged["settings"]["seed"] = 99
    with pytest.raises(ValueError, match="identity"):
        trainer.training_identity(child_script, **damaged)
    assert sha256_file(checkpoint / "manifest.json") == parent_sha


@pytest.mark.parametrize("mutation", ["cursor", "stamp", "pin", "source", "cohort"])
def test_vast_binding_rejects_wrong_parent_or_order(tmp_path, monkeypatch, mutation):
    from workflow_distillation import vast_continuation as vast

    index, source, cohort, _, filtered_script, _, checkpoint, _ = setup_migration(
        tmp_path, monkeypatch
    )
    path = checkpoint / "manifest.json"
    parent_sha = sha256_file(path)
    manifest = json.loads(path.read_text())
    if mutation == "cursor":
        manifest["progress"]["next_batch"] += 1
    elif mutation == "stamp":
        manifest["identity"]["settings"]["continuation"]["plan_sha256"] = "wrong"
    elif mutation == "source":
        manifest["identity"]["model"] = "other"
    elif mutation == "cohort":
        cohort["ordered_indices"][-2:] = cohort["ordered_indices"][-2:][::-1]
    else:
        parent_sha = "0" * 64
    if mutation in ("cursor", "stamp", "source"):
        path.write_text(json.dumps(manifest))
        parent_sha = sha256_file(path)  # Even a pinned but semantically wrong manifest is rejected.
    with pytest.raises(ValueError):
        vast.bind_parent(
            index,
            source,
            cohort,
            path,
            expected_sha256=parent_sha,
            filtered_entrypoint=filtered_script,
        )


def test_vast_hub_transition_keeps_parent_until_child_and_rejects_rollback(
    tmp_path,
    monkeypatch,
    fake_hub,  # noqa: F811
):
    from workflow_distillation import vast_continuation as vast
    from workflow_distillation.training_checkpoints import HuggingFacePublisher

    index, source, cohort, script, filtered_script, kwargs, checkpoint, training = setup_migration(
        tmp_path, monkeypatch
    )
    old = HuggingFacePublisher(
        "owner/dataset", prefix="first-run", repo_type="dataset", keep_last=3
    )
    fake_hub.transport.repo_type = "dataset"
    old(checkpoint)
    binding = vast.bind_parent(
        index,
        source,
        cohort,
        checkpoint / "manifest.json",
        expected_sha256=sha256_file(checkpoint / "manifest.json"),
        filtered_entrypoint=filtered_script,
    )
    child_script = tmp_path / "vast.py"
    child_script.write_text("# vast\n")
    trainer = trainer_module(script)
    vast.configure_trainer(trainer, cohort, binding, entrypoint=child_script)
    publisher = trainer.HuggingFacePublisher(
        "owner/dataset", prefix="first-run", repo_type="dataset", keep_last=3
    )
    manager = trainer.CheckpointManager(
        tmp_path / "child", trainer.training_identity(child_script, **kwargs), publisher=publisher
    )
    manager.load(
        checkpoint, model=training[0], optimizer=training[1], scaler=training[2], scheduler=None
    )
    manager.republish(model=training[0])
    assert len(fake_hub.commits) == 1
    update(training, cohort["ordered_indices"][4:6])
    child = manager.save(
        model=training[0],
        optimizer=training[1],
        scaler=training[2],
        scheduler=None,
        progress={"global_step": 3, "next_batch": 3, "samples_consumed": 6},
    )
    publisher(child)
    tree = fake_hub.revisions[fake_hub.head[0]]
    assert json.loads(tree["first-run/latest.json"])["checkpoint"] == child.name
    with pytest.raises(ValueError):
        publisher(checkpoint)
    assert len(fake_hub.commits) == 2


def test_vast_bounded_pilot_stops_after_one_update_and_saves(tmp_path, monkeypatch):
    from workflow_distillation import vast_continuation as vast
    from workflow_distillation.epoch_training import run_epoch_loop

    index, source, cohort, script, filtered_script, _, checkpoint, _ = setup_migration(
        tmp_path, monkeypatch
    )
    binding = vast.bind_parent(
        index,
        source,
        cohort,
        checkpoint / "manifest.json",
        expected_sha256=sha256_file(checkpoint / "manifest.json"),
        filtered_entrypoint=filtered_script,
    )
    trainer = trainer_module(script)
    trainer.run_epoch_loop = run_epoch_loop
    child = tmp_path / "vast.py"
    child.write_text("# vast\n")
    vast.configure_trainer(trainer, cohort, binding, entrypoint=child, stop_at_step=3)
    visited, saved = [], []
    result = trainer.run_epoch_loop(
        trainer.epoch_batches(cohort["total_records"]),
        progress=cohort["progress"],
        step=lambda batch, n: visited.append(n) or {},
        save=lambda progress: saved.append(dict(progress)),
        stop=lambda: False,
        emit=lambda event: None,
        save_every=5,
    )
    assert visited == [3]
    assert saved[-1] == {"global_step": 3, "next_batch": 3, "samples_consumed": 6}
    assert result["status"] == "completed"


def test_vast_stop_file_requests_checkpoint_at_completed_boundary(tmp_path, monkeypatch):
    from workflow_distillation import vast_continuation as vast
    from workflow_distillation.epoch_training import run_epoch_loop

    index, source, cohort, script, filtered_script, _, checkpoint, _ = setup_migration(
        tmp_path, monkeypatch
    )
    binding = vast.bind_parent(
        index,
        source,
        cohort,
        checkpoint / "manifest.json",
        expected_sha256=sha256_file(checkpoint / "manifest.json"),
        filtered_entrypoint=filtered_script,
    )
    trainer = trainer_module(script)
    trainer.run_epoch_loop = run_epoch_loop
    child = tmp_path / "vast.py"
    child.write_text("# vast\n")
    stop_file = tmp_path / "STOP"
    stop_file.touch()
    vast.configure_trainer(trainer, cohort, binding, entrypoint=child, stop_file=stop_file)
    visited, saved = [], []
    result = trainer.run_epoch_loop(
        trainer.epoch_batches(cohort["total_records"]),
        progress=cohort["progress"],
        step=lambda batch, n: visited.append(n) or {},
        save=lambda progress: saved.append(dict(progress)),
        stop=lambda: False,
        emit=lambda event: None,
        save_every=5,
    )
    assert visited == []
    assert saved == [cohort["progress"]]
    assert result["status"] == "stopped_checkpointed"
