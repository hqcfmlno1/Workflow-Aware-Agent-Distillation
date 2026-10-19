"""Filtering after a checkpoint must preserve its prefix and all training state."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_training_checkpoints import fake_hub  # noqa: F401 (shared in-memory Hub fixture)

from workflow_distillation.epoch_data import EpochDataset, epoch_batches
from workflow_distillation.epoch_training import run_epoch_loop
from workflow_distillation.training_checkpoints import (
    CheckpointManager,
    HuggingFacePublisher,
    training_identity,
)

HOOKS = Path(__file__).resolve().parents[1] / "cloud/kaggle/recovery"


@pytest.fixture
def continuation(monkeypatch):
    monkeypatch.syspath_prepend(str(HOOKS))
    return importlib.import_module("filtered_continuation")


def fixture_data(root):
    lengths = [29900, 27999, 28000, 5000, 24000, 28500, 26000, 7000, 28000]
    rows, offset = [], 0
    for i, length in enumerate(lengths):
        rows.append({"trajectory_id": f"trajectory-{i}", "task_key": f"task-{i // 2}",
                     "offset": offset, "length": length, "full_token_count": length,
                     "denominator": float(length - 1), "group_counts": [1, length - 1, 0, 0, 0]})
        offset += length
    np.arange(offset, dtype="<u4").tofile(root / "tokens.bin")
    np.ones(offset, dtype="u1").tofile(root / "groups.bin")
    manifest = {"format_version": 1, "selection": "all train_pool resolved=1 token_count<30000",
                "records": len(rows), "tokens": offset, "data": rows,
                "group_counts": [9, offset - 9, 0, 0, 0], "weights": [0, 1, .75, 1.5, 2],
                "file_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest()
                                for n in ("tokens.bin", "groups.bin")}}
    path = root / "index.json"
    path.write_text(json.dumps(manifest))
    return path


def parent_files(root):
    index = fixture_data(root)
    script = root / "parent.py"
    script.write_text("# immutable parent trainer\n")
    kwargs = {"model": "toy", "revision": "fixed", "settings": {"seed": 42},
              "records_sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    identity = training_identity(script, **kwargs)
    manifest = {"format_version": 1, "complete": True, "world_size": 1,
                "identity": identity, "progress": {"global_step": 2, "next_batch": 2,
                                                   "samples_consumed": 4}, "files": {}}
    path = root / "parent-manifest.json"
    path.write_text(json.dumps(manifest))
    return index, script, kwargs, path


def trainer_module(script):
    def check_dataset(dataset):
        assert dataset.manifest["selection"] == "all train_pool resolved=1 token_count<30000"

    return SimpleNamespace(__file__=str(script), EpochDataset=EpochDataset,
                           epoch_batches=epoch_batches, check_dataset=check_dataset,
                           training_identity=training_identity, CheckpointManager=CheckpointManager,
                           HuggingFacePublisher=HuggingFacePublisher)


def test_plan_retains_trained_long_samples_and_filters_only_the_suffix(continuation, tmp_path):
    index, _, _, parent = parent_files(tmp_path)
    plan = continuation.build_plan(index, parent, max_tokens=28000)
    # Seed 42 order is [3,0,7,2,4,6,1,5,8]. Prefix includes long samples 0 and 2.
    assert plan["ordered_indices"] == [3, 0, 7, 2, 4, 6, 1]
    assert plan["retained_prefix_count"] == 4
    assert plan["retained_suffix_count"] == 3
    assert plan["skipped_suffix_count"] == 2  # includes exactly 28,000, not only >28k.
    assert plan["total_records"] == 7
    assert plan["total_optimizer_steps"] == 4
    assert continuation.validate_plan(index, plan) is None


def test_filtered_view_uses_original_tokens_and_never_shuffles_again(continuation, tmp_path):
    index, parent_script, _, parent = parent_files(tmp_path)
    plan = continuation.build_plan(index, parent, max_tokens=28000)
    child = tmp_path / "child.py"
    child.write_text("# filtered entrypoint\n")
    trainer = trainer_module(parent_script)
    continuation.configure_trainer(trainer, plan, entrypoint=child)
    ds = trainer.EpochDataset(index)
    trainer.check_dataset(ds)
    assert [r["trajectory_id"] for r in ds.records] == [
        "trajectory-3", "trajectory-0", "trajectory-7", "trajectory-2",
        "trajectory-4", "trajectory-6", "trajectory-1"]
    assert trainer.epoch_batches(len(ds), 42) == [[0, 1], [2, 3], [4, 5], [6]]
    source = EpochDataset(index)
    for position, source_index in enumerate([3, 0, 7, 2, 4, 6, 1]):
        np.testing.assert_array_equal(ds.record(position)["input_ids"],
                                      source.record(source_index)["input_ids"])
    assert ds.manifest["tokens"] == sum(r["length"] for r in ds.records)
    with pytest.raises(ValueError, match="seed"):
        trainer.epoch_batches(len(ds), 43)


@pytest.mark.parametrize("mutation", ["shuffle", "prefix", "threshold", "source"])
def test_plan_rejects_reordering_changed_boundary_or_source(continuation, tmp_path, mutation):
    index, _, _, parent = parent_files(tmp_path)
    plan = continuation.build_plan(index, parent)
    if mutation == "shuffle":
        plan["ordered_indices"][-1], plan["ordered_indices"][-2] = (
            plan["ordered_indices"][-2], plan["ordered_indices"][-1])
    elif mutation == "prefix":
        plan["parent"]["progress"]["samples_consumed"] = 3
    elif mutation == "threshold":
        plan["max_tokens"] = 29000
    else:
        index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError):
        continuation.validate_plan(index, plan)


def make_training():
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(.2), torch.nn.Linear(4, 1))
    return (model, torch.optim.AdamW(model.parameters(), lr=.001),
            torch.amp.GradScaler("cpu", init_scale=8))


def update(training, indices):
    import torch

    model, optimizer, scaler = training
    optimizer.zero_grad(set_to_none=True)
    loss = sum(model(torch.tensor([[i / 10, .5]])).square().mean() for i in indices) / len(indices)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


def test_migrate_parent_then_resume_child_without_replay_or_reset(continuation, tmp_path):
    import torch

    index, parent_script, kwargs, _ = parent_files(tmp_path)
    torch.manual_seed(17)
    training = make_training()
    for indices in ([3, 0], [7, 2]):
        update(training, indices)
    parent_identity = training_identity(parent_script, **kwargs)
    original = CheckpointManager(tmp_path / "parent", parent_identity)
    parent = original.save(model=training[0], optimizer=training[1], scaler=training[2],
                           scheduler=None, progress={"global_step": 2, "next_batch": 2,
                                                     "samples_consumed": 4})
    manifest_before = (parent / "manifest.json").read_bytes()
    plan = continuation.build_plan(index, parent / "manifest.json")
    expected_losses = [update(training, ids) for ids in ([4, 6], [1])]
    expected_weights = copy.deepcopy(training[0].state_dict())
    expected_optimizer = copy.deepcopy(training[1].state_dict())
    expected_scaler, expected_rng = training[2].state_dict(), torch.get_rng_state().clone()

    child_script = tmp_path / "child.py"
    child_script.write_text("# filtered entrypoint\n")
    trainer = trainer_module(parent_script)
    continuation.configure_trainer(trainer, plan, entrypoint=child_script)
    child_identity = trainer.training_identity(child_script, **kwargs)
    assert child_identity != parent_identity
    assert child_identity["settings"]["continuation"]["plan_sha256"]
    manager = trainer.CheckpointManager(tmp_path / "child", child_identity, keep_last=1)
    training = make_training()
    progress = manager.load(parent, model=training[0], optimizer=training[1],
                            scheduler=None, scaler=training[2])
    assert manager.identity == child_identity

    assert (parent / "manifest.json").read_bytes() == manifest_before
    ds = trainer.EpochDataset(index)
    batches = trainer.epoch_batches(len(ds), 42)
    visited, losses = [], []

    def step(batch, number):
        source_indices = [plan["ordered_indices"][i] for i in batch]
        visited.extend(source_indices)
        losses.append(update(training, source_indices))
        return {"loss": losses[-1]}

    def save(position):
        manager.save(model=training[0], optimizer=training[1], scheduler=None,
                     scaler=training[2], progress=position)

    result = run_epoch_loop(batches, progress=progress, step=step, save=save,
                            stop=lambda: len(losses) == 1, emit=lambda e: None, save_every=1)
    assert result["status"] == "stopped_checkpointed"
    training = make_training()
    progress = manager.load("latest", model=training[0], optimizer=training[1],
                            scheduler=None, scaler=training[2])
    result = run_epoch_loop(batches, progress=progress, step=step, save=save,
                            stop=lambda: False, emit=lambda e: None, save_every=1)
    assert result["progress"] == {"global_step": 4, "next_batch": 4, "samples_consumed": 7}
    assert visited == [4, 6, 1]
    assert losses == expected_losses
    for name, value in training[0].state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
    for key, state in training[1].state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, expected_optimizer["state"][key][field],
                                       atol=0, rtol=0)
    assert training[2].state_dict() == expected_scaler
    assert torch.equal(torch.get_rng_state(), expected_rng)

    # A different mathematical identity must never use the parent migration path.
    changed = {**kwargs, "settings": {"seed": 99}}
    with pytest.raises(ValueError, match="parent.*identity"):
        trainer.training_identity(child_script, **changed)
    damaged = json.loads((parent / "manifest.json").read_text())
    damaged["progress"]["samples_consumed"] = 2
    (parent / "manifest.json").write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="parent"):
        manager.load(parent, model=training[0], optimizer=training[1],
                     scheduler=None, scaler=training[2])
    assert manager.identity == child_identity

def test_parent_republish_and_child_hub_transition_keep_same_prefix_and_retention(
        continuation, tmp_path, fake_hub):  # noqa: F811
    from workflow_distillation.training_checkpoints import download_latest_checkpoint

    index, script, kwargs, _ = parent_files(tmp_path)
    training = make_training()
    parent_identity = training_identity(script, **kwargs)
    old_publisher = HuggingFacePublisher('owner/dataset', prefix='first-run',
                                       repo_type='dataset', keep_last=3)
    fake_hub.transport.repo_type = 'dataset'
    old_manager = CheckpointManager(tmp_path / 'parent', parent_identity, publisher=old_publisher)
    parent = old_manager.save(model=training[0], optimizer=training[1], scheduler=None,
                              scaler=training[2], progress={
                                  'global_step': 2, 'next_batch': 2, 'samples_consumed': 4})
    plan = continuation.build_plan(index, parent / 'manifest.json')
    trainer = trainer_module(script)
    child_script = tmp_path / 'child.py'
    child_script.write_text('# child\n')
    continuation.configure_trainer(trainer, plan, entrypoint=child_script)
    child_identity = trainer.training_identity(child_script, **kwargs)
    publisher = trainer.HuggingFacePublisher('owner/dataset', prefix='first-run',
                                            repo_type='dataset', keep_last=3)
    manager = trainer.CheckpointManager(tmp_path / 'child', child_identity,
                                        publisher=publisher, keep_last=1)
    progress = manager.load(parent, model=training[0], optimizer=training[1],
                            scheduler=None, scaler=training[2])
    assert progress['global_step'] == 2
    manager.republish(model=training[0])  # Actual original trainer behavior.
    assert len(fake_hub.commits) == 1
    for number in (3, 4, 5):
        update(training, [4, 6] if number == 3 else [1])
        child = manager.save(model=training[0], optimizer=training[1], scheduler=None,
                              scaler=training[2], progress={
                                  'global_step': number, 'next_batch': number,
                                  'samples_consumed': 6 if number == 3 else 7})
        publisher(child)  # Retry does not commit again.
        assert len(fake_hub.commits) == number - 1
    tree = fake_hub.revisions[fake_hub.head[0]]
    assert json.loads(tree['first-run/latest.json'])['checkpoint'] == 'checkpoint-00000005'
    assert {name.split('/')[1] for name in tree if '/checkpoint-' in name} == {
        'checkpoint-00000003', 'checkpoint-00000004', 'checkpoint-00000005'}
    assert len(fake_hub.revisions) == 1  # History still bounded by squash.
    downloaded = download_latest_checkpoint('owner/dataset', tmp_path / 'download',
                                            prefix='first-run', repo_type='dataset')
    assert continuation.check_resume_manifest(downloaded, plan) == 'continuation'
    manager.load(downloaded, model=training[0], optimizer=training[1],
                 scheduler=None, scaler=training[2])
    with pytest.raises(ValueError):
        publisher(parent)  # Cannot roll back from a child to the old parent.


def test_hub_migration_rejects_a_parent_with_same_identity_but_different_manifest(
        continuation, tmp_path, fake_hub):  # noqa: F811
    index, script, kwargs, _ = parent_files(tmp_path)
    training = make_training()
    parent_identity = training_identity(script, **kwargs)
    publisher = HuggingFacePublisher('owner/model', prefix='first-run')
    manager = CheckpointManager(tmp_path / 'parent', parent_identity, publisher=publisher)
    parent = manager.save(model=training[0], optimizer=training[1], scheduler=None,
                           scaler=training[2], progress={
                               'global_step': 2, 'next_batch': 2, 'samples_consumed': 4})
    plan = continuation.build_plan(index, parent / 'manifest.json')
    tree = fake_hub.revisions[fake_hub.head[0]]
    name = 'first-run/checkpoint-00000002/manifest.json'
    remote = json.loads(tree[name])
    remote['progress']['samples_consumed'] = 3
    tree[name] = json.dumps(remote).encode()
    trainer = trainer_module(script)
    child_script = tmp_path / 'child.py'
    child_script.write_text('# child\n')
    continuation.configure_trainer(trainer, plan, entrypoint=child_script)
    child_identity = trainer.training_identity(child_script, **kwargs)
    child_manager = trainer.CheckpointManager(tmp_path / 'child', child_identity,
        publisher=trainer.HuggingFacePublisher('owner/model', prefix='first-run'))
    with pytest.raises(RuntimeError, match='identity'):
        child_manager.save(model=training[0], optimizer=training[1], scheduler=None,
                            scaler=training[2], progress={
                                'global_step': 3, 'next_batch': 3, 'samples_consumed': 6})
    assert len(fake_hub.commits) == 1


def test_hub_migration_rejects_copied_stamp_with_changed_model(
        continuation, tmp_path, fake_hub):  # noqa: F811
    index, script, kwargs, _ = parent_files(tmp_path)
    model, optimizer, scaler = make_training()
    parent_manager = CheckpointManager(tmp_path / 'parent', training_identity(script, **kwargs),
        publisher=HuggingFacePublisher('owner/model', prefix='first-run'))
    parent = parent_manager.save(model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
        progress={'global_step': 2, 'next_batch': 2, 'samples_consumed': 4})
    plan = continuation.build_plan(index, parent / 'manifest.json')
    trainer = trainer_module(script)
    child_script = tmp_path / 'child.py'
    child_script.write_text('# child\n')
    continuation.configure_trainer(trainer, plan, entrypoint=child_script)
    wrong_identity = trainer.training_identity(child_script, **kwargs)
    wrong_identity['model'] = 'unrelated-model'
    child = CheckpointManager(tmp_path / 'wrong', wrong_identity).save(
        model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
        progress={'global_step': 3, 'next_batch': 3, 'samples_consumed': 6})
    publisher = trainer.HuggingFacePublisher('owner/model', prefix='first-run')
    with pytest.raises(ValueError, match='identity'):
        publisher(child)
    assert len(fake_hub.commits) == 1


def test_bootstrap_validates_plan_and_checkpoint_before_copying_entrypoint(
        continuation, tmp_path, monkeypatch):
    from scripts.cloud import resume_first_run as bootstrap

    index, _, _, parent = parent_files(tmp_path)
    plan = continuation.build_plan(index, parent)
    assets, project, checkpoint = (tmp_path / n for n in ('assets', 'project', 'checkpoint'))
    assets.mkdir()
    checkpoint.mkdir()
    (project / 'scripts/training').mkdir(parents=True)
    (assets / 'continuation-plan.json').write_text(json.dumps(plan))
    (assets / 'train_qlora_filtered.py').write_text('# new entrypoint\n')
    (checkpoint / 'manifest.json').write_bytes(parent.read_bytes())
    monkeypatch.setattr(bootstrap, 'PLAN_SHA256', hashlib.sha256(
        (assets / 'continuation-plan.json').read_bytes()).hexdigest())
    entrypoint, selection = bootstrap.prepare_continuation(assets, project, index, checkpoint)
    assert entrypoint.read_text() == '# new entrypoint\n'
    assert selection['total_records'] == 7
    assert selection['resume_kind'] == 'parent'
    entrypoint.unlink()
    (checkpoint / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='neither'):
        bootstrap.prepare_continuation(assets, project, index, checkpoint)
    assert not entrypoint.exists()
    (assets / 'continuation-plan.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        bootstrap.prepare_continuation(assets, project, index, checkpoint)


def test_child_payload_corruption_still_rejected(continuation, tmp_path):
    index, script, kwargs, parent = parent_files(tmp_path)
    plan = continuation.build_plan(index, parent)
    child_script = tmp_path / 'child.py'
    child_script.write_text('# child\n')
    trainer = trainer_module(script)
    continuation.configure_trainer(trainer, plan, entrypoint=child_script)
    manager = trainer.CheckpointManager(tmp_path / 'child',
                                        trainer.training_identity(child_script, **kwargs))
    model, optimizer, scaler = make_training()
    saved = manager.save(model=model, optimizer=optimizer, scheduler=None, scaler=scaler,
                          progress={'global_step': 3, 'next_batch': 3, 'samples_consumed': 6})
    (saved / 'adapter.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        manager.load(saved, model=model, optimizer=optimizer, scheduler=None, scaler=scaler)
