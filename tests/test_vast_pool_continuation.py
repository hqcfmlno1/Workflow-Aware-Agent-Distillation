"""Pool expansion preserves every approved sample and the optimizer cursor."""

import copy

import pytest


def rows():
    return [{"task_key": key, "length": 10} for key in [
        "o/a#1", "o/b#1", "o/a#2", "o/c#1", "o/a#2", "o/b#1", "o/d#1",
    ]]


def test_pool_suffix_prioritizes_new_tasks_then_balances_repos_without_repeats():
    from workflow_distillation.vast_pool_continuation import extend_order

    source = rows()
    result = extend_order(source, list(range(7)), [0, 1], 28000)
    assert result == [0, 1, 3, 6, 2, 5, 4]
    assert len(result) == len(set(result)) == 7
    assert source == rows()


@pytest.mark.parametrize("bad", ["long", "duplicate", "outside"])
def test_pool_extension_rejects_ineligible_or_ambiguous_sources(bad):
    from workflow_distillation.vast_pool_continuation import extend_order

    source, order, prefix = rows(), list(range(7)), [0, 1]
    if bad == "long":
        source[6]["length"] = 28000
    elif bad == "duplicate":
        order.append(6)
    else:
        prefix.append(7)
    with pytest.raises(ValueError):
        extend_order(source, order, prefix, 28000)


def test_extension_identity_keeps_math_and_embeds_ancestor_metadata(tmp_path):
    from test_vast_existing_child import fixture

    from workflow_distillation.vast_pool_continuation import extension_identity, manifest_digest

    _, _, parent = fixture(tmp_path)
    original = copy.deepcopy(parent)
    entrypoint = tmp_path / "pool.py"
    entrypoint.write_text("# pool launcher\n")
    plan = {"ordered_indices": list(range(12)), "preserved_cohort_count": 8}
    identity, binding = extension_identity(parent, manifest_digest(parent), plan, entrypoint)
    assert parent == original
    assert identity["settings"]["config"] == parent["identity"]["settings"]["config"]
    assert identity["math_sources"] == parent["identity"]["math_sources"]
    assert identity["packages"] == parent["identity"]["packages"]
    assert identity["settings"]["pool_continuation"]["parent_manifest"] == parent
    assert binding["parent"]["progress"]["samples_consumed"] == 8
    with pytest.raises(ValueError):
        extension_identity(parent, "wrong-checksum", plan, entrypoint)


def test_expansion_restores_optimizer_and_rng_from_the_exact_parent(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace

    import torch
    from test_filtered_continuation import make_training, update
    from test_vast_existing_child import fixture

    from workflow_distillation import training_checkpoints as checkpoints
    from workflow_distillation.training_checkpoints import CheckpointManager, _manifest
    from workflow_distillation.vast_microbatch import migration_manager
    from workflow_distillation.vast_pool_continuation import extension_identity, manifest_digest

    _, _, old = fixture(tmp_path)
    training = make_training()
    update(training, [1, 2])
    parent_path = CheckpointManager(tmp_path / "parent", old["identity"]).save(
        model=training[0], optimizer=training[1], scaler=training[2], scheduler=None,
        progress=old["progress"],
    )
    parent = _manifest(parent_path)
    # Model the second rank's committed payload while exercising rank zero's
    # real optimizer/RNG restoration on CPU.
    shutil.copyfile(parent_path / "rank-00000.pt", parent_path / "rank-00001.pt")
    parent["world_size"] = 2
    parent["files"]["rank-00001.pt"] = parent["files"]["rank-00000.pt"]
    checkpoints._json_file(parent_path / "manifest.json", parent)
    monkeypatch.setattr(checkpoints, "_group", lambda: SimpleNamespace(
        get_rank=lambda: 0, get_world_size=lambda: 2,
    ))
    monkeypatch.setattr(checkpoints, "_gather", lambda value: [value, value])
    entrypoint = tmp_path / "pool.py"
    entrypoint.write_text("# expansion\n")
    identity, binding = extension_identity(parent, manifest_digest(parent), {
        "ordered_indices": list(range(12)), "preserved_cohort_count": 8,
    }, entrypoint)
    expected_rng = torch.get_rng_state().clone()
    restored = make_training()
    manager = migration_manager(binding)(tmp_path / "child", identity)
    progress = manager.load(parent_path, model=restored[0], optimizer=restored[1],
                            scaler=restored[2], scheduler=None)
    assert progress == {"global_step": 3, "next_batch": 3, "samples_consumed": 8}
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for x, y in zip(training[0].parameters(), restored[0].parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    for key, state in training[1].state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, restored[1].state_dict()["state"][key][field])
