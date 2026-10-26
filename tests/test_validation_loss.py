"""Catch causal masking, changed objectives, corrupt weights and partial-resume bugs."""

import json

import pytest
import torch
import torch.nn.functional as F

from workflow_distillation.validation_loss import (
    aggregate,
    evaluate_record,
    load_adapter,
    load_results,
    summarize_groups,
)


def test_chunked_metrics_match_dense_causal_ce_without_changing_model():
    from test_fp16_training import tiny_state

    state = tiny_state()
    state.model.eval()
    before = {k: v.clone() for k, v in state.model.state_dict().items()}
    record = dict(input_ids=[1, 2, 3, 4, 5, 6, 7], groups=[4, 0, 1, 1, 2, 3, 4])
    ids = torch.tensor([record["input_ids"]])
    with torch.inference_mode():
        logits = state.head(state.decoder(input_ids=ids).last_hidden_state)
        ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), ids[:, 1:], reduction="none")[0]
    actual = evaluate_record(state, record, torch.device("cpu"), chunk_size=2)
    assert actual["counts"] == [0, 2, 1, 1, 1]
    assert actual["ce_sums"] == pytest.approx(
        [0, float(ce[1:3].sum()), float(ce[3]), float(ce[4]), float(ce[5])], rel=1e-6
    )
    for key, value in before.items():
        torch.testing.assert_close(value, state.model.state_dict()[key], rtol=0, atol=0)
    assert all(p.grad is None for p in state.model.parameters())


def test_metrics_use_fixed_shared_objectives_and_distinguish_macro_micro():
    a = dict(counts=[0, 4, 0, 1, 1], ce_sums=[0, 8, 0, 4, 6])
    b = dict(counts=[0, 1, 0, 1, 0], ce_sums=[0, 1, 0, 2, 0])
    one = summarize_groups(a)
    assert one["ce"] == pytest.approx(3)
    assert one["token_weighted_loss"] == pytest.approx(26 / 7.5)
    assert one["group_weighted_loss"] == pytest.approx(7.6 / 1.8)
    assert one["content_ce"] is None
    result = aggregate([a, b])
    assert result["ce_macro"] == pytest.approx(2.25)
    assert result["ce_micro"] == pytest.approx(21 / 8)
    assert result["reasoning_ce"] == pytest.approx(9 / 5)
    assert result["content_ce"] is None
    with pytest.raises(ValueError):
        aggregate([])


def test_resume_ignores_only_unterminated_tail_and_rejects_conflicting_rows(tmp_path):
    path = tmp_path / "events.jsonl"
    row = dict(
        key="fingerprint", trajectory_id="one", counts=[0, 1, 0, 0, 0], ce_sums=[0, 2, 0, 0, 0]
    )
    path.write_text(json.dumps(row) + "\n" + '{"key":')
    assert load_results(path) == {("fingerprint", "one"): row}
    # Reading also repairs an interrupted tail before new lines are appended.
    assert path.read_text() == json.dumps(row) + "\n"
    path.write_text(json.dumps(row) + "\n" + json.dumps(dict(row, ce_sums=[0, 3, 0, 0, 0])) + "\n")
    with pytest.raises(ValueError, match="Conflicting"):
        load_results(path)
    path.write_text("{bad}\n")
    with pytest.raises(ValueError):
        load_results(path)


def test_adapter_load_checks_hash_keys_and_finiteness_before_any_copy(tmp_path):
    from workflow_distillation.epoch_data import sha256_file

    model = torch.nn.Linear(2, 2, bias=False)
    path = tmp_path / "adapter.pt"
    original = model.weight.detach().clone()
    torch.save({"weight": torch.ones(2, 2)}, path)
    manifest = {"files": {"adapter.pt": sha256_file(path)}}
    load_adapter(model, path, manifest)
    assert torch.equal(model.weight, torch.ones(2, 2))
    with pytest.raises(ValueError, match="checksum"):
        load_adapter(model, path, {"files": {"adapter.pt": "bad"}})
    for bad in ({"wrong": original}, {"weight": torch.full((2, 2), float("nan"))}):
        torch.save(bad, path)
        with pytest.raises(ValueError):
            load_adapter(model, path, {"files": {"adapter.pt": sha256_file(path)}})
        assert torch.equal(model.weight, torch.ones(2, 2))


def test_local_discovery_ignores_staging_and_rejects_wrong_dataset(tmp_path):
    from workflow_distillation.lora_fp16.config import RunConfig
    from workflow_distillation.validation_checkpoints import discover

    config = RunConfig()
    root = tmp_path / "checkpoints"
    staging = root / ".checkpoint-00000010.pending"
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_text("{}")
    spec = dict(name="old", local=str(root))
    assert discover(spec, config, "dataset", local_only=True) == []
    saved = root / "checkpoint-00000010"
    saved.mkdir()
    manifest = dict(
        format_version=1,
        complete=True,
        world_size=1,
        identity=dict(
            model=config.model,
            revision=config.revision,
            records_sha256="dataset",
            settings=dict(config=config.to_dict()),
        ),
        progress=dict(global_step=10),
        files={"adapter.pt": "a" * 64},
    )
    (saved / "adapter.pt").write_bytes(b"example")
    (saved / "manifest.json").write_text(json.dumps(manifest))
    result = discover(spec, config, "dataset", local_only=True)
    assert [r["step"] for r in result] == [10]
    with pytest.raises(ValueError, match="mismatch"):
        discover(spec, config, "different-data", local_only=True)


def test_partial_resume_completes_only_missing_trajectory_and_exports(tmp_path):
    import csv
    from contextlib import nullcontext

    from test_fp16_training import records, tiny_state

    from workflow_distillation.validation_checkpoints import evaluate_items

    state = tiny_state()
    state.model.disable_adapter = nullcontext
    data = records()
    dataset = type("Dataset", (), dict(records=data, record=lambda self, i: data[i]))()
    items = [dict(run="base", step=0, source="tiny")]
    specs = [dict(name="old", known_last_step=2)]
    kwargs = dict(
        state=state,
        dataset=dataset,
        validation=[0, 1, 2],
        items=items,
        specs=specs,
        output=tmp_path,
        identity={"fixture": True},
        steps_per_epoch=2,
        device=torch.device("cpu"),
        chunk_size=2,
    )
    evaluate_items(**kwargs)
    complete = (tmp_path / "events.jsonl").read_text().splitlines(keepends=True)
    (tmp_path / "events.jsonl").write_text("".join(complete[:2]) + "{interrupted")
    evaluate_items(**kwargs)
    resumed = (tmp_path / "events.jsonl").read_text().splitlines(keepends=True)
    assert resumed[:2] == complete[:2]
    assert len(resumed) == 3
    assert len({json.loads(row)["trajectory_id"] for row in resumed}) == 3
    unchanged = (tmp_path / "events.jsonl").read_text()
    evaluate_items(**kwargs)
    assert (tmp_path / "events.jsonl").read_text() == unchanged
    with (tmp_path / "validation-loss.csv").open() as stream:
        result = list(csv.DictReader(stream))
    assert len(result) == 1
    assert result[0]["trajectories"] == "3"
    assert float(result[0]["ce_macro_delta_vs_base"]) == 0
    with (tmp_path / "checkpoint-coverage.csv").open() as stream:
        coverage = list(csv.DictReader(stream))
    assert [r["status"] for r in coverage] == [
        "shared_base_measured",
        "weights_unavailable",
        "weights_unavailable",
    ]


def test_export_never_leaves_stale_metrics_for_changed_evaluation(tmp_path):
    from workflow_distillation.validation_checkpoints import export

    (tmp_path / "validation-loss.csv").write_text("old result\n")
    (tmp_path / "trajectory-loss.csv").write_text("old result\n")
    export(
        tmp_path,
        [dict(run="base", step=0, source="new", key="new")],
        [dict(name="old", known_last_step=2, available_steps=[2])],
        {},
        ["one"],
        2,
    )
    assert not (tmp_path / "validation-loss.csv").exists()
    assert not (tmp_path / "trajectory-loss.csv").exists()
    assert "available_not_selected" in (tmp_path / "checkpoint-coverage.csv").read_text()


def test_changed_fingerprint_recomputes_instead_of_reusing_old_losses(tmp_path):
    from contextlib import nullcontext

    from test_fp16_training import records, tiny_state

    from workflow_distillation.validation_checkpoints import evaluate_items

    state = tiny_state()
    state.model.disable_adapter = nullcontext
    data = records()
    dataset = type("Dataset", (), dict(records=data, record=lambda self, i: data[i]))()
    kwargs = dict(
        state=state,
        dataset=dataset,
        validation=[0],
        items=[dict(run="base", step=0, source="tiny")],
        specs=[],
        output=tmp_path,
        steps_per_epoch=1,
        device=torch.device("cpu"),
    )
    evaluate_items(**kwargs, identity={"source_hash": "old"})
    first = (tmp_path / "events.jsonl").read_text()
    evaluate_items(**kwargs, identity={"source_hash": "new"})
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["key"] != json.loads(lines[1])["key"]
    assert (tmp_path / "events.jsonl").read_text().startswith(first)
    # The exported comparison must contain only the current evaluation.
    assert json.loads(lines[0])["key"] not in (tmp_path / "validation-loss.csv").read_text()
