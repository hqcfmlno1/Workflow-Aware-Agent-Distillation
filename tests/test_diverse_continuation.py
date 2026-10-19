"""Protect the trained prefix and select genuinely new tasks across repositories."""

import copy
import json

import pytest

from workflow_distillation import diverse_continuation as diversity
from workflow_distillation.epoch_data import epoch_batches, sha256_file


def source_files(tmp_path):
    # Place these records in the source shuffle's order, independently of selection.
    ordered = [
        ("a/repo#1", 29000), ("a/repo#1", 100),  # Already consumed, including repeats.
        ("a/repo#2", 100), ("b/repo#1", 100),
        ("b/repo#1", 100), ("c/repo#1", 100),
        ("b/repo#2", 100), ("c/repo#2", 27999),
        ("d/repo#1", 28000), ("a/repo#3", 100),
    ]
    order = [i for batch in epoch_batches(10, 42) for i in batch]
    rows = [None] * len(order)
    for position, (task, length) in enumerate(ordered):
        rows[order[position]] = {
            "trajectory_id": f"trajectory-{position}", "task_key": task,
            "length": length, "full_token_count": length,
        }
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": 10, "tokens": sum(r["length"] for r in rows),
                                 "data": rows}))
    indices = order[:2] + [i for i in order[2:] if rows[i]["length"] < 28000]
    source = {
        "format_version": 1, "policy": "preserve_consumed_prefix_filter_suffix",
        "max_tokens": 28000, "seed": 42, "source_records": 10,
        "source_tokens": sum(r["length"] for r in rows),
        "source_index_sha256": sha256_file(index),
        "parent": {"progress": {"global_step": 1, "next_batch": 1, "samples_consumed": 2}},
        "retained_prefix_count": 2, "total_records": 9,
        "total_tokens": sum(rows[i]["length"] for i in indices),
        "ordered_indices": indices,
    }
    path = tmp_path / "source-plan.json"
    path.write_text(json.dumps(source))
    return index, path


def build(tmp_path, **kwargs):
    index, source = source_files(tmp_path)
    return index, source, diversity.build_plan(index, source, resume_step=1,
                                               target_step=3, **kwargs)


def test_selects_unseen_tasks_from_underrepresented_repos_and_preserves_prefix(tmp_path):
    index, source, plan = build(tmp_path)
    rows = json.loads(index.read_text())["data"]
    chosen = [rows[i] for i in plan["ordered_indices"]]
    # a already has two samples; new repositories b/c each receive a turn first.
    assert [r["trajectory_id"] for r in chosen] == [
        "trajectory-0", "trajectory-1", "trajectory-3", "trajectory-5",
        "trajectory-6", "trajectory-7",
    ]
    assert plan["ordered_indices"][:2] == json.loads(source.read_text())["ordered_indices"][:2]
    assert plan["coverage"]["selected"]["unique_tasks"] == 5
    assert plan["coverage"]["selected"]["unique_repos"] == 3
    assert plan["coverage"]["selected"]["repeated_task_trajectories"] == 1
    assert plan["progress"] == {"global_step": 1, "next_batch": 1, "samples_consumed": 2}
    assert plan["target_optimizer_steps"] == 3
    assert all(r["length"] < 28000 for r in chosen[2:])
    assert chosen[0]["length"] == 29000
    assert plan["checkpoint_binding_required"] is True
    assert diversity.build_plan(index, source, resume_step=1, target_step=3) == plan


def test_refuses_to_fill_target_by_repeating_tasks_when_new_tasks_run_out(tmp_path):
    index, source = source_files(tmp_path)
    with pytest.raises(ValueError, match="unseen tasks"):
        diversity.build_plan(index, source, resume_step=1, target_step=5)


@pytest.mark.parametrize("resume,target", [(0, 3), (1, 1), (True, 3), (1, 20)])
def test_rejects_invalid_or_out_of_range_optimizer_boundaries(tmp_path, resume, target):
    index, source = source_files(tmp_path)
    with pytest.raises(ValueError):
        diversity.build_plan(index, source, resume_step=resume, target_step=target)


@pytest.mark.parametrize("change", ["prefix", "source", "duplicate_id", "truncated"])
def test_refuses_a_changed_source_schedule_or_trajectory_payload(tmp_path, change):
    index, source = source_files(tmp_path)
    original = json.loads(source.read_text())
    if change == "prefix":
        original["ordered_indices"][:2] = original["ordered_indices"][:2][::-1]
        source.write_text(json.dumps(original))
    else:
        data = json.loads(index.read_text())
        if change == "duplicate_id":
            data["data"][1]["trajectory_id"] = data["data"][0]["trajectory_id"]
        elif change == "truncated":
            data["data"][0]["full_token_count"] += 1
        index.write_text(json.dumps(data) + "\n")
        if change != "source":
            original["source_index_sha256"] = sha256_file(index)
            source.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        diversity.build_plan(index, source, resume_step=1, target_step=3)


@pytest.mark.parametrize("change", ["order", "coverage", "progress", "source_plan"])
def test_validation_detects_tampered_selection_and_provenance(tmp_path, change):
    index, source, plan = build(tmp_path)
    changed = copy.deepcopy(plan)
    if change == "order":
        changed["ordered_indices"][-2:] = changed["ordered_indices"][-2:][::-1]
    elif change == "coverage":
        changed["coverage"]["selected"]["unique_tasks"] = 6
    elif change == "progress":
        changed["progress"]["samples_consumed"] = 0
    else:
        source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError):
        diversity.validate_plan(index, source, changed)


def test_cli_writes_reproducible_plan_coverage_and_ordered_trajectory_manifest(tmp_path):
    import subprocess
    import sys

    index, source = source_files(tmp_path)
    output = tmp_path / "selection"
    cmd = [sys.executable, "-m", "scripts.data.prepare_diverse_continuation", "--index", str(index),
           "--source-plan", str(source), "--resume-step", "1", "--target-step", "3",
           "--output", str(output)]
    subprocess.run(cmd, check=True, capture_output=True)
    plan = json.loads((output / "diversity-plan.json").read_text())
    assert plan["total_records"] == 6
    rows = [json.loads(line) for line in (output / "trajectories.jsonl").read_text().splitlines()]
    assert [r["global_batch"] for r in rows] == [1, 1, 2, 2, 3, 3]
    assert [r["already_trained"] for r in rows] == [True, True, False, False, False, False]
    assert rows[2]["task_key"] == "b/repo#1"
    receipt = json.loads((output / "receipt.json").read_text())
    for name, digest in receipt["sha256"].items():
        assert sha256_file(output / name) == digest
    first = {p.name: p.read_bytes() for p in output.iterdir()}
    subprocess.run(cmd, check=True, capture_output=True)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == first
    mismatch = subprocess.run([*cmd[:-2], "--output", str(output), "--target-step", "4"],
                              capture_output=True)
    assert mismatch.returncode != 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == first
