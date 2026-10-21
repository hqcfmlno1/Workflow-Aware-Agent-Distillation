"""A saved batch-four child remains resumable after ancestor payload retention."""

import copy

import pytest

from workflow_distillation.vast_microbatch import transition_identity


def fixture(tmp_path):
    script = tmp_path / "batch4.py"
    script.write_text("# original launcher\n")
    baseline = {
        "settings": {"config": {
            "per_gpu_batch_size": 1, "global_batch_size": 2,
            "gradient_accumulation_steps": 1, "rank": 32,
        }},
        "math_sources": {"loss.py": "checked-source"},
        "packages": {"torch": "pinned"},
    }
    parent = {"complete": True, "world_size": 2, "identity": baseline,
              "progress": {"global_step": 2, "next_batch": 2, "samples_consumed": 4}}
    child = {"complete": True, "world_size": 2,
             "identity": transition_identity(parent, baseline, "parent-sha", script),
             "progress": {"global_step": 3, "next_batch": 3, "samples_consumed": 8}}
    return script, baseline, child


def test_existing_child_validates_without_any_ancestor_state_files(tmp_path):
    from workflow_distillation.vast_existing_child import existing_child_binding

    script, baseline, child = fixture(tmp_path)
    binding, identity = existing_child_binding(child, baseline, "parent-sha", script)
    assert identity == child["identity"]
    assert binding["parent"]["progress"] == {
        "global_step": 2, "next_batch": 2, "samples_consumed": 4,
    }
    assert binding["parent"]["identity"]["settings"]["config"]["global_batch_size"] == 2


@pytest.mark.parametrize("change", ["config", "source", "package", "script", "parent", "cursor"])
def test_existing_child_rejects_changed_identity_or_invalid_progress(tmp_path, change):
    from workflow_distillation.vast_existing_child import existing_child_binding

    script, baseline, child = fixture(tmp_path)
    child = copy.deepcopy(child)
    if change == "config":
        child["identity"]["settings"]["config"]["rank"] = 16
    elif change == "source":
        child["identity"]["math_sources"]["loss.py"] = "different"
    elif change == "package":
        baseline["packages"]["torch"] = "different"
    elif change == "script":
        script.write_text("# changed launcher\n")
    elif change == "parent":
        child["identity"]["settings"]["microbatch_transition"]["parent_manifest_sha256"] = "other"
    else:
        child["progress"]["samples_consumed"] = 9
    with pytest.raises(ValueError):
        existing_child_binding(child, baseline, "parent-sha", script)


def test_worker_parser_receives_the_launcher_manifest_pin_through_environment(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/training/resume_qlora_vast_batch4.py"
    spec = importlib.util.spec_from_file_location("resume_parser_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("FIRST_RUN_RESUME_MANIFEST_SHA256", "pinned-child")
    flags = []
    for name in ("index", "source-plan", "cohort", "parent-checkpoint",
                 "parent-manifest-sha256", "output", "batch-parent",
                 "batch-parent-manifest-sha256"):
        flags.extend(["--" + name, "test-value"])
    args = module.parser().parse_args(flags)
    assert args.resume_manifest_sha256 == "pinned-child"
