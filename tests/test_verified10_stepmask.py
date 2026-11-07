import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    path = ROOT / "cloud/vast/verified10_stepmask" / (name + ".py")
    spec = importlib.util.spec_from_file_location("stepmask_" + name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_exact_original_ten_and_new_masked_pool_are_disjoint():
    prep = module("prepare")
    rows, protocol = prep.frozen_inputs()
    original = json.loads((ROOT / "configs/evaluation/verified10_bf16_triplet.json").read_text())
    assert [r["instance_id"] for r in rows] == original["instance_ids"]
    assert protocol["rows_sha256"] == original["rows_sha256"]
    assert protocol["training_overlap"] == []
    assert protocol["training_index_sha256"] == prep.digest(prep.TRAIN_INDEX)
    assert protocol["variant_order"] == ["span-stepmask"]
    assert protocol["sources"]["span-stepmask"]["repo"].endswith("span-stepmask-run1")
    assert protocol["sources"]["span-stepmask"]["revision"] is None
    assert protocol["checkpoint"] == "checkpoint-00000590"
    assert protocol["tool_call_limit"] == 200
    assert protocol["turn_timeout_seconds"] == 1200
    assert protocol["task_budget_seconds"] == 1800
    assert protocol["campaign_budget_seconds"] is None
    cfg = protocol["config"]
    assert (cfg["temperature"], cfg["top_p"], cfg["top_k"]) == (0.6, 0.95, 20)
    assert cfg["request_timeout_seconds"] == 1200
    assert cfg["max_context_tokens"] is cfg["max_new_tokens"] is None
    assert cfg["base_dtype"] == "bfloat16"


def test_missing_historical_output_fetches_pinned_verified_rows(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    prep = module("prepare")
    original_rows, original_protocol = prep.frozen_inputs()
    calls = []

    def load_dataset(name, *, split, revision):
        calls.append((name, split, revision))
        return list(reversed(original_rows))

    monkeypatch.setattr(prep, "REFERENCE", tmp_path)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    rows, protocol = prep.frozen_inputs()
    assert rows == original_rows
    assert protocol == original_protocol
    assert calls == [
        ("princeton-nlp/SWE-bench_Verified", "test", original_protocol["dataset_revision"])
    ]


def test_checkpoint_manifest_must_be_new_masked_training():
    prep = module("prepare")
    _, protocol = prep.frozen_inputs()
    manifest = {
        "complete": True,
        "progress": {"global_step": 590},
        "identity": {
            "records_sha256": protocol["training_index_sha256"],
            "model": protocol["base_model"],
            "revision": protocol["base_revision"],
            "settings": {
                "config": {
                    "base_dtype": "bfloat16",
                    "loss_normalization": "offline_span_anchor",
                    "anchor_beta": 0.03,
                },
                "span_sidecar_sha256": protocol["span_sidecar_sha256"],
            },
        },
    }
    prep.validate_training(manifest, protocol)
    manifest["identity"]["records_sha256"] = "old-unmasked-index"
    with pytest.raises(ValueError, match="masked training"):
        prep.validate_training(manifest, protocol)


def test_unavailable_final_checkpoint_does_not_mark_ready(tmp_path, monkeypatch):
    prep = module("prepare")
    _, protocol = prep.frozen_inputs()
    prep.write_json(tmp_path / "protocol.json", protocol)

    def missing(*args):
        raise FileNotFoundError("checkpoint 590 is not published")

    monkeypatch.setattr(prep, "fetch_checkpoint", missing)
    with pytest.raises(FileNotFoundError):
        prep.finalize(tmp_path)
    assert not (tmp_path / "prepared.json").exists()
    assert (
        json.loads((tmp_path / "protocol.json").read_text())["sources"]["span-stepmask"]["revision"]
        is None
    )


def test_host_rejects_old_adapter_and_unpinned_source():
    prep, host = module("prepare"), module("host")
    _, protocol = prep.frozen_inputs()
    export = {"source": protocol["sources"]["span-stepmask"], "checkpoint": protocol["checkpoint"]}
    with pytest.raises(ValueError, match="pin"):
        host.validate_export(export, protocol)
    protocol["sources"]["span-stepmask"]["revision"] = "a" * 40
    host.validate_export(export, protocol)
    export["source"] = {**export["source"], "repo": "old-span-anchor-run1"}
    with pytest.raises(ValueError, match="source"):
        host.validate_export(export, protocol)


def test_offline_stage_archive_contains_no_tasks_or_old_rollouts(tmp_path):
    import tarfile

    prep = module("prepare")
    prep.stage(tmp_path)
    receipt = json.loads((tmp_path / "prepared.json").read_text())
    assert receipt["adapters_ready"] is False
    assert receipt["paid_inference_started"] is False
    assert not (tmp_path / "results").exists()
    with tarfile.open(receipt["archive"]) as archive:
        names = archive.getnames()
        assert all("grading-dataset" not in n and "tasks-agent-visible" not in n for n in names)
        assert all("results/" not in n and "resume.json" not in n for n in names)
        assert all(not n.endswith((".key", "bench_api_key")) for n in names)
        assert "verified10-stepmask/runtime/benchmarks/swe_bench_verified/policy.py" in names
        exporter = (
            archive.extractfile(
                "verified10-stepmask/runtime/benchmarks/swe_bench_verified/export_adapter.py"
            )
            .read()
            .decode()
        )
        assert '"base_dtype": "bfloat16"' in exporter
    agent_rows = json.loads((tmp_path / "tasks-agent-visible.json").read_text())
    assert len(agent_rows) == 10
    assert all("patch" not in r and "test_patch" not in r for r in agent_rows)
    # Each input still contains the user-visible problem, never grading solutions.
    assert all(r["problem_statement"] for r in agent_rows)


def test_report_only_works_before_adapter_exists_and_includes_turns(tmp_path):
    import subprocess
    import sys

    prep = module("prepare")
    prep.stage(tmp_path)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "run.py"), "--report-only"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "results/comparison.json").read_text())["summary"][
        "span-stepmask"
    ]
    assert (summary["total"], summary["done"], summary["resolved"]) == (10, 0, 0)
    report = (tmp_path / "results/comparison.md").read_text()
    assert "| Task | span-stepmask |" in report
    assert "0 turns" in report
    assert "0/10 PASS" in report
