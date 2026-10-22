import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help_is_cpu_safe_and_never_imports_historical_launchers():
    code = """
import sys
from workflow_distillation.training.cli import parser
parser().parse_args(['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--gpu-profile" in result.stdout
    assert "--migration-parent-manifest-sha256" in result.stdout
    # Import the entire CLI without torch, PEFT, BNB or cloud recovery side effects.
    code = """
import sys
from workflow_distillation.training import cli, engine, model
assert 'torch' not in sys.modules
assert 'bitsandbytes' not in sys.modules
assert 'filtered_continuation' not in sys.modules
assert not any('vast_original' in name for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_one_entrypoint_preserves_all_worker_arguments_and_explicit_stop(tmp_path):
    from workflow_distillation.training.cli import parser, worker_command

    arguments = [
        "--index",
        "index.json",
        "--source-plan",
        "source.json",
        "--cohort",
        "cohort.json",
        "--pool-plan",
        "pool.json",
        "--resume-from",
        "checkpoint-00001028",
        "--gpu-profile",
        "configs/training/gpus/rtx4080_32gb/profile.json",
        "--output",
        str(tmp_path / "run"),
        "--stop-at-step",
        "1030",
        "--migration-parent-manifest",
        "parent.json",
        "--migration-parent-manifest-sha256",
        "f" * 64,
    ]
    args = parser().parse_args(arguments)
    command = worker_command(args, arguments)
    assert command[-len(arguments) :] == arguments
    assert "--max_restarts=0" in command
    assert command[command.index("--tee") + 1] == "3"
    with pytest.raises(ValueError):
        from workflow_distillation.training.cli import validate_args

        args.stop_at_step = -1
        validate_args(args)


def test_preflight_verifies_real_pool_and_pinned_parent_without_model_load(tmp_path, monkeypatch):
    import numpy as np
    from test_diverse_continuation import source_files

    from workflow_distillation.diverse_continuation import build_plan, fingerprint
    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.training.cli import parser, prepare
    from workflow_distillation.training.compatibility import source_hashes
    from workflow_distillation.training.config import MODEL, REVISION, Experiment, load_profile
    from workflow_distillation.training.identity import package_versions
    from workflow_distillation.vast_pool_continuation import build_pool_plan

    index, source = source_files(tmp_path)
    meta = json.loads(index.read_text())
    offset = 0
    for row in meta["data"]:
        row.update(offset=offset, denominator=row["length"] - 1)
        offset += row["length"]
    np.zeros(offset, dtype="<u4").tofile(tmp_path / "tokens.bin")
    np.ones(offset, dtype="u1").tofile(tmp_path / "groups.bin")
    meta.update(
        format_version=1,
        weights=[0, 1, 0.75, 1.5, 2],
        file_sha256={n: sha256_file(tmp_path / n) for n in ("tokens.bin", "groups.bin")},
    )
    index.write_text(json.dumps(meta))
    old_source = json.loads(source.read_text())
    old_source["source_index_sha256"] = sha256_file(index)
    source.write_text(json.dumps(old_source))
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(build_plan(index, source, resume_step=1, target_step=3)))
    plan = build_pool_plan(index, source, cohort)
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(plan))
    profile_path = Path("configs/training/gpus/rtx4080_32gb/profile.json")
    profile = load_profile(profile_path)
    hashes = source_hashes()
    parent_identity = {
        "model": MODEL,
        "revision": REVISION,
        "records_sha256": sha256_file(index),
        "packages": package_versions(),
        "script_sha256": hashes["scripts/training/gpus/rtx4080_32gb/train.py"],
        "math_sources": {
            n: hashes["src/workflow_distillation/" + n]
            for n in (
                "optimized_qlora.py",
                "weighted_sft_loss.py",
                "t4_attention.py",
                "ddp_probe.py",
            )
        },
        "settings": {
            "seed": 42,
            "config": Experiment().effective_config(profile),
            "pool_continuation": {"plan_sha256": fingerprint(plan)},
            "microbatch_transition": {
                "parent_progress": {"global_step": 1, "next_batch": 1, "samples_consumed": 2}
            },
        },
    }
    # A genuinely committed payload is validated by the regular manifest reader;
    # here a two-rank metadata fixture supplies small correctly hashed files.
    parent = tmp_path / "checkpoint-00000001"
    parent.mkdir()
    files = {}
    for name in ("adapter.pt", "rank-00000.pt", "rank-00001.pt"):
        (parent / name).write_bytes(b"payload")
        files[name] = sha256_file(parent / name)
    manifest = {
        "format_version": 1,
        "complete": True,
        "world_size": 2,
        "identity": parent_identity,
        "progress": {"global_step": 1, "next_batch": 1, "samples_consumed": 2},
        "files": files,
    }
    mpath = parent / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    args = parser().parse_args(
        [
            "--index",
            str(index),
            "--source-plan",
            str(source),
            "--cohort",
            str(cohort),
            "--pool-plan",
            str(pool),
            "--resume-from",
            str(parent),
            "--gpu-profile",
            str(profile_path),
            "--output",
            str(tmp_path / "run"),
            "--preflight-only",
            "--migration-parent-manifest",
            str(mpath),
            "--migration-parent-manifest-sha256",
            sha256_file(mpath),
        ]
    )
    prepared = prepare(args)
    assert prepared.progress == manifest["progress"]
    assert prepared.batches == [[0, 1], [2, 3, 4, 5], [6, 7, 8]]
    assert prepared.identity["settings"]["config"]["global_batch_size"] == 4
    assert prepared.identity["settings"]["training_refactor"]["parent"][
        "manifest_sha256"
    ] == sha256_file(mpath)
    # Pinning a different valid checkpoint cannot silently reset/replay progress.
    args.seed = 12
    with pytest.raises(ValueError):
        prepare(args)
    args.seed = 42

    # Child resume must reject implementation/profile drift even while its
    # mathematical parent remains valid; do not silently assign a new identity.
    import copy
    import shutil

    from workflow_distillation.training import identity

    child = tmp_path / "child" / parent.name
    shutil.copytree(parent, child)
    child_manifest = {**manifest, "identity": prepared.identity}
    (child / "manifest.json").write_text(
        json.dumps(child_manifest, indent=2, sort_keys=True) + "\n"
    )
    args.resume_from = child
    args.migration_parent_manifest = args.migration_parent_manifest_sha256 = None
    assert prepare(args).identity == prepared.identity
    with monkeypatch.context() as patch:
        patch.setattr(identity, "implementation_hashes", lambda: {"modified.py": "other"})
        with pytest.raises(ValueError, match="exact child"):
            prepare(args)
    modified = copy.deepcopy(profile.to_dict())
    modified["gpu_budget_bytes"] = 2 * 2**30
    args.gpu_profile = tmp_path / "different-profile.json"
    args.gpu_profile.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="exact child"):
        prepare(args)


def test_stop_at_step_saves_only_completed_optimizer_boundaries(tmp_path):
    from workflow_distillation.training.engine import execute_loop

    saved, events = [], []
    progress = {"global_step": 2, "next_batch": 2, "samples_consumed": 4}
    result = execute_loop(
        [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9]],
        progress=progress,
        step=lambda batch, number: {"loss": 0.7},
        save=lambda value: saved.append(dict(value)),
        should_stop=lambda: False,
        emit=events.append,
        save_every=10,
        stop_at_step=3,
    )
    assert result["status"] == "stopped_checkpointed"
    assert result["progress"] == {"global_step": 3, "next_batch": 3, "samples_consumed": 8}
    assert saved == [result["progress"]]
    assert len(events) == 1


def test_output_reuse_refused_so_old_rank_failures_cannot_poison_monitor(tmp_path):
    from workflow_distillation.training.cli import ensure_fresh_output

    (tmp_path / "rank0.jsonl").write_text('{"event":"failed"}\n')
    with pytest.raises(ValueError, match="fresh output"):
        ensure_fresh_output(tmp_path)
