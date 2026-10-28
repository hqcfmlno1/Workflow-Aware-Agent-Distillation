import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_checkpoints import fake_hub  # noqa: F401


def config():
    from workflow_distillation.lora_fp16.config import RunConfig

    return RunConfig(
        base_dtype="bfloat16",
        loss_normalization="offline_span_anchor",
        group_weights=(0, 1, 1, 1, 1),
    )


def test_preflight_rejects_missing_partial_and_wrong_reference_before_gpu(tmp_path):
    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.lora_fp16.cli import prepare
    from workflow_distillation.lora_fp16.metacurate import write_corpus
    from workflow_distillation.lora_fp16.span_cache import ScoreStore
    from workflow_distillation.lora_fp16.span_scoring import reference_identity

    row = dict(
        trajectory_id="a",
        task_key="t",
        input_ids=[1, 2, 3, 4],
        groups=[0, 1, 3, 4],
        resolved=True,
        full_token_count=4,
    )
    path = tmp_path / "corpus"
    write_corpus([row], path, {})
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(dict(index_sha256=sha256_file(path / "index.json"), selected_ids=["a"]))
    )
    with pytest.raises(ValueError, match="sidecar"):
        prepare(config(), path / "index.json", selection)
    identity = reference_identity(config(), "provenance")
    store = ScoreStore(tmp_path / "sidecar", sha256_file(path / "index.json"), identity)
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(config(), path / "index.json", selection, tmp_path / "sidecar")
    store.write_record(0, row["input_ids"], row["groups"], [-1, -2, -3], [0.5, 1, 1], [1, 1e-6, 1])
    store.finalize([row])
    dataset, train, valid, batches, run_identity = prepare(
        config(), path / "index.json", selection, tmp_path / "sidecar"
    )
    assert train == [0] and valid == []
    assert dataset.record(0)["weights"].tolist() == [0.5, 1, 1]
    assert run_identity["settings"]["span_sidecar_sha256"] == sha256_file(
        tmp_path / "sidecar/index.json"
    )
    with pytest.raises(ValueError, match="reference"):
        prepare(
            replace(config(), span_floor=0.2), path / "index.json", selection, tmp_path / "sidecar"
        )


def test_epoch_only_publication_retention_and_failed_upload_retry(tmp_path, fake_hub):  # noqa: F811
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.checkpoints import CheckpointManager
    from workflow_distillation.lora_fp16.span_run import EpochPublisher
    from workflow_distillation.training_checkpoints import HuggingFacePublisher

    state = tiny_state()
    (tmp_path / "events.jsonl").write_text('{"event":"step","loss":1.0}\n')
    (tmp_path / "train.log").write_text("log line\n")
    publisher = EpochPublisher(HuggingFacePublisher("owner/span", prefix="span"), [2, 4], tmp_path)
    manager = CheckpointManager(
        tmp_path / "checkpoints", {}, keep_last=1, pinned_steps=[2, 4], publisher=publisher
    )
    for step in [1, 2, 3, 4]:
        manager.save(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=None,
            scaler=state.scaler,
            progress={"global_step": step},
        )
    files = fake_hub.revisions[fake_hub.head[0]]
    assert {n.split("/")[1] for n in files if "/checkpoint-" in n} == {
        "checkpoint-00000002",
        "checkpoint-00000004",
    }
    assert "span/checkpoint-00000004/events.jsonl" in files
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("checkpoint-*")) == [
        "checkpoint-00000002",
        "checkpoint-00000004",
    ]


def test_single_longest_pilot_does_only_one_trajectory(monkeypatch, tmp_path):
    from workflow_distillation.lora_fp16 import cli

    calls = []

    class Dataset:
        records = [dict(length=10), dict(length=20), dict(length=30)]

        def record(self, i):
            return {"id": i, "input_ids": [0] * self.records[i]["length"]}

    def update(state, records, config, device, emit):
        calls.append(records)
        return dict(tokens=30, seconds=1, peak_reserved=10, cuda_capacity=100, cuda_free=90)

    monkeypatch.setattr(cli, "train_update", update)
    cli.pilot(
        None,
        Dataset(),
        [0, 1, 2],
        replace(config(), minimum_headroom_bytes=1),
        None,
        {},
        {},
        lambda e: None,
        tmp_path / "pilot.json",
    )
    assert len(calls) == 1 and [r["id"] for r in calls[0]] == [2]


def test_metrics_recovery_uses_checkpoint_cursor_without_duplicate_steps(tmp_path):
    from workflow_distillation.lora_fp16.span_run import recover_events

    path = tmp_path / "events.jsonl"
    lines = [dict(event="step", global_step=i, loss=1 / i) for i in range(1, 5)]
    path.write_text("".join(json.dumps(v) + "\n" for v in lines))
    recover_events(path, 2)
    remaining = [json.loads(v) for v in path.read_text().splitlines()]
    assert [r["global_step"] for r in remaining] == [1, 2]
    assert len(list(tmp_path.glob("events.before-resume-*.jsonl"))) == 1
    path.write_text(path.read_text() + '{"event":"step",')
    recover_events(path, 2)
    assert len(path.read_text().splitlines()) == 2


def test_offline_cli_resumes_partial_turns_and_reuses_reference_without_gpu(tmp_path, monkeypatch):
    import gzip

    from test_span_lora_scoring import Characters, reference
    from transformers import AutoTokenizer

    from scripts.training import score_span_lora as cli
    from workflow_distillation.epoch_data import sha256_file
    from workflow_distillation.lora_fp16.metacurate import write_corpus

    text = "hAlpha.Beta.TTGamma.Delta.UU"
    ids = Characters().encode(text)
    groups = [0] + [1] * 11 + [3] * 2 + [1] * 12 + [3, 4]
    assert len(ids) == len(groups)
    row = dict(
        trajectory_id="a",
        task_key="t",
        input_ids=ids,
        groups=groups,
        resolved=True,
        full_token_count=len(ids),
    )
    corpus = tmp_path / "corpus"
    write_corpus([row], corpus, {})
    receipt = tmp_path / "selection.json"
    receipt.write_text(
        json.dumps(dict(index_sha256=sha256_file(corpus / "index.json"), selected_ids=["a"]))
    )
    source = dict(
        text=text,
        length=len(ids),
        offsets=[[i, i + 1] for i in range(len(ids))],
        anchor_weights=[0 if g == 0 else 1 if g == 1 else 1e-6 for g in groups],
        turns=[
            dict(
                message_index=2,
                status="eligible",
                spans=[[1, 7], [7, 12]],
                action_chars=[12, 14],
                action_ids=ids[12:14],
            ),
            dict(
                message_index=4,
                status="eligible",
                spans=[[14, 20], [20, 26]],
                action_chars=[26, 28],
                action_ids=ids[26:28],
            ),
        ],
    )
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    with gzip.open(provenance / "a.json.gz", "wt") as stream:
        json.dump(source, stream)
    manifest = dict(
        corpus_sha256=sha256_file(corpus / "index.json"),
        eligible_turns=2,
        scoring_variants=6,
        data=[
            dict(trajectory_id="a", file="a.json.gz", sha256=sha256_file(provenance / "a.json.gz"))
        ],
    )
    (provenance / "index.json").write_text(json.dumps(manifest))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config().to_dict()))
    state = reference()
    monkeypatch.setattr(cli, "build_reference", lambda config: state)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Characters())
    output = tmp_path / "scores"
    args = [
        "--index",
        str(corpus / "index.json"),
        "--selection",
        str(receipt),
        "--provenance",
        str(provenance / "index.json"),
        "--config",
        str(cfg),
        "--output",
        str(output),
    ]
    cli.main([*args, "--limit-turns", "1"])
    assert not (output / "index.json").exists()
    q = (output / "parts/000000.ref.npz").read_bytes()
    first = (output / "parts/000000.turn-2.json").stat().st_mtime_ns
    cli.main(args)
    assert (output / "index.json").exists()
    assert (output / "parts/000000.ref.npz").read_bytes() == q
    assert (output / "parts/000000.turn-2.json").stat().st_mtime_ns == first
    assert json.loads((output / "summary.json").read_text())["complete"] is True


def test_failed_epoch_upload_keeps_endpoint_and_retry_precedes_training(tmp_path):
    from test_fp16_training import tiny_state

    from workflow_distillation.lora_fp16.span_run import EpochPublisher, SpanCheckpointManager

    class Remote:
        fail = True
        calls = []

        def __call__(self, path):
            self.calls.append(Path(path).name)
            if self.fail:
                raise RuntimeError("network unavailable")

    remote = Remote()
    publisher = EpochPublisher(remote, [2, 4], tmp_path)
    state = tiny_state()
    manager = SpanCheckpointManager(
        tmp_path / "checkpoints", {}, keep_last=1, pinned_steps=[2, 4], publisher=publisher
    )

    def save(step):
        return manager.save(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=None,
            scaler=state.scaler,
            progress={"global_step": step},
        )

    save(1)
    assert remote.calls == []
    with pytest.raises(RuntimeError, match="network unavailable"):
        save(2)
    assert (tmp_path / "checkpoints/checkpoint-00000001").is_dir()
    assert (tmp_path / "checkpoints/checkpoint-00000002/manifest.json").is_file()
    remote.fail = False
    progress = manager.load(
        "latest", model=state.model, optimizer=state.optimizer, scheduler=None, scaler=state.scaler
    )
    assert progress["global_step"] == 2
    manager.republish(model=state.model)
    assert remote.calls == ["checkpoint-00000002", "checkpoint-00000002"]


@pytest.mark.parametrize("marker", ["events.jsonl", "STOP"])
def test_launcher_refuses_used_output_or_stop_before_gpu_calls(tmp_path, marker):
    import os
    import subprocess

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    called = tmp_path / "called"
    fake = venv / "bin/python"
    fake.write_text('#!/bin/sh\ntouch "$SPAN_TEST_CALLED"\n')
    fake.chmod(0o755)
    output = tmp_path / "run"
    output.mkdir()
    (output / marker).write_text("existing")
    result = subprocess.run(
        ["bash", "scripts/training/run_metacurate_span_anchor.sh", "--stage", "train"],
        env={
            **os.environ,
            "SFT5090_VENV": str(venv),
            "SFT5090_OUTPUT": str(output),
            "SPAN_TEST_CALLED": str(called),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not called.exists()


def launch_with_fake_python(tmp_path, args, **environment):
    import os
    import subprocess

    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    calls = tmp_path / "calls.jsonl"
    fake = venv / "bin/python"
    fake.write_text(
        "#!/usr/bin/python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['SPAN_TEST_CALLS'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if 'scripts.training.score_span_lora' in sys.argv:\n"
        "    if os.environ.get('SPAN_TEST_SCORE_EXIT'):\n"
        "        sys.exit(int(os.environ['SPAN_TEST_SCORE_EXIT']))\n"
        "    if os.environ.get('SPAN_TEST_FINISH') == '1':\n"
        "        p=Path(sys.argv[sys.argv.index('--output')+1])\n"
        "        (p/'index.json').write_text('{}')\n"
    )
    fake.chmod(0o755)
    result = subprocess.run(
        ["bash", "scripts/training/run_metacurate_span_anchor.sh", *args],
        env={
            **os.environ,
            "SFT5090_VENV": str(venv),
            "SFT5090_OUTPUT": str(tmp_path / "run"),
            "SFT5090_SCORES": str(tmp_path / "scores_cached"),
            "SPAN_TEST_CALLS": str(calls),
            **environment,
        },
        text=True,
        capture_output=True,
    )
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    return result, invocations


def test_launcher_defaults_to_cached_scoring_and_keeps_explicit_full_override(tmp_path):
    result, calls = launch_with_fake_python(tmp_path, ["--stage", "estimate"])
    assert result.returncode == 0
    assert calls[0][calls[0].index("--config") + 1].endswith(
        "metacurate_span_anchor_bf16_cached.json"
    )
    other = tmp_path / "full"
    other.mkdir()
    result, calls = launch_with_fake_python(
        other, ["--stage", "estimate"], SFT5090_CONFIG="/explicit/full.json"
    )
    assert result.returncode == 0
    assert calls[0][calls[0].index("--config") + 1] == "/explicit/full.json"


@pytest.mark.parametrize("score_exit", ["0", "7"])
def test_pipeline_never_starts_training_after_partial_or_failed_scoring(tmp_path, score_exit):
    result, calls = launch_with_fake_python(
        tmp_path, ["--stage", "all", "--foreground"], SPAN_TEST_SCORE_EXIT=score_exit
    )
    assert result.returncode != 0
    assert len(calls) == 1
    assert "scripts.training.score_span_lora" in calls[0]


def test_pipeline_transitions_to_supervised_training_after_complete_scoring(tmp_path):
    (tmp_path / "run").mkdir()
    result, calls = launch_with_fake_python(
        tmp_path, ["--stage", "all", "--foreground"], SPAN_TEST_FINISH="1"
    )
    assert result.returncode == 0, result.stderr
    assert "scripts.training.score_span_lora" in calls[0]
    assert calls[-1][:5] == ["-u", "-m", "workflow_distillation.lora_fp16", "--mode", "train"]
    assert "workflow_distillation.lora_fp16.launch" not in str(calls)
    assert calls[-1][calls[-1].index("--span-sidecar") + 1] == str(tmp_path / "scores_cached")


def test_pipeline_reuses_completed_scores_and_resumes_local_checkpoint(tmp_path):
    (tmp_path / "run/checkpoints").mkdir(parents=True)
    (tmp_path / "run/checkpoints/latest.json").write_text("{}")
    (tmp_path / "scores_cached").mkdir()
    (tmp_path / "scores_cached/index.json").write_text("{}")
    result, calls = launch_with_fake_python(tmp_path, ["--stage", "all", "--foreground"])
    assert result.returncode == 0, result.stderr
    assert "scripts.training.score_span_lora" not in str(calls)
    assert calls[-1][calls[-1].index("--resume") + 1] == "latest"


def test_interrupted_log_copy_never_becomes_a_completed_upload(tmp_path, monkeypatch):
    import shutil

    from workflow_distillation.lora_fp16.span_run import EpochPublisher

    output = tmp_path / "run"
    output.mkdir()
    (output / "events.jsonl").write_text("complete event log\n")
    checkpoint = tmp_path / "checkpoint-00000002"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"progress": {"global_step": 2}}))
    uploaded = []
    publisher = EpochPublisher(
        lambda path: uploaded.append((path / "events.jsonl").read_text()), [2], output
    )
    real_copy = shutil.copyfile
    interrupted = [False]

    def copy(source, target, *args, **kwargs):
        if Path(source).name == "events.jsonl" and not interrupted[0]:
            interrupted[0] = True
            Path(target).write_text("partial")
            raise OSError("disk full")
        return real_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)
    with pytest.raises(OSError, match="disk full"):
        publisher(checkpoint)
    assert uploaded == []
    publisher(checkpoint)
    assert uploaded == ["complete event log\n"]


def test_incomplete_peft_export_is_rebuilt_before_upload(tmp_path):
    import torch
    from safetensors.torch import save_file

    from workflow_distillation.lora_fp16.span_run import EpochPublisher

    checkpoint = tmp_path / "checkpoint-00000002"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"progress": {"global_step": 2}}))

    class Model:
        attempts = 0

        def save_pretrained(self, path, **kwargs):
            path = Path(path)
            path.mkdir(exist_ok=True)
            (path / "adapter_config.json").write_text("{}")
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("interrupted adapter export")
            save_file({"lora_A": torch.ones(2, 2)}, str(path / "adapter_model.safetensors"))

    uploaded = []
    publisher = EpochPublisher(
        lambda p: uploaded.append((p / "adapter_model.safetensors").is_file()), [2], tmp_path
    )
    publisher.model = Model()
    with pytest.raises(OSError):
        publisher(checkpoint)
    publisher(checkpoint)
    assert publisher.model.attempts == 2 and uploaded == [True]


def test_hub_resume_restores_prior_step_curve_and_keeps_old_stdout(tmp_path):
    from workflow_distillation.lora_fp16 import span_run

    checkpoint = tmp_path / "checkpoint-00000002"
    checkpoint.mkdir()
    previous = tmp_path / "previous"
    previous.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"progress": {"global_step": 2}}))
    lines = [dict(event="step", global_step=i, loss=1 / i) for i in [1, 2]]
    (previous / "events.jsonl").write_text("".join(json.dumps(v) + "\n" for v in lines))
    (previous / "train.log").write_text("previous server stdout\n")
    span_run.EpochPublisher(lambda p: None, [2], previous)(checkpoint)
    output = tmp_path / "new-server"
    output.mkdir()
    (output / "events.jsonl").write_text('{"event":"initializing"}\n')
    span_run.restore_checkpoint_logs(checkpoint, output)
    span_run.recover_events(output / "events.jsonl", 2)
    events = [json.loads(v) for v in (output / "events.jsonl").read_text().splitlines()]
    assert [v["global_step"] for v in events] == [1, 2]
    assert (output / "train-restored-step-00000002.log").read_text() == "previous server stdout\n"
