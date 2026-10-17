"""Check interrupted writes, integrity, and continuation of real optimizer/RNG state."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from workflow_distillation import training_checkpoints as checkpoints


def test_save_schedule_counts_completed_updates_and_forces_a_final_save(tmp_path):
    manager = checkpoints.CheckpointManager(tmp_path, {"dataset_sha256": "a"})
    assert not manager.due(0)
    assert not manager.due(4)
    assert manager.due(5)
    assert manager.due(7, force=True)
    for settings in ({"save_every": 0}, {"keep_last": 0}):
        with pytest.raises(ValueError):
            checkpoints.CheckpointManager(tmp_path, {}, **settings)


def test_cli_accepts_single_retained_checkpoint_for_bounded_storage():
    from types import SimpleNamespace

    args = SimpleNamespace(save_every_steps=5, keep_checkpoints=1,
                           stop_margin_seconds=600, expected_step_seconds=240,
                           checkpoint_hub_repo=None)
    checkpoints.validate_checkpoint_arguments(args)


def test_budget_stops_before_next_step_and_signal_only_requests_a_boundary_stop():
    budget = checkpoints.StopBudget(max_seconds=1000, margin_seconds=100,
                                    expected_step_seconds=200, started=0)
    assert not budget.should_stop(now=699)
    assert budget.should_stop(now=700)
    budget.request_stop(15, None)
    assert budget.should_stop(now=1)


def test_hard_timeout_terminates_launcher_and_its_worker_process(tmp_path):
    import subprocess
    import sys
    import time

    pid_file = tmp_path / "worker.pid"
    child_code = "import time; time.sleep(60)"
    launcher_code = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"worker = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"Path({str(pid_file)!r}).write_text(str(worker.pid)); "
        "worker.wait()"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        checkpoints.run_worker_process([sys.executable, "-c", launcher_code], timeout=1)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 2
    while True:
        status = Path(f"/proc/{pid}/stat")
        try:
            state = status.read_text().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            break  # The killed process may disappear while /proc is being read.
        if state == "Z":
            break
        assert time.monotonic() < deadline, "A worker survived the launcher timeout"
        time.sleep(0.01)


def setup_training():
    torch = pytest.importorskip("torch")
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(3, 3).requires_grad_(False)
            self.adapter = torch.nn.Linear(3, 3, bias=False)
            self.dropout = torch.nn.Dropout(0.2)

        def forward(self, x):
            return self.base(x) + self.adapter(self.dropout(x))

    model = TinyAdapter()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cpu", init_scale=8, growth_interval=1)
    return model, optimizer, scheduler, scaler


def step(training):
    import torch

    model, optimizer, scheduler, scaler = training
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 3) * (random.random() + float(np.random.random()))
    loss = model(x).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach().clone()


def save(manager, training, number):
    model, optimizer, scheduler, scaler = training
    return manager.save(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        progress={"global_step": number, "next_batch": number})


def load(manager, training, path="latest"):
    model, optimizer, scheduler, scaler = training
    return manager.load(path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)


def test_resume_reproduces_uninterrupted_optimizer_scheduler_scaler_and_dropout(tmp_path):
    import torch

    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {"data": "same", "world_size": 1})
    step(training)
    path = save(manager, training, 1)
    expected_losses = [step(training), step(training)]
    expected_adapter = training[0].adapter.weight.detach().clone()
    expected_lr = training[2].get_last_lr()
    expected_scale = training[3].state_dict()
    resumed = setup_training()
    progress = load(manager, resumed, path)
    assert progress == {"global_step": 1, "next_batch": 1}
    actual_losses = [step(resumed), step(resumed)]
    for actual, expected in zip(actual_losses, expected_losses, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(resumed[0].adapter.weight, expected_adapter, atol=0, rtol=0)
    assert resumed[2].get_last_lr() == expected_lr
    assert resumed[3].state_dict() == expected_scale
    assert int(resumed[1].state[resumed[0].adapter.weight]["step"]) == 3


def test_partial_corrupt_or_wrong_dataset_checkpoint_is_not_loaded(tmp_path):
    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {"dataset_sha256": "a"})
    step(training)
    first = save(manager, training, 1)
    step(training)
    second = save(manager, training, 2)
    (tmp_path / ".checkpoint-00000003-incomplete").mkdir()
    (second / "adapter.pt").write_bytes(b"interrupted download")
    assert checkpoints.latest_checkpoint(tmp_path) == first
    with pytest.raises(ValueError, match="checksum"):
        load(manager, training, second)
    other = checkpoints.CheckpointManager(tmp_path, {"dataset_sha256": "changed"})
    before = training[0].adapter.weight.detach().clone()
    with pytest.raises(ValueError, match="identity"):
        load(other, training, first)
    assert training[0].adapter.weight.equal(before)


def test_malformed_manifest_does_not_prevent_fallback_to_previous_checkpoint(tmp_path):
    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {})
    first = save(manager, training, 1)
    second = save(manager, training, 2)
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["files"] = list(manifest["files"])
    (second / "manifest.json").write_text(json.dumps(manifest))
    assert checkpoints.latest_checkpoint(tmp_path) == first


def test_failed_publication_keeps_prior_checkpoint_and_local_recovery(tmp_path):
    training = setup_training()

    def publish(path):
        if path.name == "checkpoint-00000002":
            raise OSError("network unavailable")

    manager = checkpoints.CheckpointManager(tmp_path, {"data": "a"}, keep_last=1,
                                            publisher=publish)
    step(training)
    save(manager, training, 1)
    step(training)
    with pytest.raises(RuntimeError, match="network unavailable"):
        save(manager, training, 2)
    assert (tmp_path / "checkpoint-00000001").is_dir()
    assert checkpoints.latest_checkpoint(tmp_path).name == "checkpoint-00000002"
    assert json.loads((tmp_path / "latest.json").read_text())["checkpoint"] == "checkpoint-00000002"
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-00000001", "checkpoint-00000002",
    ]


def test_completed_local_checkpoint_can_retry_publication_without_replaying_training(tmp_path):
    import torch

    published = []

    def publish(path):
        # IO retries can consume RNG; they must not change subsequent dropout/data.
        random.random()
        np.random.random()
        torch.rand(1)
        if not published:
            published.append("failed")
            raise OSError("network unavailable")
        published.append(path.name)

    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {}, publisher=publish)
    step(training)
    with pytest.raises(RuntimeError, match="network unavailable"):
        save(manager, training, 1)
    expected_loss = step(training)
    resumed = setup_training()
    load(manager, resumed)
    manager.republish(model=resumed[0])
    assert published == ["failed", "checkpoint-00000001"]
    actual_loss = step(resumed)
    torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)


def test_successful_saves_keep_two_complete_checkpoints_without_overwriting(tmp_path):
    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {"data": "a"})
    for number in (1, 2, 3):
        step(training)
        save(manager, training, number)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-00000002", "checkpoint-00000003",
    ]
    with pytest.raises(RuntimeError, match="exists"):
        save(manager, training, 3)


def test_single_retention_prunes_only_after_new_checkpoint_is_published(tmp_path):
    observed = []

    def publish(path):
        observed.append(sorted(p.name for p in tmp_path.glob("checkpoint-*")))
        assert checkpoints.latest_checkpoint(tmp_path) == path

    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {"data": "a"}, keep_last=1,
                                            publisher=publish)
    for number in (1, 2, 3):
        step(training)
        save(manager, training, number)
    assert observed == [
        ["checkpoint-00000001"],
        ["checkpoint-00000001", "checkpoint-00000002"],
        ["checkpoint-00000002", "checkpoint-00000003"],
    ]
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-00000003",
    ]
    assert load(manager, setup_training()) == {"global_step": 3, "next_batch": 3}


def test_failed_staged_write_removes_its_temporary_files_and_keeps_prior(tmp_path, monkeypatch):
    training = setup_training()
    manager = checkpoints.CheckpointManager(tmp_path, {}, keep_last=1)
    step(training)
    first = save(manager, training, 1)
    original = checkpoints._torch_file

    def fail_after_rank_write(path, value):
        original(path, value)
        if path.name == "rank-00000.pt":
            raise OSError("disk full")

    monkeypatch.setattr(checkpoints, "_torch_file", fail_after_rank_write)
    step(training)
    with pytest.raises(RuntimeError, match="disk full"):
        save(manager, training, 2)
    assert checkpoints.latest_checkpoint(tmp_path) == first
    assert json.loads((tmp_path / "latest.json").read_text())["checkpoint"] == first.name
    assert sorted(p.name for p in tmp_path.iterdir()) == [first.name, "latest.json"]


def test_hub_prefix_cannot_escape_checkpoint_namespace():
    for prefix in ("../other", "/absolute", "a/../../b", ""):
        with pytest.raises(ValueError):
            checkpoints.HuggingFacePublisher("owner/private-model", prefix=prefix)


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    from fnmatch import fnmatch
    from types import SimpleNamespace

    import huggingface_hub

    # A filesystem-backed transport double replaces only the external Hub API.
    # Checkpoint creation, hashing, commit payloads, downloads, and resume are real.
    revisions = {"0": {}}
    head = ["0"]
    commits, squashes = [], []
    failures = {"squash": False}
    transport = SimpleNamespace(repo_type="model")

    def check_type(kwargs):
        assert kwargs.get("repo_type", "model") == transport.repo_type

    class Hub:
        def auth_check(self, *_args, **kwargs):
            check_type(kwargs)
            assert kwargs["write"] is True
            if failures.get("write_permission"):
                raise PermissionError("Token cannot write to this repository")

        def repo_info(self, *_args, **_kwargs):
            check_type(_kwargs)
            return SimpleNamespace(private=True, sha=head[0])

        def list_repo_files(self, _repo, *, revision, **_kwargs):
            check_type(_kwargs)
            return list(revisions[revision])

        def list_repo_refs(self, *_args, **_kwargs):
            check_type(_kwargs)
            return SimpleNamespace(branches=[SimpleNamespace(name="main")],
                                   tags=[], converts=[], pull_requests=[])

        def create_commit(self, *, operations, parent_commit=None, **_kwargs):
            check_type(_kwargs)
            assert parent_commit is not None
            if parent_commit is not None and parent_commit != head[0]:
                raise RuntimeError("concurrent Hub modification")
            updated = dict(revisions[head[0]])
            for operation in operations:
                if isinstance(operation, huggingface_hub.CommitOperationDelete):
                    del updated[operation.path_in_repo]
                else:
                    source = operation.path_or_fileobj
                    updated[operation.path_in_repo] = (
                        source if isinstance(source, bytes) else Path(source).read_bytes()
                    )
            commits.append(updated)
            head[0] = f"commit-{len(commits)}"
            revisions[head[0]] = updated
            return SimpleNamespace(oid=head[0])

        def super_squash_history(self, **_kwargs):
            check_type(_kwargs)
            if failures["squash"]:
                raise OSError("Hub history cleanup failed")
            tree = revisions[head[0]]
            squashes.append(dict(tree))
            head[0] = f"squash-{len(squashes)}"
            revisions.clear()
            revisions[head[0]] = tree

    def download(_repo, filename, *, revision, **kwargs):
        check_type(kwargs)
        path = tmp_path / "transport-cache" / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(revisions[revision][filename])
        return str(path)

    def snapshot(_repo, *, revision, local_dir, allow_patterns, **kwargs):
        check_type(kwargs)
        for name, content in revisions[revision].items():
            if any(fnmatch(name, pattern) for pattern in allow_patterns):
                path = Path(local_dir) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return SimpleNamespace(revisions=revisions, head=head, commits=commits,
                           squashes=squashes, failures=failures, transport=transport)


@pytest.mark.parametrize("keep_last", [2, 3])
def test_dataset_repo_supports_complete_backup_retention_and_resume(tmp_path, fake_hub, keep_last):
    fake_hub.transport.repo_type = "dataset"
    publisher = checkpoints.HuggingFacePublisher(
        "owner/private-dataset", prefix="first-run", repo_type="dataset", keep_last=keep_last,
    )
    manager = checkpoints.CheckpointManager(tmp_path / "source", {}, publisher=publisher)
    training = setup_training()
    for number in range(1, keep_last + 2):
        step(training)
        save(manager, training, number)
    restored = checkpoints.download_latest_checkpoint(
        "owner/private-dataset", tmp_path / "restored", prefix="first-run", repo_type="dataset",
    )
    assert load(manager, setup_training(), restored)["global_step"] == keep_last + 1
    assert len(fake_hub.revisions) == 1
    assert all("checkpoint-00000001" not in name
               for name in fake_hub.revisions[fake_hub.head[0]])
    retained = {name.split("/")[1] for name in fake_hub.revisions[fake_hub.head[0]]
                if "/checkpoint-" in name}
    assert len(retained) == keep_last


def test_cpu_preflight_requires_dataset_write_permission_without_mutating_repo(fake_hub):
    from scripts.cloud.checkpoint_preflight import probe_repo

    fake_hub.transport.repo_type = "dataset"
    assert probe_repo("owner/dataset", repo_type="dataset")["private"] is True
    assert not fake_hub.commits
    fake_hub.failures["write_permission"] = True
    with pytest.raises(PermissionError, match="cannot write"):
        probe_repo("owner/dataset", repo_type="dataset")
    assert not fake_hub.commits


def test_hub_backup_is_atomic_downloadable_and_cannot_overwrite_another_run(tmp_path, fake_hub):
    revisions, head = fake_hub.revisions, fake_hub.head
    training = setup_training()
    publisher = checkpoints.HuggingFacePublisher("owner/private-model", prefix="run-a")
    manager = checkpoints.CheckpointManager(tmp_path / "source", {"data": "a"},
                                            publisher=publisher)
    step(training)
    save(manager, training, 1)
    # The first durable commit already contains the pointer AND all checkpoint files.
    published = revisions["commit-1"]
    assert json.loads(published["run-a/latest.json"])["checkpoint"] == "checkpoint-00000001"
    assert set(published) == {
        "run-a/latest.json", "run-a/checkpoint-00000001/adapter.pt",
        "run-a/checkpoint-00000001/rank-00000.pt",
        "run-a/checkpoint-00000001/manifest.json",
    }
    restored = checkpoints.download_latest_checkpoint(
        "owner/private-model", tmp_path / "download", prefix="run-a",
    )
    assert load(manager, setup_training(), restored) == {"global_step": 1, "next_batch": 1}
    publisher(tmp_path / "source/checkpoint-00000001")
    assert head[0] == "commit-1"  # Idempotent retry after a lost commit response.
    other = checkpoints.CheckpointManager(tmp_path / "other", {"data": "different"},
                                          publisher=publisher)
    with pytest.raises(RuntimeError, match="identity"):
        save(other, setup_training(), 1)
    assert head[0] == "commit-1"
    step(training)
    save(manager, training, 2)
    with pytest.raises(ValueError, match="already at this step or newer"):
        publisher(tmp_path / "source/checkpoint-00000001")
    assert head[0] == "commit-2"


def test_explicit_hub_retention_keeps_two_snapshots_and_discards_history(tmp_path, fake_hub):
    training = setup_training()
    publisher = checkpoints.HuggingFacePublisher(
        "owner/dedicated-first-run", prefix="first-run", keep_last=2,
    )
    manager = checkpoints.CheckpointManager(tmp_path / "source", {},
                                            keep_last=1, publisher=publisher)
    for number in range(1, 4):
        step(training)
        save(manager, training, number)
    files = fake_hub.revisions[fake_hub.head[0]]
    assert len(fake_hub.revisions) == 1
    assert len(fake_hub.squashes) == 3
    assert all("checkpoint-00000001" not in name for name in files)
    for number in (2, 3):
        base = f"first-run/checkpoint-{number:08d}"
        manifest = json.loads(files[f"{base}/manifest.json"])
        for name, checksum in manifest["files"].items():
            import hashlib
            assert hashlib.sha256(files[f"{base}/{name}"]).hexdigest() == checksum
    assert json.loads(files["first-run/latest.json"])["checkpoint"] == "checkpoint-00000003"
    restored = checkpoints.download_latest_checkpoint(
        "owner/dedicated-first-run", tmp_path / "download", prefix="first-run",
    )
    assert load(manager, setup_training(), restored)["global_step"] == 3


def test_hub_history_failure_stops_and_retry_finishes_without_duplicate_upload(tmp_path, fake_hub):
    training = setup_training()
    publisher = checkpoints.HuggingFacePublisher("owner/dedicated", prefix="run", keep_last=2)
    manager = checkpoints.CheckpointManager(tmp_path / "source", {},
                                            keep_last=1, publisher=publisher)
    step(training)
    save(manager, training, 1)
    fake_hub.failures["squash"] = True
    step(training)
    with pytest.raises(RuntimeError, match="history cleanup failed"):
        save(manager, training, 2)
    assert len(fake_hub.commits) == 2  # Upload succeeded before the history API failed.
    assert len(list((tmp_path / "source").glob("checkpoint-*"))) == 2
    fake_hub.failures["squash"] = False
    publisher(tmp_path / "source/checkpoint-00000002")
    assert len(fake_hub.commits) == 2
    assert len(fake_hub.revisions) == 1


def test_hub_history_cleanup_rejects_repo_with_another_run_before_any_write(tmp_path, fake_hub):
    fake_hub.revisions["0"]["other-run/important.txt"] = b"keep this"
    training = setup_training()
    publisher = checkpoints.HuggingFacePublisher("owner/shared", prefix="run", keep_last=2)
    manager = checkpoints.CheckpointManager(tmp_path / "source", {}, publisher=publisher)
    with pytest.raises(RuntimeError, match="dedicated"):
        save(manager, training, 1)
    assert not fake_hub.commits
    assert not fake_hub.squashes
    assert fake_hub.revisions["0"]["other-run/important.txt"] == b"keep this"


def test_hub_history_cleanup_requires_at_least_two_recovery_snapshots():
    with pytest.raises(ValueError, match="at least two"):
        checkpoints.HuggingFacePublisher("owner/dedicated", keep_last=1)


def test_cli_help_does_not_load_weights_or_start_remote_training():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    for script in ("pilot_optimized_qlora.py", "pilot_ddp_once.py", "train_qlora_epoch.py"):
        result = subprocess.run(
            [sys.executable, str(root / "scripts/training" / script), "--help"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "--save-every-steps" in result.stdout
        assert "--resume-from" in result.stdout
    result = subprocess.run(
        [sys.executable, str(root / "scripts/training/fetch_training_checkpoint.py"), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--prefix" in result.stdout
