"""Committed training checkpoints and restoration at optimizer boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

_NAME = re.compile(r"checkpoint-\d{8,}$")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_file(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)


def _manifest(path):
    path = Path(path)
    if path.is_symlink() or not _NAME.fullmatch(path.name):
        raise ValueError("Not a committed checkpoint directory")
    try:
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["format_version"] != 1 or manifest["complete"] is not True
                or type(manifest["world_size"]) is not int or manifest["world_size"] < 1
                or not isinstance(manifest["files"], dict)
                or not isinstance(manifest["identity"], dict)
                or not isinstance(manifest["progress"], dict)
                or type(manifest["progress"]["global_step"]) is not int
                or manifest["progress"]["global_step"] != int(path.name.split("-")[1])):
            raise ValueError("Incomplete checkpoint")
        required = {"adapter.pt", *(f"rank-{r:05d}.pt" for r in range(manifest["world_size"]))}
        if set(manifest["files"]) != required:
            raise ValueError("Checkpoint rank files are incomplete")
        for name, checksum in manifest["files"].items():
            artifact = path / name
            if artifact.is_symlink() or not artifact.is_file() or _sha256(artifact) != checksum:
                raise ValueError(f"Checkpoint checksum mismatch: {name}")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checkpoint: {path}") from exc


def latest_checkpoint(directory):
    """Ignore incomplete/corrupt saves, even when the latest pointer is stale."""
    candidates = sorted((p for p in Path(directory).glob("checkpoint-*")
                         if _NAME.fullmatch(p.name)),
                        key=lambda p: int(p.name.split("-")[1]), reverse=True)
    for path in candidates:
        try:
            _manifest(path)
        except ValueError:
            continue
        return path
    raise ValueError(f"No complete checkpoint in {directory}")


def _group():
    import torch.distributed as dist

    return dist if dist.is_available() and dist.is_initialized() else None


def _gather(value):
    dist = _group()
    if dist is None:
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _agree(error):
    failures = [message for message in _gather(error) if message]
    if failures:
        raise RuntimeError("Checkpoint operation failed: " + "; ".join(failures))


def _cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _torch_file(path, value):
    import torch

    with path.open("wb") as stream:
        torch.save(_cpu_tree(value), stream)
        stream.flush()
        os.fsync(stream.fileno())


def _rng_state(model):
    import numpy as np
    import torch

    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": {index: torch.cuda.get_rng_state(index) for index in devices}}


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for device, value in state["cuda"].items():
        torch.cuda.set_rng_state(value, device)


class CheckpointManager:
    """Save only committed optimizer boundaries; all DDP ranks call together.

    The directory must be shared by ranks on the same host. Rank zero commits
    adapter weights; every rank retains its own optimizer, scaler, and RNG.
    """

    def __init__(self, directory, identity, *, save_every=5, keep_last=2, publisher=None):
        if save_every < 1 or keep_last < 1:
            raise ValueError("Require a positive interval and at least one retained checkpoint")
        self.directory = Path(directory)
        self.identity = json.loads(json.dumps(identity, sort_keys=True))
        self.save_every = save_every
        self.keep_last = keep_last
        self.publisher = publisher
        self.last_step = None
        self.last_checkpoint = None

    def due(self, global_step, *, force=False):
        return global_step != self.last_step and (
            force or (global_step > 0 and global_step % self.save_every == 0)
        )

    def save(self, *, model, optimizer, scheduler, scaler, progress):
        dist = _group()
        rank = dist.get_rank() if dist else 0
        world_size = dist.get_world_size() if dist else 1
        progress = json.loads(json.dumps(progress))
        if type(progress.get("global_step")) is not int or progress["global_step"] < 0:
            raise ValueError("Checkpoint requires a nonnegative completed global_step")
        agreement = _gather({"identity": self.identity, "progress": progress})
        if any(item != agreement[0] for item in agreement):
            raise ValueError("Checkpoint identity/progress differs between ranks")
        name = f"checkpoint-{progress['global_step']:08d}"
        final = self.directory / name
        stage = [None]
        error = None
        if rank == 0:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if final.exists():
                    raise FileExistsError(f"Checkpoint already exists: {final}")
                stage[0] = tempfile.mkdtemp(prefix=f".{name}-", dir=self.directory)
            except Exception as exc:
                error = str(exc)
        _agree(error)
        if dist:
            dist.broadcast_object_list(stage, src=0)
        staging = Path(stage[0])
        error = None
        try:
            if any(p.grad is not None for p in model.parameters() if p.requires_grad):
                raise ValueError("Clear gradients before saving an optimizer-boundary checkpoint")
            _torch_file(staging / f"rank-{rank:05d}.pt", {
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": _rng_state(model),
            })
            if rank == 0:
                _torch_file(staging / "adapter.pt", {
                    name: param for name, param in model.named_parameters() if param.requires_grad
                })
        except Exception as exc:
            error = str(exc)
        try:
            _agree(error)
        except RuntimeError:
            # All ranks have finished their writes before agreement returns.
            if rank == 0:
                with suppress(OSError):
                    shutil.rmtree(staging)
            raise
        error = None
        if rank == 0:
            try:
                manifest = {"format_version": 1, "complete": True,
                            "world_size": world_size, "identity": self.identity,
                            "progress": progress,
                            "files": {p.name: _sha256(p) for p in sorted(staging.glob("*.pt"))}}
                _json_file(staging / "manifest.json", manifest)
                staging.rename(final)
                _sync_directory(self.directory)
                _json_file(self.directory / "latest.json", {"checkpoint": name})
                if self.publisher is not None:
                    self._publish_local(final, model)
                # Never prune the previous recovery points if publication failed.
                own = []
                for candidate in self.directory.glob("checkpoint-*"):
                    try:
                        saved = _manifest(candidate)
                    except ValueError:
                        continue
                    if saved["identity"] == self.identity:
                        own.append(candidate)
                own.sort(key=lambda p: int(p.name.split("-")[1]), reverse=True)
                for obsolete in own[self.keep_last:]:
                    shutil.rmtree(obsolete)
                print(json.dumps({"event": "checkpoint_saved", "path": str(final),
                                  "global_step": progress["global_step"],
                                  "published": self.publisher is not None}), flush=True)
            except Exception as exc:
                # A committed checkpoint survives a failed publication. Only
                # pre-commit staging files are disposable.
                if staging.exists():
                    with suppress(OSError):
                        shutil.rmtree(staging)
                error = str(exc)
        _agree(error)
        self.last_step = progress["global_step"]
        self.last_checkpoint = final
        return final

    def _publish_local(self, checkpoint, model):
        # Network retry jitter must not move training/data/dropout RNG cursors.
        rng = _rng_state(model)
        try:
            self.publisher(checkpoint)
        finally:
            _restore_rng(rng)

    def republish(self, *, model):
        """Retry a loaded local checkpoint, including an already completed pilot.

        Call on every rank, after validating the restored data cursor.
        """
        if self.publisher is None:
            return
        dist = _group()
        error = None
        if dist is None or dist.get_rank() == 0:
            try:
                if self.last_checkpoint is None:
                    raise ValueError("Load a checkpoint before retrying publication")
                self._publish_local(self.last_checkpoint, model)
                print(json.dumps({"event": "checkpoint_republished",
                                  "path": str(self.last_checkpoint),
                                  "global_step": self.last_step, "published": True}), flush=True)
            except Exception as exc:
                error = str(exc)
        _agree(error)

    def load(self, checkpoint, *, model, optimizer, scheduler, scaler):
        if _group() is None:
            return self._load_local(checkpoint, model=model, optimizer=optimizer,
                                    scheduler=scheduler, scaler=scaler)
        error, progress = None, None
        try:
            progress = self._load_local(checkpoint, model=model, optimizer=optimizer,
                                        scheduler=scheduler, scaler=scaler)
        except Exception as exc:
            error = str(exc)
        _agree(error)
        positions = _gather(progress)
        if any(value != positions[0] for value in positions):
            raise ValueError("Resume positions differ between ranks")
        return progress

    def _load_local(self, checkpoint, *, model, optimizer, scheduler, scaler):
        import torch

        dist = _group()
        rank = dist.get_rank() if dist else 0
        world_size = dist.get_world_size() if dist else 1
        path = (latest_checkpoint(self.directory)
                if str(checkpoint) == "latest" else Path(checkpoint))
        manifest = _manifest(path)
        if manifest["identity"] != self.identity or manifest["world_size"] != world_size:
            raise ValueError("Checkpoint identity differs: data/model/config/world size must match")
        # Load only checkpoints created by this project and trusted by the user.
        adapter = torch.load(path / "adapter.pt", map_location="cpu", weights_only=True)
        current = {n: p for n, p in model.named_parameters() if p.requires_grad}
        if adapter.keys() != current.keys() or any(
            adapter[n].shape != current[n].shape or adapter[n].dtype != current[n].dtype
            for n in current
        ):
            raise ValueError("Adapter parameter names/shapes/dtypes differ")
        state = torch.load(path / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False)
        if (state["scheduler"] is None) != (scheduler is None):
            raise ValueError("Scheduler configuration differs")
        devices = {p.device.index for p in model.parameters() if p.device.type == "cuda"}
        if set(state["rng"]["cuda"]) != devices:
            raise ValueError("CUDA device placement differs")
        with torch.no_grad():
            for name, param in current.items():
                param.copy_(adapter[name])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        # Restore RNG last: model construction and state loading must not move the cursor.
        _restore_rng(state["rng"])
        self.last_step = manifest["progress"]["global_step"]
        self.last_checkpoint = path
        return manifest["progress"]


class StopBudget:
    """Signals request a stop at a safe boundary; never run IO in a handler."""

    def __init__(self, *, max_seconds, margin_seconds=120, expected_step_seconds=180, started=None):
        if max_seconds <= 0 or margin_seconds < 0 or expected_step_seconds < 0:
            raise ValueError("Invalid session time budget")
        self.max_seconds = max_seconds
        self.margin_seconds = margin_seconds
        self.expected_step_seconds = expected_step_seconds
        self.started = time.monotonic() if started is None else started
        self.requested = False
        self._previous = {}

    def request_stop(self, _signum, _frame):
        self.requested = True

    def should_stop(self, *, now=None):
        now = time.monotonic() if now is None else now
        return self.requested or (
            now - self.started + self.margin_seconds + self.expected_step_seconds
            >= self.max_seconds
        )

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.signal(signum, self.request_stop)
        return self

    def __exit__(self, *_args):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def run_worker_process(command, *, timeout, env=None, check=False):
    """Bound the entire worker group, including torchrun's descendant ranks.

    The hard deadline is after the worker's cooperative save allowance. Killing
    only torchrun can leave GPU ranks alive after the launcher has timed out.
    """
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return subprocess.CompletedProcess(command, returncode)


class HuggingFacePublisher:
    """Publish complete state atomically; optional retention destroys repo history.

    keep_last is an explicit opt-in for a dedicated, single-writer repository.
    It keeps snapshots in the current tree and permanently squashes main history.
    The default preserves all checkpoint files and all repository history.
    """

    def __init__(self, repo_id, *, prefix="training", keep_last=None, repo_type="model"):
        path = PurePosixPath(prefix)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in prefix:
            raise ValueError("Checkpoint Hub prefix must be a nonempty relative path")
        self.repo_id = repo_id
        if repo_type not in ("model", "dataset"):
            raise ValueError("Checkpoint repo type must be model or dataset")
        self.repo_type = repo_type
        self.prefix = str(path)
        if keep_last is not None and (type(keep_last) is not int or keep_last < 2):
            raise ValueError("Hub retention requires at least two recovery snapshots")
        self.keep_last = keep_last

    def _dedicated_files(self, files):
        if self.keep_last is None:
            return
        for name in files:
            if name in ("README.md", ".gitattributes"):
                continue
            if name == f"{self.prefix}/latest.json":
                continue
            relative = name.removeprefix(f"{self.prefix}/")
            parts = PurePosixPath(relative).parts
            if (relative == name or len(parts) != 2 or not _NAME.fullmatch(parts[0])
                    or (parts[1] not in ("adapter.pt", "manifest.json")
                        and not re.fullmatch(r"rank-\d{5}\.pt", parts[1]))):
                raise ValueError("History cleanup requires a dedicated checkpoint repository")

    def validate(self):
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.repo_info(self.repo_id, repo_type=self.repo_type)
        if not info.private:
            raise ValueError("Checkpoint repository must be private")
        if self.keep_last is not None:
            self._dedicated_files(api.list_repo_files(
                self.repo_id, repo_type=self.repo_type, revision=info.sha,
            ))
            refs = api.list_repo_refs(self.repo_id, repo_type=self.repo_type,
                                      include_pull_requests=True)
            if (any(branch.name != "main" for branch in refs.branches)
                    or refs.tags or refs.converts or refs.pull_requests):
                raise ValueError("History cleanup requires dedicated main without other refs")
        return info

    def _squash(self, api, revision):
        if self.keep_last is not None:
            if self.validate().sha != revision:
                raise RuntimeError("Concurrent Hub modification before history cleanup")
            api.super_squash_history(
                repo_id=self.repo_id, repo_type=self.repo_type, branch="main",
                commit_message=f"Keep latest {self.keep_last} full training checkpoints",
            )

    def __call__(self, checkpoint):
        from huggingface_hub import (
            CommitOperationAdd,
            CommitOperationDelete,
            HfApi,
            hf_hub_download,
        )

        revision = self.validate().sha
        manifest = _manifest(checkpoint)
        api = HfApi()
        pointer_name = f"{self.prefix}/latest.json"
        files = api.list_repo_files(self.repo_id, repo_type=self.repo_type, revision=revision)
        self._dedicated_files(files)
        if pointer_name in files:
            pointer = hf_hub_download(self.repo_id, pointer_name, revision=revision,
                                      repo_type=self.repo_type)
            name = json.loads(Path(pointer).read_text())["checkpoint"]
            if not _NAME.fullmatch(name):
                raise ValueError("Invalid remote checkpoint name")
            remote_path = hf_hub_download(
                self.repo_id, f"{self.prefix}/{name}/manifest.json", revision=revision,
                repo_type=self.repo_type,
            )
            remote = json.loads(Path(remote_path).read_text())
            if (remote["identity"] != manifest["identity"]
                    or remote["world_size"] != manifest["world_size"]):
                raise ValueError("Remote checkpoint identity differs; use a new Hub prefix")
            if remote == manifest:
                self._squash(api, revision)  # Finish a cleanup whose response was lost.
                return  # Retry after a successful commit whose response was lost.
            if remote["progress"]["global_step"] >= manifest["progress"]["global_step"]:
                raise ValueError("Remote checkpoint is already at this step or newer")
        operations = [CommitOperationAdd(
            path_in_repo=f"{self.prefix}/{checkpoint.name}/{path.name}", path_or_fileobj=path,
        ) for path in sorted(checkpoint.iterdir()) if path.is_file()]
        operations.append(CommitOperationAdd(
            path_in_repo=f"{self.prefix}/latest.json",
            path_or_fileobj=json.dumps({"checkpoint": checkpoint.name}).encode(),
        ))
        if self.keep_last is not None:
            names = {PurePosixPath(name).parts[-2] for name in files
                     if name.startswith(f"{self.prefix}/checkpoint-")}
            if any(int(name.split("-")[1]) > manifest["progress"]["global_step"]
                   for name in names):
                raise ValueError("Remote contains a checkpoint newer than the latest pointer")
            retained = set(sorted(names | {checkpoint.name},
                                  key=lambda name: int(name.split("-")[1]))[-self.keep_last:])
            operations.extend(CommitOperationDelete(path_in_repo=name) for name in files
                              if name.startswith(f"{self.prefix}/checkpoint-")
                              and PurePosixPath(name).parts[-2] not in retained)
        committed = api.create_commit(
            repo_id=self.repo_id, repo_type=self.repo_type, operations=operations,
            parent_commit=revision, commit_message=f"Save training checkpoint {checkpoint.name}",
        )
        self._squash(api, committed.oid)


def download_latest_checkpoint(repo_id, destination, *, prefix="training", repo_type="model"):
    """Pin the Hub revision so the pointer and all state files refer to one commit."""
    from huggingface_hub import hf_hub_download, snapshot_download

    store = HuggingFacePublisher(repo_id, prefix=prefix, repo_type=repo_type)
    revision = store.validate().sha
    pointer = hf_hub_download(repo_id, f"{store.prefix}/latest.json", revision=revision,
                              repo_type=store.repo_type)
    name = json.loads(Path(pointer).read_text())["checkpoint"]
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid remote checkpoint name")
    downloaded = snapshot_download(repo_id, revision=revision, local_dir=destination,
                                   allow_patterns=[f"{store.prefix}/{name}/*"],
                                   repo_type=store.repo_type)
    path = Path(downloaded) / store.prefix / name
    _manifest(path)
    return path


def add_checkpoint_arguments(parser):
    parser.add_argument("--save-every-steps", type=int, default=5)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-from", help="Committed checkpoint directory, or latest")
    parser.add_argument("--checkpoint-hub-repo", help="Existing private HF repo; HF_TOKEN")
    parser.add_argument("--checkpoint-hub-prefix", default="training")
    parser.add_argument("--stop-margin-seconds", type=int, default=120)
    parser.add_argument("--expected-step-seconds", type=int, default=180)


def checkpoint_arguments(args):
    """Preserve checkpoint policy when a launcher creates its worker processes."""
    values = ["--save-every-steps", str(args.save_every_steps),
              "--keep-checkpoints", str(args.keep_checkpoints),
              "--checkpoint-hub-prefix", args.checkpoint_hub_prefix,
              "--stop-margin-seconds", str(args.stop_margin_seconds),
              "--expected-step-seconds", str(args.expected_step_seconds)]
    for flag, value in (("--checkpoint-dir", args.checkpoint_dir),
                        ("--resume-from", args.resume_from),
                        ("--checkpoint-hub-repo", args.checkpoint_hub_repo)):
        if value is not None:
            values.extend([flag, str(value)])
    if getattr(args, "checkpoint_hub_keep_last", None) is not None:
        values.extend(["--checkpoint-hub-keep-last", str(args.checkpoint_hub_keep_last)])
    if getattr(args, "checkpoint_hub_repo_type", None) is not None:
        values.extend(["--checkpoint-hub-repo-type", args.checkpoint_hub_repo_type])
    return values


def validate_checkpoint_arguments(args):
    if args.save_every_steps < 1 or args.keep_checkpoints < 1:
        raise ValueError("Require save-every-steps >= 1 and keep-checkpoints >= 1")
    if args.stop_margin_seconds < 0 or args.expected_step_seconds < 0:
        raise ValueError("Stop margin and expected step duration must be nonnegative")
    if args.checkpoint_hub_repo:
        HuggingFacePublisher(args.checkpoint_hub_repo, prefix=args.checkpoint_hub_prefix,
                             keep_last=getattr(args, "checkpoint_hub_keep_last", None),
                             repo_type=getattr(args, "checkpoint_hub_repo_type", "model"))
    elif getattr(args, "checkpoint_hub_keep_last", None) is not None:
        raise ValueError("Hub retention requires checkpoint-hub-repo")


def training_identity(script, *, model, revision, records_sha256, settings):
    """Changing the data, numerical implementation, or runtime rejects continuation."""
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("torch", "numpy", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    source = Path(__file__).parent
    return {"model": model, "revision": revision, "records_sha256": records_sha256,
            "settings": settings, "packages": packages, "script_sha256": _sha256(script),
            "math_sources": {name: _sha256(source / name) for name in (
                "optimized_qlora.py", "weighted_sft_loss.py", "t4_attention.py", "ddp_probe.py")}}
