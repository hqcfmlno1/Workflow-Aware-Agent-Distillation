"""Strict parent migration over the shared atomic checkpoint transport."""

import json
from pathlib import Path, PurePosixPath

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.training_checkpoints import (
    CheckpointManager,
    HuggingFacePublisher,
    _manifest,
    latest_checkpoint,
)


def parent_binding(manifest_path, expected_sha256):
    path = Path(manifest_path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Pinned parent manifest checksum differs")
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True or manifest.get("format_version") != 1:
        raise ValueError("Parent checkpoint is not committed")
    return {
        "manifest_sha256": expected_sha256,
        "manifest": manifest,
        **{k: manifest[k] for k in ("identity", "progress", "world_size")},
    }


class MigrationManager(CheckpointManager):
    def __init__(self, *args, binding, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding = binding

    def load(self, checkpoint, **kwargs):
        path = (
            latest_checkpoint(self.directory) if str(checkpoint) == "latest" else Path(checkpoint)
        )
        child = self.identity
        try:
            if sha256_file(path / "manifest.json") == self.binding["manifest_sha256"]:
                self.identity = self.binding["identity"]
            return super().load(path, **kwargs)
        finally:
            self.identity = child


class MigrationPublisher(HuggingFacePublisher):
    def __init__(self, *args, binding, child_identity, **kwargs):
        super().__init__(*args, **kwargs)
        self.binding = binding
        self.child_identity = child_identity

    def __call__(self, checkpoint):
        return publish_migration(
            self,
            checkpoint,
            {"parent": self.binding},
            self.child_identity,
            HuggingFacePublisher.__call__,
        )


def publish_migration(publisher, checkpoint, plan, child_identity, original_publish):
    """Original atomic Hub protocol with ONE exact parent-to-child transition.

    Keep the same dedicated prefix and retention budget. The immutable original
    publisher refuses all identity changes; this protocol additionally accepts
    the byte-exact approved parent as remote latest when publishing its child.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    from workflow_distillation.training_checkpoints import _NAME

    manifest = _manifest(checkpoint)
    if sha256_file(Path(checkpoint) / "manifest.json") == plan["parent"]["manifest_sha256"]:
        # The original trainer republishes its loaded parent before training.
        return original_publish(publisher, checkpoint)
    if manifest["identity"] != child_identity:
        raise ValueError("Local checkpoint continuation identity differs")
    revision = publisher.validate().sha
    api = HfApi()
    pointer_name = f"{publisher.prefix}/latest.json"
    files = api.list_repo_files(publisher.repo_id, repo_type=publisher.repo_type, revision=revision)
    publisher._dedicated_files(files)
    if pointer_name not in files:
        raise ValueError("Continuation requires the existing parent checkpoint on Hub")
    pointer = hf_hub_download(
        publisher.repo_id, pointer_name, revision=revision, repo_type=publisher.repo_type
    )
    name = json.loads(Path(pointer).read_text())["checkpoint"]
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid remote checkpoint name")
    remote_path = hf_hub_download(
        publisher.repo_id,
        f"{publisher.prefix}/{name}/manifest.json",
        revision=revision,
        repo_type=publisher.repo_type,
    )
    remote = json.loads(Path(remote_path).read_text())
    exact_parent = (
        sha256_file(Path(remote_path)) == plan["parent"]["manifest_sha256"]
        and remote["identity"] == plan["parent"]["identity"]
        and remote["progress"] == plan["parent"]["progress"]
    )
    if remote["world_size"] != manifest["world_size"] or (
        remote["identity"] != manifest["identity"] and not exact_parent
    ):
        raise ValueError("Remote checkpoint identity differs from child or exact approved parent")
    if remote == manifest:
        publisher._squash(api, revision)
        return
    if remote["progress"]["global_step"] >= manifest["progress"]["global_step"]:
        raise ValueError("Remote checkpoint is already at this step or newer")
    operations = [
        CommitOperationAdd(
            path_in_repo=f"{publisher.prefix}/{checkpoint.name}/{p.name}",
            path_or_fileobj=p,
        )
        for p in sorted(checkpoint.iterdir())
        if p.is_file()
    ]
    operations.append(
        CommitOperationAdd(
            path_in_repo=pointer_name,
            path_or_fileobj=json.dumps({"checkpoint": checkpoint.name}).encode(),
        )
    )
    if publisher.keep_last is not None:
        names = {
            PurePosixPath(n).parts[-2]
            for n in files
            if n.startswith(f"{publisher.prefix}/checkpoint-")
        }
        if any(int(n.split("-")[1]) > manifest["progress"]["global_step"] for n in names):
            raise ValueError("Remote contains a checkpoint newer than the latest pointer")
        retained = set(
            sorted(names | {checkpoint.name}, key=lambda n: int(n.split("-")[1]))[
                -publisher.keep_last :
            ]
        )
        operations.extend(
            CommitOperationDelete(path_in_repo=n)
            for n in files
            if n.startswith(f"{publisher.prefix}/checkpoint-")
            and PurePosixPath(n).parts[-2] not in retained
        )
    committed = api.create_commit(
        repo_id=publisher.repo_id,
        repo_type=publisher.repo_type,
        operations=operations,
        parent_commit=revision,
        commit_message=f"Save training checkpoint {checkpoint.name}",
    )
    publisher._squash(api, committed.oid)
