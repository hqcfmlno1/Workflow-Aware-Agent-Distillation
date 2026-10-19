"""A checked suffix-only data view with explicit parent checkpoint migration.

The original dataset binaries, trainer and mathematical sources stay intact.
The filtered view preserves the already consumed shuffled prefix, then packs
only eligible remaining records, without reshuffling or truncating any sample.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from workflow_distillation.epoch_data import epoch_batches, sha256_file
from workflow_distillation.epoch_training import validate_progress


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _make_plan(index, parent, parent_sha, *, max_tokens, seed):
    if type(max_tokens) is not int or not 2 < max_tokens <= 30000:
        raise ValueError("Invalid strict trajectory limit")
    source = json.loads(Path(index).read_text())
    rows = source["data"]
    source_sha = sha256_file(Path(index))
    if (parent.get("complete") is not True or parent.get("format_version") != 1
            or parent["identity"]["records_sha256"] != source_sha
            or parent["identity"]["settings"]["seed"] != seed):
        raise ValueError("Continuation parent identity differs from source/seed")
    batches = epoch_batches(len(rows), seed)
    progress = parent["progress"]
    validate_progress(batches, progress)
    order = [i for batch in batches for i in batch]
    consumed = progress["samples_consumed"]
    prefix, suffix = order[:consumed], order[consumed:]
    kept = [i for i in suffix if rows[i]["length"] < max_tokens]
    selected = prefix + kept
    return {
        "format_version": 1, "policy": "preserve_consumed_prefix_filter_suffix",
        "max_tokens": max_tokens, "seed": seed,
        "source_index_sha256": source_sha, "source_records": len(rows),
        "source_tokens": source["tokens"],
        "parent": {"manifest_sha256": parent_sha, "identity": parent["identity"],
                   "progress": progress, "world_size": parent["world_size"]},
        "retained_prefix_count": len(prefix), "retained_suffix_count": len(kept),
        "skipped_suffix_count": len(suffix) - len(kept),
        "total_records": len(selected), "total_optimizer_steps": math.ceil(len(selected) / 2),
        "total_tokens": sum(rows[i]["length"] for i in selected),
        "ordered_indices": selected,
        "ordered_trajectory_ids_sha256": fingerprint([rows[i]["trajectory_id"] for i in selected]),
    }


def build_plan(index, parent_manifest, *, max_tokens=28000, seed=42):
    parent_manifest = Path(parent_manifest)
    return _make_plan(index, json.loads(parent_manifest.read_text()), sha256_file(parent_manifest),
                      max_tokens=max_tokens, seed=seed)


def validate_plan(index, plan):
    if (plan.get("format_version") != 1
            or plan.get("policy") != "preserve_consumed_prefix_filter_suffix"):
        raise ValueError("Unknown continuation policy")
    parent = {"format_version": 1, "complete": True, **plan["parent"]}
    expected = _make_plan(index, parent, plan["parent"]["manifest_sha256"],
                          max_tokens=plan["max_tokens"], seed=plan["seed"])
    if plan != expected:
        raise ValueError("Continuation plan differs from the source/order/boundary")


def load_plan(path, *, expected_sha256):
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Continuation plan checksum differs")
    return json.loads(path.read_text())


def check_resume_manifest(checkpoint, plan):
    """Accept the exact parent or a checkpoint explicitly carrying this policy."""
    path = Path(checkpoint) / "manifest.json"
    manifest = json.loads(path.read_text())
    if (sha256_file(path) == plan["parent"]["manifest_sha256"]
            and manifest["identity"] == plan["parent"]["identity"]
            and manifest["progress"] == plan["parent"]["progress"]
            and manifest["world_size"] == plan["parent"]["world_size"]):
        return "parent"
    continuation = manifest.get("identity", {}).get("settings", {}).get("continuation", {})
    if continuation.get("plan_sha256") == fingerprint(plan):
        return "continuation"
    raise ValueError("Checkpoint is neither the approved parent nor this filtered continuation")


def publish_continuation(publisher, checkpoint, plan, child_identity, original_publish):
    """Original atomic Hub protocol with ONE exact parent-to-child transition.

    Keep the same dedicated prefix and retention budget. The immutable original
    publisher refuses all identity changes; this protocol additionally accepts
    the byte-exact approved parent as remote latest when publishing its child.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

    from workflow_distillation.training_checkpoints import _NAME, _manifest

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
    pointer = hf_hub_download(publisher.repo_id, pointer_name, revision=revision,
                              repo_type=publisher.repo_type)
    name = json.loads(Path(pointer).read_text())["checkpoint"]
    if not _NAME.fullmatch(name):
        raise ValueError("Invalid remote checkpoint name")
    remote_path = hf_hub_download(publisher.repo_id, f"{publisher.prefix}/{name}/manifest.json",
                                 revision=revision, repo_type=publisher.repo_type)
    remote = json.loads(Path(remote_path).read_text())
    exact_parent = (sha256_file(Path(remote_path)) == plan["parent"]["manifest_sha256"]
                    and remote["identity"] == plan["parent"]["identity"]
                    and remote["progress"] == plan["parent"]["progress"])
    if (remote["world_size"] != manifest["world_size"]
            or (remote["identity"] != manifest["identity"] and not exact_parent)):
        raise ValueError("Remote checkpoint identity differs from child or exact approved parent")
    if remote == manifest:
        publisher._squash(api, revision)
        return
    if remote["progress"]["global_step"] >= manifest["progress"]["global_step"]:
        raise ValueError("Remote checkpoint is already at this step or newer")
    operations = [CommitOperationAdd(
        path_in_repo=f"{publisher.prefix}/{checkpoint.name}/{p.name}", path_or_fileobj=p,
    ) for p in sorted(checkpoint.iterdir()) if p.is_file()]
    operations.append(CommitOperationAdd(path_in_repo=pointer_name,
                      path_or_fileobj=json.dumps({"checkpoint": checkpoint.name}).encode()))
    if publisher.keep_last is not None:
        names = {PurePosixPath(n).parts[-2] for n in files
                 if n.startswith(f"{publisher.prefix}/checkpoint-")}
        if any(int(n.split("-")[1]) > manifest["progress"]["global_step"] for n in names):
            raise ValueError("Remote contains a checkpoint newer than the latest pointer")
        retained = set(sorted(names | {checkpoint.name}, key=lambda n: int(n.split("-")[1]))[
            -publisher.keep_last:])
        operations.extend(CommitOperationDelete(path_in_repo=n) for n in files
                          if n.startswith(f"{publisher.prefix}/checkpoint-")
                          and PurePosixPath(n).parts[-2] not in retained)
    committed = api.create_commit(repo_id=publisher.repo_id, repo_type=publisher.repo_type,
                                  operations=operations, parent_commit=revision,
                                  commit_message=f"Save training checkpoint {checkpoint.name}")
    publisher._squash(api, committed.oid)


def configure_trainer(trainer, plan, *, entrypoint):
    """Adapt the immutable epoch trainer's imported interfaces, not its source.

    Parent state is loaded with the SAME strict identity and payload checks as
    before. Only the one parent manifest named by the plan may migrate. Future
    saves/resumes use a new identity recording the plan and this implementation.
    """
    plan = copy.deepcopy(plan)
    plan_sha = fingerprint(plan)
    original_dataset, original_check = trainer.EpochDataset, trainer.check_dataset
    original_identity, original_manager = trainer.training_identity, trainer.CheckpointManager
    original_publish = trainer.HuggingFacePublisher.__call__
    original_script = Path(trainer.__file__).resolve()
    entrypoint = Path(entrypoint).resolve()
    stamp = {"plan_sha256": plan_sha, "max_tokens": plan["max_tokens"],
             "parent_manifest_sha256": plan["parent"]["manifest_sha256"],
             "parent_global_step": plan["parent"]["progress"]["global_step"],
             "implementation_sha256": sha256_file(Path(__file__))}
    expected_child_identity = copy.deepcopy(plan["parent"]["identity"])
    expected_child_identity["script_sha256"] = sha256_file(entrypoint)
    expected_child_identity["settings"]["continuation"] = stamp

    class FilteredDataset(original_dataset):
        def __init__(self, index_path, *, verify=True):
            super().__init__(index_path, verify=verify)
            original_check(self)
            validate_plan(index_path, plan)
            self.records = [self.records[i] for i in plan["ordered_indices"]]
            self.manifest = {**self.manifest, "data": self.records,
                             "records": plan["total_records"], "tokens": plan["total_tokens"],
                             "source_tokens": plan["source_tokens"],
                             "selection": plan["policy"], "continuation": stamp,
                             "group_counts": [sum(row["group_counts"][g] for row in self.records)
                                              for g in range(5)]}

    def check_dataset(dataset):
        if (dataset.manifest.get("continuation") != stamp
                or len(dataset) != plan["total_records"]
                or any(row["length"] >= plan["max_tokens"] for row in
                       dataset.records[plan["retained_prefix_count"]:])):
            raise ValueError("Filtered continuation dataset differs")

    def batches(count, seed=42):
        if seed != plan["seed"] or count != plan["total_records"]:
            raise ValueError("Continuation count/seed differs")
        # The view is ALREADY in the approved shuffled-prefix/filtered-suffix order.
        return [list(range(i, min(i + 2, count))) for i in range(0, count, 2)]

    def identity(script, **kwargs):
        # Check all parent math/config/package hashes before permitting migration.
        parent_expected = original_identity(original_script, **kwargs)
        if parent_expected != plan["parent"]["identity"]:
            raise ValueError("Continuation parent numerical identity differs")
        if Path(script).resolve() != entrypoint:
            raise ValueError("Unexpected continuation entrypoint")
        return original_identity(script, **{**kwargs, "settings": {
            **kwargs["settings"], "continuation": stamp}})

    class ContinuationManager(original_manager):
        def load(self, checkpoint, **kwargs):
            from workflow_distillation.training_checkpoints import latest_checkpoint

            path = (latest_checkpoint(self.directory)
                    if str(checkpoint) == "latest" else Path(checkpoint))
            kind = check_resume_manifest(path, plan)
            if kind == "continuation":
                return super().load(path, **kwargs)  # Normal exact child identity/payload checks.
            child_identity = self.identity
            try:
                self.identity = plan["parent"]["identity"]
                progress = super().load(path, **kwargs)  # Exact old identity AND payload checks.
            finally:
                self.identity = child_identity
            print(json.dumps({"event": "continuation_parent_loaded", "plan_sha256": plan_sha,
                              "max_tokens": plan["max_tokens"], **progress}), flush=True)
            return progress

    class ContinuationPublisher(trainer.HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, plan,
                                        expected_child_identity, original_publish)

    trainer.EpochDataset, trainer.check_dataset = FilteredDataset, check_dataset
    trainer.epoch_batches, trainer.training_identity = batches, identity
    trainer.CheckpointManager = ContinuationManager
    trainer.HuggingFacePublisher = ContinuationPublisher
    # Launcher subprocesses and checkpoint identity must refer to the NEW entrypoint.
    trainer.__file__ = str(entrypoint)
