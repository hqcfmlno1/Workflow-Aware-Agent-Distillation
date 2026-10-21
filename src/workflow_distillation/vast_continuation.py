"""Explicit migration from one pinned filtered checkpoint to the diversity schedule.

The source trainer, numerical code and strict checkpoint loader remain intact.
Only the pinned parent may cross the identity boundary; later resumes are exact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from workflow_distillation.diverse_continuation import fingerprint, validate_plan
from workflow_distillation.epoch_data import sha256_file


def bind_parent(index, source_plan, cohort, manifest_path, *, expected_sha256, filtered_entrypoint):
    import filtered_continuation
    from filtered_continuation import validate_plan as validate_filtered

    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("Pinned parent manifest checksum differs")
    source = json.loads(Path(source_plan).read_text())
    validate_filtered(index, source)
    validate_plan(index, source_plan, cohort)
    manifest = json.loads(Path(manifest_path).read_text())
    stamp = {
        "plan_sha256": fingerprint(source),
        "max_tokens": source["max_tokens"],
        "parent_manifest_sha256": source["parent"]["manifest_sha256"],
        "parent_global_step": source["parent"]["progress"]["global_step"],
        "implementation_sha256": sha256_file(Path(filtered_continuation.__file__)),
    }
    expected = copy.deepcopy(source["parent"]["identity"])
    expected["script_sha256"] = sha256_file(filtered_entrypoint)
    expected["settings"]["continuation"] = stamp
    if (
        manifest.get("format_version") != 1
        or manifest.get("complete") is not True
        or manifest.get("identity") != expected
        or manifest.get("progress") != cohort["progress"]
        or manifest.get("world_size") != source["parent"]["world_size"]
    ):
        raise ValueError("Parent identity, continuation policy or cursor differs")
    root = Path(__file__).resolve().parents[2]
    hooks = root / "cloud/kaggle/recovery"
    return {
        "parent": {
            "manifest_sha256": expected_sha256,
            "identity": expected,
            "progress": manifest["progress"],
            "world_size": manifest["world_size"],
        },
        "baseline_identity": source["parent"]["identity"],
        "policy": {
            "cohort_sha256": fingerprint(cohort),
            "source_plan_sha256": sha256_file(source_plan),
            "activation_offload": "CPU pinned",
            "gradient_sync": "manual_mean",
            "ddp_bucket_bytes": 0,
            "execution": "host_memory_adaptive_parallel_or_serial",
            "platform": "Vast RTX3090 x2",
            "sources": {
                str(p.relative_to(root)): sha256_file(p)
                for p in (
                    Path(__file__),
                    Path(__file__).with_name("diverse_continuation.py"),
                    hooks / "adaptive_memory.py",
                    hooks / "host_memory_recovery.py",
                )
            },
        },
    }


def configure_trainer(trainer, cohort, binding, *, entrypoint, stop_at_step=None, stop_file=None):
    from filtered_continuation import publish_continuation

    from workflow_distillation.training_checkpoints import latest_checkpoint

    cohort, binding = copy.deepcopy(cohort), copy.deepcopy(binding)
    original_dataset, original_check = trainer.EpochDataset, trainer.check_dataset
    original_identity, original_manager = trainer.training_identity, trainer.CheckpointManager
    original_publish = trainer.HuggingFacePublisher.__call__
    original_script = Path(trainer.__file__).resolve()
    entrypoint = Path(entrypoint).resolve()
    child_identity = copy.deepcopy(binding["parent"]["identity"])
    child_identity["script_sha256"] = sha256_file(entrypoint)
    child_identity["settings"]["vast_continuation"] = binding["policy"]
    if stop_at_step is not None and not (
        cohort["progress"]["global_step"] < stop_at_step <= cohort["target_optimizer_steps"]
    ):
        raise ValueError("Invalid bounded pilot stop step")

    class SelectedDataset(original_dataset):
        def __init__(self, index_path, *, verify=True):
            super().__init__(index_path, verify=verify)
            original_check(self)
            if sha256_file(index_path) != cohort["source_index_sha256"]:
                raise ValueError("Cohort source index differs")
            self.records = [self.records[i] for i in cohort["ordered_indices"]]
            self.manifest = {
                **self.manifest,
                "data": self.records,
                "records": len(self.records),
                "tokens": cohort["coverage"]["selected"]["tokens"],
                "selection": cohort["policy"],
                "vast_continuation": binding["policy"],
            }

    def check_dataset(dataset):
        if (
            dataset.manifest.get("vast_continuation") != binding["policy"]
            or len(dataset) != cohort["total_records"]
            or any(
                r["length"] >= cohort["max_tokens"]
                for r in dataset.records[cohort["retained_prefix_count"] :]
            )
        ):
            raise ValueError("Diversity dataset differs")

    def batches(count, seed=42):
        if count != cohort["total_records"] or seed != cohort["seed"]:
            raise ValueError("Diversity count or seed differs")
        return [list(range(i, i + 2)) for i in range(0, count, 2)]

    def identity(script, **kwargs):
        if (
            Path(script).resolve() != entrypoint
            or original_identity(original_script, **kwargs) != binding["baseline_identity"]
        ):
            raise ValueError("Parent numerical identity differs")
        return copy.deepcopy(child_identity)

    class VastManager(original_manager):
        def load(self, checkpoint, **kwargs):
            path = (
                latest_checkpoint(self.directory)
                if str(checkpoint) == "latest"
                else Path(checkpoint)
            )
            # The original loader still verifies every payload hash, tensor signature,
            # optimizer/scaler and per-rank RNG state. Only the exact parent may migrate.
            is_parent = sha256_file(path / "manifest.json") == binding["parent"]["manifest_sha256"]
            current = self.identity
            try:
                if is_parent:
                    self.identity = binding["parent"]["identity"]
                return super().load(path, **kwargs)
            finally:
                self.identity = current

    class VastPublisher(trainer.HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, binding, child_identity, original_publish)

    if stop_at_step is not None or stop_file is not None:
        original_loop = trainer.run_epoch_loop

        def bounded_loop(batches, *, progress, step, stop, **kwargs):
            completed = progress["global_step"]

            def bounded_step(batch, number):
                nonlocal completed
                result = step(batch, number)
                completed = number
                return result

            return original_loop(
                batches,
                progress=progress,
                step=bounded_step,
                stop=lambda: (
                    (stop_at_step is not None and completed >= stop_at_step)
                    or (stop_file is not None and Path(stop_file).exists())
                    or stop()
                ),
                **kwargs,
            )

        trainer.run_epoch_loop = bounded_loop
    trainer.EpochDataset, trainer.check_dataset = SelectedDataset, check_dataset
    trainer.epoch_batches, trainer.training_identity = batches, identity
    trainer.CheckpointManager, trainer.HuggingFacePublisher = VastManager, VastPublisher
    trainer.__file__ = str(entrypoint)
