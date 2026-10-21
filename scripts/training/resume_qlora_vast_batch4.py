"""Resume an authenticated, pinned batch-four child after ancestor retention.

The original numerical worker and its complete saved identity are preserved.
Only the launcher preflight no longer needs the ancestor's optimizer payload.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ORIGINAL = Path(__file__).with_name("train_qlora_vast_batch4.py")
spec = importlib.util.spec_from_file_location("original_vast_batch4", ORIGINAL)
batch4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch4
spec.loader.exec_module(batch4)

from workflow_distillation.vast_existing_child import existing_child_binding  # noqa: E402

original_parser = batch4.parser


def parser():
    result = original_parser()
    result.description = __doc__
    checksum = os.environ.get("FIRST_RUN_RESUME_MANIFEST_SHA256")
    result.add_argument("--resume-manifest-sha256", default=checksum, required=not checksum)
    return result


def prepare(args):
    from filtered_continuation import publish_continuation

    if not args.resume_from or args.resume_from == "latest":
        raise ValueError("Existing-child resume requires an explicit pinned checkpoint")
    path = Path(args.resume_from)
    checksum = batch4.sha256_file(path / "manifest.json")
    if checksum != args.resume_manifest_sha256:
        raise ValueError("Resume manifest checksum differs")
    os.environ["FIRST_RUN_RESUME_MANIFEST_SHA256"] = checksum
    resumed = batch4._manifest(path)  # Verify every current state file before model allocation.
    local_rank = os.environ.pop("LOCAL_RANK", None)
    try:
        trainer, cohort, _, baseline_identity = batch4.baseline.prepare(args)
    finally:
        if local_rank is not None:
            os.environ["LOCAL_RANK"] = local_rank
    binding, identity = existing_child_binding(
        resumed, baseline_identity, args.batch_parent_manifest_sha256, ORIGINAL,
        cohort["total_records"],
    )
    batches = batch4.batch_schedule(cohort["total_records"], binding["parent"]["progress"])
    dataset = trainer.EpochDataset(args.index, verify=False)
    if any(r["length"] >= 28000
           for r in dataset.records[binding["parent"]["progress"]["samples_consumed"]:]):
        raise ValueError("Untrained cohort must remain below 28000 tokens")

    def schedule(count, seed=42):
        if count != cohort["total_records"] or seed != cohort["seed"]:
            raise ValueError("Cohort count or seed differs")
        return batches

    original_publish = batch4.HuggingFacePublisher.__call__

    class Publisher(batch4.HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, binding, identity, original_publish)

    trainer.CONFIG = dict(identity["settings"]["config"])
    trainer.CheckpointManager = batch4.migration_manager(binding)
    trainer.HuggingFacePublisher = Publisher
    trainer.epoch_batches = schedule
    if local_rank is None:
        (args.output / "resume-audit.json").write_text(json.dumps({
            "mode": "existing_exact_child",
            "resume_checkpoint": str(path),
            "resume_manifest_sha256": checksum,
            "historical_parent_payload_required": False,
            "original_entrypoint_sha256": batch4.sha256_file(ORIGINAL),
            "resume_entrypoint_sha256": batch4.sha256_file(Path(__file__)),
            "validation_source_sha256": batch4.sha256_file(
                Path(sys.modules[existing_child_binding.__module__].__file__)),
        }, indent=2) + "\n")
    return trainer, {**cohort, "target_optimizer_steps": len(batches)}, binding, identity


if __name__ == "__main__":
    batch4.parser = parser
    batch4.prepare = prepare
    # Original main launches this preflight for each rank; its worker is unchanged.
    batch4.__file__ = __file__
    batch4.main()
