"""Continue the original cohort, then the remaining filtered pool, exactly once."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ORIGINAL = Path(__file__).with_name("train_qlora_vast_batch4.py")
spec = importlib.util.spec_from_file_location("pool_original_batch4", ORIGINAL)
batch4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch4
spec.loader.exec_module(batch4)

from workflow_distillation.diverse_continuation import fingerprint  # noqa: E402
from workflow_distillation.epoch_data import EpochDataset  # noqa: E402
from workflow_distillation.vast_existing_child import existing_child_binding  # noqa: E402
from workflow_distillation.vast_pool_continuation import (  # noqa: E402
    build_pool_plan,
    extension_identity,
    manifest_digest,
)

original_parser = batch4.parser
ENV_FIELDS = {
    "pool-plan": "FIRST_RUN_POOL_PLAN",
    "pool-parent-manifest-sha256": "FIRST_RUN_POOL_PARENT_SHA256",
}


def parser():
    result = original_parser()
    result.description = __doc__
    for flag, variable in ENV_FIELDS.items():
        value = os.environ.get(variable)
        result.add_argument("--" + flag, default=value, required=not value)
    result.add_argument("--pool-parent-manifest", type=Path)
    return result


def prepare(args):
    from filtered_continuation import publish_continuation

    for flag, variable in ENV_FIELDS.items():
        os.environ[variable] = str(getattr(args, flag.replace("-", "_")))
    if not args.resume_from or args.resume_from == "latest":
        raise ValueError("Use an explicit committed resume checkpoint")
    resumed = batch4._manifest(Path(args.resume_from))
    if args.pool_parent_manifest:
        parent = json.loads(args.pool_parent_manifest.read_text())
        if batch4.sha256_file(args.pool_parent_manifest) != args.pool_parent_manifest_sha256:
            raise ValueError("Pool parent manifest checksum differs")
    elif manifest_digest(resumed) == args.pool_parent_manifest_sha256:
        parent = resumed
    else:
        parent = resumed["identity"]["settings"]["pool_continuation"]["parent_manifest"]
    if manifest_digest(parent) != args.pool_parent_manifest_sha256:
        raise ValueError("Embedded pool parent manifest checksum differs")
    local_rank = os.environ.pop("LOCAL_RANK", None)
    try:
        trainer, cohort, _, baseline_identity = batch4.baseline.prepare(args)
    finally:
        if local_rank is not None:
            os.environ["LOCAL_RANK"] = local_rank
    existing_child_binding(parent, baseline_identity, args.batch_parent_manifest_sha256,
                           ORIGINAL, cohort["total_records"])
    plan = json.loads(Path(args.pool_plan).read_text())
    if plan != build_pool_plan(args.index, args.source_plan, args.cohort):
        raise ValueError("Pool plan, preserved cohort, or source schedule differs")
    identity, binding = extension_identity(parent, args.pool_parent_manifest_sha256,
                                           plan, Path(__file__))
    if (manifest_digest(resumed) != args.pool_parent_manifest_sha256
            and resumed["identity"] != identity):
        raise ValueError("Resume is neither the exact pool parent nor its exact child")
    batches = batch4.batch_schedule(
        plan["total_records"],
        parent["identity"]["settings"]["microbatch_transition"]["parent_progress"],
    )
    batch4.validate_progress(batches, resumed["progress"])
    stamp = fingerprint(plan)

    class PoolDataset(EpochDataset):
        def __init__(self, index_path, *, verify=True):
            super().__init__(index_path, verify=verify)
            if batch4.sha256_file(index_path) != plan["source_index_sha256"]:
                raise ValueError("Pool source index checksum differs")
            self.records = [self.records[i] for i in plan["ordered_indices"]]
            self.manifest = {**self.manifest, "data": self.records,
                             "records": len(self.records), "tokens": plan["coverage"]["tokens"],
                             "selection": plan["policy"], "pool_plan_sha256": stamp}

    def check_dataset(dataset):
        if (dataset.manifest.get("pool_plan_sha256") != stamp
                or len(dataset) != plan["total_records"]
                or fingerprint([r["trajectory_id"] for r in dataset.records])
                != plan["ordered_trajectory_ids_sha256"]
                or any(r["length"] >= plan["max_tokens"] for r in
                       dataset.records[cohort["retained_prefix_count"]:])):
            raise ValueError("Expanded pool dataset differs")

    def schedule(count, seed=42):
        if count != plan["total_records"] or seed != plan["seed"]:
            raise ValueError("Expanded pool count or seed differs")
        return batches

    original_publish = batch4.HuggingFacePublisher.__call__

    class Publisher(batch4.HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, binding, identity, original_publish)

    trainer.CONFIG = dict(identity["settings"]["config"])
    trainer.CheckpointManager = batch4.migration_manager(binding)
    trainer.HuggingFacePublisher = Publisher
    trainer.EpochDataset = PoolDataset
    trainer.check_dataset = check_dataset
    trainer.epoch_batches = schedule
    if local_rank is None:
        (args.output / "pool-resume-audit.json").write_text(json.dumps({
            "parent_manifest_sha256": args.pool_parent_manifest_sha256,
            "parent_progress": parent["progress"], "resume_progress": resumed["progress"],
            "preserved_cohort_count": plan["preserved_cohort_count"],
            "pool_records": plan["total_records"], "total_optimizer_steps": len(batches),
            "pool_plan_sha256": stamp, "identity_changed": True,
            "numerical_worker_changed": False,
        }, indent=2) + "\n")
    return trainer, {**cohort, "total_records": plan["total_records"],
                     "target_optimizer_steps": len(batches)}, binding, identity


if __name__ == "__main__":
    batch4.parser = parser
    batch4.prepare = prepare
    batch4.__file__ = __file__
    batch4.main()
