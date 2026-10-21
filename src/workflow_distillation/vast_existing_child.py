"""Validate an existing child without restoring a retained-away ancestor."""

from __future__ import annotations

import copy

from workflow_distillation.epoch_training import validate_progress
from workflow_distillation.vast_microbatch import batch_schedule, transition_identity


def existing_child_binding(manifest, baseline_identity, parent_sha, entrypoint, total_records=4000):
    if manifest.get("complete") is not True or manifest.get("world_size") != 2:
        raise ValueError("Expected a complete two-rank child checkpoint")
    stamp = manifest["identity"]["settings"]["microbatch_transition"]
    if stamp["parent_manifest_sha256"] != parent_sha:
        raise ValueError("Child refers to a different batch-two parent")
    # This is historical metadata, never a fabricated checkpoint or load source.
    parent = {
        "complete": True,
        "world_size": 2,
        "identity": copy.deepcopy(baseline_identity),
        "progress": copy.deepcopy(stamp["parent_progress"]),
    }
    expected = transition_identity(parent, baseline_identity, parent_sha, entrypoint)
    if manifest["identity"] != expected:
        raise ValueError("Existing child numerical identity differs")
    schedule = batch_schedule(total_records, parent["progress"])
    validate_progress(schedule, manifest["progress"])
    if manifest["progress"]["global_step"] <= parent["progress"]["global_step"]:
        raise ValueError("Expected a checkpoint after the batch-four transition")
    return {"parent": {"manifest_sha256": parent_sha,
                       **{k: parent[k] for k in ("identity", "progress", "world_size")}}}, expected
