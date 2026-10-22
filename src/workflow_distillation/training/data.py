"""Approved dataset views and deterministic schedules, without trainer mutation."""

from __future__ import annotations

import json
from pathlib import Path

from workflow_distillation.epoch_data import EpochDataset, sha256_file
from workflow_distillation.epoch_training import validate_progress
from workflow_distillation.vast_pool_continuation import build_pool_plan


def read_pool_plan(index, source_plan, cohort, pool_plan):
    saved = json.loads(Path(pool_plan).read_text())
    if saved != build_pool_plan(index, source_plan, cohort):
        raise ValueError("Pool plan differs from approved source, cohort or order")
    return saved


def batch_schedule(count, per_gpu_batch_size, transition=None):
    if type(count) is not int or count < 1 or per_gpu_batch_size not in (1, 2):
        raise ValueError("Positive dataset count and verified batch size required")
    cursor = 0
    batches = []
    if transition is not None:
        if per_gpu_batch_size != 2:
            raise ValueError("Cannot revert the batch-four transition to batch two")
        cursor = transition.get("samples_consumed")
        step = transition.get("global_step")
        if (
            type(cursor) is not int
            or type(step) is not int
            or cursor != 2 * step
            or step < 0
            or cursor > count
            or transition.get("next_batch") != step
        ):
            raise ValueError("Invalid batch-two transition cursor")
        batches = [list(range(i, i + 2)) for i in range(0, cursor, 2)]
        validate_progress(batches, transition)
    width = 2 * per_gpu_batch_size
    return batches + [list(range(i, min(i + width, count))) for i in range(cursor, count, width)]


class TrainingDataset(EpochDataset):
    def __init__(self, index, plan, *, verify=True):
        super().__init__(index, verify=verify)
        if sha256_file(Path(index)) != plan["source_index_sha256"]:
            raise ValueError("Dataset index checksum differs")
        if self.manifest["weights"] != [0, 1, 0.75, 1.5, 2]:
            raise ValueError("Dataset loss weights differ")
        indices = plan["ordered_indices"]
        if len(set(indices)) != len(indices) or len(indices) != plan["total_records"]:
            raise ValueError("Dataset plan has duplicate or missing trajectories")
        self.records = [self.records[i] for i in indices]
        if any(r["length"] != r["full_token_count"] or r["denominator"] <= 0 for r in self.records):
            raise ValueError("Expected complete trajectories with positive loss denominator")
        self.manifest = {**self.manifest, "data": self.records, "records": len(self.records)}
