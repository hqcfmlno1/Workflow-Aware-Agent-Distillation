"""Append the remaining filtered pool without changing an approved prefix."""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from workflow_distillation.diverse_continuation import (
    _source,
    coverage,
    fingerprint,
    repository,
    validate_plan,
)
from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.epoch_training import validate_progress
from workflow_distillation.vast_microbatch import batch_schedule


def extend_order(rows, source_order, prefix, max_tokens):
    if (len(source_order) != len(set(source_order)) or len(prefix) != len(set(prefix))
            or not set(prefix) <= set(source_order)
            or any(type(i) is not int or not 0 <= i < len(rows) for i in source_order)):
        raise ValueError("Source or preserved cohort has invalid/duplicate indices")
    chosen = set(prefix)
    seen = {rows[i]["task_key"] for i in prefix}
    counts = Counter(repository(rows[i]) for i in prefix)
    fresh, repeats = defaultdict(deque), defaultdict(deque)
    for position, index in enumerate(source_order):
        if index in chosen:
            continue
        row = rows[index]
        if not 2 <= row["length"] < max_tokens:
            raise ValueError("Pending pool trajectory exceeds token threshold")
        bucket = repeats if row["task_key"] in seen else fresh
        bucket[repository(row)].append((position, index))
        seen.add(row["task_key"])
    result = list(prefix)
    for buckets in (fresh, repeats):
        queue = [(counts[repo], items[0][0], repo) for repo, items in buckets.items()]
        heapq.heapify(queue)
        while queue:
            _, _, repo = heapq.heappop(queue)
            _, index = buckets[repo].popleft()
            result.append(index)
            counts[repo] += 1
            if buckets[repo]:
                heapq.heappush(queue, (counts[repo], buckets[repo][0][0], repo))
    return result


def build_pool_plan(index_path, source_path, cohort_path):
    rows, source = _source(index_path, source_path)
    cohort = json.loads(Path(cohort_path).read_text())
    validate_plan(index_path, source_path, cohort)
    ordered = extend_order(rows, source["ordered_indices"], cohort["ordered_indices"],
                           source["max_tokens"])
    return {
        "format_version": 1,
        "policy": "preserve_full_cohort_append_unseen_tasks_then_repeats_balance_repos",
        "source_index_sha256": sha256_file(Path(index_path)),
        "source_plan_sha256": sha256_file(Path(source_path)),
        "original_cohort_sha256": sha256_file(Path(cohort_path)),
        "preserved_cohort_count": len(cohort["ordered_indices"]),
        "preserved_cohort_indices_sha256": fingerprint(cohort["ordered_indices"]),
        "seed": source["seed"], "max_tokens": source["max_tokens"],
        "total_records": len(ordered), "ordered_indices": ordered,
        "ordered_trajectory_ids_sha256": fingerprint([rows[i]["trajectory_id"] for i in ordered]),
        "coverage": coverage([rows[i] for i in ordered]),
        "added_coverage": coverage([rows[i] for i in ordered[len(cohort["ordered_indices"]):]]),
    }


def manifest_digest(manifest):
    # CheckpointManager commits manifests with precisely this serialization.
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def extension_identity(parent, parent_sha, plan, entrypoint):
    if (parent.get("complete") is not True or parent.get("world_size") != 2
            or manifest_digest(parent) != parent_sha):
        raise ValueError("Pinned pool migration parent manifest differs")
    old_identity = parent["identity"]
    if "pool_continuation" in old_identity["settings"]:
        raise ValueError("Pool migration parent must be the original batch-four child")
    base_progress = old_identity["settings"]["microbatch_transition"]["parent_progress"]
    prefix = plan["preserved_cohort_count"]
    if (prefix - base_progress["samples_consumed"]) % 4:
        raise ValueError("Preserved cohort ends with a partial batch; cannot regroup it")
    validate_progress(batch_schedule(prefix, base_progress), parent["progress"])
    result = copy.deepcopy(old_identity)
    result["script_sha256"] = sha256_file(Path(entrypoint))
    result["settings"]["pool_continuation"] = {
        "parent_manifest_sha256": parent_sha,
        "parent_manifest": copy.deepcopy(parent),
        "plan_sha256": fingerprint(plan),
        "implementation_sha256": sha256_file(Path(__file__)),
        "preserved_cohort_count": prefix,
        "total_records": len(plan["ordered_indices"]),
    }
    return result, {"parent": {
        "manifest_sha256": parent_sha,
        **{k: copy.deepcopy(parent[k]) for k in ("identity", "progress", "world_size")},
    }}
