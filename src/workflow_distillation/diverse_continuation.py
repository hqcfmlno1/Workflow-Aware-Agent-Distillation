"""Deterministic selection of new tasks while preserving a consumed training prefix."""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

from workflow_distillation.epoch_data import epoch_batches, sha256_file
from workflow_distillation.epoch_training import validate_progress


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def repository(row):
    repo, separator, number = row["task_key"].rpartition("#")
    if not separator or "/" not in repo or not number.isdigit() or repo != repo.lower():
        raise ValueError("Expected a canonical lower-case repository#PR task key")
    return repo


def coverage(rows):
    tasks = Counter(row["task_key"] for row in rows)
    repos = Counter(repository(row) for row in rows)
    return {
        "trajectories": len(rows), "unique_tasks": len(tasks), "unique_repos": len(repos),
        "repeated_task_trajectories": len(rows) - len(tasks),
        "tokens": sum(row["length"] for row in rows),
        "top10_repo_fraction": sum(n for _, n in repos.most_common(10)) / len(rows) if rows else 0,
        "top_repositories": [[repo, count] for repo, count in
                             sorted(repos.items(), key=lambda item: (-item[1], item[0]))[:10]],
    }


def _source(index_path, source_plan_path):
    index = json.loads(Path(index_path).read_text())
    source = json.loads(Path(source_plan_path).read_text())
    rows = index["data"]
    if (source.get("format_version") != 1
            or source.get("policy") != "preserve_consumed_prefix_filter_suffix"
            or source["source_index_sha256"] != sha256_file(Path(index_path))
            or index["records"] != len(rows) or source["source_records"] != len(rows)
            or index["tokens"] != sum(row["length"] for row in rows)
            or source["source_tokens"] != index["tokens"]
            or type(source["max_tokens"]) is not int or not 2 < source["max_tokens"] <= 30000):
        raise ValueError("Source index or filtered schedule identity differs")
    identifiers = [row["trajectory_id"] for row in rows]
    if len(set(identifiers)) != len(rows):
        raise ValueError("Duplicate source trajectory IDs")
    for row in rows:
        repository(row)
        if not 2 <= row["length"] < 30000 or row["length"] != row["full_token_count"]:
            raise ValueError("Source must retain complete, untruncated trajectories below30k")
    batches = epoch_batches(len(rows), source["seed"])
    validate_progress(batches, source["parent"]["progress"])
    boundary = source["parent"]["progress"]["samples_consumed"]
    order = [i for batch in batches for i in batch]
    expected = order[:boundary] + [i for i in order[boundary:]
                                  if rows[i]["length"] < source["max_tokens"]]
    if (source["retained_prefix_count"] != boundary or source["ordered_indices"] != expected
            or source["total_records"] != len(expected)
            or source["total_tokens"] != sum(rows[i]["length"] for i in expected)):
        raise ValueError("Source schedule changed its trained prefix or filtered suffix")
    return rows, source


def build_plan(index_path, source_plan_path, *, resume_step, target_step):
    """Select a total cohort ending at target_step, with one new task per new sample.

    The existing schedule determines the consumed prefix and candidate tie order.
    Among unconsumed, previously unseen tasks, choose the earliest trajectory of
    each task. Repeatedly select from the repository with the fewest total samples,
    including the fixed prefix. Ties follow the original candidate order. This
    is a deterministic greedy balance, not a guarantee of optimal model quality.

    This manifest selects data only. A launcher must separately verify and bind
    the actual checkpoint and perform any strict training-identity migration.
    """
    rows, source = _source(index_path, source_plan_path)
    count = len(source["ordered_indices"])
    if (type(resume_step) is not int or type(target_step) is not int
            or not 0 <= resume_step < target_step or resume_step * 2 > count
            or resume_step * 2 < source["retained_prefix_count"]):
        raise ValueError("Invalid resume/target optimizer boundary")
    consumed, target = resume_step * 2, target_step * 2
    order = source["ordered_indices"]
    prefix, pending = order[:consumed], order[consumed:]
    seen_tasks = {rows[i]["task_key"] for i in prefix}
    prefix_rows = [rows[i] for i in prefix]
    repo_counts = Counter(repository(row) for row in prefix_rows)
    candidates = defaultdict(deque)
    for position, index in enumerate(pending):
        row = rows[index]
        if row["length"] >= source["max_tokens"]:
            raise ValueError("Pending trajectory exceeds the strict token threshold")
        if row["task_key"] in seen_tasks:
            continue
        seen_tasks.add(row["task_key"])
        candidates[repository(row)].append((position, index))
    available = sum(map(len, candidates.values()))
    if available < target - consumed:
        raise ValueError(f"Not enough unseen tasks: need {target - consumed}, have {available}")
    queue = [(repo_counts[repo], tasks[0][0], repo) for repo, tasks in candidates.items()]
    heapq.heapify(queue)
    selected = []
    while len(selected) < target - consumed:
        _, _, repo = heapq.heappop(queue)
        _, index = candidates[repo].popleft()
        selected.append(index)
        repo_counts[repo] += 1
        if candidates[repo]:
            heapq.heappush(queue, (repo_counts[repo], candidates[repo][0][0], repo))
    ordered = prefix + selected
    chosen = [rows[i] for i in ordered]
    added = [rows[i] for i in selected]
    return {
        "format_version": 1, "policy": "preserve_prefix_unseen_tasks_balance_repositories",
        "source_index_sha256": sha256_file(Path(index_path)),
        "source_plan_sha256": sha256_file(Path(source_plan_path)),
        "source_plan_fingerprint": fingerprint(source),
        "source_records": len(rows), "source_schedule_records": count,
        "seed": source["seed"], "max_tokens": source["max_tokens"],
        "progress": {"global_step": resume_step, "next_batch": resume_step,
                     "samples_consumed": consumed},
        "target_optimizer_steps": target_step, "total_records": target,
        "remaining_optimizer_steps": target_step - resume_step,
        "retained_prefix_count": consumed, "new_trajectory_count": len(selected),
        "new_unique_tasks_available": available,
        "ordered_indices": ordered,
        "ordered_trajectory_ids_sha256": fingerprint([row["trajectory_id"] for row in chosen]),
        "consumed_trajectory_ids_sha256": fingerprint(
            [row["trajectory_id"] for row in prefix_rows]),
        "new_max_tokens": max(row["length"] for row in added),
        "checkpoint_binding_required": True,
        "coverage": {
            "consumed": coverage(prefix_rows),
            "original_target": coverage([rows[i] for i in order[:target]]),
            "selected": coverage(chosen), "added": coverage(added),
            "new_repositories": len({repository(row) for row in added}
                                    - {repository(row) for row in prefix_rows}),
        },
    }


def validate_plan(index_path, source_plan_path, plan):
    expected = build_plan(index_path, source_plan_path, resume_step=plan["progress"]["global_step"],
                          target_step=plan["target_optimizer_steps"])
    if expected != plan:
        raise ValueError("Diversity selection, prefix, coverage or provenance differs")
