"""CPU-only, reproducible rule selection from the full Qwen-rendered success corpus.

No weights, teacher commands, inference, labels, or context truncation. Original
JSONL lines are copied unchanged. Rules estimate quality/complexity, not student NLL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.data.export_success_trajectories import sha256_file
from scripts.data.prepare_action_focused_sft import (
    CATEGORIES,
    ROOT,
    digest,
    terminal_candidate,
    write_json,
)

RULE_VERSION = 1
RULES = {
    "version": RULE_VERSION,
    "rank_order": [
        "bash_usage_errors",
        "redundant_repeats",
        "token_count",
        "tool_call_count",
        "trajectory_id",
    ],
    "eligibility": (
        "train_pool; >=2 original successful attempts/task; "
        ">=1 valid candidate; no invalid schema calls"
    ),
    "bash_errors": (
        "Matching shell diagnostic AND returncode 127 (command not found) or 2 (shell syntax)"
    ),
    "repeats": (
        "Adjacent action blocks with identical functions/arguments "
        "and original observations; ignore IDs/reasoning"
    ),
    "sampling": (
        "Largest-remainder quotas proportional to eligible tasks; seeded repository round robin"
    ),
    "unverified": (
        "Positional observation links and missing terminal observation allowed; "
        "unknown failures not penalized"
    ),
}
_BOUNDARY = r"(?:^|\n|\\n)"
_SHELL = r"(?:/[^:\s]+/)?(?:bash|sh):\s*(?:-c:\s*)?(?:line \d+:\s*)?"
_DIAGNOSTIC = r"(?:(?!\\n)[^\n]){0,250}"
_NOT_FOUND = re.compile(_BOUNDARY + _SHELL + _DIAGNOSTIC + r": command not found(?:$|\n|\\n)")
_SYNTAX = re.compile(_BOUNDARY + _SHELL + r"syntax error(?: near unexpected token|:|$|\n|\\n)")


def shell_error(observation):
    code, output = observation.get("returncode"), observation.get("output")
    if not isinstance(output, str):
        return False
    return bool(
        (code == 127 and _NOT_FOUND.search(output)) or (code == 2 and _SYNTAX.search(output))
    )


def features(row):
    result = {
        k: row.get(k)
        for k in (
            "trajectory_id",
            "task_key",
            "category",
            "language",
            "hf_dataset_name",
            "partition",
            "resolved",
            "token_count",
        )
    }
    result["repo"] = (row.get("task_key") or "").rsplit("#", 1)[0].lower()
    reasons = []
    if row.get("partition") != "train_pool":
        reasons.append(f"partition:{row.get('partition')}")
    if row.get("resolved") != 1:
        reasons.append("not_successful")
    if row.get("serialization_status") != "ok":
        reasons.append("serialization_error")
    if any(
        c.get("kind") == "arguments" and c.get("status") == "invalid" for c in row.get("checks", [])
    ):
        reasons.append("invalid_tool_arguments")
    if row.get("category") not in CATEGORIES:
        reasons.append("unsupported_category")
    if not isinstance(row.get("token_count"), int) or row["token_count"] <= 0:
        reasons.append("invalid_token_count")
    result.update(
        bash_usage_errors=0,
        redundant_repeats=0,
        tool_call_count=0,
        unverified_observations=0,
        other_nonzero_observations=0,
        submit_calls=0,
    )
    messages = row.get("messages") or []
    previous = None
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            previous = None
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        result["tool_call_count"] += len(calls)
        observed = []
        following = index + 1
        while following < len(messages) and messages[following].get("role") == "tool":
            observed.append(messages[following])
            following += 1
        pending = list(range(len(calls)))
        ordered_observations = [None] * len(calls)
        for obs in observed:
            call_id = obs.get("tool_call_id")
            if call_id:
                matches = [i for i in pending if calls[i].get("id") == call_id]
                ci = matches[0] if matches else None
            else:
                ci = pending[0] if pending else None
            if ci is None:
                result["unverified_observations"] += 1
                continue
            pending.remove(ci)
            content = obs.get("content")
            ordered_observations[ci] = content
            try:
                parsed = json.loads(content)
            except (ValueError, TypeError):
                parsed = None
            if (
                not isinstance(parsed, dict)
                or type(parsed.get("returncode")) is not int
                or not isinstance(parsed.get("output"), str)
            ):
                result["unverified_observations"] += 1
            elif calls[ci]["function"]["name"] == "bash" and shell_error(parsed):
                result["bash_usage_errors"] += 1
            elif parsed["returncode"] != 0:
                result["other_nonzero_observations"] += 1
        signatures = []
        for ci, call in enumerate(calls):
            fn = call["function"]
            signatures.append({"type": call.get("type"), "function": fn})
            args = fn.get("arguments")
            terminal = (
                fn.get("name") == "bash"
                and isinstance(args, dict)
                and isinstance(args.get("command"), str)
                and "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in args["command"]
                and terminal_candidate(args["command"])
            )
            result["submit_calls"] += int(terminal)
            if ci in pending and not terminal:
                result["unverified_observations"] += 1
        # Digest only signatures, never reinterpret command escapes or observation JSON.
        block = digest(
            json.dumps([signatures, ordered_observations], sort_keys=True, ensure_ascii=False)
        )
        if not pending and block == previous:
            result["redundant_repeats"] += 1
        previous = block if not pending else None
    if result["tool_call_count"] == 0:
        reasons.append("no_tool_calls")
    result["exclusion_reasons"] = reasons
    return result


def rank(row):
    return tuple(row[k] for k in RULES["rank_order"])


def proportional_quotas(counts, total):
    size = sum(counts.values())
    if total <= 0 or total > size:
        raise ValueError(f"Insufficient eligible tasks: requested={total}, available={size}")
    quotas = {c: total * counts.get(c, 0) // size for c in CATEGORIES}
    order = sorted(
        CATEGORIES, key=lambda c: (-(total * counts.get(c, 0) % size), CATEGORIES.index(c))
    )
    for category in order[: total - sum(quotas.values())]:
        quotas[category] += 1
    return quotas


def select(rows, total=4000, seed=42):
    tasks = defaultdict(list)
    ids = set()
    for row in rows:
        if row["trajectory_id"] in ids:
            raise ValueError("Duplicate trajectory ID")
        ids.add(row["trajectory_id"])
        tasks[row["task_key"]].append(row)
    winners, exclusions = [], []
    for key, attempts in sorted(tasks.items()):
        partitions = {r["partition"] for r in attempts}
        if len(partitions) != 1:
            raise ValueError(f"Task spans multiple partitions: {key}")
        if partitions != {"train_pool"}:
            continue
        successes = [r for r in attempts if r["resolved"] == 1]
        valid = [r for r in successes if not r["exclusion_reasons"]]
        reason = "single_success" if len(successes) < 2 else "no_valid_candidate"
        if len(successes) < 2 or not valid:
            exclusions.append({"task_key": key, "reason": reason, "successes": len(successes)})
            continue
        winners.append(min(valid, key=rank))
    counts = Counter(r["category"] for r in winners)
    quotas = proportional_quotas(counts, total)
    rng = random.Random(seed)
    chosen = []
    for category in CATEGORIES:
        repos = defaultdict(list)
        for row in winners:
            if row["category"] == category:
                repos[row["repo"]].append(row)
        repository_order = sorted(repos)
        rng.shuffle(repository_order)
        queues = {}
        for repo in repository_order:
            candidates = sorted(repos[repo], key=lambda r: r["task_key"])
            rng.shuffle(candidates)
            queues[repo] = deque(candidates)
        active = deque(repository_order)
        for _ in range(quotas[category]):
            if not active:
                raise ValueError(f"Insufficient tasks in category {category}")
            repo = active.popleft()
            chosen.append(queues[repo].popleft())
            if queues[repo]:
                active.append(repo)
    return {
        "selection": chosen,
        "winners": winners,
        "quotas": quotas,
        "eligible_tasks_by_category": {c: counts[c] for c in CATEGORIES},
        "task_exclusions": exclusions,
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    write_json(temporary, value)
    temporary.replace(path)


def rows_digest(rows):
    return digest(json.dumps(rows, sort_keys=True, ensure_ascii=False))


def extract_part(source, artifact, cache, scoring_digest, verify_source=False):
    import hashlib

    if sha256_file(source) != artifact["sha256"]:
        raise ValueError(f"Source checksum mismatch: {source}")
    identity = {
        "source_sha256": artifact["sha256"],
        "scoring_sha256": scoring_digest,
        "rules": RULES,
    }
    path = cache / (source.name.replace(".jsonl.gz", ".features.json"))
    saved_rows = None
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] == identity:
            if saved["features_sha256"] != rows_digest(saved["rows"]):
                raise ValueError(f"Feature cache checksum mismatch: {path}")
            if len(saved["rows"]) != artifact["rows"]:
                raise ValueError("Feature cache row count mismatch")
            if not verify_source:
                return saved["rows"]
            saved_rows = saved["rows"]
    rows = []
    with gzip.open(source, "rb") as stream:
        for index, line in enumerate(stream):
            row = features(json.loads(line))
            row.update(
                source_part=source.name,
                source_line_index=index,
                payload_sha256=hashlib.sha256(line).hexdigest(),
            )
            rows.append(row)
    if len(rows) != artifact["rows"]:
        raise ValueError(f"Source row count mismatch: {source}")
    if saved_rows is not None and saved_rows != rows:
        raise ValueError(f"Cache features differ from actual source: {path}")
    atomic_json(path, {"identity": identity, "features_sha256": rows_digest(rows), "rows": rows})
    return rows


def export_part(source, target, selected, source_sha256):
    import hashlib

    if sha256_file(source) != source_sha256:
        raise ValueError(f"Source checksum mismatch: {source}")
    identity = {"source_sha256": source_sha256, "selection_sha256": rows_digest(selected)}
    receipt_path = target.with_suffix(".receipt.json")
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["identity"] != identity or receipt["sha256"] != sha256_file(target):
            raise ValueError(f"Existing output checksum/selection mismatch: {target}")
        return receipt
    wanted = {r["source_line_index"]: r for r in selected}
    if len(wanted) != len(selected):
        raise ValueError("Duplicate selected source locator")
    temporary = target.with_suffix(".partial")
    count = 0
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=3, mtime=0) as destination,
        gzip.open(source, "rb") as stream,
    ):
        for index, line in enumerate(stream):
            if index not in wanted:
                continue
            expected = wanted[index]
            if hashlib.sha256(line).hexdigest() != expected["payload_sha256"]:
                raise ValueError("Selected source payload checksum mismatch")
            row = json.loads(line)
            if (
                row["trajectory_id"] != expected["trajectory_id"]
                or row["partition"] != "train_pool"
            ):
                raise ValueError("Selected source membership/partition mismatch")
            destination.write(line)  # Exact source bytes, including reasoning/command escapes.
            count += 1
    if count != len(selected):
        raise ValueError("Missing selected source records")
    temporary.replace(target)
    receipt = {
        "identity": identity,
        "sha256": sha256_file(target),
        "rows": count,
        "bytes": target.stat().st_size,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def write_rows(path, rows):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def length_summary(rows):
    lengths = sorted(r["token_count"] for r in rows)
    return {
        "total": sum(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        **{f"p{p}": lengths[int((len(lengths) - 1) * p / 100)] for p in (50, 90, 95, 99)},
        "above_context_lengths": {
            str(n): sum(v > n for v in lengths) for n in (8192, 16384, 32768, 65536, 131072, 262144)
        },
    }


def coverage(rows, key):
    return dict(sorted(Counter(str(r.get(key) or "unknown") for r in rows).items()))


def make_report(output, provenance, result, rows, artifacts):
    selected = result["selection"]
    total = provenance["count"]
    chosen_keys = {r["task_key"] for r in selected}
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_key"]].append(row)
    ranking_path = output / "task_candidate_rankings.jsonl"
    report = {
        **provenance,
        "rows": total,
        "unique_tasks": len(chosen_keys),
        "original_train_tasks_with_multiple_successes": sum(
            len([r for r in attempts if r["resolved"] == 1]) >= 2
            for attempts in tasks.values()
            if attempts[0]["partition"] == "train_pool"
        ),
        "eligible_tasks": len(result["winners"]),
        "eligible_tasks_by_category": result["eligible_tasks_by_category"],
        "quotas": result["quotas"],
        "categories": coverage(selected, "category"),
        "repositories": coverage(selected, "repo"),
        "repository_count": len({r["repo"] for r in selected}),
        "languages": coverage(selected, "language"),
        "sources": coverage(selected, "hf_dataset_name"),
        "length_tokens": length_summary(selected),
        "metric_totals": {
            k: sum(r[k] for r in selected)
            for k in (
                "bash_usage_errors",
                "redundant_repeats",
                "tool_call_count",
                "other_nonzero_observations",
                "unverified_observations",
                "submit_calls",
            )
        },
        "candidate_exclusion_reasons": dict(
            Counter(reason for r in rows for reason in r["exclusion_reasons"])
        ),
        "task_exclusion_reasons": dict(Counter(r["reason"] for r in result["task_exclusions"])),
        "remaining_eligible_tasks": len(result["winners"]) - total,
        "total_bytes": sum(r["bytes"] for r in artifacts.values()),
        "limitations": [
            "Heuristic quality/complexity ranking, not student capability measurement",
            "No commands executed; observations are historical evidence only",
            "No SFT labels or weights; 1 epoch intent does not guarantee Kaggle budget",
        ],
        "metadata_artifacts": {
            p.name: sha256_file(p)
            for p in (
                ranking_path,
                output / "candidate_exclusions.jsonl",
                output / "task_exclusions.jsonl",
                output / "remaining_task_winners.jsonl",
            )
        },
        "artifacts": dict(sorted(artifacts.items())),
    }
    return report


def run(source, output, cache, total=4000, seed=42, workers=4, verify=False):
    import inspect

    source_report = json.loads((source / "report.json").read_text())
    source_verification = json.loads((source / "verification_report.json").read_text())
    source_sha = sha256_file(source / "report.json")
    if (
        source_verification.get("report_sha256") != source_sha
        or source_verification.get("rows") != source_report["rows"]
        or not source_verification.get("all_source_payloads_and_official_renders_verified")
    ):
        raise ValueError("Full-corpus source conversion has not been verified")
    scoring_digest = digest(
        inspect.getsource(features)
        + inspect.getsource(shell_error)
        + _NOT_FOUND.pattern
        + _SYNTAX.pattern
    )
    sampling_digest = digest(
        inspect.getsource(select) + inspect.getsource(proportional_quotas) + inspect.getsource(rank)
    )
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_part, source / name, artifact, cache, scoring_digest, verify
            ): name
            for name, artifact in sorted(source_report["artifacts"].items())
        }
        for index, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"Indexed {index}/{len(futures)} parts ({len(rows)} rows)", flush=True)
    if len(rows) != source_report["rows"]:
        raise ValueError("Feature index/source totals differ")
    rows.sort(key=lambda r: (r["source_part"], r["source_line_index"]))
    # Quotas are based on eligible TASKS, not attempt counts.
    result = select(rows, total, seed)
    if result != select(list(reversed(rows)), total, seed):
        raise ValueError("Selection depends on source input order")
    selected = result["selection"]
    provenance = {
        "rules": RULES,
        "count": total,
        "seed": seed,
        "scoring_sha256": scoring_digest,
        "sampling_sha256": sampling_digest,
        "source_report_sha256": source_sha,
        "dataset": source_report["dataset"],
        "tokenizer": source_report["tokenizer"],
        "commands_executed": 0,
        "model_inference": False,
        "training_executed": False,
        "labels_created": False,
        "truncation": False,
    }
    if verify:
        return verify_outputs(output, result, rows, provenance, all_candidates_replayed=True)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise ValueError(
            "Output belongs to another rule/source/seed configuration; use a new directory"
        )
    atomic_json(provenance_path, provenance)
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_key"]].append(row)
    chosen_keys = {r["task_key"] for r in selected}
    ranking_path = output / "task_candidate_rankings.jsonl"
    write_rows(
        ranking_path,
        (
            {
                "task_key": winner["task_key"],
                "selected_for_train": winner["task_key"] in chosen_keys,
                "winner_id": winner["trajectory_id"],
                "candidates": [
                    {**r, "rank_tuple": list(rank(r)) if not r["exclusion_reasons"] else None}
                    for r in sorted(tasks[winner["task_key"]], key=lambda r: r["trajectory_id"])
                ],
            }
            for winner in result["winners"]
        ),
    )
    write_rows(output / "candidate_exclusions.jsonl", (r for r in rows if r["exclusion_reasons"]))
    write_rows(output / "task_exclusions.jsonl", result["task_exclusions"])
    write_rows(
        output / "remaining_task_winners.jsonl",
        (r for r in result["winners"] if r["task_key"] not in chosen_keys),
    )
    by_part = defaultdict(list)
    for row in selected:
        by_part[row["source_part"]].append(row)
    artifacts = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                export_part,
                source / name,
                output / name,
                candidates,
                source_report["artifacts"][name]["sha256"],
            ): name
            for name, candidates in sorted(by_part.items())
        }
        for index, future in enumerate(as_completed(futures), 1):
            receipt = future.result()
            artifacts[futures[future]] = {k: receipt[k] for k in ("sha256", "rows", "bytes")}
            print(
                f"Copied {index}/{len(futures)} selected parts: {receipt['rows']} records",
                flush=True,
            )
    if sum(r["rows"] for r in artifacts.values()) != total:
        raise ValueError("Export count differs from selection")
    manifest = {
        **provenance,
        "eligible_tasks_by_category": result["eligible_tasks_by_category"],
        "quotas": result["quotas"],
        "selection": selected,
        "artifacts": dict(sorted(artifacts.items())),
    }
    atomic_json(output / "task_selection_manifest.json", manifest)
    report = make_report(output, provenance, result, rows, artifacts)
    atomic_json(output / "report.json", report)
    verify_outputs(output, result, rows, provenance)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("rows", "eligible_tasks", "quotas", "length_tokens", "metric_totals")
            },
            indent=2,
        )
    )


def verify_outputs(output, result, rows, provenance, all_candidates_replayed=False):
    import hashlib

    selected = result["selection"]

    manifest_path = output / "task_selection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        any(manifest.get(k) != v for k, v in provenance.items())
        or manifest["selection"] != selected
        or manifest["quotas"] != result["quotas"]
        or manifest["eligible_tasks_by_category"] != result["eligible_tasks_by_category"]
    ):
        raise ValueError("Manifest/source/rules or reproducible selection differ")
    expected = {r["trajectory_id"]: r for r in selected}
    if len(expected) != len(selected) or len({r["task_key"] for r in selected}) != len(selected):
        raise ValueError("Selected trajectories/tasks not unique")
    if set(p.name for p in output.glob("part-*.jsonl.gz")) != set(manifest["artifacts"]):
        raise ValueError("Unexpected/missing output parts")
    seen, token_total = set(), 0
    for name, artifact in manifest["artifacts"].items():
        path = output / name
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Output checksum mismatch: {path}")
        count = 0
        with gzip.open(path, "rb") as stream:
            for line in stream:
                row = json.loads(line)
                tid = row["trajectory_id"]
                if tid not in expected or tid in seen:
                    raise ValueError("Unexpected/duplicate exported trajectory")
                feature = expected[tid]
                if hashlib.sha256(line).hexdigest() != feature["payload_sha256"]:
                    raise ValueError("Original source payload changed")
                actual = features(row)
                if actual != {
                    k: v
                    for k, v in feature.items()
                    if k not in ("source_part", "source_line_index", "payload_sha256")
                }:
                    raise ValueError("Exported rule features differ from manifest")
                if (
                    row["partition"] != "train_pool"
                    or actual["exclusion_reasons"]
                    or "reference_patch" in row
                ):
                    raise ValueError("Invalid or heldout record exported")
                seen.add(tid)
                token_total += row["token_count"]
                count += 1
        if count != artifact["rows"]:
            raise ValueError("Output part count differs")
    report = json.loads((output / "report.json").read_text())
    if (
        seen != set(expected)
        or token_total != report["length_tokens"]["total"]
        or report["rows"] != len(seen)
        or coverage(selected, "category") != report["categories"]
    ):
        raise ValueError("Output/report totals differ")
    if report != make_report(output, provenance, result, rows, manifest["artifacts"]):
        raise ValueError("Derived report fields differ from all candidate features")
    for name, checksum in report["metadata_artifacts"].items():
        if sha256_file(output / name) != checksum:
            raise ValueError(f"Metadata checksum mismatch: {name}")
    verification = {
        "rows": len(seen),
        "unique_tasks": len(seen),
        "total_tokens": token_total,
        "source_payloads_unchanged": True,
        "train_pool_only": True,
        "selection_reproduced_from_all_candidates": True,
        "source_index_order_invariant": True,
        "rules_recomputed_for_all_selected_records": True,
        "all_candidate_features_recomputed_from_source": all_candidates_replayed,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(output / "report.json"),
        "commands_executed": 0,
    }
    atomic_json(output / "verification_report.json", verification)
    print(f"Verified {len(seen)} unchanged selected trajectories/tasks", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/processed/nvidia_mini_swe_agent_v1_2_success_qwen35_4b",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/mini_swe_agent_rule_baseline_4000"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/interim/nvidia_success_rule_features_v1"
    )
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    run(args.source, args.output, args.cache, args.count, args.seed, args.workers, args.verify)


if __name__ == "__main__":
    main()
