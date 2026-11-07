"""Judge whole assistant steps through a local API; retain source context verbatim."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import signal
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path

from workflow_distillation.trajectory_step_masking import (
    atomic_json,
    fingerprint,
    judge_record,
    prepare_record,
    read_completed,
    request_identity,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / (
    "data/processed/metacurate_deepseek_v4pro_success_le36k_tqs/trajectories-ranked.jsonl.gz"
)
DEFAULT_OUTPUT = ROOT / "outputs/data_audits/deepseek1180_llm_step_mask_v1"


def completed_results(rows, output, config, policy):
    results = {}
    for row in rows:
        _, sources, steps = prepare_record(row)
        prior = read_completed(
            output / "results" / f"{row['id']}.json",
            request_identity(row, config, policy),
            row["id"],
            sources,
            steps,
        )
        if prior:
            results[row["id"]] = prior
    return results


def export_results(rows, output, results):
    """Only completed records exported; failed/pending never become empty masks."""
    masks = output / "mask_steps.jsonl"
    corpus = output / "annotated_trajectories.jsonl.gz"
    masks_tmp = masks.with_suffix(".tmp")
    corpus_tmp = corpus.with_suffix(".tmp")
    try:
        with masks_tmp.open("w") as a, gzip.open(corpus_tmp, "wt") as b:
            for row in rows:
                result = results.get(row["id"])
                if not result:
                    continue
                annotation = {
                    **result["annotation"],
                    "mask_message_indices": result["mask_message_indices"],
                    "request_sha256": result["request_sha256"],
                    "source_record_sha256": result["source_record_sha256"],
                }
                a.write(json.dumps(annotation, ensure_ascii=False) + "\n")
                b.write(
                    json.dumps({"source_record": row, "loss_mask": annotation}, ensure_ascii=False)
                    + "\n"
                )
        masks_tmp.replace(masks)
        corpus_tmp.replace(corpus)
    finally:
        masks_tmp.unlink(missing_ok=True)
        corpus_tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs/data/step_mask_policy_v1.txt"
    )
    parser.add_argument("--endpoint", default="http://localhost:20128/v1")
    parser.add_argument("--model", default="cx/gpt-6-luna[1m]")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path.home() / ".cache/workflow-distillation/llm-step-mask.key",
    )
    parser.add_argument("--api-kind", choices=["responses", "chat"], default="responses")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-records", type=int, default=1180)
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument(
        "--no-quotes",
        action="store_true",
        help="Use approved v2 output: step_id, reason_codes, reason only",
    )
    parser.add_argument(
        "--run", action="store_true", help="Actually call API; otherwise prepare only"
    )
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.timeout <= 0
        or args.retries < 0
        or args.max_output_tokens < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("Invalid worker/token/time/retry/limit setting")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A runner already owns this output directory")
    with gzip.open(args.source, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    ids = [row["id"] for row in rows]
    if len(rows) != args.expected_records or len(set(ids)) != len(ids):
        parser.error("Unexpected source count or duplicate trajectory IDs")
    policy = args.policy.read_text()
    config = {
        "model": args.model,
        "api_kind": args.api_kind,
        "effort": args.effort or None,
        "max_output_tokens": args.max_output_tokens,
        "structured": not args.no_structured_output,
    }
    if args.no_quotes:
        config["schema_version"] = "no_quote_v2"
    manifest = {
        "source": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "records": len(rows),
        "ids_sha256": fingerprint(ids),
        "policy_sha256": hashlib.sha256(policy.encode()).hexdigest(),
        "config": config,
        "step_definition": "one assistant message; step_id starts at 1",
        "mask_indices": "zero-based original message indices; assistant only",
        "source_modified": False,
        "teacher_commands_executed": 0,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error("Run identity changed: choose a different output directory")
    atomic_json(manifest_path, manifest)
    completed = completed_results(rows, output, config, policy)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _sig, _frame: stop.set())
    started = time.monotonic()
    log_lock = threading.Lock()
    active, failures = {}, set()

    def emit(event):
        line = json.dumps({"time_utc": datetime.now(UTC).isoformat(), **event}, ensure_ascii=False)
        with log_lock:
            with (output / "campaign.log").open("a") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

    def save_summary(status):
        reasons = Counter(
            code
            for result in completed.values()
            for mask in result["annotation"]["mask_steps"]
            for code in mask["reason_codes"]
        )
        usage = Counter()
        for result in completed.values():
            u = result.get("usage", {})
            usage["input_tokens"] += u.get("input_tokens", u.get("prompt_tokens", 0))
            usage["output_tokens"] += u.get("output_tokens", u.get("completion_tokens", 0))
        summary = {
            "status": status,
            "total": len(rows),
            "completed": len(completed),
            "errors_this_invocation": len(failures),
            "pending": len(rows) - len(completed),
            "masked_steps": sum(len(r["mask_step_ids"]) for r in completed.values()),
            "reviewed_assistant_steps": sum(r["assistant_steps"] for r in completed.values()),
            "trajectories_with_masks": sum(bool(r["mask_step_ids"]) for r in completed.values()),
            "reason_counts": dict(reasons),
            "completed_result_usage": dict(usage),
            "usage_note": "Failed-attempt usage, if returned, is retained in attempts/ receipts",
            "active": [
                {
                    "trajectory_id": k,
                    "seconds": round(time.monotonic() - v["start"]),
                    "assistant_steps": v["steps"],
                    "source_tokens": v["tokens"],
                }
                for k, v in active.items()
            ],
            "invocation_seconds": round(time.monotonic() - started, 1),
            "updated_utc": datetime.now(UTC).isoformat(),
        }
        atomic_json(output / "summary.json", summary)
        return summary

    emit(
        {
            "event": "prepared",
            "total": len(rows),
            "resumed": len(completed),
            "model": args.model,
            "api_kind": args.api_kind,
            "workers": args.workers,
            "pid": os.getpid(),
            "source_student_tokens": sum(r["total_tokens"] for r in rows),
        }
    )
    export_results(rows, output, completed)
    if not args.run:
        save_summary("ready")
        return
    key = args.api_key_file.read_text().strip()
    if not key:
        parser.error("Empty API key file")
    atomic_json(
        output / "process.json",
        {
            "pid": os.getpid(),
            "endpoint": args.endpoint,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    pending = [r for r in rows if r["id"] not in completed]
    if args.limit:
        pending = pending[: args.limit]
    iterator = iter(pending)
    rank = {ident: n for n, ident in enumerate(ids, 1)}
    futures = {}
    fatal = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:

        def dispatch():
            if stop.is_set():
                return False
            row = next(iterator, None)
            if row is None:
                return False
            ident = row["id"]
            active[ident] = {
                "start": time.monotonic(),
                "steps": sum(m["role"] == "assistant" for m in row["messages"]),
                "tokens": row["total_tokens"],
            }
            emit(
                {
                    "event": "trajectory_started",
                    "trajectory_id": ident,
                    "rank": rank[ident],
                    "task": row.get("instance_id"),
                    "assistant_steps": active[ident]["steps"],
                    "source_tokens": row["total_tokens"],
                }
            )
            f = pool.submit(
                judge_record,
                row,
                output,
                args.endpoint,
                key,
                config,
                policy,
                timeout=args.timeout,
                retries=args.retries,
                emit=emit,
            )
            futures[f] = ident
            return True

        for _ in range(args.workers):
            dispatch()
        save_summary("running")
        while futures:
            done, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
            if not done:
                emit(
                    {
                        "event": "heartbeat",
                        **save_summary("stopping" if stop.is_set() else "running"),
                    }
                )
                continue
            for future in done:
                ident = futures.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "error",
                        "trajectory_id": ident,
                        "error_type": type(error).__name__,
                    }
                active.pop(ident, None)
                if result["status"] in {"completed", "resumed"}:
                    completed[ident] = result
                else:
                    failures.add(ident)
                    attempt = result.get("attempt_file")
                    if attempt and Path(attempt).exists():
                        error = json.loads(Path(attempt).read_text())
                        if error.get("http_status") in {401, 403, 404}:
                            fatal = True
                            stop.set()
                emit(
                    {
                        "event": "trajectory_finished",
                        "trajectory_id": ident,
                        "status": result["status"],
                        "completed": len(completed),
                        "total": len(rows),
                        "errors": len(failures),
                        "mask_step_ids": result.get("mask_step_ids"),
                        "seconds": result.get("seconds"),
                        "usage": result.get("usage"),
                    }
                )
                save_summary("stopping" if stop.is_set() else "running")
                dispatch()
            if len(completed) % 20 == 0:
                export_results(rows, output, completed)
    export_results(rows, output, completed)
    status = (
        "completed"
        if len(completed) == len(rows)
        else "blocked_api"
        if fatal
        else "paused"
        if stop.is_set()
        else "partial"
    )
    emit({"event": "run_finished", **save_summary(status)})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
