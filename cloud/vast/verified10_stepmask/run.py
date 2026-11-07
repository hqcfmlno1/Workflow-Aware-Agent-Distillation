"""New masked span-anchor benchmark; local Docker, durable steps and official grading."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import os
import shutil
import signal
import sys
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "server/runtime" if (HERE / "server/runtime").exists() else HERE / "runtime"
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(HERE))


def load_json(path):
    return json.loads(Path(path).read_text())


def task_stats(state):
    folder = Path(state.get("attempt_dir", "/nonexistent"))
    stats = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "generation_seconds": 0.0,
        "model_turns": 0,
        "tool_calls": 0,
        "submission": False,
    }
    messages = folder / "agent-messages.jsonl"
    if messages.exists():
        for line in messages.read_text().splitlines():
            message = json.loads(line)["message"]
            response = message.get("extra", {}).get("response", {})
            if response:
                stats["model_turns"] += 1
                for key in ("prompt_tokens", "completion_tokens"):
                    stats[key] += response.get("usage", {}).get(key, 0)
                stats["generation_seconds"] += response.get("backend_metrics", {}).get("seconds", 0)
    saved = folder / "resume.json"
    if saved.exists():
        stats["tool_calls"] = load_json(saved).get("tool_calls_started", 0)
        stats["termination"] = load_json(saved).get("result", {}).get("exit_status")
    stats["submission"] = bool(state.get("prediction", {}).get("model_patch", "").strip())
    seconds = stats["generation_seconds"]
    stats["tokens_per_second"] = stats["completion_tokens"] / seconds if seconds else None
    stats["task_seconds"] = state.get("task_seconds", 0.0)
    return stats


def eligible(protocol, variant, iid):
    return variant in protocol["variant_order"] and iid in protocol["instance_ids"]


def report(root, protocol):
    from benchmarks.swe_bench_verified.campaign import Store, atomic_json

    records = []
    summary = {}
    for variant in protocol["variant_order"]:
        store = Store(root / variant)
        for iid in protocol["instance_ids"]:
            if not eligible(protocol, variant, iid):
                continue
            state = store.task(iid)
            result = state.get("result", {})
            records.append(
                {
                    "instance_id": iid,
                    "variant": variant,
                    "phase": state["phase"],
                    "resolved": result.get("resolved"),
                    "kind": result.get("kind", "pending"),
                    "turn_timeout_seconds": state.get(
                        "turn_timeout_seconds", protocol.get("turn_timeout_seconds")
                    ),
                    "inherited_from": state.get("inherited_from"),
                    **task_stats(state),
                }
            )
        own = [r for r in records if r["variant"] == variant]
        summary[variant] = {
            "total": len(own),
            "done": sum(r["phase"] == "done" for r in own),
            "resolved": sum(r["resolved"] is True for r in own),
            "infrastructure_or_pending": sum(
                r["kind"] == "infrastructure_error" or r["phase"] != "done" for r in own
            ),
            "budget_stops": sum(r["kind"] == "task_budget_exhausted" for r in own),
            "turn_timeouts": sum(r["kind"] == "turn_timeout" for r in own),
            "official_graded": sum(r["kind"] == "graded" for r in own),
            "infrastructure_errors": sum(r["kind"] == "infrastructure_error" for r in own),
            "pending": sum(r["phase"] != "done" for r in own),
            "completion_tokens": sum(r["completion_tokens"] for r in own),
            "tool_calls": sum(r["tool_calls"] for r in own),
        }
    atomic_json(
        root / "comparison.json",
        {
            "summary": summary,
            "tasks": records,
            "token_accounting": (
                "Completed responses only; interrupted generation may use extra tokens."
            ),
        },
    )
    with (root / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for r in records for k in r))
        )
        writer.writeheader()
        writer.writerows(records)
    headers = ["Task", *protocol["variant_order"]]
    lines = [
        f"# Span-anchor + mask ({protocol['tool_call_limit']} tool calls)",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for iid in protocol["instance_ids"]:
        if not any(eligible(protocol, v, iid) for v in protocol["variant_order"]):
            continue
        cells = []
        for variant in protocol["variant_order"]:
            if not eligible(protocol, variant, iid):
                cells.append("not selected")
                continue
            record = next(r for r in records if r["instance_id"] == iid and r["variant"] == variant)
            label = (
                "BUDGET"
                if record["kind"] == "task_budget_exhausted"
                else "TIMEOUT"
                if record["kind"] == "turn_timeout"
                else "PASS"
                if record["resolved"] is True
                else "FAIL"
                if record["resolved"] is False
                else record["kind"]
            )
            timeout = record.get("turn_timeout_seconds")
            suffix = f"; {timeout}s/turn" if timeout else ""
            cells.append(
                f"{label}; {record['model_turns']} turns; {record['tool_calls']} calls; "
                f"{record['completion_tokens']} tokens{suffix}"
            )
        lines.append("| " + " | ".join([iid, *cells]) + " |")
    lines.extend(
        [
            "",
            *[
                f"{v}: {s['resolved']}/{s['total']} PASS; {s['done']} completed."
                for v, s in summary.items()
            ],
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    return summary


def check_host(endpoint, key, protocol):
    import re

    from benchmarks.swe_bench_verified.config import fingerprint
    from benchmarks.swe_bench_verified.runner import server_info

    source = protocol["sources"]["span-stepmask"]
    if not re.fullmatch(r"[0-9a-f]{40}", source.get("revision") or ""):
        raise ValueError("Finalize preparation to pin checkpoint 590 before benchmarking")
    info = server_info(endpoint, key, base_dtype="bfloat16")
    if info.get("config") != protocol["config"]:
        raise ValueError("Serving config differs from this frozen benchmark")
    if info.get("benchmark_protocol_sha256") != fingerprint(protocol):
        raise ValueError("Serving protocol differs from local tasks/checkpoint")
    if info.get("backend") != "vllm" or info.get("adapter_dtype") != "bfloat16":
        raise ValueError("Expected vLLM BF16 base and BF16 runtime adapter")
    exported = info.get("export") or {}
    if exported.get("source") != source or exported.get("checkpoint") != protocol["checkpoint"]:
        raise ValueError("Wrong checkpoint: expected new span-stepmask epoch 2")
    return info


def finish_error(store, iid, exc, resources):
    import httpx
    from benchmarks.swe_bench_verified.campaign import atomic_json

    state = store.task(iid)
    # An inferred patch is committed: a grading failure must never repeat inference.
    if state["phase"] == "inferred":
        store.event("grading_pending", instance_id=iid, error=str(exc))
        return
    timeout = isinstance(exc, TimeoutError) or (
        isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 408
    )
    kind = getattr(exc, "kind", "turn_timeout" if timeout else "infrastructure_error")
    result = {
        "kind": kind,
        "resolved": False if kind == "turn_timeout" else None,
        "official_graded": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    folder = Path(state.get("attempt_dir", store.output / "tasks" / iid))
    checkpoint = folder / "resume.json"
    if checkpoint.exists():
        saved = load_json(checkpoint)
        saved.update(phase="complete", result={**result, "exit_status": result["kind"]})
        atomic_json(checkpoint, saved)
    atomic_json(folder / "terminal-error.json", result)
    store.put(iid, phase="done", result=result)
    store.predictions()
    resources.cleanup()
    store.event("task_error_continue", instance_id=iid, **result)


def cleanup_task_image(group):
    from benchmarks.swe_bench_verified.campaign import atomic_json
    from benchmarks.swe_bench_verified.docker_resources import DockerResources

    owner = group / "ownership.json"
    if not owner.exists():
        return
    ownership = load_json(owner)
    if ownership["preexisting"] or ownership.get("cleaned"):
        return
    DockerResources("v10-images", group, [ownership["image"]]).cleanup_images()
    inspected = DockerResources.command("image", "inspect", ownership["image"])
    if inspected.returncode == 0 or "No such image" not in inspected.stderr:
        raise RuntimeError("Owned task image cleanup incomplete; inspect " + str(group))
    atomic_json(owner, {**ownership, "cleaned": True})


def main():
    from benchmarks.swe_bench_verified import campaign
    from benchmarks.swe_bench_verified.config import BenchConfig, fingerprint, sha256_file
    from benchmarks.swe_bench_verified.docker_resources import DockerResources
    from benchmarks.swe_bench_verified.policy import ToolBudgetAgent
    from benchmarks.swe_bench_verified.runner import agent_config, check_docker, grader_preflight
    from benchmarks.swe_bench_verified.step_resume import UncertainToolExecution
    from budget import RunBudget, RunBudgetExceeded

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--retry-infrastructure", action="store_true")
    args = parser.parse_args()
    protocol = load_json(HERE / "protocol.json")
    rows = load_json(HERE / "grading-dataset.json")
    if fingerprint(rows) != protocol["rows_sha256"]:
        raise ValueError("Frozen grading data changed")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.report_only:
            print(json.dumps(report(root, protocol), indent=2))
            return
        if version("mini-swe-agent") != "2.4.6":
            raise RuntimeError("Install mini-swe-agent==2.4.6")
        check_docker()
        grader_preflight(rows)
        key_file = Path(
            os.environ.get(
                "BENCH_API_KEY_FILE",
                str(Path.home() / ".config/workflow-distillation/bench_api_key"),
            )
        )
        key = os.environ.get("OPENAI_API_KEY") or key_file.read_text().strip()
        os.environ["OPENAI_API_KEY"] = key
        info = check_host(args.endpoint, key, protocol)
        campaign.CHECKPOINT = protocol["checkpoint"]
        identity = {
            "protocol": protocol,
            "server_fingerprint": info["fingerprint"],
            "runtime": {
                str(p.relative_to(RUNTIME)): sha256_file(p) for p in sorted(RUNTIME.rglob("*.py"))
            },
            "runner_sha256": sha256_file(Path(__file__)),
            "packages": {p: version(p) for p in ("mini-swe-agent", "swebench", "httpx")},
        }
        path = root / "identity.json"
        if path.exists() and load_json(path) != identity:
            raise ValueError("Resume model/config/data/runtime changed; use a new output")
        campaign.atomic_json(path, identity)
        stores = {variant: campaign.Store(root / variant) for variant in protocol["variant_order"]}
        if args.retry_infrastructure:
            for store in stores.values():
                for iid in protocol["instance_ids"]:
                    state = store.task(iid)
                    if state.get("result", {}).get("kind") == "infrastructure_error":
                        store.put(
                            iid,
                            phase="pending",
                            attempt_dir="",
                            error=None,
                            result={},
                            inference_seconds=0.0,
                            task_seconds=0.0,
                        )

        def stop(signum, frame):
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        logging.basicConfig(level=logging.INFO)
        from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

        budget = RunBudget(root / "run-budget.json", protocol["campaign_budget_seconds"])
        if budget.remaining() <= 0:
            print("Saved campaign budget exhausted; no further requests sent.", flush=True)
            report(root, protocol)
            return
        budget.start()
        try:
            for task_index, row in enumerate(rows):
                iid = row["instance_id"]
                if (root / "STOP").exists():
                    break
                active_stores = [store for v, store in stores.items() if eligible(protocol, v, iid)]
                if not active_stores:
                    continue
                if all(store.task(iid)["phase"] == "done" for store in active_stores):
                    cleanup_task_image(root / "images" / iid)
                    continue
                if shutil.disk_usage(root).free < 20 * 1024**3:
                    raise RuntimeError("Less than 20GiB free before task image pull")
                check_host(args.endpoint, key, protocol)
                image = get_swebench_docker_image_name(row)
                group = root / "images" / iid
                group.mkdir(parents=True, exist_ok=True)
                owner = group / "ownership.json"
                if not owner.exists():
                    inspected = DockerResources.command("image", "inspect", image)
                    if inspected.returncode and "No such image" not in inspected.stderr:
                        raise RuntimeError(inspected.stderr)
                    campaign.atomic_json(
                        owner, {"preexisting": inspected.returncode == 0, "image": image}
                    )
                # Pull once per task; release only images owned by this campaign.
                # TaskResources records preexisting=True and cleans only its own containers.
                try:
                    campaign.ensure_task_image(image, next(iter(stores.values())).event)
                except campaign.TaskImageUnavailable as exc:
                    for store in stores.values():
                        if store.task(iid)["phase"] != "done":
                            store.put(
                                iid,
                                phase="done",
                                result={
                                    "kind": "infrastructure_error",
                                    "resolved": None,
                                    "error": str(exc),
                                },
                            )
                    report(root, protocol)
                    continue
                variants = protocol["variant_order"]
                offset = task_index % len(variants)
                for variant in variants[offset:] + variants[:offset]:
                    if not eligible(protocol, variant, iid):
                        continue
                    store = stores[variant]
                    if store.task(iid)["phase"] == "done":
                        continue
                    config = BenchConfig(**protocol["config"])
                    resolved = agent_config(config, args.endpoint, info["fingerprint"])
                    resolved["model"].update(model_name=variant)
                    resolved["agent"].update(
                        tool_call_limit=protocol["tool_call_limit"],
                        max_consecutive_format_errors=0,
                        wall_time_limit_seconds=0,
                    )
                    resolved["environment"]["container_timeout"] = "infinity"
                    run_id = "v10mask-" + variant + "-" + fingerprint(str(root))[:10]
                    resources = campaign.TaskResources(
                        run_id, store.output / "tasks" / iid, image, iid
                    )
                    started = time.monotonic()
                    print(
                        json.dumps(
                            {
                                "event": "stepmask_task_start",
                                "variant": variant,
                                "instance_id": iid,
                                "remaining_campaign_seconds": (
                                    None
                                    if budget.limit_seconds is None
                                    else round(budget.remaining(), 1)
                                ),
                            }
                        ),
                        flush=True,
                    )

                    def infer_bounded(
                        r, p, *, store=store, iid=iid, resolved=resolved, resources=resources
                    ):
                        spent = store.task(iid).get("inference_seconds", 0.0)
                        infer_start = time.monotonic()
                        try:
                            with budget.task(protocol["task_budget_seconds"] - spent):
                                return campaign.infer_task(
                                    r, p, resolved, store, resources, agent_class=ToolBudgetAgent
                                )
                        finally:
                            store.put(iid, inference_seconds=spent + time.monotonic() - infer_start)

                    try:
                        campaign.run_task(
                            store,
                            row,
                            resources,
                            infer_bounded,
                            lambda r, pred, p, run_id=run_id: campaign.grade_task(
                                r, pred, p, run_id
                            ),
                        )
                    except UncertainToolExecution:
                        raise
                    except Exception as exc:
                        finish_error(store, iid, exc, resources)
                    finally:
                        store.put(
                            iid,
                            task_seconds=store.task(iid).get("task_seconds", 0.0)
                            + time.monotonic()
                            - started,
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "stepmask_progress",
                                    "remaining_campaign_seconds": (
                                        None
                                        if budget.limit_seconds is None
                                        else round(budget.remaining(), 1)
                                    ),
                                    "summary": report(root, protocol),
                                }
                            ),
                            flush=True,
                        )
                        budget.save()
                # A STOP drains the current task. Ctrl-C retains the current step/container.
                cleanup_task_image(group)
        except RunBudgetExceeded as exc:
            print(
                json.dumps(
                    {
                        "event": "campaign_budget_pause",
                        "detail": str(exc),
                        "elapsed_seconds": budget.elapsed(),
                    }
                ),
                flush=True,
            )
        finally:
            budget.close()
            report(root, protocol)


if __name__ == "__main__":
    main()
