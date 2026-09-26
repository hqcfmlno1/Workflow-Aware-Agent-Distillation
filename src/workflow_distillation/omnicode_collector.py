"""Incremental teacher-trajectory collection on top of OmniCode's SWE-agent runner."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LANGUAGES = ("python", "java", "cpp")
WORKFLOWS = ("bugfixing", "test_generation", "style_review", "review_response")
COLLECTOR_VERSION = "0.1.0"
OMNICODE_CONTAINER_PREFIX = "omnicodeorgomnicode"
DEFAULT_TASK_TIMEOUT_SECONDS = 720.0
CLEANUP_RESERVE_SECONDS = 60.0
DEFAULT_SWE_REX_STARTUP_TIMEOUT_SECONDS = 300
PROCESS_TERMINATION_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class TaskSpec:
    workflow: str
    language: str
    dataset_path: Path
    sweagent_mode: str
    require_reviewfix_patch: bool = False


@dataclass(frozen=True)
class PlannedTask:
    spec: TaskSpec
    instance: dict[str, Any]
    attempt_id: int
    run_key: str


def build_task_specs(
    omnicode_root: Path,
    workflows: Iterable[str] | None = None,
    languages: Iterable[str] = LANGUAGES,
) -> dict[str, list[TaskSpec]]:
    selected_workflows = list(workflows or WORKFLOWS)
    unknown_workflows = set(selected_workflows) - set(WORKFLOWS)
    if unknown_workflows:
        raise ValueError(f"Unknown workflow(s): {sorted(unknown_workflows)}")

    selected_languages = list(languages)
    unknown_languages = set(selected_languages) - set(LANGUAGES)
    if unknown_languages:
        raise ValueError(f"Unknown language(s): {sorted(unknown_languages)}")

    mode_map = {
        "bugfixing": {"python": "bugfixing", "java": "bugfixing-java", "cpp": "bugfixing-cpp"},
        "test_generation": {"python": "testgen", "java": "testgen-java", "cpp": "testgen-cpp"},
        "style_review": {
            "python": "stylereview-python",
            "java": "stylereview-java-checkstyle",
            "cpp": "stylereview-cpp-clangtidy",
        },
        "review_response": {language: "reviewfix" for language in LANGUAGES},
    }

    specs: dict[str, list[TaskSpec]] = {workflow: [] for workflow in selected_workflows}
    for workflow in selected_workflows:
        for language in selected_languages:
            filename = (
                f"omnicode_style_instances_{language}.json"
                if workflow == "style_review"
                else f"omnicode_instances_{language}.json"
            )
            specs[workflow].append(
                TaskSpec(
                    workflow=workflow,
                    language=language,
                    dataset_path=omnicode_root / "data" / filename,
                    sweagent_mode=mode_map[workflow][language],
                    require_reviewfix_patch=workflow == "review_response",
                )
            )
    return specs


def load_instances(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"OmniCode dataset not found: {path}")
    if path.suffix.lower() == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("test", payload.get("train", payload))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Expected a list of JSON objects in {path}")
    return payload


def _has_reviewfix_patch(instance: dict[str, Any]) -> bool:
    return any(
        isinstance(patch, dict) and "agentless" in str(patch.get("source", ""))
        for patch in instance.get("bad_patches", [])
    )


def select_tasks(
    instances: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    completed_ids: set[str],
    require_reviewfix_patch: bool = False,
) -> list[dict[str, Any]]:
    candidates = [
        item
        for item in instances
        if item.get("instance_id") not in completed_ids
        and (not require_reviewfix_patch or _has_reviewfix_patch(item))
    ]
    candidates.sort(key=lambda item: str(item["instance_id"]))
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _experiment_id(omnicode_root: Path, model: str, api_base: str, seed: int) -> str:
    payload = {
        "collector_version": COLLECTOR_VERSION,
        "omnicode_revision": _git_revision(omnicode_root),
        "model": model,
        "api_base": api_base,
        "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def _group_seed(seed: int, workflow: str, language: str) -> int:
    digest = hashlib.sha256(f"{seed}:{workflow}:{language}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _command_for_task(
    *,
    spec: TaskSpec,
    instance_id: str,
    omnicode_root: Path,
    output_dir: Path,
    model: str,
    python_executable: str,
    use_apptainer: bool,
) -> list[str]:
    runner = omnicode_root / "baselines" / "sweagent" / "sweagent_regular.py"
    return [
        python_executable,
        str(runner),
        "--input_tasks",
        str(spec.dataset_path),
        "--output_dir",
        str(output_dir),
        "--instance_ids",
        instance_id,
        "--mode",
        spec.sweagent_mode,
        "--model_name",
        model,
        "--use_apptainer",
        str(use_apptainer).lower(),
        "--startup_timeout",
        str(int(DEFAULT_SWE_REX_STARTUP_TIMEOUT_SECONDS)),
        "--output_file",
        str(output_dir / "all_preds.jsonl"),
    ]


def _build_subprocess_env(api_base: str, api_key: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OPENAI_API_KEY": api_key,
            "OPENAI_API_BASE": api_base,
            "OPENAI_BASE_URL": api_base,
            "LITELLM_DROP_PARAMS": "true",
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return env


def _split_task_budget(
    total_timeout_seconds: float,
    cleanup_reserve_seconds: float = CLEANUP_RESERVE_SECONDS,
) -> tuple[float, float]:
    """Reserve time for cleanup while keeping at least one second for execution."""
    total = max(1.0, float(total_timeout_seconds))
    reserve = min(max(0.0, float(cleanup_reserve_seconds)), total - 1.0)
    return total - reserve, reserve


def _process_tree_termination_command(pid: int, *, platform: str | None = None) -> list[str]:
    current_platform = platform or os.name
    if current_platform == "nt" or current_platform.startswith("win"):
        return ["taskkill", "/PID", str(pid), "/T", "/F"]
    return ["kill", "-TERM", f"-{pid}"]


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    command = _process_tree_termination_command(process.pid)
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS,
        )
    try:
        process.wait(timeout=PROCESS_TERMINATION_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    stop_event: threading.Event,
    immediate_stop: threading.Event,
) -> tuple[str, int | None]:
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        popen_options: dict[str, Any] = {
            "cwd": cwd,
            "env": env,
            "stdout": stdout,
            "stderr": stderr,
        }
        if os.name == "nt":
            popen_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(command, **popen_options)
        started = time.monotonic()
        while process.poll() is None:
            if immediate_stop.is_set():
                _terminate_process_tree(process)
                return "interrupted", process.returncode
            if time.monotonic() - started >= timeout_seconds:
                _terminate_process_tree(process)
                return "timeout", process.returncode
            time.sleep(0.5)
    return ("completed" if process.returncode == 0 else "infra_failed"), process.returncode


def _classify_run_result(status: str, returncode: int | None, stderr_path: Path) -> str:
    if status != "completed" or returncode != 0:
        return status
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    infrastructure_markers = (
        "litellm.",
        "AuthenticationError",
        "APIError",
        "ContextWindowExceededError",
        "Exiting due to unknown error",
    )
    if any(marker in stderr for marker in infrastructure_markers):
        return "infra_failed"
    return "completed"


def _completed_run_keys(rows: list[dict[str, Any]]) -> set[str]:
    latest_status: dict[str, str] = {}
    for row in rows:
        run_key = row.get("run_key")
        if run_key:
            latest_status[run_key] = str(row.get("status", ""))
    return {
        run_key
        for run_key, status in latest_status.items()
        if status.startswith("completed_") or status == "completed"
    }


def _reset_run_output(raw_dir: Path) -> None:
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)


def _containers_to_remove(
    rows: list[dict[str, str]],
    task_id: str | None,
    *,
    include_running: bool,
    exclude_ids: set[str] | None = None,
) -> list[str]:
    target = _safe_name(task_id) if task_id else None
    exclude_ids = exclude_ids or set()
    removable_states = {"exited", "dead", "created"}
    selected: list[str] = []
    for row in rows:
        name = row.get("name", "")
        if row.get("id") in exclude_ids:
            continue
        state = row.get("status", "").split(maxsplit=1)[0].lower()
        if not name.startswith(OMNICODE_CONTAINER_PREFIX):
            continue
        if target and target not in name and not exclude_ids:
            continue
        if state in removable_states or (include_running and state == "up"):
            selected.append(row["id"])
    return selected


def _docker_container_rows() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Status}}\t{{.Names}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        container_id, separator, remainder = line.partition("\t")
        status, separator, name = remainder.partition("\t")
        if separator and name:
            rows.append({"id": container_id, "status": status, "name": name})
    return rows


def cleanup_omnicode_containers(
    task_id: str | None = None,
    *,
    include_running: bool = False,
    dry_run: bool = False,
    exclude_ids: set[str] | None = None,
    timeout_seconds: float | None = None,
) -> list[str]:
    """Remove finished OmniCode containers, optionally scoped to one task."""
    rows = _docker_container_rows()
    selected = _containers_to_remove(
        rows, task_id, include_running=include_running, exclude_ids=exclude_ids
    )
    if dry_run:
        return selected
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    for container_id in selected:
        command = ["docker", "rm"]
        if include_running:
            command.append("-f")
        command.append(container_id)
        remaining = None if deadline is None else max(0.1, deadline - time.monotonic())
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=remaining,
            )
        except subprocess.TimeoutExpired:
            break
    return selected


def _find_artifact(run_dir: Path, task_id: str, suffix: str) -> Path | None:
    task_output_dir = run_dir / "sweagent_output" / task_id
    preferred = task_output_dir / f"{task_id}{suffix}"
    if preferred.exists():
        return preferred
    matches = sorted(task_output_dir.glob(f"*{suffix}")) if task_output_dir.exists() else []
    return matches[0] if matches else None


def _load_trajectory_payload(run_dir: Path, task_id: str) -> dict[str, Any] | None:
    path = _find_artifact(run_dir, task_id, ".traj")
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_final_patch(run_dir: Path, task_id: str) -> str | None:
    patch_path = _find_artifact(run_dir, task_id, ".patch")
    if patch_path is not None:
        try:
            return patch_path.read_text(encoding="utf-8")
        except OSError:
            pass

    predictions_path = run_dir / "sweagent_output" / "all_preds.jsonl"
    if not predictions_path.exists():
        return None
    prediction_lines = predictions_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    for line in reversed(prediction_lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("instance_id") != task_id:
            continue
        model_patch = row.get("model_patch")
        if isinstance(model_patch, dict):
            value = model_patch.get("model_patch")
            return value if isinstance(value, str) else None
    return None


def _load_verifier_result(run_dir: Path) -> tuple[bool | None, float | None]:
    verifier_path = run_dir / "verifier.json"
    if not verifier_path.exists():
        return None, None
    try:
        payload = json.loads(verifier_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    success = payload.get("success")
    score = payload.get("verifier_score", payload.get("score"))
    return (
        success if isinstance(success, bool) else None,
        score if isinstance(score, (int, float)) else None,
    )


def _build_distillation_record(
    run_dir: Path,
    *,
    task_id: str,
    language: str,
    workflow: str,
    status: str,
    runtime: float,
) -> dict[str, Any]:
    trajectory = _load_trajectory_payload(run_dir, task_id)
    info = trajectory.get("info", {}) if trajectory else {}
    info = info if isinstance(info, dict) else {}
    stats = info.get("model_stats", {})
    stats = stats if isinstance(stats, dict) else {}
    verifier_success, verifier_score = _load_verifier_result(run_dir)

    if verifier_success is not None:
        success = verifier_success
    elif status in {"infra_failed", "timeout", "interrupted"}:
        success = None
    else:
        exit_status = str(info.get("exit_status", ""))
        success = exit_status == "submitted" if exit_status else status == "completed_success"

    return {
        "task_id": task_id,
        "language": language,
        "workflow": workflow,
        "status": status,
        "success": success,
        "verifier_score": verifier_score,
        "tokens_input": stats.get("tokens_sent"),
        "tokens_output": stats.get("tokens_received"),
        "api_calls": stats.get("api_calls"),
        "cost": stats.get("instance_cost", stats.get("total_cost")),
        "runtime": runtime,
        "trajectory": trajectory,
        "final_patch": _load_final_patch(run_dir, task_id),
    }


def run_one(
    planned: PlannedTask,
    *,
    omnicode_root: Path,
    output_root: Path,
    model: str,
    python_executable: str,
    api_base: str,
    api_key: str,
    use_apptainer: bool,
    timeout_seconds: float,
    cleanup_reserve_seconds: float,
    stop_event: threading.Event,
    immediate_stop: threading.Event,
) -> dict[str, Any]:
    task_id = str(planned.instance["instance_id"])
    run_dir = output_root / "runs" / _safe_name(task_id) / f"attempt_{planned.attempt_id:03d}"
    raw_dir = run_dir / "sweagent_output"
    _reset_run_output(raw_dir)
    existing_container_ids = {row["id"] for row in _docker_container_rows()}
    started = time.monotonic()
    execution_timeout, cleanup_budget = _split_task_budget(
        timeout_seconds, cleanup_reserve_seconds
    )
    command = _command_for_task(
        spec=planned.spec,
        instance_id=task_id,
        omnicode_root=omnicode_root,
        output_dir=raw_dir,
        model=model,
        python_executable=python_executable,
        use_apptainer=use_apptainer,
    )
    env = _build_subprocess_env(api_base, api_key)
    status, returncode = _run_subprocess(
        command,
        cwd=omnicode_root,
        env=env,
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
        timeout_seconds=execution_timeout,
        stop_event=stop_event,
        immediate_stop=immediate_stop,
    )
    status = _classify_run_result(status, returncode, run_dir / "stderr.log")
    cleanup_omnicode_containers(
        task_id,
        include_running=True,
        exclude_ids=existing_container_ids,
        timeout_seconds=cleanup_budget,
    )
    duration = time.monotonic() - started
    if duration > timeout_seconds and status in {"completed", "infra_failed"}:
        status = "timeout"
    manifest_status = "completed_success" if status == "completed" else status
    record = _build_distillation_record(
        run_dir,
        task_id=task_id,
        language=planned.spec.language,
        workflow=planned.spec.workflow,
        status=manifest_status,
        runtime=duration,
    )
    record_path = run_dir / "record.json"
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "task_id": task_id,
        "attempt_id": planned.attempt_id,
        "run_key": planned.run_key,
        "workflow": planned.spec.workflow,
        "language": planned.spec.language,
        "dataset_path": str(planned.spec.dataset_path),
        "sweagent_mode": planned.spec.sweagent_mode,
        "teacher_model": model,
        "api_base": api_base,
        "omnicode_revision": _git_revision(omnicode_root),
        "collector_version": COLLECTOR_VERSION,
        "started_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": duration,
        "status": manifest_status,
        "outcome": "agent_completed" if status == "completed" else status,
        "returncode": returncode,
        "command": command,
        "verifier_status": "not_run",
        "record_path": str(record_path),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_plan(
    specs: dict[str, list[TaskSpec]],
    *,
    target_per_group: int,
    attempts: int,
    seed: int,
    experiment_id: str,
    output_root: Path,
    skip_task_ids: set[str] | None = None,
) -> list[PlannedTask]:
    selection_path = output_root / "selection_manifest.jsonl"
    manifest_path = output_root / "manifest.jsonl"
    selection_rows = [
        row for row in _read_jsonl(selection_path) if row.get("experiment_id") == experiment_id
    ]
    manifest_rows = _read_jsonl(manifest_path)
    completed = _completed_run_keys(manifest_rows)
    selected_by_group: dict[tuple[str, str], list[str]] = {}
    for row in selection_rows:
        selected_by_group.setdefault((row["workflow"], row["language"]), []).append(row["task_id"])

    plan: list[PlannedTask] = []
    skip_task_ids = skip_task_ids or set()
    new_selection_rows: list[dict[str, Any]] = []
    known_selection_keys = {
        (row["workflow"], row["language"], row["task_id"])
        for row in selection_rows
    }
    for workflow, group_specs in specs.items():
        for spec in group_specs:
            group_key = (workflow, spec.language)
            existing_ids = selected_by_group.get(group_key, [])
            instances = load_instances(spec.dataset_path)
            by_id = {str(instance["instance_id"]): instance for instance in instances}
            missing = target_per_group - len(existing_ids)
            if missing > 0:
                candidates = select_tasks(
                    instances,
                    limit=missing,
                    seed=_group_seed(seed, workflow, spec.language),
                    completed_ids=set(existing_ids),
                    require_reviewfix_patch=spec.require_reviewfix_patch,
                )
                for instance in candidates:
                    task_id = str(instance["instance_id"])
                    existing_ids.append(task_id)
                    if (workflow, spec.language, task_id) not in known_selection_keys:
                        new_selection_rows.append(
                            {
                                "experiment_id": experiment_id,
                                "workflow": workflow,
                                "language": spec.language,
                                "task_id": task_id,
                                "seed": seed,
                            }
                        )
                        known_selection_keys.add((workflow, spec.language, task_id))
            for task_id in existing_ids[:target_per_group]:
                if task_id in skip_task_ids:
                    continue
                instance = by_id.get(task_id)
                if instance is None:
                    continue
                for attempt_id in range(1, attempts + 1):
                    run_key = f"{experiment_id}:{workflow}:{spec.language}:{task_id}:{attempt_id}"
                    if run_key not in completed:
                        plan.append(PlannedTask(spec, instance, attempt_id, run_key))
    if new_selection_rows:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("a", encoding="utf-8") as stream:
            for row in new_selection_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return plan


def collect(args: argparse.Namespace) -> int:
    load_dotenv()
    omnicode_root = Path(args.omnicode_root).resolve()
    output_root = Path(args.output_dir).resolve()
    specs = build_task_specs(omnicode_root, args.workflows, args.languages)
    experiment_id = _experiment_id(omnicode_root, args.model, args.api_base, args.seed)
    plan = _build_plan(
        specs,
        target_per_group=args.target_per_group,
        attempts=args.num_attempts,
        seed=args.seed,
        experiment_id=experiment_id,
        output_root=output_root,
        skip_task_ids=set(args.skip_task_ids),
    )
    print(f"experiment_id={experiment_id}")
    print(f"planned_runs={len(plan)}")
    if args.dry_run:
        for planned in plan:
            print(
                f"{planned.spec.language}/{planned.spec.workflow}/"
                f"{planned.instance['instance_id']}/attempt_{planned.attempt_id}"
            )
        return 0
    if not args.api_key:
        raise RuntimeError("OPENAI_API_KEY is required unless --dry-run is used")
    runner = omnicode_root / "baselines" / "sweagent" / "sweagent_regular.py"
    if not runner.exists():
        raise FileNotFoundError(f"OmniCode SWE-agent runner not found: {runner}")

    stop_event = threading.Event()
    immediate_stop = threading.Event()

    def handle_sigint(_signum: int, _frame: Any) -> None:
        if stop_event.is_set():
            immediate_stop.set()
            print("Immediate stop requested; terminating active run.", file=sys.stderr)
        else:
            stop_event.set()
            print("Graceful stop requested; no new tasks will be started.", file=sys.stderr)

    previous_handler = signal.signal(signal.SIGINT, handle_sigint)
    manifest_path = output_root / "manifest.jsonl"
    records_path = output_root / "records.jsonl"
    manifest_lock = threading.Lock()
    records_lock = threading.Lock()
    try:
        index = 0
        while index < len(plan) and not immediate_stop.is_set():
            batch = plan[index : index + max(1, args.workers)]
            if stop_event.is_set():
                break
            if args.workers == 1:
                results = [
                    run_one(
                        item,
                        omnicode_root=omnicode_root,
                        output_root=output_root,
                        model=args.model,
                        python_executable=args.omnicode_python,
                        api_base=args.api_base,
                        api_key=args.api_key,
                        use_apptainer=args.use_apptainer,
                        timeout_seconds=args.timeout_seconds,
                        cleanup_reserve_seconds=args.cleanup_reserve_seconds,
                        stop_event=stop_event,
                        immediate_stop=immediate_stop,
                    )
                    for item in batch
                ]
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [
                        executor.submit(
                            run_one,
                            item,
                            omnicode_root=omnicode_root,
                            output_root=output_root,
                            model=args.model,
                            python_executable=args.omnicode_python,
                            api_base=args.api_base,
                            api_key=args.api_key,
                            use_apptainer=args.use_apptainer,
                            timeout_seconds=args.timeout_seconds,
                            cleanup_reserve_seconds=args.cleanup_reserve_seconds,
                            stop_event=stop_event,
                            immediate_stop=immediate_stop,
                        )
                        for item in batch
                    ]
                    results = [future.result() for future in futures]
            for result in results:
                record_path = Path(result["record_path"])
                if record_path.exists():
                    record = json.loads(record_path.read_text(encoding="utf-8"))
                    _append_jsonl(records_path, record, records_lock)
                _append_jsonl(manifest_path, result, manifest_lock)
            index += len(batch)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(f"finished_runs={index}")
    print(f"remaining_runs={max(0, len(plan) - index)}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omnicode-root", default=os.getenv("OMNICODE_ROOT", "external/OmniCode"))
    parser.add_argument(
        "--output-dir", default=os.getenv("COLLECTOR_OUTPUT_DIR", "teacher_dataset")
    )
    parser.add_argument("--model", default=os.getenv("TEACHER_MODEL", "openai/cx/gpt-5.6-luna"))
    parser.add_argument("--omnicode-python", default=os.getenv("OMNICODE_PYTHON", "python"))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_API_BASE", "http://localhost:20128/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES))
    parser.add_argument("--workflows", nargs="+", choices=WORKFLOWS, default=list(WORKFLOWS))
    parser.add_argument("--target-per-group", type=int, default=10)
    parser.add_argument("--num-attempts", type=int, default=1)
    parser.add_argument(
        "--skip-task-ids",
        nargs="*",
        default=[],
        help="Task IDs to leave out of this collection run without marking them successful.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TASK_TIMEOUT_SECONDS)
    parser.add_argument(
        "--cleanup-reserve-seconds",
        type=float,
        default=CLEANUP_RESERVE_SECONDS,
        help="Budget reserved for process termination, container cleanup, and record writing.",
    )
    parser.add_argument("--use-apptainer", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    return collect(make_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
