"""Sequential official Verified evaluation with durable task/grade resume.

STOP drains the current task. SIGINT/SIGTERM checkpoints its next unfinished
step and retains the owned container for resume. Inference committed before
grading is never repeated. Infrastructure exceptions pause the entire campaign.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from .config import DATASET, MINI_VERSION, fingerprint, load_config, sha256_file
from .docker_resources import DockerResources
from .runner import agent_config, agent_instance, check_docker, grader_preflight, server_info

ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
SOURCE = {
    "repo": "hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-vanilla-run1",
    "repo_type": "model",
    "prefix": "bf16-metacurate1180-vanilla",
    "revision": "c090a4592a5f1c6d7b75462f6ca6bfe50f8b4c76",
}
CHECKPOINT = "checkpoint-00000590"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Store:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / "state.json"
        self.state = json.loads(path.read_text()) if path.exists() else {"tasks": {}}

    def task(self, iid):
        return self.state["tasks"].get(iid, {"phase": "pending", "attempts": 0})

    def put(self, iid, **fields):
        self.state["tasks"][iid] = {**self.task(iid), **fields, "updated": time.time()}
        atomic_json(self.output / "state.json", self.state)

    def event(self, event, **fields):
        item = {"time": time.time(), "event": event, **fields}
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def predictions(self):
        predictions = {
            iid: value["prediction"]
            for iid, value in self.state["tasks"].items()
            if value.get("phase") in ("inferred", "done") and "prediction" in value
        }
        atomic_json(self.output / "preds.json", predictions)
        target = self.output / "predictions.jsonl"
        temp = target.with_suffix(".tmp")
        temp.write_text("".join(json.dumps(pred) + "\n" for pred in predictions.values()))
        os.replace(temp, target)


def run_task(store, row, resources, infer, grade):
    iid = row["instance_id"]
    state = store.task(iid)
    if state["phase"] == "done":
        return
    try:
        if state["phase"] != "inferred":
            folder = Path(state.get("attempt_dir", store.output / "no-attempt"))
            resuming = (folder / "resume.json").exists()
            if resuming:
                attempt = state["attempts"]
                store.put(iid, resumes=state.get("resumes", 0) + 1)
                store.event("step_resume", instance_id=iid, attempt=attempt)
            else:
                attempt = state.get("attempts", 0) + 1
                folder = store.output / "tasks" / iid / f"attempt-{attempt:04d}"
                while folder.exists():
                    attempt += 1
                    folder = store.output / "tasks" / iid / f"attempt-{attempt:04d}"
                folder.mkdir(parents=True, exist_ok=False)
            store.put(iid, phase="running", attempts=attempt, attempt_dir=str(folder), error=None)
            store.event("inference_start", instance_id=iid, attempt=attempt)
            prediction = infer(row, folder)
            # State is the transaction authority; aggregate preds is reconstructible.
            store.put(iid, phase="inferred", prediction=prediction)
            store.predictions()
            store.event("inference_committed", instance_id=iid)
        prediction = store.task(iid)["prediction"]
        store.event("grading_start", instance_id=iid)
        result = grade(row, prediction, store.output / "tasks" / iid)
        store.put(iid, phase="done", result=result, error=None)
        store.event("task_done", instance_id=iid, **result)
    except BaseException as exc:
        phase = store.task(iid)["phase"]
        store.put(
            iid,
            phase="inferred" if phase == "inferred" else "pending",
            error=f"{type(exc).__name__}: {exc}",
        )
        store.event("task_paused", instance_id=iid, error=type(exc).__name__)
        raise
    finally:
        resources.cleanup()


def campaign_summary(store, ids):
    done = {iid: store.task(iid) for iid in ids if store.task(iid)["phase"] == "done"}
    resolved = sum(bool(v["result"].get("resolved")) for v in done.values())
    return {
        "total": len(ids),
        "done": len(done),
        "pending": len(ids) - len(done),
        "resolved": resolved,
        "resolved_fraction_of_all": resolved / len(ids),
        "complete": len(done) == len(ids),
        "results": {iid: value["result"] for iid, value in done.items()},
    }


class TaskResources:
    """Remember image ownership before pulling; recover cleanup after process death."""

    def __init__(self, run_id, output, image, iid):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.iid = iid
        self.resources = DockerResources(run_id, self.output, [image])
        path = self.output / "image-ownership.json"
        if path.exists():
            self.preexisting = json.loads(path.read_text())["preexisting"]
        else:
            checked = self.resources.command("image", "inspect", image)
            if checked.returncode and "No such image" not in checked.stderr:
                raise RuntimeError("Cannot determine Docker image ownership: " + checked.stderr)
            self.preexisting = checked.returncode == 0
        atomic_json(path, {"image": image, "preexisting": self.preexisting, "cleaned": False})

    def cleanup(self):
        from .step_resume import latest_checkpoint, park_container

        checkpoint = latest_checkpoint(self.output)
        if checkpoint and checkpoint.get("phase") != "complete" and checkpoint.get("container_id"):
            if park_container(checkpoint, self.resources):
                self.resources.record(kind="container", id=checkpoint["container_id"],
                                      status="retained_for_step_resume")
                atomic_json(self.output / "image-ownership.json", {
                    "image": self.resources.images[0], "preexisting": self.preexisting,
                    "cleaned": False, "retained_for_resume": checkpoint["container_id"],
                })
                return
        self.resources.cleanup_containers()
        self.resources.cleanup_grader_containers(self.resources.run_id, [self.iid])
        if not self.preexisting:
            self.resources.cleanup_images()
        # Fail closed on cleanup failures: do not accumulate hundreds of images.
        containers = self.resources.command(
            "ps", "-aq", "--filter", "label=workflow.bench.run=" + self.resources.run_id
        )
        if containers.returncode or containers.stdout.strip():
            raise RuntimeError(
                "Benchmark container cleanup incomplete; inspect docker-cleanup.json"
            )
        if not self.preexisting:
            present = self.resources.command("image", "inspect", self.resources.images[0])
            if present.returncode == 0:
                raise RuntimeError(
                    "Benchmark image cleanup incomplete; inspect docker-cleanup.json"
                )
        atomic_json(
            self.output / "image-ownership.json",
            {"image": self.resources.images[0], "preexisting": self.preexisting, "cleaned": True},
        )


def check_host(endpoint, config):
    info = server_info(endpoint, os.environ["OPENAI_API_KEY"])
    exported = info.get("export", {})
    if exported.get("source") != SOURCE or exported.get("checkpoint") != CHECKPOINT:
        raise ValueError("Host must serve the pinned epoch-2 checkpoint 590")
    if info.get("config") != config.to_dict():
        raise ValueError("Host inference configuration differs from benchmark")
    if info.get("reasoning_loop_guard") is not False:
        raise ValueError("Disable custom reasoning-loop guard for this benchmark")
    validation = info.get("inference_validation", {})
    if not validation.get("native_generate_match") or not validation.get("native_sampling_match"):
        raise ValueError("Host native generation/sampling self-check failed")
    if exported.get("progress", {}).get("global_step") != 590:
        raise ValueError("Unexpected training step")
    return info


class TaskImageUnavailable(RuntimeError):
    """Registry failure before inference; pause without consuming a task."""


def ensure_task_image(image, event):
    checked = DockerResources.command("image", "inspect", image)
    if checked.returncode == 0:
        return
    if "No such image" not in checked.stderr:
        raise RuntimeError("Cannot inspect task image: " + checked.stderr)
    event("image_pull_start", image=image)
    # Pull separately so the native 120-second container-start timeout cannot
    # kill an otherwise healthy multi-GB download. Inherit stdout/stderr so
    # Docker layer progress is visible in campaign.log.
    for attempt in range(1,4):
        try:
            subprocess.run(["docker", "pull", image], check=True, timeout=1800)
            event("image_pull_complete", image=image)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            event("image_pull_retry", image=image, attempt=attempt, error_type=type(exc).__name__)
            if attempt==3:
                raise TaskImageUnavailable(f"Image unavailable after three pulls: {image}; task remains pending") from exc
            time.sleep(5*attempt)


def infer_task(row, folder, resolved, store, resources, *, agent_class=None):
    import httpx
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import (
        get_sb_environment,
        get_swebench_docker_image_name,
    )
    from .step_resume import ResumableAgent, UncertainToolExecution, attach_container

    class TraceAgent(agent_class or ResumableAgent):
        def add_messages(self, *messages):
            with (folder / "agent-messages.jsonl").open("a") as stream:
                for message in messages:
                    stream.write(
                        json.dumps({"time": time.time(), "message": message}, ensure_ascii=False)
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            return super().add_messages(*messages)

        def query(self):
            store.event("turn_start", instance_id=row["instance_id"], turn=self.n_calls + 1)
            result = super().query()
            store.event(
                "turn_complete",
                instance_id=row["instance_id"],
                turn=self.n_calls,
                usage=result.get("extra", {}).get("response", {}).get("usage", {}),
            )
            return result

    cfg = deepcopy(resolved)
    cfg["environment"]["pull_timeout"] = 1800
    run_args = cfg["environment"].setdefault("run_args", ["--rm"])
    run_args[:] = [arg for arg in run_args if arg != "--rm"]
    run_args.extend(["--label", "workflow.bench.run=" + resources.resources.run_id,
                     "--label", "workflow.bench.task=" + row["instance_id"]])
    checkpoint_path = folder / "resume.json"
    saved = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    if saved and saved["phase"] == "action_running":
        raise UncertainToolExecution("Shell action result is uncertain; retained container needs inspection")
    if saved and saved["phase"] == "complete":
        result = saved["result"]
        return {"instance_id": row["instance_id"], "model_name_or_path": CHECKPOINT,
                "model_patch": result.get("submission") or ""}
    model = get_model(config=cfg["model"])
    agent = env = None
    try:
        if saved and saved.get("container_id"):
            env = attach_container(saved, resources.resources)
        else:
            ensure_task_image(get_swebench_docker_image_name(row), store.event)
            # Gold test patches and grading scripts are not visible to the agent.
            env = get_sb_environment(cfg, agent_instance(row))
            if saved:
                if saved.get("restore_kind") != "audited_tracked_patch":
                    raise RuntimeError("No saved container for this step; refusing to replay tools")
                patch_path = Path(saved["workspace_patch"])
                if sha256_file(patch_path) != saved["workspace_patch_sha256"]:
                    raise ValueError("Legacy recovery patch changed")
                if patch_path.stat().st_size:
                    subprocess.run(["docker", "cp", str(patch_path),
                                    env.container_id + ":/tmp/workflow-resume.patch"], check=True)
                    restored = env.execute({"command": "git apply --binary /tmp/workflow-resume.patch"})
                    if restored.get("returncode"):
                        raise RuntimeError("Cannot restore saved tracked patch: " + restored.get("output", ""))
                store.event("legacy_workspace_restored", instance_id=row["instance_id"],
                            next_turn=saved["n_calls"] + 1,
                            limitation="Tracked files restored; old container/caches were already removed")
        environment_config = env.config.model_dump(mode="json") if hasattr(env, "config") else {}
        agent = TraceAgent(
            model, env, checkpoint_path=checkpoint_path, write_checkpoint=atomic_json,
            checkpoint_metadata={"instance_id": row["instance_id"],
                                 "task_sha256": fingerprint(agent_instance(row)),
                                 "run_id": resources.resources.run_id,
                                 "image": get_swebench_docker_image_name(row),
                                 "container_id": env.container_id,
                                 "environment_config": environment_config,
                                 "restore_kind": "full_container"},
            **{**cfg["agent"], "output_path": folder / "trajectory.json"}
        )
        try:
            result = agent.run(row["problem_statement"])
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            context_error = (
                "Context budget exceeded:" in detail or "Model native context exceeded:" in detail
            )
            if exc.response.status_code != 400 or not context_error:
                raise
            result = {
                "exit_status": "NativeContextLimit"
                if "Model native context exceeded:" in detail else "ContextLimit",
                "submission": "",
                "detail": detail,
            }
        patch = result.get("submission") or ""
        agent.mark_finished(result)
        (folder / "model.patch").write_text(patch)
        atomic_json(folder / "result.json", result)
        return {
            "instance_id": row["instance_id"],
            "model_name_or_path": CHECKPOINT,
            "model_patch": patch,
        }
    finally:
        if agent is not None:
            agent.save(folder / "trajectory.json")
        if env is not None:
            try:
                diff = env.execute({"command": "git diff --binary"})
                (folder / "workspace-at-stop.patch").write_text(diff.get("output", ""))
            finally:
                # TaskResources parks interrupted work or removes completed work.
                env.container_id = None


def grade_task(row, prediction, folder, run_id):
    if not prediction["model_patch"].strip():
        return {"resolved": False, "kind": "empty_patch"}
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / "grading-dataset.json", [row])
    (folder / "prediction.jsonl").write_text(json.dumps(prediction) + "\n")
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(folder / "grading-dataset.json"),
        "--split",
        "test",
        "--predictions_path",
        str(folder / "prediction.jsonl"),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--cache_level",
        "instance",
        "--clean",
        "false",
    ]
    atomic_json(
        folder / "grading-command.json",
        {"command": command, "swebench_version": version("swebench")},
    )
    with (folder / "grading.log").open("a") as log:
        child = subprocess.Popen(
            command, cwd=folder, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = child.wait()
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    report_path = (
        folder / "logs/run_evaluation" / run_id / CHECKPOINT / row["instance_id"] / "report.json"
    )
    if code or not report_path.exists():
        raise RuntimeError(
            f"Official grading incomplete for {row['instance_id']}; "
            f"inspect {folder / 'grading.log'}"
        )
    report = json.loads(report_path.read_text())[row["instance_id"]]
    atomic_json(folder / "official-report.json", report)
    return {"resolved": bool(report["resolved"]), "kind": "graded", "report": str(report_path)}


def prepare(output):
    from datasets import load_dataset

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "grading-dataset.json"
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        rows = sorted(
            list(load_dataset(DATASET, split="test", revision=DATASET_REVISION)),
            key=lambda row: row["instance_id"],
        )
    if len(rows) != 500 or len({r["instance_id"] for r in rows}) != 500:
        raise ValueError("Expected all 500 distinct SWE-bench Verified instances")
    grader_preflight(rows)
    meta = {
        "dataset": DATASET,
        "revision": DATASET_REVISION,
        "split": "test",
        "count": len(rows),
        "sha256": fingerprint(rows),
    }
    meta_path = output / "dataset-manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError("Pinned dataset content changed")
    atomic_json(path, rows)
    atomic_json(meta_path, meta)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = prepare(output)
        if args.prepare_only:
            print(json.dumps({"status": "prepared", "tasks": len(rows)}), flush=True)
            return
        endpoint = args.endpoint or (output / "endpoint.txt").read_text().strip()
        os.environ["OPENAI_API_KEY"] = (
            (Path.home() / ".config/workflow-distillation/bench_api_key").read_text().strip()
        )
        if version("mini-swe-agent") != MINI_VERSION:
            raise RuntimeError("mini-swe-agent version mismatch")
        check_docker()
        config = load_config(ROOT / "benchmarks/swe_bench_verified/config_unlimited.json")
        info = check_host(endpoint, config)
        resolved = agent_config(config, endpoint, info["fingerprint"])
        resolved["agent"].update(max_consecutive_format_errors=0, wall_time_limit_seconds=0)
        resolved["environment"]["container_timeout"] = "infinity"
        comparable = deepcopy(resolved)
        comparable["model"].pop("api_base")
        manifest = {
            "checkpoint": CHECKPOINT,
            "source": SOURCE,
            "server": info,
            "agent_config": comparable,
            "dataset_sha256": fingerprint(rows),
            "packages": {
                p: version(p) for p in ("mini-swe-agent", "swebench", "datasets", "httpx")
            },
            "implementation": {
                p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "resume_policy": (
                "Completed tasks skipped; inferred patches graded; "
                "interrupted inference restarted in preserved new attempt"
            ),
        }
        manifest_path = output / "run-manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Resume identity changed: checkpoint/config/data/code must remain unchanged"
            )
        atomic_json(manifest_path, manifest)
        store = Store(output)
        store.predictions()
        run_id = "verified590-" + fingerprint(str(output))[:12]
        from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

        # Recovery cleanup runs before any new task, including after an unclean local shutdown.
        for row in rows:
            owned_dir = output / "tasks" / row["instance_id"]
            ownership = owned_dir / "image-ownership.json"
            if ownership.exists() and not json.loads(ownership.read_text()).get("cleaned", False):
                TaskResources(
                    run_id, owned_dir, get_swebench_docker_image_name(row), row["instance_id"]
                ).cleanup()

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        store.event("campaign_started", checkpoint=CHECKPOINT, tasks=len(rows), endpoint=endpoint)
        try:
            for row in rows:
                if (output / "STOP").exists():
                    store.event("campaign_paused", reason="STOP file; current task drained")
                    break
                if store.task(row["instance_id"])["phase"] == "done":
                    continue
                if shutil.disk_usage(output).free < 20 * 1024**3:
                    raise RuntimeError(
                        "Less than 20 GiB free disk; campaign paused before pulling another image"
                    )
                if store.task(row["instance_id"])["phase"] != "inferred":
                    current = check_host(endpoint, config)
                    if current["fingerprint"] != info["fingerprint"]:
                        raise ValueError("Host identity changed during benchmark")
                resources = TaskResources(
                    run_id,
                    output / "tasks" / row["instance_id"],
                    get_swebench_docker_image_name(row),
                    row["instance_id"],
                )
                run_task(
                    store,
                    row,
                    resources,
                    lambda r, p, owned=resources: infer_task(r, p, resolved, store, owned),
                    lambda r, pred, p: grade_task(r, pred, p, run_id),
                )
                atomic_json(
                    output / "summary.json",
                    campaign_summary(store, [r["instance_id"] for r in rows]),
                )
        except BaseException as exc:
            store.event("campaign_paused", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            atomic_json(
                output / "summary.json", campaign_summary(store, [r["instance_id"] for r in rows])
            )


if __name__ == "__main__":
    main()
