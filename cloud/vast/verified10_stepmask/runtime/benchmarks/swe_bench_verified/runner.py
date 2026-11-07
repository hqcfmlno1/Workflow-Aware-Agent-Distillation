"""Use mini-SWE-agent's native runner and grade its patches with official SWE-bench."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import httpx

from .config import DATASET, MINI_VERSION, MODEL, REVISION, fingerprint, sha256_file

GRADER_VERSION = "4.1.0"


def grader_preflight(rows):
    """Construct official evaluation specs before spending any inference quota."""
    if version("swebench") != GRADER_VERSION:
        raise RuntimeError(f"Install swebench=={GRADER_VERSION} in the grading environment")
    from swebench.harness.test_spec.test_spec import make_test_spec

    return [make_test_spec(row, namespace="swebench") for row in rows]


def check_docker():
    if not shutil.which("docker"):
        raise RuntimeError("Docker is missing: install/start Docker on the local agent machine")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL, timeout=30)


def server_info(endpoint, api_key, *, base_dtype="float16"):
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY to the server's BENCH_API_KEY")
    if not endpoint.rstrip("/").endswith("/v1"):
        raise ValueError("Endpoint must end with /v1")
    for attempt in range(4):
        try:
            response = httpx.get(
                endpoint.rstrip("/")[:-3] + "/health",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            break
        except httpx.ConnectError:
            if attempt == 3:
                raise
            time.sleep(0.5 * 2**attempt)
    response.raise_for_status()
    info = response.json()
    identity = {k: v for k, v in info.items() if k != "fingerprint"}
    if (
        info.get("base_model") != MODEL
        or info.get("base_revision") != REVISION
        or info.get("base_dtype") != base_dtype
        or info.get("fingerprint") != fingerprint(identity)
    ):
        raise ValueError("Server identity is not the expected pinned model/weight format")
    if base_dtype == "nf4":
        quant = info.get("quantization", {})
        if not (
            quant.get("load_in_4bit") is True
            and quant.get("bnb_4bit_quant_type") == "nf4"
            and quant.get("bnb_4bit_use_double_quant") is True
            and quant.get("bnb_4bit_compute_dtype") == "float16"
        ):
            raise ValueError("NF4 server quantization differs from training")
    return info


def select_instances(rows, *, ids=None, limit=1):
    rows = sorted(rows, key=lambda row: row["instance_id"])
    if ids is not None:
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Instance IDs must be nonempty and unique")
        unknown = set(ids) - {row["instance_id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown Verified instance IDs: {sorted(unknown)}")
        return [row for row in rows if row["instance_id"] in set(ids)]
    if not 1 <= limit <= len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    return rows[:limit]


def agent_instance(row):
    """Gold patches/tests stay in the grader dataset, never in agent template variables."""
    allowed = ("instance_id", "repo", "base_commit", "version", "problem_statement")
    return {key: row[key] for key in allowed if key in row}


def agent_config(config, endpoint, server_fingerprint=None):
    import yaml
    from minisweagent.config import builtin_config_dir

    base = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())
    base["agent"].update(step_limit=config.agent_step_limit, cost_limit=0)
    base["environment"]["environment_class"] = "docker"
    templates = {
        key: value
        for key, value in base["model"].items()
        if key in ("observation_template", "format_error_template")
    }
    base["model"] = {
        **templates,
        "model_class": "benchmarks.swe_bench_verified.client.Fp16Model",
        "model_name": config.model_name,
        "api_base": endpoint.rstrip("/"),
        "server_fingerprint": server_fingerprint or "",
        "max_new_tokens": config.max_new_tokens,
        "request_timeout_seconds": config.request_timeout_seconds,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "seed": config.seed,
    }
    # The adapter reads the key from the environment; trajectories never serialize it.
    return base


def convert_predictions(source, destination, expected_ids=None):
    predictions = json.loads(Path(source).read_text())
    if not isinstance(predictions, dict):
        raise ValueError("Expected mini-SWE-agent's preds.json mapping")
    if expected_ids is not None and set(predictions) != set(expected_ids):
        raise ValueError("Predictions do not cover exactly the pinned instance list")
    converted = []
    for instance, row in sorted(predictions.items()):
        if instance != row.get("instance_id") or not isinstance(row.get("model_patch"), str):
            raise ValueError(f"Invalid prediction for {instance}")
        converted.append(
            {
                key: row[key]
                for key in (
                    "instance_id",
                    "model_name_or_path",
                    "model_patch",
                )
            }
        )
    Path(destination).write_text("".join(json.dumps(row) + "\n" for row in converted))
    return len(converted)


def run(
    config,
    endpoint,
    output,
    *,
    dataset_revision=None,
    ids=None,
    limit=1,
    retry_failed=False,
    limits=None,
):
    import yaml
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from minisweagent.run.benchmarks.swebench import process_instance
    from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager
    from rich.live import Live

    if version("mini-swe-agent") != MINI_VERSION:
        raise RuntimeError(f"Install mini-swe-agent=={MINI_VERSION}")
    check_docker()
    info = server_info(
        endpoint,
        os.environ.get("OPENAI_API_KEY"),
        **({"base_dtype": "nf4"} if config.base_dtype == "nf4" else {}),
    )
    if info["config"] != config.to_dict():
        raise ValueError("Local benchmark config differs from the serving config")
    if limits is not None:
        required = {
            "cancel_jobs",
            "client_job_identity",
            "bounded_generation",
            "prefill_self_check",
            "reasoning_loop_guard",
        }
        if not required.issubset(info.get("capabilities", [])):
            raise ValueError(
                "Update the Kaggle host: bounded benchmark requires cancellation and self-check"
            )
        validation = info.get("inference_validation", {})
        if not validation.get("cache_logits_match") or not validation.get("native_generate_match"):
            raise ValueError("Inference self-check did not pass")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if (output / "preds.json").exists() and previous is None:
        raise ValueError("Existing predictions have no provenance; use a new output directory")
    revision = dataset_revision or (previous["dataset_revision"] if previous else "main")
    revision = HfApi().dataset_info(DATASET, revision=revision).sha
    rows = select_instances(
        list(load_dataset(DATASET, split="test", revision=revision)),
        ids=ids,
        limit=limit,
    )
    if limits is not None:
        grader_preflight(rows)
    resolved = agent_config(config, endpoint, info["fingerprint"])
    # Endpoint may change after a tunnel restarts; model/config identity must remain identical.
    comparable_config = json.loads(json.dumps(resolved))
    comparable_config["model"].pop("api_base")
    manifest = {
        "dataset": DATASET,
        "dataset_revision": revision,
        "split": "test",
        "instance_ids": [row["instance_id"] for row in rows],
        "grading_dataset_sha256": fingerprint(rows),
        "server": info,
        "agent_config": comparable_config,
        "packages": {p: version(p) for p in ("mini-swe-agent", "httpx", "datasets")},
        "model_retry_attempts": 1,
        "implementation": {
            p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    }
    resources = None
    if limits is not None:
        from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

        from .docker_resources import DockerResources

        manifest["run_limits"] = limits.to_dict()
        resources = DockerResources(
            "bench-" + fingerprint(str(output.resolve()))[:16],
            output,
            [get_swebench_docker_image_name(row) for row in rows],
        )
        manifest["docker_resources"] = {"run_id": resources.run_id, "images": resources.images}
    if previous is not None and previous != manifest:
        raise ValueError(
            "Resume identity changed: use a new output for a different model/task/config"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "agent.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (output / "grading-dataset.json").write_text(json.dumps(rows) + "\n")
    predictions_path = output / "preds.json"
    existing = json.loads(predictions_path.read_text()) if predictions_path.exists() else {}
    if set(existing) - set(manifest["instance_ids"]):
        raise ValueError("Unexpected instances in existing predictions")
    pending = [
        row
        for row in rows
        if row["instance_id"] not in existing
        or (retry_failed and not existing[row["instance_id"]].get("model_patch"))
    ]
    print(f"Verified: {len(rows)} selected, {len(pending)} pending; one agent worker", flush=True)
    if pending:
        progress = RunBatchProgressManager(len(pending), output / "exit-statuses.yaml")
        with Live(progress.render_group, refresh_per_second=2):
            for row in pending:
                current = server_info(
                    endpoint,
                    os.environ.get("OPENAI_API_KEY"),
                    **({"base_dtype": "nf4"} if config.base_dtype == "nf4" else {}),
                )
                if current["fingerprint"] != info["fingerprint"]:
                    raise ValueError("Serving model changed during evaluation; stop this run")
                if limits is None:
                    process_instance(agent_instance(row), output, resolved, progress)
                else:
                    from .bounded_run import process_bounded_instance

                    try:
                        process_bounded_instance(
                            agent_instance(row), output, resolved, progress, limits, resources
                        )
                    except BaseException:
                        resources.cleanup_containers()
                        resources.cleanup_images()
                        raise
    convert_predictions(predictions_path, output / "predictions.jsonl", manifest["instance_ids"])
    return manifest


def grade(output, run_id, *, max_workers=1):
    output = Path(output).resolve()
    manifest = json.loads((output / "run-manifest.json").read_text())
    try:
        return _grade(output, run_id, max_workers=max_workers)
    finally:
        if "docker_resources" in manifest:
            from .docker_resources import DockerResources

            owned = manifest["docker_resources"]
            resources = DockerResources(owned["run_id"], output, owned["images"])
            resources.cleanup_containers()
            resources.cleanup_grader_containers(run_id, manifest["instance_ids"])
            resources.cleanup_images()


def _grade(output, run_id, *, max_workers=1):
    if version("swebench") != GRADER_VERSION:
        raise RuntimeError(f"Install swebench=={GRADER_VERSION} in the grading environment")
    check_docker()
    output = Path(output).resolve()
    manifest = json.loads((output / "run-manifest.json").read_text())
    dataset_path = output / "grading-dataset.json"
    if fingerprint(json.loads(dataset_path.read_text())) != manifest["grading_dataset_sha256"]:
        raise ValueError("Pinned grading dataset was modified")
    grader_preflight(json.loads(dataset_path.read_text()))
    convert_predictions(
        output / "preds.json", output / "predictions.jsonl", manifest["instance_ids"]
    )
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_path),
        "--split",
        "test",
        "--predictions_path",
        str(output / "predictions.jsonl"),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--cache_level",
        "instance",
        "--clean",
        "false",
    ]
    (output / "grading-command.json").write_text(
        json.dumps(
            {
                "command": command,
                "swebench_version": version("swebench"),
                "predictions_sha256": sha256_file(output / "predictions.jsonl"),
            },
            indent=2,
        )
        + "\n"
    )
    subprocess.run(command, check=True, cwd=output, timeout=2400)
