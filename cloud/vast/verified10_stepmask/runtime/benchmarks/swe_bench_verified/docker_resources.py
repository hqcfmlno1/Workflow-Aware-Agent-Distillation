"""Explicit Docker ownership; never prune global containers, images or caches."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


class DockerResources:
    def __init__(self, run_id, output, images=()):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
            raise ValueError("Invalid Docker run identity")
        self.run_id = run_id
        self.output = Path(output)
        self.images = list(images)
        self.report = []
        path = self.output / "docker-cleanup.json"
        if path.exists():
            self.report = json.loads(path.read_text())

    @staticmethod
    def command(*args):
        return subprocess.run(["docker", *args], text=True, capture_output=True, timeout=60)

    def record(self, **entry):
        self.report.append(entry)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "docker-cleanup.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def cleanup_containers(self):
        result = self.command("ps", "-aq", "--filter", "label=workflow.bench.run=" + self.run_id)
        if result.returncode:
            self.record(kind="containers", status="failed", error=result.stderr[-1000:])
            return
        for container in result.stdout.split():
            checked = self.command("inspect", "--format", "{{json .Config.Labels}}", container)
            if (
                checked.returncode
                or json.loads(checked.stdout or "{}").get("workflow.bench.run") != self.run_id
            ):
                continue
            removed = self.command("rm", "-f", container)
            self.record(
                kind="container",
                id=container,
                status="removed" if not removed.returncode else "retained",
                error=removed.stderr[-500:],
            )

    def cleanup_grader_containers(self, grader_run_id, instances):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", grader_run_id):
            raise ValueError("Invalid grader run identity")
        for instance in instances:
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", instance):
                raise ValueError("Invalid SWE-bench instance ID")
            name = f"sweb.eval.{instance.lower()}.{grader_run_id}"
            checked = self.command("inspect", "--format", "{{.Name}}", name)
            if checked.returncode == 0 and checked.stdout.strip() == "/" + name:
                removed = self.command("rm", "-f", name)
                self.record(
                    kind="grader_container",
                    id=name,
                    status="removed" if not removed.returncode else "retained",
                )

    def cleanup_images(self):
        for image in self.images:
            if not re.fullmatch(
                r"(?:docker.io/)?swebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+(?::[a-z0-9_.-]+)?", image
            ):
                raise ValueError(
                    "Refusing to remove an image outside the selected benchmark: " + image
                )
            present = self.command("image", "inspect", image)
            if present.returncode:
                self.record(kind="image", id=image, status="already_absent")
                continue
            users = self.command("ps", "-aq", "--filter", "ancestor=" + image)
            if users.returncode or users.stdout.strip():
                self.record(kind="image", id=image, status="retained_in_use")
                continue
            removed = self.command("image", "rm", image)  # no --force, no prune
            self.record(
                kind="image",
                id=image,
                status="removed" if not removed.returncode else "retained",
                error=removed.stderr[-500:],
            )
