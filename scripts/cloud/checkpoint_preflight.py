"""Validate the mounted epoch data and private checkpoint access on Kaggle CPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from workflow_distillation.epoch_data import EpochDataset, sha256_file  # noqa: E402
from workflow_distillation.training_checkpoints import HuggingFacePublisher  # noqa: E402


def find_index(path, expected_sha256, *, root=Path("/kaggle/input")):
    path = Path(path)
    if not path.is_file():
        matches = [p for p in Path(root).rglob(path.name) if p.parent.name == path.parent.name]
        if len(matches) != 1:
            raise ValueError("Expected one unique mounted epoch index")
        path = matches[0]
    if sha256_file(path) != expected_sha256:
        raise ValueError("Mounted epoch index checksum differs from the approved dataset")
    return path


def probe_repo(repo, *, repo_type):
    from huggingface_hub import HfApi

    info = HuggingFacePublisher(repo, repo_type=repo_type).validate()
    HfApi().auth_check(repo_id=repo, repo_type=repo_type, write=True)
    return {"repo": repo, "repo_type": repo_type, "private": info.private,
            "revision": info.sha, "content_write_permission": True}


def secret_error_details(error):
    """Report allowlisted diagnostics without exception messages or credentials."""
    details = {"error_type": type(error).__name__}
    current = error
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, HTTPError):
            details["http_status"] = current.code
            if current.code in (401, 403):
                details["secret_access_hint"] = "not_authorized_for_session"
                return details
            if current.code == 400:
                try:
                    # Inspect only in memory; never log response bodies or free-form messages.
                    body = current.read(16384).decode("utf-8", errors="replace").lower()
                except Exception:
                    body = ""
                details["response_format"] = (
                    "html" if body.lstrip().startswith("<") else
                    "json" if body.lstrip().startswith(("{", "[")) else "other"
                )
                vocabulary = {
                    "secret", "secrets", "label", "kernel", "session", "missing",
                    "invalid", "expired", "jwt", "authorization", "permission",
                    "forbidden", "attached", "enabled", "decrypt", "csrf", "xsrf",
                    "origin", "referer", "malformed", "illegal", "request", "notebook",
                    "script", "interactive", "committed", "unsupported", "proxy",
                }
                details["service_response_clues"] = sorted(
                    set(re.findall(r"\b[a-z]+\b", body)) & vocabulary
                )
                categories = {
                    "secret_not_attached_to_session": (
                        "no user secrets exist", "secret not found", "no secrets found",
                        "not attached", "not enabled", "does not have access to secret",
                    ),
                    "session_credentials_invalid": (
                        "jwt", "token expired", "invalid token", "invalid credential",
                    ),
                }
                for hint, phrases in categories.items():
                    if any(phrase in body for phrase in phrases):
                        details["secret_access_hint"] = hint
                        return details
                details["secret_access_hint"] = "secret_service_rejected_request"
                return details
        current = current.__cause__ or current.__context__
    details["secret_access_hint"] = "network_or_service_error"
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--secret-label", default="HF_TOKEN")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--wait-secret-seconds", type=int, default=1200)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/training/first run/preflight"
    )
    args = parser.parse_args()
    if not Path("/kaggle/working").is_dir():
        raise SystemExit("This preflight reads Secrets only inside the remote Kaggle CPU notebook")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"run_name": "first run", "stage": "cpu_preflight", "gpu_training_started": False}

    def emit(event):
        value = {"time": datetime.now(UTC).isoformat(), **report, **event}
        print(json.dumps(value), flush=True)
        with (args.output / "preflight.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        (args.output / "report.json").write_text(json.dumps(value, indent=2) + "\n")

    try:
        index = find_index(args.index, args.index_sha256)
        dataset = EpochDataset(index)
        report.update(resolved_index=str(index), index_sha256=args.index_sha256,
                      samples=len(dataset), tokens=dataset.manifest["tokens"])
        emit({"status": "data_verified"})
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
        deadline = time.monotonic() + args.wait_secret_seconds
        next_notice = 0
        while True:
            try:
                token = client.get_secret(args.secret_label)
                if not token or not token.startswith("hf_"):
                    raise ValueError("Expected an HF token value in the enabled Secret")
                os.environ["HF_TOKEN"] = token
                del token
                break
            except Exception as exc:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("Enable the named Secret for this Kaggle notebook") from None
                if now >= next_notice:
                    emit({"status": "awaiting_secret_access", "secret_label": args.secret_label,
                          **secret_error_details(exc)})
                    next_notice = now + 60
                time.sleep(min(10, max(0, deadline - now)))
        report.update(probe_repo(args.repo, repo_type=args.repo_type))
        emit({"status": "ready", "model_loaded": False, "repository_mutated": False})
    except Exception as exc:
        # Avoid raw server responses, headers or Secret values in logs.
        emit({"status": "failed", "error_type": type(exc).__name__,
              "hint": "Check the enabled Secret, private repo ID/type and read/write permissions"})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
