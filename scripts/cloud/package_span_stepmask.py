"""Build a verified standalone span-anchor+step-mask server bundle without secrets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from scripts.cloud.check_span_stepmask import audit
from workflow_distillation.epoch_data import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    files = {str(p.relative_to(ROOT)): p for p in (ROOT / "src").rglob("*.py")}
    for directory in (
        "data/processed/metacurate1180_stepmask_v1",
        "data/processed/metacurate1180_span_stepmask_scores_v1",
        "third_party/webkd",
    ):
        for p in (ROOT / directory).iterdir():
            if p.is_file():
                files[f"{directory}/{p.name}"] = p
    for name in (
        "pyproject.toml", "README.md", "scripts/__init__.py",
        "scripts/data/prepare_span_stepmask.py", "scripts/cloud/check_span_stepmask.py",
        "scripts/cloud/package_span_stepmask.py", "cloud/vast/rtx5090/requirements.txt",
        "cloud/vast/rtx5090/run-span-stepmask-bf16.sh",
        "configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_stepmask_bf16.json",
        "docs/research/span_anchor_stepmask_training_2026-11-07.md",
    ):
        files[name] = ROOT / name
    for name in ("step_mask_policy_v1.txt", "step_mask_policy_noquote_v2.txt"):
        files[f"configs/data/{name}"] = args.policies / name
    hashes = {n: sha256_file(p) for n, p in sorted(files.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".building")
    with tarfile.open(temporary, "w:gz") as archive:
        for name, p in sorted(files.items()):
            archive.add(p, arcname=name, recursive=False)
        for name, value in (
            ("PACKAGE_MANIFEST.json", dict(format_version=1, files_sha256=hashes)),
            ("CPU_DATA_AUDIT.json", report),
        ):
            content = (json.dumps(value, indent=2) + "\n").encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
    with tarfile.open(temporary, "r:gz") as archive:
        if len(archive.getnames()) != len(hashes) + 2:
            raise ValueError("Unexpected package content")
        for name, expected in hashes.items():
            with archive.extractfile(name) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError(f"Archive checksum mismatch: {name}")
    temporary.replace(args.output)
    digest = sha256_file(args.output)
    args.output.with_name(args.output.name + ".sha256").write_text(
        f"{digest}  {args.output.name}\n"
    )
    print(json.dumps(dict(
        archive=str(args.output), bytes=args.output.stat().st_size, sha256=digest,
        files=len(files), **report,
    ), indent=2))


if __name__ == "__main__":
    main()
