"""Fetch only pinned private token assets and the pinned public base model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from workflow_distillation.epoch_data import sha256_file
from workflow_distillation.lora_fp16.config import RunConfig

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "configs/training/gpus/rtx5090_32gb"


def verify_assets(directory, receipt):
    for name, expected in receipt["files"].items():
        path = Path(directory) / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Missing or corrupt pinned asset: {name}")


def copy_verified(cached, target, expected):
    cached, target = Path(cached), Path(target)
    if sha256_file(cached) != expected:
        raise ValueError(f"Remote checksum differs: {target.name}")
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(cached, temporary)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=PROFILE / "data.json")
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "data/processed/tqs1140_holdout20_fp16"
    )
    parser.add_argument("--with-model", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    receipt = json.loads(args.receipt.read_text())
    if not HfApi().repo_info(receipt["repo_id"], repo_type=receipt["repo_type"]).private:
        raise ValueError("Expected private training dataset")
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, expected in receipt["files"].items():
        target = args.destination / name
        if target.is_file() and sha256_file(target) == expected:
            continue
        cached = hf_hub_download(
            receipt["repo_id"],
            f"{receipt['path']}/{name}",
            repo_type=receipt["repo_type"],
            revision=receipt["revision"],
        )
        copy_verified(cached, target, expected)
    verify_assets(args.destination, receipt)
    print("Pinned token corpus downloaded and SHA256 verified.", flush=True)
    if args.with_model:
        config = RunConfig()
        snapshot_download(
            config.model,
            revision=config.revision,
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json"],
        )
        print("Pinned base cached; no model loaded and no GPU used.", flush=True)


if __name__ == "__main__":
    main()
