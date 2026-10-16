"""Download the latest complete checkpoint from an existing private HF model repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from workflow_distillation.training_checkpoints import download_latest_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Private model repo, owner/name; HF_TOKEN")
    parser.add_argument("--prefix", default="training")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = download_latest_checkpoint(args.repo, args.output, prefix=args.prefix)
    print(path.resolve())


if __name__ == "__main__":
    main()
