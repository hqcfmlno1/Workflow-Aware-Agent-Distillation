"""Canonical QLoRA entrypoint. Use --help; preflight needs no GPU allocations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from workflow_distillation.training.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
