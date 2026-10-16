"""Run numerical loss/gradient tests in Kaggle's PyTorch environment."""

import sys
from pathlib import Path

if not Path("/kaggle/working").exists():
    raise SystemExit("Run this check remotely; local preparation does not require PyTorch.")

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.exit(pytest.main(["tests/test_weighted_sft_loss.py", "-q", "-p", "no:cacheprovider"]))
