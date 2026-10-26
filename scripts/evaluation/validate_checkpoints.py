"""Run from a checkout without relying on an editable installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from workflow_distillation.validation_checkpoints import main  # noqa: E402

if __name__ == "__main__":
    main()
