"""Read-only inference-checkpoint fetcher, usable directly from a checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from workflow_distillation.checkpoint_download import main  # noqa: E402

if __name__ == "__main__":
    main()
