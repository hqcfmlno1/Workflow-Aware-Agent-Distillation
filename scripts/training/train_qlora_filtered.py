"""Resume the approved checkpoint with an immutable, filtered suffix schedule."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    if "--resume-from" not in sys.argv:
        raise ValueError("Filtered continuation requires a resume checkpoint")
    from filtered_continuation import configure_trainer, load_plan

    plan = load_plan(os.environ["FIRST_RUN_CONTINUATION_PLAN"],
                     expected_sha256=os.environ["FIRST_RUN_CONTINUATION_PLAN_SHA256"])
    original = root / "scripts/training/train_qlora_epoch.py"
    spec = importlib.util.spec_from_file_location("first_run_original_epoch", original)
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    configure_trainer(trainer, plan, entrypoint=__file__)
    # Fail on changed packages/config/math before allocating model weights.
    source = root / "src/workflow_distillation"
    trainer.training_identity(
        __file__, model=trainer.MODEL, revision=trainer.REVISION,
        records_sha256=plan["source_index_sha256"],
        settings={"config": trainer.CONFIG, "seed": plan["seed"], "runtime_sources": {
            name: trainer.sha256_file(source / name) for name in
            ("epoch_data.py", "epoch_training.py", "qlora_runtime.py",
             "weighted_sft_batches.py", "training_checkpoints.py")}},
    )
    trainer.main()


if __name__ == "__main__":
    main()
