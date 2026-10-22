"""RTX4080-only execution continuation with explicit checkpoint identity migration."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cloud/kaggle/recovery"))

from workflow_distillation.epoch_data import sha256_file  # noqa: E402
from workflow_distillation.gpu_profiles.rtx4080_32gb import runtime  # noqa: E402
from workflow_distillation.training_checkpoints import (  # noqa: E402
    HuggingFacePublisher,
    _manifest,
)
from workflow_distillation.vast_microbatch import migration_manager  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "rtx4080_legacy", ROOT / "scripts/training/train_qlora_vast_4080.py",
)
legacy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)
original_parser, original_prepare = legacy.pool.parser, legacy.pool.prepare


def parser():
    result = original_parser()
    result.add_argument("--execution-profile", type=Path, required=True)
    result.add_argument("--execution-parent", type=Path, required=True)
    result.add_argument("--execution-parent-manifest-sha256", required=True)
    return result


def prepare(args):
    from filtered_continuation import publish_continuation

    parent = _manifest(args.execution_parent)
    if (sha256_file(args.execution_parent / "manifest.json")
            != args.execution_parent_manifest_sha256):
        raise ValueError("Execution parent manifest checksum differs")
    resumed = args.resume_from
    try:
        # Reconstruct the unchanged pool identity against its genuine saved parent.
        args.resume_from = str(args.execution_parent)
        trainer, cohort, _, base_identity = original_prepare(args)
    finally:
        args.resume_from = resumed
    profile = json.loads(args.execution_profile.read_text())
    if profile["attention"] != "t4_repeated_kv":
        raise ValueError("Flash SDPA has not passed the GPU gradient comparison; benchmark only")
    identity, binding = runtime.execution_identity(
        parent, args.execution_parent_manifest_sha256, base_identity, profile,
        launcher_sha=sha256_file(Path(__file__)), runtime_sha=sha256_file(Path(runtime.__file__)),
    )
    actual = _manifest(Path(resumed))
    if (sha256_file(Path(resumed) / "manifest.json") != args.execution_parent_manifest_sha256
            and actual["identity"] != identity):
        raise ValueError("Resume is neither exact execution parent nor exact profile child")

    class Publisher(HuggingFacePublisher):
        def __call__(self, checkpoint):
            return publish_continuation(self, checkpoint, binding, identity,
                                        HuggingFacePublisher.__call__)

    trainer.CONFIG = dict(identity["settings"]["config"])
    trainer.CheckpointManager = migration_manager(binding)
    trainer.HuggingFacePublisher = Publisher
    rank = os.environ.get("LOCAL_RANK", "launcher")
    (args.output / f"execution-audit-{rank}.json").write_text(json.dumps({
        "profile": profile, "identity": identity, "resume_progress": actual["progress"],
        "checkpoint_migration": binding, "loss_optimizer_and_sample_order_changed": False,
    }, indent=2) + "\n")
    if "LOCAL_RANK" in os.environ:
        runtime.install_execution(profile)
    return trainer, cohort, binding, identity


if __name__ == "__main__":
    legacy.pool.parser = parser
    legacy.pool.prepare = prepare
    legacy.__file__ = __file__
    legacy.main()
