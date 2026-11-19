# Historical compatibility

The canonical local training entrypoint is [`scripts/train.py`](../scripts/train.py).
See the [review map](../docs/training.md) before reading historical launchers.

[`first_run/source-hashes.json`](first_run/source-hashes.json) inventories the
source files used by historical training identities. Migration verifies shared
source files against the recorded hashes and uses the retired launcher hashes
to identify older checkpoints.

This directory holds the compatibility inventory, not a second implementation.
Some inventoried files are also shared mathematical/data/checkpoint primitives
called directly by the canonical engine. Training orchestration lives in
`src/workflow_distillation/training/`.

The 2x RTX 4080 profile, launcher, runtime and pilot notes are available in
[commit `e31b4bd`](https://github.com/hqcfmlno1/Workflow-Aware-Agent-Distillation/tree/e31b4bd086b4ca79a660216ce08fc6684b7df22a/configs/training/gpus/rtx4080_32gb).
To resume a pinned 4080 run, use the original checkout or the revision matching
its checkpoint manifest.
