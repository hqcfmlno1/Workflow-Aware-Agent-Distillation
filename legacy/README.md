# Historical compatibility

The canonical local training entrypoint is [`scripts/train.py`](../scripts/train.py).
See the [review map](../docs/training.md) before reading historical launchers.

[`first_run/source-hashes.json`](first_run/source-hashes.json) inventories the
source files used by existing training identities. Their original paths and
bytes remain intact because checkpoint manifests hash them. Physically moving
or replacing those files would invalidate old resume checks.

This directory holds the compatibility inventory, not a second implementation.
Some inventoried files are also shared mathematical/data/checkpoint primitives
called directly by the canonical engine. Historical launchers/recovery adapters
remain at their original paths for reproducing older sessions. New training
orchestration belongs in `src/workflow_distillation/training/`.
