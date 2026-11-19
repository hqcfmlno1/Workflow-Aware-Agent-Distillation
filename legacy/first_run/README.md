# First-run source inventory

`source-hashes.json` records SHA256 values before the local training refactor.
`training.compatibility.source_hashes()` verifies the local files during migration.
Entries for the retired 4080 launcher and runtime identify older checkpoints.
Do not refresh these values to make an incompatible change pass validation.

Groups in the inventory:

- `scripts/training/train_qlora_*.py`, `gpus/rtx4080_32gb/train.py`: historical
  launcher identities.
- `cloud/kaggle/recovery/`: historical bootstrap, diagnostics and runtime adapters.
- `src/workflow_distillation/qlora_runtime.py`, `vast_*.py` and the retired
  `gpu_profiles/`: historical runtime and continuation identities.
- `weighted_sft_loss.py`, `weighted_sft_batches.py`, `t4_attention.py`,
  `optimized_qlora.py`, `ddp_probe.py`, `epoch_data.py`, `epoch_training.py`,
  `training_checkpoints.py`, `diverse_continuation.py`, `vast_pool_continuation.py`:
  shared primitives also used directly by the canonical pipeline where needed.

An implementation change gets its own identity and an explicit pinned migration;
it must not impersonate an old checkpoint identity.
