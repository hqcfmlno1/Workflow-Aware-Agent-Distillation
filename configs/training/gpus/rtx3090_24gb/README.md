# Two RTX3090 cards with 24 GiB each

Canonical local profile: `profile.json`, two trajectories/GPU and pinned CPU
storage. The explicit engine uses this profile without a separate GPU trainer.
It still needs a CUDA migration pilot; see [training](../../../../docs/training.md).

Historical launchers: `scripts/training/train_qlora_vast_pool.py` and
`scripts/training/train_qlora_vast_batch4.py`. Global batch 4 uses pinned CPU
activation offload and nonreentrant checkpointing. Do not apply a 32 GiB profile.
Historical sources stay in place to preserve checkpoint hash validation.
