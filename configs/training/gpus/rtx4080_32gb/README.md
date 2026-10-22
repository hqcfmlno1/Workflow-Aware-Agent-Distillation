# Two RTX4080 cards with actual 32 GiB each

Canonical local profile: `profile.json`, global batch 4 and hybrid 4 GiB/GPU,
passed explicitly to `scripts/train.py`. The refactored implementation has not
been deployed or CUDA-piloted; see [training](../../../../docs/training.md).
The measurements and launcher below describe the prior production implementation.

Current verified baseline: global batch 4, two trajectories per GPU, pinned CPU
activation offload, nonreentrant gradient checkpointing, repeated-KV efficient SDPA.

Experiment sequence: same longest untrained batch and same restored checkpoint;
compare baseline, Flash SDPA with CPU storage, Flash SDPA with GPU storage, then
bounded GPU retention if full GPU storage fails. Probe weights are never published.
Record peak allocated/reserved memory, forward/backward times, loss, gradient norm,
and differences against the baseline. Require at least 2 GiB free at measured peak.

Actual power limit on machine 43318: 225 W/card; container cannot raise it.

`flash_*.json` are experimental inputs only: the full-model GPU gradient comparison
failed despite a close loss. The production launcher rejects Flash profiles.
`efficient_gpu.json` also lacks safe peak headroom on the longest batch. Use only
a measured hybrid profile with the explicit execution-parent checkpoint binding.

The historical launcher is `scripts/training/gpus/rtx4080_32gb/train.py`. It accepts the
existing pool launch arguments plus `--execution-profile`, `--execution-parent`
and `--execution-parent-manifest-sha256`. It restores adapters, optimizer, scaler,
RNG and data cursor, and records the placement change in checkpoint identity.

## Measured on machine 43318, 2026-10-21 UTC

Same checkpoint 1027, same four longest untrained trajectories (27998, 27996,
27995, 27994 tokens), one full optimizer update per trial. Seconds below cover
forward, backward and manual gradient synchronization; exclude checkpoint IO.
VRAM is the larger peak reserved value across the two ranks.

| Activation storage | Seconds / global batch 4 | Peak VRAM | Decision |
| --- | ---: | ---: | --- |
| Native pinned CPU baseline | 99.12 | 24.71 GiB | Reference |
| Entirely GPU | 84.81 | 31.02 GiB | Only 0.48 GiB headroom; rejected |
| GPU up to 4 GiB, remainder directly into pinned CPU | 92.33 | 28.13 GiB | Deployed; checkpoint 1028 verified on HF |

The selected trial reduces compute time by 6.85% (throughput increases 7.36%).
Its loss was identical at logged precision (0.7992426157), and the full adapter
gradient relative L2 difference was 1.04%, within the declared 2% FP16 comparison
tolerance. This is a placement optimization, not a promise of bitwise identity.
All probe weights were discarded. The production launcher resumed checkpoint
1027 and saved checkpoint 1028 (2692 consumed trajectories). At HF revision
`e554b3f535dee0f49b699121f9d76194d31654bd`, `first-run/latest.json` points to
1028 and its manifest matches the server copy (SHA256
`c7b8976dfaa2c6820ec5771919f0ab1671ab8b42ca4ccd89435bea4adc0b3769`).
The checkpoint records the hybrid 4 GiB profile and the original attention.
The continuous job then resumed 1028 and entered backward for step 1029.

An earlier 2 GiB hybrid prototype took 109.35 seconds because its CPU path first
created pageable storage, then copied it again to pinned storage. That double
copy was removed before the 4 GiB trial. Flash CPU matched loss closely but failed
the full-model gradient comparison (relative L2 1.29); it was not deployed.

Production integration additionally requires the activation context to remain a
class: the existing host cache cleanup wraps its `__enter__` method. The launcher
test covers this real recovery integration. The first deployment attempt exposed
that incompatibility and stopped before training; it was fixed locally before
resuming deployment verification.
