# Kaggle two T4 cards (approximately 14.6 GiB available each)

Canonical local profile: `profile.json`, one trajectory/GPU, pinned CPU storage.
The explicit engine retains observed pinned-memory calibration, conservative
reclaimable-cache accounting and serial execution when only one rank fits RAM.
It still needs a CUDA migration pilot; see [training](../../../../docs/training.md).

Historical runtime: `src/workflow_distillation/qlora_runtime.py` and
`cloud/kaggle/recovery/`. Uses repeated-KV efficient SDPA and CPU activation
offload. Execution and host-RAM admission depend on the attached recovery version.
Historical sources stay in place to preserve checkpoint hash validation.
