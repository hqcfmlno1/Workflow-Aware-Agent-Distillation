# GPU execution profiles

Each GPU and VRAM variant has its own folder. `profile.json` is the full validated
profile for the canonical local `scripts/train.py --gpu-profile PATH` entrypoint.
Model, weighted trajectory loss, optimizer, dataset order and checkpoint utilities
remain shared. See the [review/run guide](../../../docs/training.md).
Results belong under `outputs/training/first run/gpus/<profile>/` (ignored by Git).

The canonical engine lives in `src/workflow_distillation/training/`; it receives
the profile explicitly. The local refactor still needs a CUDA migration pilot.

Historical launchers and `src/workflow_distillation/gpu_profiles/` adapters remain
in place because checkpoint identities pin their paths and SHA256 checksums.
Other JSON files in the 4080 folder are inputs to those historical experiments;
they are not interchangeable with the canonical `profile.json` schema.

## RTX5090: fresh FP16 LoRA

[RTX5090 32GB](rtx5090_32gb/README.md) uses a separate fresh-run entrypoint,
`python -m workflow_distillation.lora_fp16`. It trains an unquantized FP16 base
with LoRA on the frozen TQS>=0.6 selection and a 20task holdout. Its Vast
bootstrap, isolated GPU pilot, detached launcher and resume are documented there.
The legacy two-GPU NF4 continuation entrypoint remains unchanged.
