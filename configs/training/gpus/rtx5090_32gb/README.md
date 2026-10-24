# RTX5090 32 GB: fresh FP16 LoRA, TQS >=0.6

This is a new adapter from pinned `Qwen/Qwen3.5-4B`, not a continuation of
checkpoint1430. Frozen base weights are **FP16, without quantization**. LoRA
parameters, gradients and AdamW moments are FP32. Default: **1 GPU, microbatch1,
accumulation4, 2 epochs**. This profile has CPU tests; **no RTX5090 CUDA pilot or
throughput measurement has been performed yet**.

## Fixed data and validation

The unchanged upstream TQS scores select 1,140 successful Qwen3.8-27B
trajectories, belonging to 994 canonical tasks. `selection.json` freezes all IDs,
original scores, source hashes and the seed42 holdout.

| Partition | Trajectories | Tasks | Input tokens |
| --- | ---: | ---: | ---: |
| Train | 1,112 | 974 | 22,900,658 |
| Validation | 20 | 20 | 409,062 |
| Other attempts of validation tasks, excluded | 8 | 6 | 157,202 |

Validation quotas: 10 bug-fix, 6 feature-request, 2 enhancement, 2 other. Select one
seeded representative per task, favor different repos and language/length
coverage within each category. Every attempt of each validation task is excluded
from training. No validation result influences the split. This is a small
checkpoint-selection set, not a statistically precise benchmark.

All selected trajectories remain complete. The longest training trajectory is
**29,949 tokens**; 18 original candidates exceed28k. No silent 28k filtering,
truncation, packing or context windows. Token IDs and supervision groups are
copied byte-for-byte from the verified original corpus; no new template rendering.

`data.json` pins a private Hugging Face dataset revision and SHA256 for all 3
payloads. Large files remain under ignored `data/processed/`; credentials remain
in the server's HF cache, never in Git. Preparation code:
`scripts/data/prepare_fp16_tqs.py`.

For the new action-prioritized token-group objective and one-command launch,
see [GROUP_LOSS.md](GROUP_LOSS.md). The profile described below is the original
token-mean experiment; its settings remain available separately.

## Model and objective

- Text transformer and tied embedding/head on GPU; frozen vision on CPU.
- LoRA r32, alpha64, dropout0.05 on full-attention q/k/v/o projections,
  Gated DeltaNet `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`,
  and MLP gate/up/down projections. Vision, embeddings and head stay frozen.
- Standard AdamW, peak LR1e-4, weight decay0.01, gradient clipping1.0; cosine schedule
  with5% warmup. FP16 GradScaler starts at8. Nonfinite loss/gradients abort before
  committing the data cursor; no silent skipping of samples.
- Nonreentrant gradient checkpointing, GPU saved activations, chunked vocabulary
  projection/CE (128 supervised tokens/chunk). No full sequence-by-vocab tensor.
- Native PyTorch SDPA on the full-attention layers. Pinned FLA0.6.0 for Gated
  DeltaNet; an import failure stops startup instead of silently using a slow
  scan. CausalConv1d may use the correct PyTorch fallback; its optional compiled
  package needs a separate build. `gdn_backend=reference` is an explicit debug
  option using the blocked reference scan; changing it requires a new pilot.
- Original token-group weights retained: masked0, reasoning1, content0.75,
  action/protocol1.5, terminal action2. Each trajectory has its own weighted
  denominator; the optimizer loss is the **uniform mean over actual trajectories**
  in the accumulation group. The last group of an epoch uses its actual size.
- Teacher-forced validation uses exactly this objective and mask, with eval mode
  and no gradients, before training and after each epoch. It logs per-trajectory
  and mean loss. Validation loss alone cannot detect agent exposure bias or loops.

The peak LR is an initial adapter-training choice, consistent with the
[TRL SFT guidance](https://huggingface.co/docs/trl/sft_trainer#train-adapters-with-peft)
(approximately1e-4 for adapters). It is not a measured optimum for this dataset,
and increasing LR is not established as a remedy for reasoning loops.

## Checkpoints and resume

There are278 updates/epoch and556 updates for2 epochs. Save every10 completed
updates and at epoch boundaries. Retain last 3 recovery snapshots **plus pinned
steps278 and556**; for `--epochs 3`, also pin834. Epoch checkpoints include LoRA,
AdamW, scheduler, GradScaler, RNG and the exact shuffled multi-epoch cursor.
Save local PEFT adapters plus validation metrics in `adapters/epoch-NNN/` at
validated epoch boundaries; full resumable state is in `checkpoints/` and HF.

Resume rejects changed data, dtype, hyperparameters, epoch schedule, software,
source hashes or device world size. Choose2 or 3 epochs **before the pilot**;
changing the scheduler horizon mid-run is intentionally rejected. This run
never loads the old NF4 checkpoint. A resumed epoch-boundary checkpoint repeats
that boundary's validation without replaying its optimizer update.

HF history cleanup is an explicit `--prune-hub-history` opt-in for a new,
dedicated private model repo. It keeps the last 3 full checkpoints plus every
pinned epoch boundary and squashes that dedicated repo's old history. Never
point it at an existing model/data repo with unrelated artifacts. Without the
flag, HF preserves all uploaded checkpoints/history and storage grows.

## Run on Vast

Choose a CUDA12.8-compatible driver, Python3.11 template, one actual32 GB5090,
at least32 GBhost RAM (48–64 GB preferred) and60 GBdisk. The bootstrap creates an
isolated venv; it does not copy a local venv. Model files are downloaded by the
server directly from HF. An HF token with read access to the private dataset
and write access for checkpoint backup is required.

```bash
git clone --branch sft/weighted-token-ce https://github.com/hqcfmlno1/Workflow-Aware-Agent-Distillation.git
cd Workflow-Aware-Agent-Distillation
bash cloud/vast/rtx5090/bootstrap.sh
```

For an existing clone, run `git pull --ff-only` first. Bootstrap installs pinned
Torch2.11/cu128, Transformers, PEFT and FLA, prompts for HF login if needed,
verifies the pinned data, caches the pinned model, then runs CPU preflight.
It does **not** start training.

Run the isolated CUDA pilot, then inspect `pilot.json`:

```bash
bash cloud/vast/rtx5090/pilot.sh
```

Pilot performs a complete update on the four longest train trajectories and
four more updates on16 seeded representative trajectories. It records loss,
gradient norm, peak allocated/reserved VRAM, GPU power limits, seconds/trajectory,
tokens/s and a compute-only duration estimate. Initial FLA/Triton compilation
can make the first update slow. Pilot-updated weights are discarded; production
reloads a fresh adapter. Require at least2 GiBheadroom, exact profile/software/GPU
match, and finite loss/gradients before training. Compare reported compute time
with the8 hour budget; validation/setup/upload time is extra. If needed, explicitly
change `activation_storage` to `hybrid` and `gpu_budget_bytes` (e.g.4294967296),
then repeat the pilot before starting. No OOM retry or silent config changes.

Start detached training with backup to a **new dedicated private repo**, created
by the launcher. Replace the default repo for another independent experiment:

```bash
bash cloud/vast/rtx5090/start.sh --prune-hub-history
```

The launcher prints PID and absolute log path. SSH may disconnect after launch.
The kernel lock prevents concurrent writers in the same output directory.
Default output: `outputs/training/rtx5090_32gb/tqs060-fp16/`.

```bash
tail -n 30 -F outputs/training/rtx5090_32gb/tqs060-fp16/train.log
```

Each `step` event contains loss, LR, completed update, consumed trajectories,
seconds/trajectory, throughput and peak VRAM. `validation_complete` contains mean
validation loss; only `checkpoint_complete` confirms successful local/Hub save.
`summary.json` records complete/stopped progress. `failures.jsonl` records errors.

Stop safely at the next accumulation boundary:

```bash
touch outputs/training/rtx5090_32gb/tqs060-fp16/STOP
```

Resume on the same server:

```bash
rm -f outputs/training/rtx5090_32gb/tqs060-fp16/STOP
bash cloud/vast/rtx5090/start.sh --prune-hub-history --resume latest
```

For a new server: clone/bootstrap/pilot with the **same config/software**, then
run `start.sh --prune-hub-history --resume hub`. This downloads and verifies a
single pinned latest HF revision. It restores weights, AdamW, scheduler, scaler,
RNG and cursor. GPU power/memory admission is repeated on the new server.

To choose3 epochs, pass `--epochs 3` to **both** pilot and start. To change the
training wall-clock budget, pass `--max-seconds N` (default28800, including model
load and validation within the trainer). Signals/time budget/STOP request a
boundary save. Abrupt instance destruction or loss of credit cannot guarantee a
final save; the last published checkpoint is the recovery point. Stopping the
training process does not stop Vast billing: stop/destroy the instance in Vast
when finished.

## Local verification and review order

1. `selection.json`, `data.json`: frozen membership, task leakage, payload hashes.
2. `src/workflow_distillation/lora_fp16/data.py`: selection and full-token checks.
3. `config.py`, `model.py`: dtype, trainable blocks, attention/GDN and optimizer.
4. `engine.py`: per-trajectory loss, accumulation, validation and cursor.
5. `cli.py`, `training_checkpoints.py`: pilot, scheduler and pinned retention.
6. `cloud/vast/rtx5090/`, `launch.py`, `assets.py`: portable bootstrap and backup.

```bash
.venv/bin/python -m pytest tests/test_fp16_*.py tests/test_training_checkpoints.py
.venv/bin/python -m workflow_distillation.lora_fp16 --mode preflight
```

These checks do not establish RTX5090 memory fit, long-run stability, or completion
within8 hours. Those require the actual pilot and measured rental hardware.
