# Remote weighted SFT through KRun

Run KRun locally to submit code; model loading, inference, and training run only
inside Kaggle. Local preparation uses the tokenizer and CPU arrays. The local
machine does not need PyTorch or student weights for this workflow.

Use the repository root as the project root. The source bundle limit is 20 MB;
attach large arrays through a private Kaggle Dataset. KRun does not automatically
launch DDP. Two T4 GPUs do not provide a single 32 GB memory space. Budget
accounting must include every allocated GPU.

## Pinned runtime

`weighted-requirements.txt` pins Transformers, PEFT, bitsandbytes, and Accelerate.
Keep Kaggle's CUDA-enabled PyTorch. The measured image has Python 3.13,
PyTorch 2.11.0+cu128, and two Tesla T4 GPUs with 15,636,037,632 bytes each.
Explicit requirements avoid installing the repository's local Python environment.

The pilot removes the image's unused `torchao<0.16` before creating LoRA: its
installed 0.10 version conflicts with the pinned PEFT version. Neither tested
LoRA path uses torchao. No local system package is changed.

Model: `Qwen/Qwen3.5-4B`, revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Official template SHA256:
`a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.

## CPU preparation

```bash
.venv/bin/python -m scripts.data.prepare_weighted_sft
```

This verifies the official render, preserves token IDs, and labels assistant
reasoning/content/action/terminal-call regions with weights 1 / 0.75 / 1.5 / 2.
Prompt, observations, and supplied assistant prefixes are masked. If a whitespace
token spans a supplied-prefix boundary, mask the complete token and report the
count. Other ambiguous boundaries fail preparation.

Outputs under `data/processed/weighted_sft_tokens_4000/`:

- `tokens.bin`: concatenated little-endian uint32 token IDs.
- `groups.bin`: aligned uint8 groups, with 0 for masked context.
- `index.json`: offsets, lengths, target counts, denominators, and checksums.

Only a completed run writes `index.json`. Partial arrays are not a usable dataset.
No filtering or windowing occurs during annotation. Generated data stays out of Git.

The complete 4,000-record annotation has been verified: 191,891,858 tokens,
4,000 distinct tasks, and zero overlap with the 50 held-out tasks. Masked context
has 99,144,253 tokens; R/C/A/F counts are 62,224,493 / 5,293,170 / 25,072,499 /
157,443. One whitespace token crossing a supplied-prefix boundary was masked.
The official template omits older reasoning in 11 records; original normalized
messages remain in the source JSONL.

After measuring the context limit and settling the effective sample count, write
a separate selection without altering source arrays:

```bash
.venv/bin/python -m scripts.data.select_weighted_sft_context \
  --index data/processed/weighted_sft_tokens_4000/index.json \
  --max-length CONTEXT_TOKENS --expected-count AGREED_COUNT \
  --output data/processed/weighted_sft_tokens_4000/context_selection.json
```

Both values are required. A count mismatch aborts. Excluded IDs remain explicit;
there is no truncation or replacement sampling. Full-history next-step examples
can still overflow at later steps.

## Technical pilots

```bash
krun doctor
krun run scripts/cloud/gpu_probe.py --gpu T4 \
  --requirements cloud/kaggle/probe-requirements.txt \
  --output outputs/training/gpu_probe.json --detach
krun run scripts/training/pilot_weighted_sft.py --gpu T4 \
  --requirements cloud/kaggle/weighted-requirements.txt \
  --input tests/test_weighted_sft_loss.py \
  --output outputs/training/precision_pilot --detach
```

The precision pilot runs numerical CE/gradient checks, then separately tests LoRA
FP16 and NF4 QLoRA with FP32 compute. Synthetic adapters are diagnostic artifacts
and must not be used as baseline checkpoints. The loss computes exact vocabulary
CE in checkpointed chunks to avoid a full sequence-by-vocabulary tensor. This
does not shorten decoder context.

For complete-trajectory tests, pass a prepared JSON array through `--input FILE`,
then forward `--records FILE --steps 2` after KRun's `--`. Each record must contain
exact `input_ids`, `groups`, and `trajectory_id`.

| Job | Result |
| --- | --- |
| `20261015-165048-b5c123` | Hardware probe: two T4 GPUs |
| `20261015-170447-f8412e` | Three numerical tests passed; QLoRA finite forward/backward at 128, 512, 2,048 tokens; LoRA blocked by old torchao |
| `20261015-171206-812999` | Both precision paths stopped on backward OOM at the shortest complete trajectory, 5,225 tokens; torchao conflict resolved |
| `20261015-171814-845d0b` | Four numerical tests passed; both precision paths completed two finite optimizer steps at 5,225 tokens, then stopped on OOM at 8,044 tokens |

The first QLoRA probe measured about 103 input tokens/s on one T4 at 2,048 tokens,
with 8.61 GB peak allocated memory. Repeated synthetic text, target density, and
short lengths do not establish full-corpus throughput or maximum context.
Reference Gated DeltaNet kernels were used; optimized kernels were unavailable.

The second context pilot tests a specific attention dispatch hypothesis. Native
SDPA GQA can select a quadratic math fallback when the GPU lacks FlashAttention
support. The registered `t4_repeated_kv` attention repeats KV heads explicitly,
uses the official SDPA mask builder, and requires the memory-efficient CUDA
backend. CPU numerical tests compare outputs and gradients against native GQA
before loading weights. An unavailable efficient backend causes an explicit error.
This changes attention dispatch, not the represented history or training targets.

Measured complete-trajectory results with this backend:

| Method | Successful length | Mean step time | Peak allocated memory | Next attempted length |
| --- | ---: | ---: | ---: | --- |
| LoRA FP16 | 5,225 | 32.53 s | 13.99 GB | 8,044: OOM |
| NF4 QLoRA, FP32 compute | 5,225 | 64.70 s | 11.81 GB | 8,044: OOM |

Each method ran on its own single T4. This is not a DDP test or a proof of the
exact maximum context; lengths between these points and other implementations
remain unmeasured. FP16 stability beyond two steps is also unverified. Both jobs
stopped without truncating inputs. Their adapters were independent pilot updates,
not a baseline checkpoint.

## Optimized full-history QLoRA probe

Current code uses **LoRA r=32, alpha=64 and PagedAdamW32bit**; see the
[current configuration](../../docs/qlora_current_config.md). This update has not
been run on GPUs. Historical measurements below used r=16/alpha=32 and
PagedAdamW8bit; they do not establish memory feasibility for the new configuration.

`scripts/training/pilot_optimized_qlora.py` keeps the original precision probe
available for comparison. The new probe uses NF4 with double quantization,
FP16 compute and autocast, FP32 adapters, a frozen FP16 tied embedding/head,
CPU-offloaded vision weights, nonreentrant gradient checkpointing with saved
inputs offloaded to pinned CPU memory, and PagedAdamW32bit. Official explicit FP32
casts in normalization and Gated DeltaNet are preserved. Repeated-KV efficient
SDPA and checkpointed vocabulary projection remain enabled.

The probe first places all text layers on GPU 0. A fresh worker then places
layers 0–11 on GPU 0 and layers 12–31 on GPU 1; this is sequential model
parallelism, not DDP. The tied embedding/head stays on GPU 0. The loss copies
small hidden-state chunks to that head while retaining gradients across devices.
Numerical tests compare cross-device loss and hidden-state gradients with a
full-logit reference before loading base weights.

Layer checkpointing also saves RNG state for every CUDA device, including a
destination not yet present in the layer's input arguments. This preserves
LoRA dropout during recomputation across the device boundary. A two-GPU
output/gradient comparison checks this separately from memory feasibility.

Each input record must include `full_token_count`, `trajectory_id`, `input_ids`,
and aligned `groups`. A mismatched length or a supervised first token aborts.
The prepared probe contains complete original records of 5,225, 8,044, 16,320,
32,766, and 40,715 tokens; the last is near the training-set median. No histories
or targets are truncated. A worker stops at its first failure. Time budgets are
checked before each optimizer step; a subprocess timeout also bounds a stuck/slow
worker and terminates its process group.

```bash
krun run scripts/training/pilot_optimized_qlora.py --gpu T4 \
  --requirements cloud/kaggle/weighted-requirements.txt \
  --input outputs/training/optimized_probe_inputs.json \
  --input tests/test_weighted_sft_loss.py \
  --output outputs/training/optimized_qlora --cell-timeout 3300 --detach -- \
  --records outputs/training/optimized_probe_inputs.json \
  --steps 2 --max-train-seconds 900
```

Reports record per-GPU allocated/reserved peaks at forward, backward, optimizer,
and cleanup stages, losses, finite-gradient checks, and a measured adapter update
on each participating GPU. Pilot updates are not baseline checkpoints. A passing
two-step probe establishes feasibility only for those records and settings;
long-run numerical stability and full-corpus throughput require further evidence.
Baseline training and CUDA/bitsandbytes resume verification remain separate work.

The initial optimized probe completed two finite updates at 5,225, 8,044, and
16,320 tokens on one T4, then stopped on OOM at 32,766. Peak allocated memory
at these successful lengths was 6.21, 7.83, and 12.61 GiB. A separate two-T4
confirmation with checkpoint RNG preservation also completed those lengths:
at 16,320 tokens, peaks were 11.29 / 10.60 GiB, with a mean step time of
210.12 seconds. It also stopped on OOM at 32,766 tokens. Jobs:
`20261015-181132-a85728` (single-GPU results) and
`20261015-182033-822f57` (confirmed partitioned results).

The `--delta-block-size 1024` option (now the default) checkpoints the reference Gated
DeltaNet computation in internal blocks. Block boundaries align with its native
64-token chunks, and recurrent state **and its gradients** pass between every
block. The original full-attention layers still attend to the complete prefix.
This is compute blocking, not trajectory windowing or detached-state training.
Tests compare output, final state, and gradients for Q/K/V, decay, beta, and
initial state against the unblocked reference, including nested checkpointing
and saved-input offload. The reference arithmetic and original token targets
are retained. Packed variable-length inputs are explicitly unsupported.

The subsequent blocked-GDN probe (`20261015-183614-aeaf3b`) completed two
finite single-T4 updates at 8,044 and 32,766 full tokens. Peak allocated memory
was 5.85 and 14.03 GiB respectively; the 32,766-token steps took about
215 seconds each. The single-T4 worker then stopped on OOM at 40,715 tokens
inside the MLP during checkpoint recomputation. This failed allocation does
not establish the memory required for a complete successful step. The final downloaded report confirms two
partitioned 32,766-token updates, with peaks of 12.63 / 11.94 GiB and step
times of 206.85 / 202.59 seconds. The partitioned worker then also stopped
on OOM at 40,715 tokens, inside MLP LoRA dropout during checkpoint
recomputation on GPU 1. No 257,109-token success has been established.
Reports are stored locally under
`outputs/training/blocked_final_pilot/krun_outputs/outputs/training/optimized_qlora/`.

No additional GPU jobs were submitted after the user's quota concern. A fresh
Kaggle API check subsequently confirmed this notebook is `COMPLETE`: its parent
process finished after both workers reported their failures. This status does
not mean all lengths passed or baseline SFT completed. Closing a local log
stream alone does not stop a remote GPU. Use approximately 30k tokens as a
conservative planning limit for this configuration; 32,766 passed this probe,
while the range between that length and 40,715 remains untested.

## One-batch DDP feasibility probe

At the user's explicit request, job `20261016-150256-04d9cf` ran exactly one
global batch: two complete successful train-pool trajectories of 29,997 and
29,994 tokens, one per T4, with gradient accumulation 1 and one optimizer step.
The two samples came from the full success corpus and distinct tasks. There
was no training warmup, sweep, automatic restart, or second GPU job.

`scripts/training/pilot_ddp_once.py` retains the blocked-GDN QLoRA settings
above. Each process owns a full text-model replica on its own GPU and offloads
frozen vision weights to CPU. DDP wraps the text decoder, synchronizes adapter
gradients, disables unused-parameter discovery and buffer broadcast, and uses
gradient bucket views. Local nonreentrant checkpointing preserves dropout RNG.
NCCL P2P and IB are disabled; collective timeout is 600 seconds and the launcher
hard timeout is `--max-train-seconds + 300` (1,200 seconds by default).

Both ranks passed: losses were 0.852301 / 0.858676, the shared pre-clipping
gradient norm was 0.503915, and a sampled adapter matrix changed by about
`5e-5` on each GPU. SHA-256 comparisons of **all** adapter gradients and all
updated adapter parameters matched across ranks. Each rank reported exactly
one optimizer step. The measured batch, including verification, took 169.50
seconds, excluding model loading. Peak allocated VRAM was 13.088 / 13.087 GiB;
peak reserved VRAM was 14.209 / 13.711 GiB. Each process reached 12.282 GiB RSS,
with the lowest observed host-available RAM at 5.820 GiB. Kaggle reported
`COMPLETE`; this is evidence for this single batch, not long-run stability or
all trajectories below 30k. No baseline checkpoint was saved.

Verified downloaded reports are preserved under
`outputs/training/ddp_once_result/`; exact input IDs and supervision groups are
in `outputs/training/ddp_once_inputs.json`. Local validation completed with
65 tests passing and one PyTorch-dependent module skipped; lint passed.

## Periodic checkpoints and resume

The optimized QLoRA and one-batch DDP pilots now use full training checkpoints:
adapter weights, per-rank optimizer/GradScaler/RNG, and the next data cursor.
The default is `--save-every-steps 5 --keep-checkpoints 2`, plus forced saves on
normal completion or a cooperative budget/signal stop. The DDP pilot still runs
only one total optimizer step and saves at step 1; loading its completed
checkpoint does not replay that update. Earlier measured jobs predate this feature.

Local saves are staged, checksummed and atomically committed. An optional existing
private Hugging Face model repo receives complete checkpoints and a latest pointer
in one commit. Local `/kaggle/working` alone is insufficient protection against
losing a session. See the [checkpoint and resume guide](../../docs/training_checkpoint_resume.md)
for paths, flags, `HF_TOKEN` via Kaggle Secrets, download/resume examples, upload
retry behavior, and quota/timeout limitations.

CPU tests cover exact resume with dropout/optimizer/scaler/scheduler, real
two-process Gloo resume and rank failure, and simulated Hub publication/download.
CUDA/bitsandbytes resume and real Hub upload remain unverified. No additional
Kaggle/GPU job was submitted to implement this feature.

## Baseline launch status

Baseline SFT has not started. Full-history optimization probes measure actual
training feasibility; the earlier 5,225-token result is not the current limit.
The effective training count and a budget for the intended baseline remain
unfinalized. Do not silently enable windows, select replacement tasks, or train
a tiny subset as the baseline. Full-state resume on CUDA/bitsandbytes and
full-corpus throughput remain unverified. Successful model partitioning is a single-process result.
The separate one-batch DDP probe above does not establish a complete baseline
trainer or verify distributed resume.

The requested endpoint is one epoch with fixed weights, followed by evaluating
the final checkpoint on SWE-bench Verified 500 under matched base/adapter settings.
The existing 50 NVIDIA validation tasks remain held out. No second epoch or data
expansion is planned. Docker availability for evaluation on Kaggle is unverified.

Inspect jobs with `krun status JOB_ID` and `krun logs JOB_ID`. Outputs belong under
`outputs/training/`; a new job must not assume it retains earlier cache/checkpoints.
