# MetaCurate vanilla BF16 LoRA

Fresh experiment from pinned Qwen/Qwen3.5-4B, not continuation of the previous
Qwen-teacher adapters. A stopped session of THIS run can resume its optimizer,
scheduler, RNG and deterministic data cursor with `--resume latest`.

- All **1,180 resolved attempts / 477 tasks**, full rendered length <=36,000.
  No holdout, TQS cutoff, top-K selection or task deduplication. Original storage
  order is descending TQS; each epoch uses a deterministic shuffle (seed42+epoch).
- **24,256,659 tokens/epoch**, maximum35,998, no truncation/packing/windowing.
- Unquantized **BF16** frozen base, FP32 LoRA/gradients/AdamW state; no loss scaler.
  LoRA **r32, alpha64, dropout.05**, same text targets as previous experiments.
- **Vanilla assistant CE**: equal token weights for reasoning, content, action,
  submit and their supervised protocol. Supplied assistant generation prefixes,
  system/user prompts and tool observations are masked. CE is averaged over the
  supervised targets of each trajectory, then over four accumulated trajectories.
  No group length normalization or reasoning/action preference.
- Microbatch1, accumulation4, effective batch4; **2 epochs, 295 updates/epoch,
  590 updates / 2,360 trajectory passes**. LR1e-4, cosine decay, 30 warmup updates
  (5%), weight decay.01, clipping1.0. These optimizer/LoRA settings retain the
  previous base experiment; this is not a proposed LR ablation.
- SDPA, FLA and gradient checkpointing. Hybrid saved activations retain at most
  2GiB on GPU and offload the rest to pinned host RAM. This budget is NOT a limit
  on all VRAM. Earlier36k FP16 GPU-only probes passed short tests but lacked2GiB
  reserved-memory headroom. The new BF16/hybrid profile still requires a new CUDA
  pilot, including repeated longest-sequence updates, with the unchanged2GiB gate.
  No claim of measured BF16 throughput or full-epoch memory stability yet.

## Checkpoints and logs

Save/upload every **20 completed optimizer updates** (80 trajectories).
Always save/pin **checkpoint-00000295** and **checkpoint-00000590**, even though
neither step is a multiple of20. Local and dedicated private Hub retention:
**latest recovery + every epoch end**. At completion the two epoch snapshots
remain. This runner opts into pruning/squashing only its dedicated new Hub repo;
historical experiments are not cleaned up. Standard PEFT epoch adapters are
also exported locally to `adapters/epoch-001` and `epoch-002`.

Default output:
`outputs/training/rtx5090_32gb/metacurate1180-bf16-vanilla/`

`events.jsonl`: loss, LR, gradient norm, sequence lengths, throughput, memory and
checkpoint events; `train.log`: full stdout/stderr with detached launcher.
There is **no validation loss**: all selected data train, as requested. Rollout
benchmarks are a separate job, not automatically executed on the paid trainer.
Copy logs and PEFT exports from the server before deleting its disk. Checkpoint
publication uploads complete recovery state; it does not upload these run logs.

## Prepare locally (already generated for the pinned receipt)

```bash
.venv/bin/python -m scripts.data.prepare_metacurate_vanilla
tar -czf outputs/validation/metacurate1180-vanilla-tokens.tar.gz \
  data/processed/metacurate1180_vanilla
```

Source attribution is bundled; teacher commands are never executed. Every token
count was checked against the scored source. The official template omits earlier
reasoning before later user retries in **30 attempts**, matching the official
render and source length definition; these omissions are recorded per attempt.
The corpus has zero schema-invalid calls. These are provenance facts, not claims
that all trajectories have semantically correct reasoning or one call per turn.

## On the existing server, after receiving the local token archive

Use the new branch/commit supplied with this change; preserve cached model files.
At the repository root:

```bash
tar -xzf /workspace/metacurate1180-vanilla-tokens.tar.gz
bash cloud/vast/rtx5090/run-metacurate-bf16.sh --check
bash cloud/vast/rtx5090/run-metacurate-bf16.sh
```

Default environment is `/venv/main` and cache `/workspace/.hf_home`. No model
download occurs: weights must already be cached at the pinned revision. Hugging
Face credentials need write access; the detached launcher creates the dedicated
private repo if absent. Foreground/supervisor use requires creating that repo
first (see below).

```bash
# Resume only after THIS new run has written a checkpoint:
bash cloud/vast/rtx5090/run-metacurate-bf16.sh --resume latest

# Preview the exact training command without touching the GPU or Hub:
bash cloud/vast/rtx5090/run-metacurate-bf16.sh --dry-run

# If a supervisor owns the foreground process:
/venv/main/bin/python -c 'from huggingface_hub import HfApi; HfApi().create_repo("hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-vanilla-run1", private=True, exist_ok=True)'
SFT5090_FOREGROUND=1 bash cloud/vast/rtx5090/run-metacurate-bf16.sh
```

First start runs a disposable memory/numerics pilot; its weights are discarded.
Training starts from base only after the pilot passes. Stop requests use the
`STOP` file or SIGTERM to save at an optimizer boundary. A hard instance stop
can lose at most the updates since the latest completed save.
