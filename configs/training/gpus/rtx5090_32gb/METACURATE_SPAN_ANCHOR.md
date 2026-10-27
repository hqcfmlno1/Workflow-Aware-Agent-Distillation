# Offline span-weighted BF16 LoRA on one RTX 5090

Implementation of the approved coding adaptation of WebKD weighting/anchor.
This does not implement student-state correction or teacher style editing.

## Fixed experiment

- Base `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- BF16, no quantization; fresh LoRA r32/alpha64/dropout .05; FP32 adapters/AdamW.
- Same 1,180 successful MetaCurate attempts / 477 tasks, <=36k, no holdout.
- 2 epochs; microbatch 1, accumulation 4; 295 updates/epoch, 590 total.
- LR 1e-4, cosine, warmup .05, weight decay .01, gradient norm clipping 1.
- Per-trajectory weighted CE, then trajectory mean over actual accumulated batch.
- Reasoning CE weights .1–1 from action leave-one-out ranks; other target weights 1.
- Upstream singleton special case: one span gets 1; tied multiple spans get .55.
- Independent anchor gamma: whole rationale section (including supervised closing
  tags) and response envelope 1, other response tokens 1e-6; supplied prefix masked.
- Target-token k3 anchor beta .03, d clamp [-20,8], FP32 exponent/reductions.
- Context/user/tool/padding are masked. Action and submit retain separate group labels.
- One longest-trajectory optimizer update for memory admission; its weights discarded.
- Same gradient checkpointing/FLA/SDPA/hybrid activation offload as vanilla baseline.

The exact design, limitations and formula are in
[the research spec](../../../../docs/research/offline_span_lora_design_2026-10-27.md).
Public upstream segmentation/rank revision is pinned in `third_party/webkd/NOTICE`.

## Local preparation (CPU, already exported)

```bash
PYTHONPATH=src:. .venv/bin/python -m scripts.data.prepare_span_lora
```

This exporter deliberately refuses an existing output; do not regenerate provenance
for an active score cache. Original tokens/masks remain byte-for-byte unchanged.
All 1,180 exports passed full IDs/groups verification. Thirty attempts have some
reasoning omitted by the official template; that text is never restored implicitly.

Measured corpus workload, before GPU timing:

- 30,978 eligible reasoning/action turns.
- 188,276 full/ablated action scoring variants.
- 1,180 additional full-trajectory reference-cache forwards.
- These are forwards of different lengths, not a measured GPU-hour estimate.

Scoring may dominate the cost. A 1,180-trajectory training duration is not a bound
on offline scoring time. The optional limited run below leaves an incomplete cache;
training correctly refuses it. Resuming never changes weights already scored.

## Transfer and environment

Use the code branch/commit and `span-lora-assets.tar.gz` prepared with this change.
At the repo root on the server, extract assets:

```bash
tar -xzf /workspace/span-lora-assets.tar.gz
export SFT5090_VENV=/venv/main
export HF_HOME=/workspace/.hf_home
```

Assets include the original compact token corpus and the new rendered provenance;
no model weights or credentials. On an already prepared server, reuse the existing
Python 3.11 environment and cached model; install code without reinstalling dependencies:

```bash
"$SFT5090_VENV/bin/python" -m pip install --no-deps -e .
```

On a fresh server, install `cloud/vast/rtx5090/requirements.txt` plus the project
before running. `bootstrap.sh` belongs to the older experiment and fetches its
holdout data: do not run it just to start this profile. Ensure the pinned base is
cached, using the same HF_HOME in scoring and training. If missing, download once:

```bash
"$SFT5090_VENV/bin/python" -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.5-4B", revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", allow_patterns=["*.json", "*.jinja", "*.safetensors", "merges.txt", "vocab.json"])'
```

The private output repo is created by the training launcher, using the normal
HF login/environment credential. Never put a token in shell arguments or Git.
Set `SFT5090_HUB_REPO` to override the dedicated new default repo.

## Offline scoring: stop and resume

CPU-only estimate:

```bash
bash scripts/training/run_metacurate_span_anchor.sh --stage estimate
```

Use tmux for scoring so SSH disconnect does not terminate the job. Optionally
measure three new turns before paying for the full corpus:

```bash
bash scripts/training/run_metacurate_span_anchor.sh --stage score --limit-turns 3
```

Continue all remaining scores (same command also resumes after interruption):

```bash
bash scripts/training/run_metacurate_span_anchor.sh --stage score --max-seconds 28800
```

Ctrl-C/SIGTERM requests stop at a saved variant boundary. The time budget is
checked between forwards, not a hard deadline inside a CUDA kernel. Reference
logps, every completed ablation score, and complete record arrays are atomic.
A killed process can repeat at most the in-flight forward. Partial caches have
no completed final index and cannot enter training. Scoring and training lock
files prevent concurrent writers to the same output.

Score output: `data/processed/metacurate1180_span_scores/`.
Log: `tail -F data/processed/metacurate1180_span_scores/scoring.log`.
`scoring.jsonl` contains per-forward/turn timings; `summary.json` records total cost.
Software/model/template/corpus/scorer identity changes reject old caches rather
than mixing scores. Do not edit scoring files after a cache is started.

## Train: admission, checkpoint policy, logs

Once the complete scoring cache exists:

```bash
bash scripts/training/run_metacurate_span_anchor.sh --stage check
bash scripts/training/run_metacurate_span_anchor.sh --stage train --max-seconds 28800
```

`check` is CPU preflight. `train` reuses an exact matching pilot or performs exactly
one longest-trajectory optimizer update, discards that model, and detaches fresh
training to `train.log`. It does not run broad/repeated pilot batches.

Default output: `outputs/training/rtx5090_32gb/metacurate1180-bf16-span-anchor-run1/`.

```bash
tail -F outputs/training/rtx5090_32gb/metacurate1180-bf16-span-anchor-run1/train.log
```

Stop at the next completed optimizer boundary:

```bash
touch outputs/training/rtx5090_32gb/metacurate1180-bf16-span-anchor-run1/STOP
```

To resume, explicitly remove STOP then:

```bash
bash scripts/training/run_metacurate_span_anchor.sh --stage train --resume latest
```

Only trusted checkpoints from this run may be loaded. `--resume hub` restores the
latest uploaded epoch endpoint; mid-epoch local recoveries are not on HF. Keep
score cache and corpus available on the resumed server.

Local recovery every 20 completed updates and at safe stop; retain latest recovery
plus pinned epoch endpoints **295 and 590**. **HF upload only at epoch endpoints**;
no upload for local recovery. Each epoch upload includes full optimizer/RNG state,
verified standard PEFT adapter, run config/identity, sidecar manifest, and snapshots of
`events.jsonl`/`train.log`. It never uploads the full base or the training corpus.
Snapshots are staged with a completion receipt and SHA256 checks before upload;
interrupted copies/adapter exports are rebuilt, never trusted from file existence.
No history squashing; only two epoch checkpoints are uploaded.

Upload failures keep the complete local endpoint and previous recovery. Transient
409/412/429/5xx failures retry up to three times; identity/concurrent-writer errors
are not bypassed. If upload still fails, resume the local endpoint to retry before
another update. Event records newer than the restored cursor are archived under
`events.before-resume-*.jsonl` and removed from the active curve to avoid duplicate
completed-update points. Hub resume restores the previous epoch event snapshot
into the active curve. Raw `train.log` remains append-only on each server; previous
server stdout is retained as `train-restored-step-*.log` and included in the next
snapshot, without replacing the live stdout file descriptor.

`events.jsonl` logs each optimizer update: total loss, weighted CE, plain CE,
anchor and scaled anchor, per-group NLL/count/weight mass, reasoning weight
quantiles, LR, gradient norm, throughput and VRAM. There is **no validation loss**
without holdout. Do not compare total loss directly across different objectives;
use plain CE and independent task rollouts for comparison with vanilla.

Copy logs back with normal scp; do not publish experiment logs to Git:

```bash
scp -P PORT root@HOST:/workspace/Workflow-Aware-Agent-Distillation/outputs/training/rtx5090_32gb/metacurate1180-bf16-span-anchor-run1/events.jsonl ./events-span-anchor.jsonl
```

## Cached scoring profile (2026-10-27)

The separately identified `metacurate_span_anchor_bf16_cached.json` keeps the
objective, segmentation, BF16 weights and unchanged-context reference probabilities.
It advances one unablated teacher-history cache between turns and deep-copies all
KV/convolution/recurrent state for each ablation. Only the original prefix and one
branch are resident. Complete variant tokenization is checked before forwarding.
Singleton turns receive the existing constant weight 1 without diagnostic logps.

The original full-prefill score output is preserved. Cached scoring writes to
`data/processed/metacurate1180_span_scores_cached/`, with its own software/config
identity. It may reuse only original reference arrays after exact source identity
and sample-hash validation; full-prefill span scores are never mixed into it.

Measured on RTX 5090, one 35,690-token / 24-span turn took 58.93 seconds full versus
7.68 seconds cached; peak allocated 10.76 GiB, reserved 12.42 GiB. **Numerical parity
failed:** BF16 action mean logp differed by up to .00604 nats, changing some span
weights by .50870. The user explicitly chose to rescore the entire pool separately
with cache after seeing this result. This is not bitwise-equivalent scoring.
Two successive shorter turns exercised history reuse on GPU and had identical
weights to full scoring. None of these timings alone predicts full-pool duration.

Resume cached scoring directly (use supervisor on the current Vast image):

```bash
export PYTHONPATH=src:.
/venv/main/bin/python -u -m scripts.training.score_span_lora \
  --config configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_bf16_cached.json \
  --output data/processed/metacurate1180_span_scores_cached \
  --reuse-reference-cache data/processed/metacurate1180_span_scores \
  --max-seconds 172800
```

On the active server, `/workspace/server-run-span.sh` uses this profile and will
perform the one-longest admission followed by training once scoring is complete.
It is managed by `workflow-span-run`; the initial timing subset is managed by
`workflow-span-measure`. Follow `/workspace/span-cache-measure.log` during the
subset and `/workspace/span-pipeline.log` afterwards. Scoring remains resumable at
each completed variant; training does not start from a partial score index.
