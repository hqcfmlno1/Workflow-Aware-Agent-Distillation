# Span-anchor + Luna step masks — server training preparation

## Experiment

Fresh LoRA initialized from Qwen/Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16 without quantization.
Keep all **1,079 original Luna masks**, including reasoning/action conflicts.
The proposal to restore 345 repetition masks was not applied. Annotation validity
is structural, not a guarantee that every semantic decision is correct.

All 1,180 successful <=36k-token trajectories remain; no holdout, matching the
previous control. Preserve all 24,256,659 input tokens, max length35,998. Masked
text still conditions subsequent predictions. Exclude 287,513 supervised tokens
(3.7044%), leaving 7,473,970 targets. Selected assistant steps lose supervision
for their reasoning, content and tool/protocol targets, as in the prepared masks.

This is **not** continuation of the prior span-anchor adapter or the vanilla
step-mask experiment. New output, dedicated HF repo/prefix, fresh optimizer.

## Objective

For each trajectory, S contains only retained, originally supervised targets.
History x[<i] is the intact original token prefix, including masked steps.

```text
ce[i] = -log p_student(y[i] | x[<i])
q[i] = log p_frozen_base(y[i] | x[<i])
L_ce = sum(i in S, w[i] * ce[i]) / sum(i in S, w[i])
d[i] = clip(q[i] + ce[i], -20, 8)
k[i] = exp(d[i]) - d[i] - 1
L_anchor = sum(i in S, gamma[i] * k[i]) / sum(i in S, gamma[i])
L = L_ce + 0.03 * L_anchor
```

Use original offline adaptive reasoning weights w in[0.1,1]; other retained
targets have w=1. Original gamma=1 for rationale sections/tags and response
envelope, gamma=1e-6 for other supervised response tokens. Neither term includes
masked targets in its numerator or denominator. This is the same target-token
anchor as the previous run, not full-vocabulary KL. Mean trajectory losses over
the4-trajectory accumulation, not one global token mean.

## Exact cache reuse

Reconstructed1180 complete records from local backup SHA256
`e8d3ecac733bf373d1f23e54769742abd34f268777c7e16a0c7de56f1e36d3a5`.
All three reconstructed original arrays match the sidecar published with the
previous epoch2 checkpoint, byte for byte. New sidecar selects q, w and gamma by
absolute retained target positions, verifies unchanged input IDs and groups
only changing to zero. It does not rerank spans, rescore or impute missing values.

Derivation and checksums are saved in the new sidecar index/summary. Retained
values are exactly the old float32 values. No new GPU scoring or teacher calls.

## Config and checkpoints

Config: `configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_stepmask_bf16.json`.
LoRA r32/alpha64/dropout0.05, FP32 adapter/optimizer, LR1e-4, cosine,
warmup5%, weight decay0.01, grad clip1, microbatch1/accumulation4, seed42.
Two epochs,295 updates/epoch,590 total. FLA/SDPA, gradient checkpointing,
hybrid activation storage, chunked loss128 as before.

Local recovery every20 updates and on graceful stop; retain latest recovery and
epoch endpoints295/590. HF uploads **only epoch endpoints**, including logs and
standard PEFT adapter export. Upload failure retains the local checkpoint and
must be retried before continuing. No full-base copies. Epoch upload alone is
not a backup of the input corpus/score binaries: keep this deployment archive.

Default repo: `hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-span-stepmask-run1`.
Repo creation/publication happens only when the server launcher is actually run.
No GPU/server deployment/training/publication was performed during preparation.

## Server setup and start

Use a Python3.11 CUDA template and one RTX5090 32GB. Runtime must match the
historical scoring identity: torch2.11.0+cu128, transformers5.19.0, numpy2.1.3,
FLA/fla-core0.6.0, Triton3.6.0; additional requirements are pinned in the archive.
Do not run the generic legacy bootstrap: it targets another dataset/profile.

Transfer `metacurate1180-span-stepmask-server.tar.gz` and its `.sha256` to
`/workspace`, then extract into a fresh dedicated directory:

```bash
cd /workspace
sha256sum -c metacurate1180-span-stepmask-server.tar.gz.sha256
mkdir -p /workspace/span-stepmask
tar -xzf metacurate1180-span-stepmask-server.tar.gz -C /workspace/span-stepmask
cd /workspace/span-stepmask
export SFT5090_VENV=/workspace/venvs/span-stepmask
export HF_HOME=/workspace/.hf_home
python3.11 -m venv "$SFT5090_VENV"
"$SFT5090_VENV/bin/python" -m pip install -r cloud/vast/rtx5090/requirements.txt
"$SFT5090_VENV/bin/python" -m pip install -e .
"$SFT5090_VENV/bin/python" -m pip check
```

If using an existing matching Python3.11 environment, set SFT5090_VENV to it
instead of creating another. Reuse HF_HOME/model files when present; no model
weights are included. Authenticate HF in the terminal, never in source/chat:

```bash
"$SFT5090_VENV/bin/python" -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
"$SFT5090_VENV/bin/python" - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
c = json.loads(Path('configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_stepmask_bf16.json').read_text())
snapshot_download(c['model'], revision=c['revision'], allow_patterns=[
    '*.json', '*.jinja', '*.safetensors', 'merges.txt', 'vocab.json',
])
PY
bash cloud/vast/rtx5090/run-span-stepmask-bf16.sh --check
bash cloud/vast/rtx5090/run-span-stepmask-bf16.sh
```

The launcher skips the disposable longest-trajectory pilot, as explicitly
requested: this pool previously trained on32GB. It passes `--skip-pilot`, records
that choice in the initializing event, and launches training detached from SSH
after data/cache/runtime checks. Hardware checks remain in the trainer; no
passing pilot report is fabricated. Current-run GPU memory is not remeasured
before training.
It refuses used output without explicit resume and refuses an existing STOP file.
Default runtime limit8hours is resumable; override with `--max-seconds N` if needed.
For deliberate local-only operation pass `--no-hub`.

## Logs, stop and resume

```bash
export RUN_DIR=/workspace/span-stepmask/outputs/training/rtx5090_32gb/metacurate1180-bf16-span-stepmask-run1
tail -n 30 -F "$RUN_DIR/train.log"
tail -n 5 -F "$RUN_DIR/events.jsonl"
touch "$RUN_DIR/STOP"
```

Wait for a completed recovery checkpoint and process exit before stopping GPU.
Remove STOP explicitly before resume:

```bash
rm "$RUN_DIR/STOP"
bash cloud/vast/rtx5090/run-span-stepmask-bf16.sh --resume latest
# On a different server, restore the same archive/runtime and use:
bash cloud/vast/rtx5090/run-span-stepmask-bf16.sh --resume hub
```

Hub resume restores the latest **uploaded epoch endpoint**, not unpublished
mid-epoch work. Copy latest local recovery off server to preserve that work.
Events include trajectory_id/microbatch, weighted/plain CE, anchor, total loss,
group NLL/counts, LR, grad norm, throughput, VRAM and checkpoint events. There is
no independent validation loss because no holdout is configured.

## Local verification limits

Run `python -m scripts.cloud.check_span_stepmask` with PYTHONPATH=src:. for a CPU
checksum/alignment audit. It reports local/runtime differences explicitly. The
local CPU environment differs from historical CUDA/FLA, so strict training
preflight is expected to refuse it; no guard is disabled. Actual server preflight
requires the pinned packages and exact scoring sources. Hardware checks happen
on the server; no separate GPU pilot will run.
