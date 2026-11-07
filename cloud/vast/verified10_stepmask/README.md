# New span-anchor + Luna mask, same Verified10

Preparation only: no GPU loading, benchmark requests or interference with training.
New epoch-2 adapter: `hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-span-stepmask-run1`,
prefix `bf16-metacurate1180-span-stepmask`, checkpoint `checkpoint-00000590`.
Missing checkpoint is an error; it never falls back to the earlier unmasked run.

The exact original ten IDs and grading rows are reused. All ten are disjoint from
the new masked training index (same 477 training tasks). No old rollout is imported.
Sampling: temperature 0.6, top_p 0.95, top_k 20, seed 42. BF16 base/runtime adapter,
FP32 adapter storage, vLLM 0.32.0, native context, chunked prefill 2048, prefix cache.

- 200 actual tool executions (multiple Bash calls count individually).
- 1200 seconds per generation, including prefill/reasoning/content.
- 1800 seconds per task inference, including environment creation and tool runs.
- No campaign time/cost limit; image pull and official grading outside task budget.
- Timeout/error continues to next task. Infrastructure errors stay distinct from
  model failures; `--retry-infrastructure` starts a new attempt for those errors.
- Only explicitly submitted patches are officially graded. Workspace diffs at
  interruption/termination remain separate diagnostic artifacts.

## Preparation and launch

Run from repository root (the first command has already been run):

```bash
.venv/bin/python cloud/vast/verified10_stepmask/prepare.py stage
# After checkpoint 590 is uploaded: resolve an immutable HF commit, verify the
# masked dataset AND span-sidecar hashes, download/export the adapter on CPU.
.venv/bin/python cloud/vast/verified10_stepmask/prepare.py finalize
bash cloud/vast/verified10_stepmask/deploy.sh root@HOST SSH_PORT
ssh -N -p SSH_PORT -L 8000:127.0.0.1:8000 root@HOST
```

GPU inference runs on Vast; Docker task environments/official grading run locally.
Deployment refuses active training or any occupied GPU. It creates its own `.venv`
under `/workspace/verified10-stepmask`; training's venv is not modified. On the
current server it reuses `/workspace/.hf_home`, avoiding another base download.
Runtime compatibility/capacity is checked by the first real task, with no pilot.
Preparation does not assert that a new GPU can fit the entire native context.

After `MODEL READY`, run locally in tmux:

```bash
.venv/bin/python -u outputs/evaluation/verified10-span-stepmask-epoch2/run.py \
  >> outputs/evaluation/verified10-span-stepmask-epoch2/campaign.log 2>&1
```

Use the identical command to resume a Ctrl-C interruption. State includes reasoning,
observations, pending actions, tool count and the parked Docker workspace. An action
interrupted during shell execution requires inspection rather than automatic replay.
Resume rejects changed checkpoint/config/runtime/data. Clean downtime does not use
the saved task budget. Timeout/limit stops are terminal; they are not automatically
retried. A fresh model always gets a fresh campaign.

```bash
tail -f outputs/evaluation/verified10-span-stepmask-epoch2/campaign.log
touch outputs/evaluation/verified10-span-stepmask-epoch2/results/STOP
# STOP drains current task; Ctrl-C checkpoints the current step.
```

Reports: `results/comparison.md`, `.json`, `.csv`, with PASS totals, model turns,
tool calls, prompt/completion tokens and time. Full trajectories:
`results/span-stepmask/tasks/ID/attempt-0001/agent-messages.jsonl` and
`trajectory.json`. Interrupted generation may consume tokens absent from completed
response totals. Cleanup removes owned task containers/images; interrupted work is
retained for resume, preexisting images are kept. No global Docker prune.

`prepared.json` distinguishes staged code from a finalized adapter-ready archive.
The server archive contains no grading rows, gold patches, test patches or secrets.
API key is read from `~/.config/workflow-distillation/bench_api_key` (or
`BENCH_API_KEY_FILE`) and sent separately over SSH. HF login is local for export;
the public base reuses the remote cache. Preparation copies the proven runtime from the tracked `runtime/` directory,
records original/transformed hashes, and creates a self-contained deployment tar.
`runtime-origin.json` freezes the original historical source hashes. No historical
`outputs/` directory is required: missing Verified rows are fetched at the pinned
dataset revision and checked against the original ten-row hash.

This code does not stop or destroy the Vast instance automatically. The request is
to prepare the benchmark; starting inference and the controller is a later action.

## Inputs after a fresh clone

Install the project and the local harness dependencies (no GPU required):

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[data]'
.venv/bin/pip install -r cloud/vast/verified10_stepmask/requirements-client.txt \
  -r cloud/vast/verified10_stepmask/requirements-grader.txt
```

Restore the prepared masked corpus and projected sidecar under the documented
`data/processed/` paths before staging. These experiment inputs, annotations,
weights and checkpoint files are intentionally kept outside Git; see the training
preparation document for building/packaging them. A missing input is an error,
never a reason to substitute unmasked training data. The benchmark only needs the
two index manifests to audit task overlap and checkpoint training identity.
