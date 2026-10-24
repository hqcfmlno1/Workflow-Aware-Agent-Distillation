# Fresh FP16 LoRA with token-group normalized loss

This profile uses the same pinned 1112 train trajectories, 20 validation tasks,
base FP16 model, LoRA r32/alpha64, LR1e-4, accumulation4 and two-epoch schedule.
It starts a new adapter; checkpoint304 is not loaded.

For each complete trajectory, aggregate causal supervised tokens across all
turns into reasoning (1), content (2), ordinary action/protocol (3), and submit
(4). Group4 is the final explicit submit tool call plus its ending protocol;
it is separate from ordinary actions. Existing verified protocol labels remain
unchanged. Content includes ordinary assistant prose, including final prose if
present; there is no new overlapping "final response" group.

With alpha=0.5 and weights R=0.15, C=0.05, A=1.0, S=0.5:

    L_i = sum_g(w_g * N_g**(1-alpha) * mean_CE_g)
          / sum_g(w_g * N_g**(1-alpha))
    L_batch = mean_i(L_i)

Each token receives w_g*N_g**(-alpha) divided by ONE shared denominator.
Empty groups are omitted; position0 and masked context do not enter the counts.
Train and validation use the same implementation. The bundle's old denominator
is checked as provenance, then ignored for this new objective. No retokenization,
new dataset upload, truncation or change to the holdout is required.

The preflight reports the actual mean coefficient share per group. These are
loss coefficients, not measured gradient shares. Alpha=0 recovers token mean
for the chosen weights; alpha=1 recovers weighted means over content groups.
`length_alpha` is independent of LoRA `alpha`.

## One-command fresh run on a 5090

From an existing clone on the server:

```bash
git pull --ff-only origin sft/weighted-token-ce
bash cloud/vast/rtx5090/run-group-loss.sh
```

On a new server:

```bash
git clone --branch sft/weighted-token-ce https://github.com/hqcfmlno1/Workflow-Aware-Agent-Distillation.git
cd Workflow-Aware-Agent-Distillation
bash cloud/vast/rtx5090/run-group-loss.sh
```

The script bootstraps an isolated venv if missing, verifies/downloads pinned HF
assets, runs CPU preflight, and runs the exact-config CUDA pilot before launching
detached training. Keep SSH connected through setup/pilot; training detaches and
prints its PID. The pilot tests the longest trajectories and discards its updates.
Hardware/dependency/pilot failures stop the command before training.

Default output: `outputs/training/rtx5090_32gb/tqs060-fp16-group-loss/`.
Default private checkpoint repo:
`hqcfmlno1/QwenStacy-4b-fp16-tqs060-group-loss-run1`, prefix
`fp16-tqs060-group-loss`. The launcher creates it with the server's HF token;
read/write token access is needed. No history squashing is enabled by default.
Set `SFT5090_HUB_REPO` and `SFT5090_OUTPUT` for another independent experiment.
The fresh-run wrapper accepts only `--epochs N`, `--max-seconds N`, and the
optional `--prune-hub-history`. It rejects config/output/resume overrides;
use the documented environment variables or explicit resume command instead.

```bash
tail -n 30 -F outputs/training/rtx5090_32gb/tqs060-fp16-group-loss/train.log
```

Checkpoints are saved and published every10 updates, at epoch boundaries278/556,
and at graceful stop. Local retention keeps last3 plus epoch checkpoints. Hub
retains uploads/history unless the explicit dedicated-repo `--prune-hub-history`
option is passed. Train loss is logged per update; teacher-forced validation runs
before training and after each epoch. Holdout loss is not an agent success score.

Stop at a completed update and save:

```bash
touch outputs/training/rtx5090_32gb/tqs060-fp16-group-loss/STOP
```

Resume on the same server (after the old process exits):

```bash
rm -f outputs/training/rtx5090_32gb/tqs060-fp16-group-loss/STOP
bash cloud/vast/rtx5090/start.sh \
  --config configs/training/gpus/rtx5090_32gb/group_loss.json \
  --output outputs/training/rtx5090_32gb/tqs060-fp16-group-loss \
  --hub-repo hqcfmlno1/QwenStacy-4b-fp16-tqs060-group-loss-run1 \
  --hub-prefix fp16-tqs060-group-loss --resume latest
```

For another server bootstrap/assets and run `pilot.sh` with the same config and
output, then use the command above with `--resume hub`. Checkpoint identity pins
the loss settings, source/data hashes and schedule, rejecting old-objective state.
Stopping training does not stop Vast billing. The default time budget is8 hours;
it does not read balance or destroy an instance.

## Review and verification

1. `group_loss.json`: approved settings; other hyperparameters match `run.json`.
2. `src/workflow_distillation/lora_fp16/loss.py`: counts, shared denominator.
3. `engine.py`: chunked CE integration and actual accumulation mean; the historical
   `weighted_sft_loss.py` stays unchanged for old-run source checksums.
4. `cli.py`: objective fingerprint, preflight shares and pilot admission.
5. `run-group-loss.sh`: identical profile/output for preflight, pilot and train.

```bash
.venv/bin/python -m pytest tests/test_fp16_*.py tests/test_weighted_sft_loss.py
.venv/bin/python -m workflow_distillation.lora_fp16 --mode preflight \
  --config configs/training/gpus/rtx5090_32gb/group_loss.json \
  --output outputs/validation/token-group-loss-preflight
```

Numerical tests compare both loss and gradients against dense CE at alpha0/0.5/1,
including masked/empty groups, distinct submit, zero weights and a short final
accumulation group. They also verify stale legacy denominators cannot affect the
new loss. Local checks do not establish CUDA memory fit or improved agent behavior.
