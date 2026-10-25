# Reasoning-weight ablation: 0.15 → 0.5

`reasoning_050.json` differs from `group_loss.json` in exactly one setting:
reasoning weight increases from 0.15 to 0.5. Content=0.05, action=1,
submit=0.5, length_alpha=0.5. Same FP16 base, LoRA r32/alpha64, seed42,
LR1e-4, accumulation4, 1112 train trajectories and 20 held-out tasks.

Start a **fresh adapter from the pinned base**, not checkpoint170/304.
This isolates the objective change; use matching optimizer steps when comparing
rollouts. A shorter matched-step comparison is more informative than comparing
different runs after different numbers of updates. Two epochs remain the default;
no training starts during local preflight.

CPU preflight over the frozen train split gives mean coefficient shares:

| Group | Previous R=0.15 | New R=0.5 |
|---|---:|---:|
| Reasoning | 14.7765% | 36.3464% |
| Content | 2.1610% | 1.6100% |
| Action | 78.3522% | 58.5289% |
| Submit | 4.7102% | 3.5148% |

These are normalized coefficient shares, **not measured gradient shares**.
The shared denominator changes all effective shares although only reasoning's
configured weight changes. This test does not establish that more reasoning
supervision improves accuracy or prevents loops.

## Run after starting Vast

```bash
cd /workspace/Workflow-Aware-Agent-Distillation
git pull --ff-only origin sft/weighted-token-ce
bash cloud/vast/rtx5090/run-reasoning-loss.sh
```

The wrapper reuses the existing venv/assets, verifies CPU preflight and runs the
new-config GPU pilot, then launches detached training. Pilot updates are discarded.
Defaults are isolated from the previous experiment:

- Output: `outputs/training/rtx5090_32gb/tqs060-fp16-reasoning050/`.
- Private model repo: `hqcfmlno1/QwenStacy-4b-fp16-tqs060-reasoning050-run1`.
- HF prefix: `fp16-tqs060-reasoning050`.

HF read/write token is required on the server. Set `SFT5090_OUTPUT`,
`SFT5090_HUB_REPO` or `SFT5090_HUB_PREFIX` only to intentionally choose another
fresh destination. An output containing previous run events/checkpoints or STOP
is rejected. The new wrapper always selects `reasoning_050.json`.

Logging/checkpoint schedule remains the same: train loss every optimizer step,
validation before training and after each epoch (278/556), checkpoint every10
updates plus epoch boundaries and graceful stop. Validation is **not** measured
after each batch. Keep the run's `events.jsonl` and `train.log` when copying logs
to local; see [checkpoint validation](../../../../docs/checkpoint_validation.md).

```bash
tail -n 30 -F outputs/training/rtx5090_32gb/tqs060-fp16-reasoning050/train.log
```

To request a graceful stop after the current update and save:

```bash
touch outputs/training/rtx5090_32gb/tqs060-fp16-reasoning050/STOP
```

Default budget is8h and does not stop Vast billing. Local retention keeps last3
plus epoch boundaries. Hub pruning is off unless explicitly requested. The
existing HF412 publication issue has not been changed by this ablation.

## Evaluate after checkpoints exist

Stop training before loading a second model on the GPU. The new evaluation plan
includes the two previous FP16 runs and this one:

```bash
VALIDATION_OUTPUT=outputs/validation/reasoning050-checkpoint-loss \
  bash cloud/vast/rtx5090/validate-checkpoints.sh \
  --plan configs/evaluation/validation_reasoning050.json
```

The evaluator reports **fixed shared metrics** for every checkpoint: unweighted
CE, old token-weighted objective and group objective with reasoning=0.15. Its
`group_weighted_loss` column is intentionally comparable across runs; it is not
the new run's native reasoning=0.5 training loss. Native train/validation loss is
in this run's `events.jsonl`. Do not compare differently weighted scalars directly.
Compare rollout outcomes too, especially hypothesis revision after observations.

Local checks:

```bash
.venv/bin/python -m workflow_distillation.lora_fp16 --mode preflight \
  --config configs/training/gpus/rtx5090_32gb/reasoning_050.json \
  --output outputs/validation/reasoning050-preflight
.venv/bin/python -m pytest -q tests/test_reasoning_loss_launch.py tests/test_fp16_launch.py
```
