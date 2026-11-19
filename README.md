# Workflow-Aware Agent Distillation

This repository studies how to distill software engineering workflows into
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). The student learns from
successful coding-agent trajectories that include reasoning, tool calls, tool
outputs, and patch submission. We evaluate the resulting agents on fixed
subsets of [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified).

## Training methods

| Method | What it trains | Where to start |
| --- | --- | --- |
| Base | The original Qwen3.5-4B checkpoint serves as the evaluation baseline. | [Pinned base model and evaluation settings](configs/evaluation/verified10_bf16_triplet.json) |
| Action-focused SFT | Weights reasoning, content, tool actions, and submission tokens by `1.0`, `0.75`, `1.5`, and `2.0`, respectively. Each trajectory contributes equally after token-loss normalization. | [Loss design](docs/action_focused_sft_loss_design.md) and [training guide](docs/training.md) |
| Vanilla LoRA | Gives all supervised assistant tokens equal weight within each trajectory. | [Implementation](src/workflow_distillation/vanilla_sft/) and [MetaCurate runbook](configs/training/gpus/rtx5090_32gb/METACURATE_BF16.md) |
| Span-anchor | Scores reasoning spans by their effect on predicting the next teacher action. Training combines span-weighted cross-entropy with a target-token anchor to the frozen base. | [Implementation](src/workflow_distillation/span_anchor/) and [runbook](configs/training/gpus/rtx5090_32gb/METACURATE_SPAN_ANCHOR.md) |
| Span-anchor with step masking | Keeps flagged assistant steps in context while excluding their tokens from both training losses. | [Step review policy](docs/research/deepseek1180_llm_step_masking_2026-11-07.md) and [training guide](docs/research/span_anchor_stepmask_training_2026-11-07.md) |
| CoReD | Trains a council of LoRA experts, uses their predictions to weight reasoning supervision, merges the expert updates with TIES, and trains a student adapter on the merged base. | [Method and stage commands](src/workflow_distillation/cored/README.md) |

The original Action-focused SFT pipeline uses successful Qwen3.8-27B Thinking
runs from NVIDIA Open-SWE-Traces v1.2 with mini-SWE-agent. The MetaCurate
experiments use a separate corpus of 1,180 successful trajectories across 477
tasks. Training and benchmark overlap is checked using normalized repository
and pull request identifiers. See the [source manifest](configs/datasets/nvidia_mini_swe_agent_v1_2_action_focused_sft.json),
[data inventory](data/README.md), and [Verified overlap audit](reports/swebench_verified_overlap_2026-10-12/).

## Evaluation

The repository tracks the task selections for the [10-task comparison](configs/evaluation/verified10_bf16_triplet.json)
and the [20-task diagnostic pilot](configs/evaluation/verified20_span_anchor.json).
The [step-mask evaluation harness](cloud/vast/verified10_stepmask/README.md)
reuses the original ten tasks and grades submitted patches with the official
SWE-bench tests. Run outputs record task outcomes, generation time, token counts,
and tool calls. Infrastructure errors are recorded separately from graded failures.

The final CoReD checkpoint resolved **11 of 27 tasks** in a separate diagnostic
run. This is a subset result. The historical baseline runs used different budgets
and continuation rules, which limits direct comparison.

## Getting started

Use Python 3.11. From the repository root, install the data dependencies and
inspect the method entrypoints. The CoReD command also loads the training
dependencies:

```bash
uv sync --extra data
uv run python -m workflow_distillation.vanilla_sft --help
uv run python -m workflow_distillation.span_anchor --help
uv run --extra data --extra train python -m workflow_distillation.cored --help
```

Use the method runbooks for the exact model revision, input preparation, GPU
dependencies, and checkpoint commands. Vanilla and Span-anchor share the
[FP16/BF16 LoRA trainer](src/workflow_distillation/lora_sft/README.md). The
Action-focused QLoRA continuation uses [`scripts/train.py`](scripts/train.py).
For exact checkpoint resume, use the source revision and environment recorded
for that run. See the [checkpoint guide](docs/training_checkpoint_resume.md).

## Repository layout

```text
src/workflow_distillation/  Training methods, losses, and shared trainer code
configs/                    Dataset manifests, training recipes, and evaluation task lists
scripts/                    Data preparation, scoring, training, and checkpoint tools
cloud/                      Kaggle and Vast launchers and evaluation infrastructure
docs/                       Method notes and runbooks
reports/                    Tracked dataset and benchmark-overlap audits
legacy/                     Source fingerprints and compatibility notes
```

Generated corpora, model weights, caches, checkpoints, and run logs are stored
outside Git. Local artifacts are typically under `data/`, `outputs/`, or
`draft/`. See the [output inventory](outputs/README.md) for where to look.
