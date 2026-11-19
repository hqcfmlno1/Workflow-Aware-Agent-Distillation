# Shared LoRA trainer

The trainer serves unquantized FP16 and BF16 runs. `cli.py` checks data and
configuration, `engine.py` runs training and validation, and `checkpoints.py`
manages local state and Hub publication. `metacurate.py` loads the token corpus
shared by Vanilla and Span-anchor.

| Method | Command | Implementation |
| --- | --- | --- |
| Vanilla | `python -m workflow_distillation.vanilla_sft --help` | [Assistant-token cross-entropy](../vanilla_sft/loss.py) |
| Span-anchor | `python -m workflow_distillation.span_anchor --help` | [Span loss](../span_anchor/span_loss.py), [reference scoring](../span_anchor/span_scoring.py) and [score cache](../span_anchor/span_cache.py) |
| Other LoRA profiles | `python -m workflow_distillation.lora_sft --help` | [Shared group loss](loss.py) |

The method commands require an explicit `--config` and check that its
`loss_normalization` matches the method. The existing runbooks give the data,
selection and score-cache arguments for
[Vanilla](../../../configs/training/gpus/rtx5090_32gb/METACURATE_BF16.md)
and [Span-anchor](../../../configs/training/gpus/rtx5090_32gb/METACURATE_SPAN_ANCHOR.md).

The original `workflow_distillation.lora_fp16` imports and module commands
forward to the same implementations. Existing Span-anchor cache validation still
uses the original source hashes and keys. The scoring modules retain a few
compatibility imports so their exact source bytes match earlier cache manifests.

## Checkpoints from earlier runs

An exact training resume validates the paths and hashes of the trainer sources,
along with the model, data, config and package versions. For an older checkpoint,
use the source revision and environment recorded for that run. The last snapshot
before the package split on this branch was `01daf6a`. Check the checkpoint's
`identity.settings.sources` before choosing a revision.

```bash
git worktree add --detach ../lora-resume <source-revision>
```

Run the original `python -m workflow_distillation.lora_fp16` command from that
checkout with the saved paths and `--resume`. Use a fresh output and Hub prefix
for a new training run on the reorganized source.
