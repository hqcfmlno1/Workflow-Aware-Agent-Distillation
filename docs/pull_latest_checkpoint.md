# Pull latest checkpoint directly from the training repo

The module `workflow_distillation.checkpoint_download` only **reads** an existing
Hugging Face model/dataset repo. It never creates a repo, uploads a snapshot or
updates a remote pointer. The default source is the current reasoning-weight2 run:

- Repo: `hqcfmlno1/QwenStacy-4b-fp16-tqs060-reasoning200-run1`.
- Prefix: `fp16-tqs060-reasoning200`.

## Use on local or a server

Use the existing HF login or `HF_TOKEN` with **read access** to the private repo.
Do not paste tokens into commands or chat.

```bash
.venv/bin/python scripts/evaluation/pull_latest_checkpoint.py
```

Equivalent when the project is installed:

```bash
python -m workflow_distillation.checkpoint_download
```

Only inspect the current latest checkpoint and progress (no adapter download):

```bash
python -m workflow_distillation.checkpoint_download --metadata-only
```

Download another run, or pin an older checkpoint/revision:

```bash
python -m workflow_distillation.checkpoint_download \
  --repo hqcfmlno1/QwenStacy-4b-fp16-tqs060-group-loss-run1 \
  --prefix fp16-tqs060-group-loss \
  --output outputs/checkpoints/group-loss
```

Options include `--checkpoint checkpoint-00000170`, `--revision <commit>`, and
`--repo-type model|dataset`. The default checkpoint is `latest`, revision `main`.
**Use a separate output directory per training run** so equal step names cannot
silently replace weights from a different run.

## Result and integrity

Default output: `outputs/checkpoints/reasoning200/`:

```text
latest.json
checkpoint-XXXXXXXX/
  manifest.json
  adapter.pt
  source.json
```

`latest.json` and `source.json` are local receipts containing checkpoint name,
immutable HF revision, training progress/config and checksums. The module resolves
one HF commit before reading the remote pointer, then downloads all artifacts at
that same revision, even if training publishes a newer checkpoint meanwhile.

Only adapter weights and manifest are downloaded. Optimizer `rank-*.pt` shards and
base-model weights are excluded. This is an **inference** download; use the
existing `scripts/training/fetch_training_checkpoint.py` for full training resume.
The raw `adapter.pt` still needs PEFT export for inference; the existing Kaggle
host performs that export, tensor-shape/dtype/finite checks and model loading.

Downloads are checksum-verified in a staging directory and published with a
rename after success. A damaged download never becomes the selected checkpoint.
Re-running reuses a valid local checkpoint; source conflicts fail instead of
overwriting another run. A local lock prevents concurrent writes. If a checkpoint
has been removed from the source repo/history, download fails instead of silently
switching to another model. Read-only access cannot preserve deleted remote files.

Python API:

```python
from workflow_distillation.checkpoint_download import download_checkpoint, resolve_checkpoint

snapshot = resolve_checkpoint()  # small metadata only
result = download_checkpoint(destination="outputs/checkpoints/reasoning200")
print(result["checkpoint"], result["revision"], result["directory"])
```

## Kaggle test

The generated one-cell FP16 host at
`outputs/validation/reasoning200-latest/kaggle-host-fp16-latest.ipynb` embeds the
module and host sources. It reads latest directly from the training repo and pins
it at host startup, with no eval-repo copy. Enable Internet, T4×2, `HF_TOKEN`
(read) and `BENCH_API_KEY`. Restarting host selects the new latest checkpoint;
it does **not** swap weights during a running task. Keep the cell running and use
the emitted `/v1` endpoint from the local rollout runner.
