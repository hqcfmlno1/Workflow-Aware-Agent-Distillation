#!/usr/bin/env bash
# One-variable ablation: fresh base, reasoning 2.0 instead of .15.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export SFT5090_CONFIG="$TASK_REPO_ROOT/configs/training/gpus/rtx5090_32gb/reasoning_200.json"
export SFT5090_OUTPUT="${SFT5090_OUTPUT:-$TASK_REPO_ROOT/outputs/training/rtx5090_32gb/tqs060-fp16-reasoning200}"
export SFT5090_HUB_REPO="${SFT5090_HUB_REPO:-hqcfmlno1/QwenStacy-4b-fp16-tqs060-reasoning200-run1}"
export SFT5090_HUB_PREFIX="${SFT5090_HUB_PREFIX:-fp16-tqs060-reasoning200}"
exec bash "$TASK_REPO_ROOT/cloud/vast/rtx5090/run-group-loss.sh" "$@"
