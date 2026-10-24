#!/usr/bin/env bash
# Detached from SSH; only this run's PID/log are touched. No instance API calls.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/sft5090}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "$TASK_REPO_ROOT"
# The launcher handles locking and starts a new OS session without forwarding
# the SSH lifetime. Arguments remain an array, never eval/interpolated shell.
exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch "$@"
