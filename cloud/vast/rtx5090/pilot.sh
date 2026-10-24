#!/usr/bin/env bash
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/sft5090}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
cd "$TASK_REPO_ROOT"
exec "$TASK_VENV/bin/python" -u -m workflow_distillation.lora_fp16 --mode pilot "$@"
