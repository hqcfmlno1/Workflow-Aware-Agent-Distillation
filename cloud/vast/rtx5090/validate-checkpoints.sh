#!/usr/bin/env bash
# Read-only validation; does not start/resume training or upload/delete checkpoints.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/sft5090}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK_REPO_ROOT"
if [[ ! -x "$TASK_VENV/bin/python" ]]; then
    printf 'Missing existing training venv: %s\nRun cloud/vast/rtx5090/bootstrap.sh first.\n' "$TASK_VENV" >&2
    exit 1
fi
TASK_OUTPUT="${VALIDATION_OUTPUT:-$TASK_REPO_ROOT/outputs/validation/checkpoint-loss}"
mkdir -p "$TASK_OUTPUT"
# A plan-only call inspects existing assets; no unnecessary model download.
if [[ " $* " != *" --plan-only "* ]]; then
    "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.assets --with-model
fi
"$TASK_VENV/bin/python" -u scripts/evaluation/validate_checkpoints.py \
    --output "$TASK_OUTPUT" "$@" 2>&1 | tee -a "$TASK_OUTPUT/validation.log"
