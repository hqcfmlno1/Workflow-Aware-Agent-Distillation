#!/usr/bin/env bash
# Run from any directory. No GPU model is loaded by this setup script.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/sft5090}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
cd "$TASK_REPO_ROOT"
TASK_PYTHON="${SFT5090_PYTHON:-python3.11}"
"$TASK_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use a Python 3.11 Vast template"'
if [[ ! -x "$TASK_VENV/bin/python" ]]; then
    "$TASK_PYTHON" -m venv "$TASK_VENV"
fi
"$TASK_VENV/bin/python" -m pip install --upgrade pip
"$TASK_VENV/bin/python" -m pip install -r cloud/vast/rtx5090/requirements.txt
"$TASK_VENV/bin/python" -m pip install -e .
"$TASK_VENV/bin/python" -m pip check
# Causal conv is optional: avoid an unexpected source build during a paid rental.
# FLA supplies optimized Gated DeltaNet. Conv1d otherwise uses the correct Torch fallback.
if [[ "${SFT5090_BUILD_CONV1D:-0}" == 1 ]]; then
    "$TASK_VENV/bin/python" -m pip install ninja
    "$TASK_VENV/bin/python" -m pip install --no-build-isolation causal-conv1d==1.7.0
fi
"$TASK_VENV/bin/python" -c 'from huggingface_hub import get_token, login; login(add_to_git_credential=False) if not get_token() else None'
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.assets --with-model
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16 --mode preflight
printf '\nSetup finished. Pilot next: bash cloud/vast/rtx5090/pilot.sh\n'
