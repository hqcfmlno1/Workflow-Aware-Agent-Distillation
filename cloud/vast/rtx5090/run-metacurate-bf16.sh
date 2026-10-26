#!/usr/bin/env bash
# Fresh from base; --resume latest/hub resumes only this exact new experiment.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/venv/main}"
TASK_OUTPUT="${SFT5090_OUTPUT:-$TASK_REPO_ROOT/outputs/training/rtx5090_32gb/metacurate1180-bf16-vanilla}"
TASK_CONFIG="$TASK_REPO_ROOT/configs/training/gpus/rtx5090_32gb/metacurate_bf16.json"
TASK_SELECTION="$TASK_REPO_ROOT/configs/training/gpus/rtx5090_32gb/metacurate_selection.json"
TASK_INDEX="$TASK_REPO_ROOT/data/processed/metacurate1180_vanilla/index.json"
TASK_HUB_REPO="${SFT5090_HUB_REPO:-hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-vanilla-run1}"
TASK_RESUME=()
TASK_CHECK=0
TASK_DRY=0
TASK_SECONDS=28800
while (( $# )); do
    case "$1" in
        --resume)
            if [[ $# -lt 2 || ( "$2" != latest && "$2" != hub ) ]]; then
                printf 'Use --resume latest or --resume hub\n' >&2
                exit 2
            fi
            TASK_RESUME=(--resume "$2")
            shift 2
            ;;
        --check) TASK_CHECK=1; shift ;;
        --dry-run) TASK_DRY=1; shift ;;
        --max-seconds)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf 'Expected positive --max-seconds\n' >&2
                exit 2
            fi
            TASK_SECONDS="$2"; shift 2
            ;;
        *) printf 'Unsupported argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ ! -x "$TASK_VENV/bin/python" ]]; then
    printf 'Set SFT5090_VENV to the existing prepared Python 3.11 environment.\n' >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_REPO_ROOT/src:$TASK_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK_REPO_ROOT"
TASK_COMMON=(--config "$TASK_CONFIG" --selection "$TASK_SELECTION" --index "$TASK_INDEX"
             --output "$TASK_OUTPUT" --max-seconds "$TASK_SECONDS")
TASK_TRAIN=("${TASK_COMMON[@]}" "${TASK_RESUME[@]}" --hub-repo "$TASK_HUB_REPO"
            --hub-prefix bf16-metacurate1180-vanilla --prune-hub-history)
if (( TASK_DRY )); then
    exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch \
        --dry-run "${TASK_TRAIN[@]}"
fi
if [[ ! -f "$TASK_INDEX" ]]; then
    printf 'Missing token bundle. Extract metacurate1180-vanilla-tokens.tar.gz at repo root first.\n' >&2
    exit 1
fi
if [[ ${#TASK_RESUME[@]} == 0 && ( -e "$TASK_OUTPUT/events.jsonl" || -e "$TASK_OUTPUT/checkpoints/latest.json" ) ]]; then
    printf 'Existing run found. Use --resume latest; do not restart it from base.\n' >&2
    exit 1
fi
if [[ -e "$TASK_OUTPUT/STOP" ]]; then
    printf 'STOP exists: remove it explicitly when ready to continue this run.\n' >&2
    exit 1
fi
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16 --mode preflight "${TASK_COMMON[@]}"
if (( TASK_CHECK )); then exit 0; fi
# Admit exactly this corpus/config/software/GPU. This loads no model weights.
# Reuse a matching pilot after stopping/restarting the same instance.
if ! "$TASK_VENV/bin/python" - "$TASK_CONFIG" "$TASK_SELECTION" "$TASK_INDEX" "$TASK_OUTPUT/pilot.json" <<'PY'
import json
import sys
from pathlib import Path
from workflow_distillation.lora_fp16.config import RunConfig
from workflow_distillation.lora_fp16.cli import prepare, gpu_audit, validate_pilot
config = RunConfig(**json.loads(Path(sys.argv[1]).read_text()))
identity = prepare(config, Path(sys.argv[3]), Path(sys.argv[2]))[-1]
validate_pilot(json.loads(Path(sys.argv[4]).read_text()), identity, gpu_audit(config))
PY
then
    "$TASK_VENV/bin/python" -u -m workflow_distillation.lora_fp16 --mode pilot "${TASK_COMMON[@]}"
fi
if [[ "${SFT5090_FOREGROUND:-0}" == 1 ]]; then
    # For a supervisor-owned session. CLI validates an existing private Hub repo.
    exec "$TASK_VENV/bin/python" -u -m workflow_distillation.lora_fp16 --mode train "${TASK_TRAIN[@]}"
fi
exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch "${TASK_TRAIN[@]}"
