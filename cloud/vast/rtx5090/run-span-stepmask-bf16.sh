#!/usr/bin/env bash
# Fresh span-anchor LoRA on Luna-masked targets; uses complete offline scores.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/span-stepmask}"
TASK_OUTPUT="${SFT5090_OUTPUT:-$TASK_ROOT/outputs/training/rtx5090_32gb/metacurate1180-bf16-span-stepmask-run1}"
TASK_DATA="$TASK_ROOT/data/processed/metacurate1180_stepmask_v1"
TASK_SCORES="$TASK_ROOT/data/processed/metacurate1180_span_stepmask_scores_v1"
TASK_CONFIG="$TASK_ROOT/configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_stepmask_bf16.json"
TASK_REPO="${SFT5090_HUB_REPO:-hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-span-stepmask-run1}"
TASK_RESUME=()
TASK_CHECK=0
TASK_DRY=0
TASK_NO_HUB=0
TASK_SECONDS=28800
while (( $# )); do
    case "$1" in
        --resume)
            [[ $# -ge 2 && ( "$2" == latest || "$2" == hub ) ]] || exit 2
            TASK_RESUME=(--resume "$2"); shift 2 ;;
        --check) TASK_CHECK=1; shift ;;
        --dry-run) TASK_DRY=1; shift ;;
        --no-hub) TASK_NO_HUB=1; shift ;;
        --max-seconds)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || exit 2
            TASK_SECONDS="$2"; shift 2 ;;
        *) printf 'Unsupported argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK_ROOT"
TASK_COMMON=(--config "$TASK_CONFIG" --selection "$TASK_DATA/selection.json"
    --index "$TASK_DATA/index.json" --span-sidecar "$TASK_SCORES" --output "$TASK_OUTPUT")
# User requested direct training: this same pool already trained on 32GB.
TASK_TRAIN=("${TASK_COMMON[@]}" --skip-pilot --max-seconds "$TASK_SECONDS" "${TASK_RESUME[@]}")
if (( TASK_NO_HUB )); then
    [[ "${TASK_RESUME[*]}" != '--resume hub' ]] || exit 2
    TASK_PUBLISH=(--no-hub)
else
    TASK_PUBLISH=(--hub-repo "$TASK_REPO" --hub-prefix bf16-metacurate1180-span-stepmask)
fi
if (( TASK_DRY )); then
    exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch \
        --dry-run "${TASK_TRAIN[@]}" "${TASK_PUBLISH[@]}"
fi
if (( ! TASK_CHECK )); then
    if [[ -e "$TASK_OUTPUT/STOP" ]]; then
        printf 'STOP exists; remove explicitly before resuming.\n' >&2; exit 1
    fi
    if [[ ${#TASK_RESUME[@]} == 0 && ( -e "$TASK_OUTPUT/events.jsonl" || -e "$TASK_OUTPUT/checkpoints/latest.json" ) ]]; then
        printf 'Existing run: use --resume latest/hub or a fresh output.\n' >&2; exit 1
    fi
fi
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16 --mode preflight "${TASK_COMMON[@]}"
if (( TASK_CHECK )); then exit 0; fi
exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch \
    "${TASK_TRAIN[@]}" "${TASK_PUBLISH[@]}"
