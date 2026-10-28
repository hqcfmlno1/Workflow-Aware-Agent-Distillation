#!/usr/bin/env bash
# Estimate (CPU), resume offline scoring, then train with one resident model.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/venv/main}"
TASK_PYTHON="$TASK_VENV/bin/python"
TASK_OUTPUT="${SFT5090_OUTPUT:-$TASK_ROOT/outputs/training/rtx5090_32gb/metacurate1180-bf16-span-anchor-run1}"
TASK_SCORES="${SFT5090_SCORES:-$TASK_ROOT/data/processed/metacurate1180_span_scores_cached}"
TASK_CONFIG="${SFT5090_CONFIG:-$TASK_ROOT/configs/training/gpus/rtx5090_32gb/metacurate_span_anchor_bf16_cached.json}"
TASK_SELECTION="$TASK_ROOT/configs/training/gpus/rtx5090_32gb/metacurate_selection.json"
TASK_INDEX="$TASK_ROOT/data/processed/metacurate1180_vanilla/index.json"
TASK_PROVENANCE="$TASK_ROOT/data/processed/metacurate1180_span_provenance/index.json"
TASK_REPO="${SFT5090_HUB_REPO:-hqcfmlno1/QwenStacy-4b-bf16-metacurate1180-span-anchor-run1}"
TASK_STAGE=estimate
TASK_SECONDS=28800
TASK_LIMIT=()
TASK_RESUME=()
TASK_FOREGROUND=false
while (( $# )); do
    case "$1" in
        --stage)
            [[ $# -ge 2 && "$2" =~ ^(estimate|score|check|train|all)$ ]] || exit 2
            TASK_STAGE="$2"; shift 2 ;;
        --max-seconds)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || exit 2
            TASK_SECONDS="$2"; shift 2 ;;
        --limit-turns)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || exit 2
            TASK_LIMIT=(--limit-turns "$2"); shift 2 ;;
        --resume)
            [[ $# -ge 2 && "$2" =~ ^(latest|hub)$ ]] || exit 2
            TASK_RESUME=(--resume "$2"); shift 2 ;;
        --foreground)
            TASK_FOREGROUND=true; shift ;;
        *) printf 'Unsupported argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$TASK_STAGE" != score && ${#TASK_LIMIT[@]} -gt 0 ]]; then
    printf '%s\n' '--limit-turns is only for the score stage.' >&2; exit 2
fi
if [[ "$TASK_STAGE" != train && "$TASK_STAGE" != all && "$TASK_FOREGROUND" == true ]]; then
    printf '%s\n' '--foreground is only for train/all stages.' >&2; exit 2
fi
[[ -x "$TASK_PYTHON" ]] || { printf 'Set SFT5090_VENV to your Python 3.11 environment.\n' >&2; exit 1; }
export PYTHONPATH="$TASK_ROOT/src:$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$TASK_ROOT"
TASK_COMMON=(--config "$TASK_CONFIG" --selection "$TASK_SELECTION" --index "$TASK_INDEX")
case "$TASK_STAGE" in
    estimate)
        exec "$TASK_PYTHON" -m scripts.training.score_span_lora "${TASK_COMMON[@]}" \
            --provenance "$TASK_PROVENANCE" --estimate ;;
    score|all)
        if [[ "$TASK_STAGE" == score || ! -f "$TASK_SCORES/index.json" ]]; then
            mkdir -p "$TASK_SCORES"
            "$TASK_PYTHON" -u -m scripts.training.score_span_lora "${TASK_COMMON[@]}" \
                --provenance "$TASK_PROVENANCE" --output "$TASK_SCORES" \
                --max-seconds "$TASK_SECONDS" "${TASK_LIMIT[@]}" 2>&1 | tee -a "$TASK_SCORES/scoring.log"
        fi
        [[ "$TASK_STAGE" == score ]] && exit 0
        [[ -f "$TASK_SCORES/index.json" ]] || {
            printf 'Scoring is partial; training was not started. Resume the same command.\n' >&2
            exit 2
        } ;;
esac
if [[ "$TASK_STAGE" == train || "$TASK_STAGE" == all ]]; then
    if [[ "$TASK_STAGE" == all && ${#TASK_RESUME[@]} == 0 && -f "$TASK_OUTPUT/checkpoints/latest.json" ]]; then
        TASK_RESUME=(--resume latest)
    fi
    if [[ -e "$TASK_OUTPUT/STOP" ]]; then
        printf 'STOP exists; remove it explicitly before resuming.\n' >&2; exit 1
    fi
    if [[ ${#TASK_RESUME[@]} == 0 && ( -e "$TASK_OUTPUT/events.jsonl" || -e "$TASK_OUTPUT/checkpoints/latest.json" ) ]]; then
        printf 'Existing run: use --resume latest/hub or a fresh SFT5090_OUTPUT.\n' >&2; exit 1
    fi
fi
TASK_TRAIN=("${TASK_COMMON[@]}" --span-sidecar "$TASK_SCORES" --output "$TASK_OUTPUT" --max-seconds "$TASK_SECONDS")
"$TASK_PYTHON" -m workflow_distillation.lora_fp16 --mode preflight "${TASK_TRAIN[@]}"
[[ "$TASK_STAGE" == check ]] && exit 0
# Reuse matching single-longest admission on resume; no repeated broad pilots.
if ! "$TASK_PYTHON" - "$TASK_CONFIG" "$TASK_INDEX" "$TASK_SELECTION" "$TASK_SCORES" "$TASK_OUTPUT/pilot.json" <<'PY'
import json,sys
from pathlib import Path
from workflow_distillation.lora_fp16.cli import prepare,gpu_audit,validate_pilot
from workflow_distillation.lora_fp16.config import RunConfig
config=RunConfig(**json.loads(Path(sys.argv[1]).read_text()))
identity=prepare(config,Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))[-1]
validate_pilot(json.loads(Path(sys.argv[5]).read_text()),identity,gpu_audit(config))
PY
then
    "$TASK_PYTHON" -u -m workflow_distillation.lora_fp16 --mode pilot "${TASK_TRAIN[@]}"
fi
if [[ "$TASK_FOREGROUND" == true ]]; then
    # Supervisor must own the training process so stopping the service stops its GPU work.
    "$TASK_PYTHON" - "$TASK_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

api = HfApi()
try:
    if not api.repo_info(sys.argv[1]).private:
        raise ValueError("Checkpoint repo must be private and dedicated")
except RepositoryNotFoundError:
    api.create_repo(sys.argv[1], private=True)
PY
    exec "$TASK_PYTHON" -u -m workflow_distillation.lora_fp16 --mode train \
        "${TASK_TRAIN[@]}" "${TASK_RESUME[@]}" --hub-repo "$TASK_REPO" \
        --hub-prefix bf16-metacurate1180-span-anchor >> "$TASK_OUTPUT/train.log" 2>&1
fi
exec "$TASK_PYTHON" -m workflow_distillation.lora_fp16.launch "${TASK_TRAIN[@]}" \
    "${TASK_RESUME[@]}" --hub-repo "$TASK_REPO" --hub-prefix bf16-metacurate1180-span-anchor
