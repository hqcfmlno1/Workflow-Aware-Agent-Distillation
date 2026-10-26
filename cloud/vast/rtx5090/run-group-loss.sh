#!/usr/bin/env bash
# Fresh experiment only: setup if needed, verify, pilot, then detach training.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
TASK_VENV="${SFT5090_VENV:-/workspace/venvs/sft5090}"
TASK_OUTPUT="${SFT5090_OUTPUT:-$TASK_REPO_ROOT/outputs/training/rtx5090_32gb/tqs060-fp16-group-loss}"
TASK_CONFIG="${SFT5090_CONFIG:-$TASK_REPO_ROOT/configs/training/gpus/rtx5090_32gb/group_loss.json}"
TASK_HUB_REPO="${SFT5090_HUB_REPO:-hqcfmlno1/QwenStacy-4b-fp16-tqs060-group-loss-run1}"
TASK_HUB_PREFIX="${SFT5090_HUB_PREFIX:-fp16-tqs060-group-loss}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# An existing venv can point at an older clone through its editable install.
export PYTHONPATH="$TASK_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK_REPO_ROOT"
TASK_EXTRA=()
TASK_TRAIN_ONLY=()
while (( $# )); do
    case "$1" in
        --epochs|--max-seconds)
            if [[ $# -lt 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf 'Expected a positive integer after %s\n' "$1" >&2
                exit 2
            fi
            TASK_EXTRA+=("$1" "$2")
            shift 2
            ;;
        --prune-hub-history)
            TASK_TRAIN_ONLY+=("$1")
            shift
            ;;
        *)
            printf 'Unsupported fresh-run argument: %s\nAllowed: --epochs N, --max-seconds N, --prune-hub-history.\n' "$1" >&2
            exit 2
            ;;
    esac
done
if [[ -e "$TASK_OUTPUT/events.jsonl" || -e "$TASK_OUTPUT/checkpoints/latest.json" || -e "$TASK_OUTPUT/STOP" ]]; then
    printf 'Output already used or stopped: %s\nChoose SFT5090_OUTPUT for a fresh run, or use the explicit resume command in the README.\n' "$TASK_OUTPUT" >&2
    exit 1
fi
if [[ ! -x "$TASK_VENV/bin/python" ]]; then
    bash cloud/vast/rtx5090/bootstrap.sh
fi
TASK_COMMON=(--config "$TASK_CONFIG" --output "$TASK_OUTPUT" "${TASK_EXTRA[@]}")
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.assets --with-model
"$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16 --mode preflight "${TASK_COMMON[@]}"
"$TASK_VENV/bin/python" -u -m workflow_distillation.lora_fp16 --mode pilot "${TASK_COMMON[@]}"
exec "$TASK_VENV/bin/python" -m workflow_distillation.lora_fp16.launch \
    "${TASK_COMMON[@]}" "${TASK_TRAIN_ONLY[@]}" \
    --hub-repo "$TASK_HUB_REPO" --hub-prefix "$TASK_HUB_PREFIX"
