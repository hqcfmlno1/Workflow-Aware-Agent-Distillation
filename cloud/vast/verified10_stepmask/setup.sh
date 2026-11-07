#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
case "${1:-}" in
  setup)
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    if (( ${driver_version%%.*} < 580 )); then
      echo "Driver $driver_version incompatible with the pinned CUDA13 wheel; stop before install."
      exit 2
    fi
    if ! command -v uv >/dev/null 2>&1; then python3 -m pip install uv; fi
    if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.11 .venv; fi
    uv pip install --python .venv/bin/python --torch-backend=cu130 -r requirements-server.txt
    ;;
  download)
    exec .venv/bin/python -u host.py --download-only
    ;;
  serve)
    shift
    if [[ -z "${BENCH_API_KEY:-}" ]]; then
      key_file="${BENCH_API_KEY_FILE:-$HOME/.config/workflow-distillation/bench_api_key}"
      if [[ ! -s "$key_file" ]]; then echo "Missing private BENCH_API_KEY_FILE"; exit 2; fi
      BENCH_API_KEY="$(cat -- "$key_file")"
      export BENCH_API_KEY
    fi
    exec .venv/bin/python -u host.py "$@"
    ;;
  *) echo "Usage: bash setup.sh {setup|download|serve}"; exit 2;;
esac
