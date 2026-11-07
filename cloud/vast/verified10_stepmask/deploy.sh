#!/usr/bin/env bash
# Explicit deployment only, after training. Never stops another GPU process.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ $# != 2 || ! "$2" =~ ^[0-9]+$ || "$1" == -* ]]; then
  echo 'Usage: bash cloud/vast/verified10_stepmask/deploy.sh root@HOST SSH_PORT'
  exit 2
fi
server="$1"
port="$2"
prepared="${VERIFIED10_STEPMASK_DIR:-$task_root/outputs/evaluation/verified10-span-stepmask-epoch2}"
key_file="${BENCH_API_KEY_FILE:-$HOME/.config/workflow-distillation/bench_api_key}"
archive="$prepared/verified10-stepmask-server.tar.gz"
[[ -s "$key_file" && -s "$archive" ]] || { echo 'Missing archive/private API key file'; exit 2; }
"$task_root/.venv/bin/python" - "$prepared" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
receipt = json.loads((root / 'prepared.json').read_text())
if not receipt['adapters_ready']:
    raise SystemExit('Run prepare.py finalize after new checkpoint 590 is uploaded')
with (root / 'verified10-stepmask-server.tar.gz').open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != receipt['archive_sha256']:
        raise SystemExit('Archive checksum mismatch')
PY
# Gate before any installation or changes on the paid server.
ssh -p "$port" "$server" 'bash -s' <<'REMOTE'
set -euo pipefail
if command -v supervisorctl >/dev/null 2>&1; then
  status="$(supervisorctl status span-stepmask 2>/dev/null || true)"
  if [[ "$status" =~ (RUNNING|STARTING|STOPPING) ]]; then
    echo 'Span-stepmask training is active. Wait until it finishes before inference.'
    exit 2
  fi
fi
gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$gpu_pids" ]]; then
  echo 'GPU has an active process; refusing to load inference alongside it.'
  exit 2
fi
umask 077
mkdir -p /workspace ~/.config/workflow-distillation
REMOTE
scp -P "$port" "$archive" "$server:/workspace/verified10-stepmask-server.tar.gz"
scp -P "$port" "$key_file" "$server:.config/workflow-distillation/bench_api_key"
ssh -p "$port" "$server" 'bash -s' <<'REMOTE'
set -euo pipefail
tar -xzf /workspace/verified10-stepmask-server.tar.gz -C /workspace
chmod 600 ~/.config/workflow-distillation/bench_api_key
cd /workspace/verified10-stepmask
export HF_HOME=/workspace/.hf_home
bash setup.sh setup
bash setup.sh download
if command -v supervisorctl >/dev/null 2>&1; then
  cat > /etc/supervisor/conf.d/verified10-stepmask.conf <<'CONF'
[program:verified10-stepmask]
command=/bin/bash /workspace/verified10-stepmask/setup.sh serve
directory=/workspace/verified10-stepmask
environment=HF_HOME="/workspace/.hf_home"
autostart=false
autorestart=false
stopasgroup=true
killasgroup=true
stopsignal=INT
stopwaitsecs=60
stdout_logfile=/workspace/verified10-stepmask/server.log
stdout_logfile_maxbytes=0
redirect_stderr=true
CONF
  supervisorctl reread
  supervisorctl update
  supervisorctl start verified10-stepmask
else
  nohup bash setup.sh serve > server.log 2>&1 < /dev/null &
  echo $! > server.pid
fi
echo 'Inference starting: /workspace/verified10-stepmask/server.log'
REMOTE
echo "Tunnel: ssh -N -p $port -L 8000:127.0.0.1:8000 $server"
echo "Runner: $task_root/.venv/bin/python -u $prepared/run.py"
