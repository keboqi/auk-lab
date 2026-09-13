#!/usr/bin/env bash
# One-command Linux bootstrap: no host Python/Conda environment is required.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
AuK Lab quickstart (Linux / NVIDIA Docker)

  bash quickstart.sh [--model flash|base|both] [--gpu INDEX] [--port PORT]
                    [--backend sglang|official|both]
                    [--bind ADDRESS] [--no-wait] [--wait-seconds SECONDS]

Defaults: Flash, GPU 0, UI 7865, loopback binding, wait up to 30 minutes.
Creates .env from .env.example only if missing; existing settings are preserved.
Command-line flags override .env for this launch without rewriting that file.
Docker Engine, Compose v2, NVIDIA driver and Container Toolkit must be installed.
All Python/CUDA dependencies are prepared inside the pinned Docker environments.
EOF
}

variant=flash
backend=sglang
wait_for_models=1
wait_seconds=1800
while (($#)); do
  case "$1" in
    --model|--backend|--gpu|--port|--bind|--wait-seconds)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model) variant="$2" ;;
        --backend) backend="$2" ;;
        --gpu) export GPU_DEVICE="$2" ;;
        --port) export LAB_PORT="$2" ;;
        --bind) export LAB_BIND="$2" ;;
        --wait-seconds) wait_seconds="$2" ;;
      esac
      shift 2 ;;
    --no-wait) wait_for_models=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$variant" in flash|base|both) ;; *) echo 'Model must be flash, base, or both.' >&2; exit 2 ;; esac
case "$backend" in sglang|official|both) ;; *) echo 'Backend must be sglang, official, or both.' >&2; exit 2 ;; esac
[[ "$variant" != both || "$backend" != both ]] || { echo 'Compare one variant at a time: --model flash or base.' >&2; exit 2; }
[[ "$wait_seconds" =~ ^[0-9]+$ ]] && ((wait_seconds > 0)) || { echo 'Wait timeout must be a positive integer.' >&2; exit 2; }
[[ -z "${GPU_DEVICE:-}" || "$GPU_DEVICE" =~ ^[0-9]+$ ]] || { echo 'GPU index must be a nonnegative integer.' >&2; exit 2; }
if [[ -n "${LAB_PORT:-}" ]]; then
  [[ "$LAB_PORT" =~ ^[0-9]+$ ]] && ((LAB_PORT >= 1 && LAB_PORT <= 65535)) || { echo 'UI port must be 1–65535.' >&2; exit 2; }
fi

[[ "$(uname -s)" == Linux ]] || { echo 'Run this script on the Linux GPU host.' >&2; exit 1; }
command -v docker >/dev/null || { echo 'Install Docker Engine and the Compose plugin first: https://docs.docker.com/engine/install/' >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo 'Docker is not accessible. Start the daemon and give this user Docker access.' >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo 'Docker Compose v2 is required.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver and Container Toolkit first: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html' >&2; exit 1; }
echo 'Detected NVIDIA devices:'
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo 'Created .env with default settings.'
else
  echo 'Using existing .env settings.'
fi
mkdir -p data
docker compose --profile '*' config --quiet
echo 'Building application and selected inference environments, then starting services…'
bash auk-lab.sh up "$variant" "$backend"
if (( ! wait_for_models )); then exit 0; fi

echo 'Waiting for model downloads and readiness. Ctrl+C stops waiting; containers keep running.'
deadline=$((SECONDS + wait_seconds))
while ((SECONDS < deadline)); do
  if docker compose exec -T lab python - "$variant" "$backend" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:7865/api/status', timeout=10) as response:
        models = json.load(response)['models']
    wanted = ['flash', 'base'] if sys.argv[1] == 'both' else [sys.argv[1]]
    families = ['sglang', 'official'] if sys.argv[2] == 'both' else [sys.argv[2]]
    wanted = [('official_' if family == 'official' else '') + name for name in wanted for family in families]
    raise SystemExit(0 if all(models[name]['ready'] for name in wanted) else 1)
except (OSError, ValueError, KeyError):
    raise SystemExit(1)
PY
  then
    echo 'AuK Lab is ready. Open the UI URL above, or use an SSH port-forward from your workstation.'
    exit 0
  fi
  sleep 5
done
echo 'Readiness timed out. Containers are still running; inspect: bash auk-lab.sh logs' >&2
exit 1
