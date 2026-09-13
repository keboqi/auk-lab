#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null
action="${1:-up}"
variant="${2:-flash}"
case "$action" in
  up|switch)
    case "$variant" in flash|base|both) ;; *) echo 'Choose flash, base, or both.' >&2; exit 2 ;; esac
    if [[ "$variant" == flash ]]; then docker compose --profile both stop auk-base; fi
    if [[ "$variant" == base ]]; then docker compose --profile both stop auk-flash; fi
    docker compose --profile "$variant" up -d --build
    published=$(docker compose port lab 7865 2>/dev/null || true)
    echo "AuK Lab: http://${published:-127.0.0.1:${LAB_PORT:-7865}}"
    echo 'First launch downloads weights. Watch logs or the app service indicators.'
    ;;
  down) docker compose --profile both down ;;
  logs) docker compose --profile both logs --tail=150 -f ;;
  status) docker compose --profile both ps -a ;;
  doctor)
    nvidia-smi
    docker compose --profile both config --quiet
    docker compose --profile both ps -a
    ;;
  *) echo 'Usage: bash auk-lab.sh {up|switch} [flash|base|both] | down | logs | status | doctor' >&2; exit 2 ;;
esac
