#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
docker compose version >/dev/null
action="${1:-up}"
variant="${2:-flash}"
backend="${3:-sglang}"
case "$action" in
  up|switch)
    case "$variant" in flash|base|both) ;; *) echo 'Choose flash, base, or both.' >&2; exit 2 ;; esac
    case "$backend" in sglang|official|both) ;; *) echo 'Choose sglang, official, or both backends.' >&2; exit 2 ;; esac
    if [[ "$backend" == both && "$variant" == both ]]; then
      echo 'Compare one model variant at a time on the 96 GB GPU: flash or base.' >&2; exit 2
    fi
    profile="$variant"
    [[ "$backend" != official ]] || profile="official-$variant"
    [[ "$backend" != both ]] || profile="compare-$variant"
    stop=()
    for service in auk-flash auk-base official-flash official-base; do
      family=sglang; [[ "$service" != official-* ]] || family=official
      model="${service##*-}"
      if [[ ( "$backend" != both && "$backend" != "$family" ) || ( "$variant" != both && "$variant" != "$model" ) ]]; then
        stop+=("$service")
      fi
    done
    if ((${#stop[@]})); then docker compose --profile '*' stop "${stop[@]}"; fi
    docker compose --profile "$profile" up -d --build
    published=$(docker compose port lab 7865 2>/dev/null || true)
    echo "AuK Lab: http://${published:-127.0.0.1:${LAB_PORT:-7865}}"
    echo 'First launch downloads weights. Watch logs or the app service indicators.'
    ;;
  down) docker compose --profile '*' down ;;
  logs) docker compose --profile '*' logs --tail=150 -f ;;
  status) docker compose --profile '*' ps -a ;;
  doctor)
    nvidia-smi
    docker compose --profile '*' config --quiet
    docker compose --profile '*' ps -a
    ;;
  *) echo 'Usage: bash auk-lab.sh {up|switch} [flash|base|both] [sglang|official|both] | down | logs | status | doctor' >&2; exit 2 ;;
esac
