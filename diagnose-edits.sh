#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source_id="${1:?Usage: bash diagnose-edits.sh SOURCE_ID [flash|base] [sglang|official]}"
variant="${2:-flash}"
backend="${3:-sglang}"
case "$variant" in flash|base) ;; *) echo 'Choose flash or base.' >&2; exit 2 ;; esac
case "$backend" in sglang|official) ;; *) echo 'Choose sglang or official.' >&2; exit 2 ;; esac
[[ "$source_id" =~ ^[0-9a-f]{32}$ ]] || { echo 'Invalid source ID.' >&2; exit 2; }
mkdir -p data/diagnostics
if [[ "$backend" == sglang ]]; then
echo 'Inspecting the installed backend request mapping and Qwen audio features (CPU only).'
docker compose exec -T lab python tools/diagnose_edits.py --source-id "$source_id" --model "$variant" --emit-request |
  docker compose exec -T "auk-$variant" python3 /opt/auk-lab-inspect-edit.py |
  tee "data/diagnostics/preprocessing-$variant-$(date +%Y%m%d-%H%M%S).json"
fi
echo 'Running three real editing probes. This may take several minutes.'
docker compose exec -T lab python tools/diagnose_edits.py --source-id "$source_id" --model "$variant" --backend "$backend"
