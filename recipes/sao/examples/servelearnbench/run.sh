#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
reef_source_root="$(cd ../../../.. && pwd)"
export PYTHONPATH="$reef_source_root${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_DIR:?Set MODEL_DIR to a Qwen3 checkpoint}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a fresh scratch directory}"
: "${REEF_WORK:?Set REEF_WORK to a fresh run directory}"
export MODEL_DIR CHECKPOINT_DIR REEF_WORK
export REEF_PORT=${REEF_PORT:-8900}
export REEF_SERVICE_URL="http://127.0.0.1:${REEF_PORT}"
export REEF_SCENARIO=${REEF_SCENARIO:-retail-l1-w1}
export SAO_GPUS=${SAO_GPUS:-4}
export SAO_WARMUP=${SAO_WARMUP:-10}
method=${METHOD:-sao}
python=${REEF_PYTHON:-python3}
case "$method" in
    sao|blind|oracle) ;;
    *) echo "METHOD must be sao, blind, or oracle" >&2; exit 1 ;;
esac
"$python" - "$REEF_WORK" "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path
for value in sys.argv[1:]:
    path = Path(value)
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Use a fresh empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
PY
export REEF_TOKEN=${REEF_TOKEN:-$("$python" -c 'import secrets; print(secrets.token_hex(24))')}
config="$PWD/serve.yaml"
# All methods share exact-token history and engine settings. Baselines send
# no reports, so their training backend remains idle throughout the campaign.
"$python" -m reef serve -c "$config" > "$REEF_WORK/reef.log" 2>&1 &
reef_pid=$!
trap 'kill "$reef_pid" 2>/dev/null || true; wait "$reef_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((attempt=0; attempt<1800; attempt++)); do
    if ! kill -0 "$reef_pid" 2>/dev/null; then
        break
    fi
    if curl -fsS --max-time 2 "$REEF_SERVICE_URL/healthz" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" != 1 ]]; then
    echo "Reef failed to become ready. See $REEF_WORK/reef.log" >&2
    exit 1
fi
"$python" run.py --method "$method" --scenario "$REEF_SCENARIO" \
    --output "$REEF_WORK/results.jsonl" --max-tokens 8192 "$@"
