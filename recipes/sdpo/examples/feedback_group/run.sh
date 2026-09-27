#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SDPO_WORK="$PWD/work"
export REEF_SERVICE_URL=http://127.0.0.1:8900
export REEF_SCENARIO=sdpo-feedback-smoke
export REEF_TOKEN=${REEF_TOKEN:-reef-local}
export REEF_INFERENCE_HOST=$(hostname -I | awk '{print $1}')
REEF_ROOT=$(cd ../../../.. && pwd)
export PYTHONPATH="$REEF_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SDPO_WORK"

python3 -m reef serve -c "$PWD/serve.yaml" > "$SDPO_WORK/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT

deadline=$((SECONDS + 3600))
while ! curl -sf --max-time 5 "$REEF_SERVICE_URL/healthz" > /dev/null; do
    if ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= deadline )); then
        tail -n 100 "$SDPO_WORK/reef.log" >&2
        exit 1
    fi
    sleep 5
done

uv run --no-project --python 3.12 --with-editable "$PWD" --with-editable "$REEF_ROOT" python run.py "$@"
