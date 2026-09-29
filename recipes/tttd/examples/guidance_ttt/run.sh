#!/usr/bin/env bash
# Run from a Linux GPU allocation with the task judge and frozen executor ready.
set -euo pipefail
cd "$(dirname "$0")"
export GUIDANCE_TASK="${GUIDANCE_TASK:-polyomino_packing}"
case "$GUIDANCE_TASK" in
    polyomino_packing|lasso_path|ahc058|trimul) ;;
    *) echo "Unknown GUIDANCE_TASK: $GUIDANCE_TASK" >&2; exit 2 ;;
esac
export GUIDANCE_CONFIG="${GUIDANCE_CONFIG:-$PWD/serve.yaml}"
export GUIDANCE_STATE_DIR="${GUIDANCE_STATE_DIR:-$PWD/work/$GUIDANCE_TASK}"
export GUIDANCE_MODEL="${GUIDANCE_MODEL:-Qwen/Qwen3-8B}"
export GUIDANCE_MODEL_PATH="${GUIDANCE_MODEL_PATH:-$PWD/work/models/${GUIDANCE_MODEL##*/}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export GUIDANCE_STATE_DIR="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["GUIDANCE_STATE_DIR"]).resolve())')"
export GUIDANCE_MODEL_PATH="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["GUIDANCE_MODEL_PATH"]).resolve())')"
# Reject grid mismatches before downloading weights or starting GPU workers.
python3 -c 'from harness.config import RunConfig; RunConfig.load().validate_state()'
mkdir -p "$GUIDANCE_STATE_DIR"
if [ ! -f "$GUIDANCE_MODEL_PATH/config.json" ]; then
    hf download "$GUIDANCE_MODEL" --local-dir "$GUIDANCE_MODEL_PATH"
fi
export REEF_INFERENCE_HOST="${REEF_INFERENCE_HOST:-$(hostname -I | awk '{print $1}')}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$REEF_INFERENCE_HOST"
python3 -m reef serve -c "$GUIDANCE_CONFIG" > "$GUIDANCE_STATE_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
service_url="$(python3 -c 'from harness.config import RunConfig; print(RunConfig.load().service_url)')"
ready_deadline=$((SECONDS + 3600))
while ! curl -sf "$service_url/healthz" > /dev/null; do
    if ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= ready_deadline )); then
        tail -n 100 "$GUIDANCE_STATE_DIR/reef.log" >&2
        echo "run.sh: the Reef stack did not become ready" >&2
        exit 1
    fi
    sleep 5
done
python3 run.py
