#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

# This example ships three tasks. run.py and the harness derive the scenario and
# state directory from TTTD_TASK; only serve.yaml needs these three, because a
# YAML cannot compute them, and packing needs a longer context on the same GPUs.
TTTD_TASK=${TTTD_TASK:-erdos_min_overlap}
case "$TTTD_TASK" in
  erdos_min_overlap)
    export TTTD_SEQ_LENGTH=30000 TTTD_MAX_TOKENS_PER_GPU=30000 TTTD_LOG_PROBS_CHUNK_SIZE=1024
    ;;
  circle_packing_26|circle_packing_32)
    export TTTD_SEQ_LENGTH=32768 TTTD_MAX_TOKENS_PER_GPU=16384 TTTD_LOG_PROBS_CHUNK_SIZE=512
    ;;
  *)
    echo "run.sh: unknown TTTD_TASK '$TTTD_TASK' (choose erdos_min_overlap, circle_packing_26, or circle_packing_32)" >&2
    exit 1
    ;;
esac
export TTTD_TASK

# The two values this script computes, because neither can be written down in
# serve.yaml: the absolute state root (Ray workers and git resolve relative
# paths from their own directories) and this machine's IP (slime binds the
# training engines' router to it, not to localhost).
export TTTD_STATE_DIR="$PWD/work/$TTTD_TASK"
export REEF_INFERENCE_HOST=$(hostname -I | awk '{print $1}')
mkdir -p "$TTTD_STATE_DIR"

# Download the model on first run (serve.yaml expects it at work/model).
if [ ! -f work/model/config.json ]; then
    huggingface-cli download Qwen/Qwen3-8B --local-dir work/model
fi

# Start the Reef training stack. The Harbor controller waits for the final
# durable training commit before this script exits and stops the stack.
PYTHONPATH="$(cd ../../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}" python3 -m reef serve -c "$PWD/serve.yaml" > "$TTTD_STATE_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Ray + Slime/Megatron + SGLang take minutes to come up.
# Fail with the service log instead of waiting forever if boot fails.
ready_deadline=$((SECONDS + 3600))
while ! curl -sf http://127.0.0.1:8900/healthz > /dev/null; do
    if ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= ready_deadline )); then
        tail -n 100 "$TTTD_STATE_DIR/reef.log" >&2
        echo "run.sh: the Reef stack did not become ready" >&2
        exit 1
    fi
    sleep 5
done

# Run the learning loop.
python3 run.py
