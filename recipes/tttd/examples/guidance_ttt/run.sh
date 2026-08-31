#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
#
# Two services must already be reachable: the privileged FrontierCS/go-judge
# on port 8081 and the frozen executor on port 8000. This script starts the
# third, Reef's Qwen3-8B LoRA training stack, and then runs one Harbor trial
# that owns the complete Guidance-TTT trajectory.
set -e
cd "$(dirname "$0")"
mkdir -p work/polyomino_packing

# Download the model on first run (serve.yaml expects it at work/model).
if [ ! -f work/model/config.json ]; then
    huggingface-cli download Qwen/Qwen3-8B --local-dir work/model
fi

# slime binds the training engines' router to this machine's IP (not
# localhost), so the stack dials that same IP.
export REEF_INFERENCE_HOST=$(hostname -I | awk '{print $1}')
export NO_PROXY=$REEF_INFERENCE_HOST

# Start the Reef training stack. The Harbor controller waits for the final
# durable training commit before this script exits and stops the stack.
PYTHONPATH=../../../.. python3 -m reef serve -c "$PWD/serve.yaml" > work/polyomino_packing/reef.log 2>&1 &
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
        tail -n 100 work/polyomino_packing/reef.log >&2
        echo "run.sh: the Reef stack did not become ready" >&2
        exit 1
    fi
    sleep 5
done

# Run the learning loop.
python3 run.py
