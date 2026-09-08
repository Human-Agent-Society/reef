#!/bin/bash
# Serve + run for the CORAL test-time-training example.
# Setup (once): pip install -e .[coral]  — see README. State goes to ./work.
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../../../.. && pwd):${PYTHONPATH:-}"  # recipes.coral importable

# The values serve.yaml cannot compute itself.
export CORAL_TTT_STATE_DIR="$PWD/work/coral-demo"
export REEF_INFERENCE_HOST=$(hostname -I | awk '{print $1}')
mkdir -p "$CORAL_TTT_STATE_DIR"

# Download the model on first run (serve.yaml expects it at work/model).
if [ ! -f work/model/config.json ]; then
    huggingface-cli download Qwen/Qwen3-8B --local-dir work/model
fi

# Start the Reef training stack; stop it when the demo loop exits.
python3 -m reef serve -c "$PWD/serve.yaml" > "$CORAL_TTT_STATE_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Ray + Slime/Megatron + SGLang take minutes to come up.
ready_deadline=$((SECONDS + 3600))
while ! curl -sf http://127.0.0.1:8900/healthz > /dev/null; do
    if ! kill -0 "$reef_pid" 2>/dev/null || (( SECONDS >= ready_deadline )); then
        tail -n 100 "$CORAL_TTT_STATE_DIR/reef.log" >&2
        echo "run.sh: the Reef stack did not become ready" >&2
        exit 1
    fi
    sleep 5
done

# CORAL gateway + demo agents + reporting; writes work/coral-demo/bundle.json.
python3 run.py --work "$CORAL_TTT_STATE_DIR"
