#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

# Limit the locally managed Ray cluster to this training stack's GPU pool.
# On an external cluster, its node configuration determines GPU visibility.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
mkdir -p work

# Start the Reef training stack, stop it again when this script exits.
python3 -m reef serve -c "$PWD/serve.yaml" > work/reef.log 2>&1 &
reef_pid=$!
trap 'kill "$reef_pid" 2>/dev/null || true; wait "$reef_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Reef enforces the YAML ready_timeout and exits if startup fails.
while kill -0 "$reef_pid" 2>/dev/null; do
    curl -sf --max-time 5 http://127.0.0.1:8900/healthz > /dev/null && break
    sleep 1
done
if ! kill -0 "$reef_pid" 2>/dev/null; then
    echo "Reef failed to start. See $PWD/work/reef.log" >&2
    exit 1
fi

python3 run.py
