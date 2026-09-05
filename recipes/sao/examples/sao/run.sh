#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

# Limit the locally managed Ray cluster to this training stack's GPU pool.
# On an external cluster, its node configuration determines GPU visibility.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
mkdir -p work

# Start the Reef training stack, stop it again when this script exits.
PYTHONPATH=../../../.. python3 -m reef serve -c "$PWD/serve.yaml" > work/reef.log 2>&1 &
trap 'kill %1' EXIT

# Ray + Slime/Megatron + SGLang take minutes to come up.
# If this never returns, check work/reef.log.
while ! curl -s http://127.0.0.1:8900/healthz > /dev/null; do
    sleep 5
done

python3 run.py
