#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

mkdir -p work

# Start Reef from the external-provider stack, with the local example's
# credential and state directory overriding the deployment defaults
# (`reef serve -c <stack> --<reef.key> <value>`); stop it again when this
# script exits. REEF_UPSTREAM_URL and REEF_UPSTREAM_API_KEY come from your
# environment.
export REEF_TOKEN=reef-local
PYTHONPATH=../.. python3 -m reef serve -c "$PWD/external-provider.yaml" \
    --agent_record_dir work/agent-record \
    --artifact_repository work/artifacts.git \
    --artifact_work_dir work/artifact-work \
    --artifact_cache_dir work/artifact-cache \
    > work/reef.log 2>&1 &
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
