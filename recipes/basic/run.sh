#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"

startup_timeout=${REEF_STARTUP_TIMEOUT_S:-300}
if [[ ! "$startup_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "run.sh: REEF_STARTUP_TIMEOUT_S must be a positive integer" >&2
    exit 2
fi
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
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Bound both the overall wait and each HTTP probe; stop if the stack exits.
ready_deadline=$((SECONDS + startup_timeout))
while true; do
    if ! kill -0 "$reef_pid" 2>/dev/null; then
        exit_code=0
        wait "$reef_pid" || exit_code=$?
        echo "run.sh: Reef exited before ready (exit code $exit_code). Log: $PWD/work/reef.log" >&2
        tail -n 100 work/reef.log >&2
        (( exit_code != 0 )) || exit_code=1
        exit "$exit_code"
    fi
    remaining=$((ready_deadline - SECONDS))
    if (( remaining <= 0 )); then
        echo "run.sh: Reef did not become ready within ${startup_timeout}s. Log: $PWD/work/reef.log" >&2
        tail -n 100 work/reef.log >&2
        exit 1
    fi
    probe_timeout=$((remaining < 5 ? remaining : 5))
    if curl -sf --connect-timeout "$probe_timeout" --max-time "$probe_timeout" \
        http://127.0.0.1:8900/healthz > /dev/null && kill -0 "$reef_pid" 2>/dev/null; then
        break
    fi
    sleep 1
done

python3 run.py
