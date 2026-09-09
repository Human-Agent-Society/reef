#!/bin/bash
# No-GPU smoke run: real CORAL runtime + real Reef service, canned model.
#
# Exercises run.py unmodified — AgentManager, worktrees, gateway + LiteLLM,
# grader daemon, attempt watcher, result bundle — against
# smoke/echo_reef_service.py (the production Reef service with a canned
# inference backend) and smoke/scripted_runtime.py (a real AgentRuntime
# whose agent is scripted). Needs an environment with coral AND reef
# installed:
#
#   uv venv .venv && uv pip install -e .[coral] -e ../../../..
#   ./smoke/run_smoke.sh
set -e
cd "$(dirname "$0")/.."
REPO_ROOT="$(cd ../../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT:$PWD:${PYTHONPATH:-}"  # recipes.coral + smoke importable
export REEF_TOKEN="${REEF_TOKEN:-reef-local}"
WORK="$PWD/work/smoke-$(date +%s)"
mkdir -p "$WORK"

REEF_PORT="${REEF_PORT:-18900}"  # clear of any real stack on 8900
python3 smoke/echo_reef_service.py --port "$REEF_PORT" --token "$REEF_TOKEN" > "$WORK/reef-stub.log" 2>&1 &
reef_pid=$!
cleanup() {
    kill "$reef_pid" 2>/dev/null || true
    wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 60); do
    curl -sf "http://127.0.0.1:$REEF_PORT/healthz" > /dev/null && break
    sleep 1
done

python3 run.py \
    --work "$WORK" \
    --agents 2 \
    --max-attempts 4 \
    --reef-url "http://127.0.0.1:$REEF_PORT" \
    --gateway-port 18091 \
    --runtime smoke.scripted_runtime:ScriptedRuntime \
    --model reef-policy

python3 smoke/check_bundle.py "$WORK/bundle.json"
echo "smoke run OK: $WORK/bundle.json"
