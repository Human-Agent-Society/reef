#!/usr/bin/env bash
# SDPO on SciKnowEval Chemistry (recipes/sdpo/examples/sciknoweval) through
# reef-eval: run.py runs the Harbor task under this directory's harness while
# the Reef stack serves. Setup (once): see README. State goes to $SDPO_RUN_DIR.
#
#   ./run.sh                          the whole schedule
#   SDPO_STEPS=2 ./run.sh             two steps, to check the stack
#   SDPO_TRAINING_HOURS=5 ./run.sh    the paper's five-hour training budget
set -euo pipefail
cd "$(dirname "$0")"
export SDPO_RUN_DIR="${SDPO_RUN_DIR:-$PWD/work}"
export SDPO_MODEL_PATH="${SDPO_MODEL_PATH:-$HOME/models/Qwen3-8B}"
export REEF_PORT="${REEF_PORT:-28902}"
export REEF_SCENARIO="${REEF_SCENARIO:-sdpo-chemistry}"

# Prerequisites
command -v uv >/dev/null || { echo "run.sh: uv not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
[ -d "$SDPO_MODEL_PATH" ] \
    || { echo "run.sh: $SDPO_MODEL_PATH not found (hf download Qwen/Qwen3-8B)" >&2; exit 1; }
mkdir -p "$SDPO_RUN_DIR"
[ -f "$SDPO_RUN_DIR/token" ] || openssl rand -hex 16 > "$SDPO_RUN_DIR/token"
export REEF_TOKEN="$(cat "$SDPO_RUN_DIR/token")"
# The stage calls Reef from inside the task container, not from this host.
export REEF_SERVICE_URL="http://host.docker.internal:$REEF_PORT"

python -m reef serve -c "$PWD/serve.yaml" > "$SDPO_RUN_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
  kill "$reef_pid" 2>/dev/null || true
  wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for _ in $(seq 1 720); do
  if ! kill -0 "$reef_pid" 2>/dev/null; then
    tail -100 "$SDPO_RUN_DIR/reef.log"
    exit 1
  fi
  if curl --silent --fail --max-time 5 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "run.sh: Reef did not become ready; see $SDPO_RUN_DIR/reef.log" >&2; exit 1; }

# The episode, in an ephemeral uv environment: reef-eval with Harbor, the
# reef-client protocol, and this directory's harness package.
uv run --no-project --python 3.12 \
    --with "reef-eval[harbor]" --with reef-client --with-editable "$PWD" \
    run.py
