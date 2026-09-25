#!/usr/bin/env bash
# SDPO on an arithmetic grid (recipes/sdpo/examples/arithmetic_grid) through
# reef-eval: run.py runs the Harbor task under this directory's harness while
# the Reef stack serves. Setup (once): see README. State goes to $RUN_DIR.
#
#   ./run.sh                     two grids, two optimizer steps
#   SDPO_STEPS=5 ./run.sh        five
set -euo pipefail
cd "$(dirname "$0")"
export RUN_DIR="${RUN_DIR:-$PWD/work}"
export REEF_PORT="${REEF_PORT:-28902}"
export REEF_SCENARIO="${REEF_SCENARIO:-sdpo-arithmetic}"
MODEL_DIR="${MODEL_DIR:-/tmp/models}"

# Prerequisites
command -v uv >/dev/null || { echo "run.sh: uv not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
[ -d "$MODEL_DIR/Qwen3-0.6B" ] \
    || { echo "run.sh: $MODEL_DIR/Qwen3-0.6B not found (hf download Qwen/Qwen3-0.6B)" >&2; exit 1; }
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/token" ] || openssl rand -hex 16 > "$RUN_DIR/token"
export REEF_TOKEN="$(cat "$RUN_DIR/token")"
# The runner calls Reef from inside the task container, not from this host.
export REEF_SERVICE_URL="http://host.docker.internal:$REEF_PORT"

python -m reef serve -c "$PWD/serve.yaml" > "$RUN_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
  kill "$reef_pid" 2>/dev/null || true
  wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$reef_pid" 2>/dev/null; then
    tail -100 "$RUN_DIR/reef.log"
    exit 1
  fi
  if curl --silent --fail --max-time 5 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "run.sh: Reef did not become ready; see $RUN_DIR/reef.log" >&2; exit 1; }

# The episode, in an ephemeral uv environment: reef-eval with Harbor, the
# reef-client protocol, and this directory's harness package.
uv run --no-project --python 3.12 \
    --with "reef-eval[harbor]" --with reef-client --with-editable "$PWD" \
    run.py
