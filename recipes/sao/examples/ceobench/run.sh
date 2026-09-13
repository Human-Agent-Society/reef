#!/usr/bin/env bash
# The SAO training stack for CEO-Bench, then one episode through reef-eval,
# trained while it is played. Setup (once): see README. State goes to $RUN_DIR.
#
# The stack is left running and a healthy one is reused; `docker compose down`
# stops it. A trained stack is bound to its scenario, so switching
# REEF_SCENARIO or RUN_DIR needs a restart.
set -euo pipefail
cd "$(dirname "$0")"
REEF_ROOT="$(cd ../../../.. && pwd)"

REEF_IMAGE="${REEF_IMAGE:-reef}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
RUN_DIR="${RUN_DIR:-$PWD/work}"
HOST_IP="$(hostname -I | awk '{print $1}')"
export REEF_SERVICE_URL="${REEF_SERVICE_URL:-http://${HOST_IP}:28900}"
export REEF_SCENARIO="${REEF_SCENARIO:-ceobench-sao}"
# The trainer's window (serve.yaml's seq-length): turns longer than this are
# served and recorded but not reported for training.
export CEOBENCH_TRAIN_MAX_TOKENS="${CEOBENCH_TRAIN_MAX_TOKENS:-49152}"
# The agent's shell runs as this unprivileged user inside the task container.
export SAAS_BENCH_TOOL_USER="${SAAS_BENCH_TOOL_USER:-agent}"

# Prerequisites
command -v uv >/dev/null || { echo "run.sh: uv not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
docker image inspect "$REEF_IMAGE" >/dev/null 2>&1 \
    || { echo "run.sh: image $REEF_IMAGE not found (build docker/Dockerfile.reef)" >&2; exit 1; }
[ -d "$MODEL_DIR/Qwen3.6-27B" ] \
    || { echo "run.sh: $MODEL_DIR/Qwen3.6-27B not found (hf download Qwen/Qwen3.6-27B)" >&2; exit 1; }
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/token" ] || openssl rand -hex 16 > "$RUN_DIR/token"
export REEF_TOKEN="$(cat "$RUN_DIR/token")"

# 1. Compose owns the stack's startup and readiness.
echo "==> [1/2] the reef stack at $REEF_SERVICE_URL"
(
    export REEF_IMAGE MODEL_DIR RUN_DIR REEF_ROOT
    docker compose up -d --wait
) || { echo "run.sh: the stack never became healthy; docker compose logs" >&2; exit 1; }

# 2. The episodes, through reef-eval in an ephemeral uv environment.
echo "==> [2/2] seed ${CEOBENCH_SEED:-42}, ${CEOBENCH_DAYS:-500} days (agent: harness:HarborAgent)"
uv run --no-project --python 3.12 \
    --with "reef-eval[harbor]" --with reef-client --with-editable "$PWD" \
    run.py "$@"
