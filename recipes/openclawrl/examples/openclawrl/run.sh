#!/usr/bin/env bash
# The reef training stack (docker-compose.yaml), then the 72-session GSM8K
# stream through REEF Eval. Setup (once): see README. State goes to $RUN_DIR.
#
# The stack is left running and a healthy one is reused, so a re-run continues
# where the lab left off; `docker compose down` stops it. A trained
# stack is bound to its scenario, so switching STREAM_NAME or RUN_DIR needs a
# restart.
set -euo pipefail
cd "$(dirname "$0")"
REEF_ROOT="$(cd ../../../.. && pwd)"
REEF_EVAL_DIR="$REEF_ROOT/reef-eval"
if [ ! -f "$REEF_EVAL_DIR/pyproject.toml" ]; then
    echo "run.sh: local reef-eval checkout not found at $REEF_EVAL_DIR" >&2
    exit 1
fi

REEF_IMAGE="reef-openclawrl"
REEF_CONFIG="recipes/openclawrl/examples/openclawrl/serve.yaml"
MODEL_DIR="$HOME/models"
RUN_DIR="$HOME/reef-run"
STREAM_NAME="gsm8k"
TASKS="$PWD/harbor-tasks"          # all 72 sessions, in order
HOST_IP="$(hostname -I | awk '{print $1}')"
REEF_URL="http://${HOST_IP}:28900"

# Prerequisites
command -v uvx >/dev/null || { echo "run.sh: uvx not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
docker image inspect "$REEF_IMAGE" >/dev/null 2>&1 \
    || { echo "run.sh: image $REEF_IMAGE not found (build docker/Dockerfile.reef)" >&2; exit 1; }
mkdir -p "$RUN_DIR"

# The judge sidecar, one image shared by every task, tagged with user_sim/'s
# content hash. The tasks pin that tag, so a change here without a re-stamp
# would build a sidecar no task can pull; ./restamp.sh fixes it.
USER_SIM_TAG="$(cat user_sim/{Dockerfile,personas.py,pyproject.toml,student_server.py} | sha256sum | cut -c1-12)"
grep -q "FROM openclawrl-user-sim:$USER_SIM_TAG" harbor-tasks/gsm8k-s000/environment/Dockerfile.judge \
    || { echo "run.sh: user_sim/ changed; re-stamp the 72 tasks with ./restamp.sh" >&2; exit 1; }
docker build -q -t "openclawrl-user-sim:$USER_SIM_TAG" user_sim >/dev/null
[ -f "$RUN_DIR/token" ] || openssl rand -hex 16 > "$RUN_DIR/token"
REEF_TOKEN="$(cat "$RUN_DIR/token")"

# 1. The reef training stack (docker-compose.yaml holds its configuration)
echo "==> [1/2] the reef stack at $REEF_URL (a cold boot takes ~6 minutes on B200s)"
# compose reads these from the environment; the subshell keeps the token out of
# everything that runs after it.
(
    export REEF_IMAGE REEF_CONFIG MODEL_DIR RUN_DIR REEF_ROOT HOST_IP REEF_TOKEN
    docker compose up -d --wait
) || { echo "run.sh: the stack never became healthy; docker compose logs" >&2; exit 1; }

# 2. The stream, through REEF Eval in an ephemeral uvx environment

# Per-session accept/style metrics, when a key is present: learning_curve.py tails
# the lab, because the judge's verdict lands there rather than in reef.
if [ -n "${WANDB_API_KEY:-}" ]; then
    uv run --no-project --with wandb "$PWD/results/learning_curve.py" \
        --lab "$RUN_DIR/lab" --reef-url "$REEF_URL" --reef-token "$REEF_TOKEN" \
        --wandb-dir "$RUN_DIR/lab" > "$RUN_DIR/learning_curve.log" 2>&1 &
    trap 'kill $! 2>/dev/null' EXIT
fi

echo "==> [2/2] streaming $TASKS as '$STREAM_NAME' (agent: harness:HermesStreamAgent)"
REEF_TOKEN="$REEF_TOKEN" uvx \
    --from "${REEF_EVAL_DIR}[harbor]" \
    --with-editable "$PWD" \
    --with reef-client \
    reef-eval stream "$TASKS" --name "$STREAM_NAME" \
        --agent harness:HermesStreamAgent \
        --agent-arg kwargs="{\"reef_url\": \"$REEF_URL\"}" \
        --agent-arg extra_allowed_hosts="[\"$HOST_IP\"]" \
        --lab "$RUN_DIR/lab" \
        "$@"

echo "==> done; REEF Eval lab at $RUN_DIR/lab (stack still serving at $REEF_URL)"
echo "    sessions-to-adaptation = reef_eval.metrics.learning_curve over the stream's rewards"
