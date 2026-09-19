#!/usr/bin/env bash
# The SAO training stack for CEO-Bench, then one episode through reef-eval:
# the benchmark's bash agent, played by this harness against the stack and
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
# The longest turn the trainer takes (serve.yaml explains the budget): turns
# longer than this are served and recorded but not reported for training.
export CEOBENCH_TRAIN_MAX_TOKENS="${CEOBENCH_TRAIN_MAX_TOKENS:-24576}"
# The reward values a week's opening state as cash plus the subscription
# run-rate (subscribers at the lowest listed price, enterprise seats at plan
# C's) over the weeks left in the episode, at most this many of them.
export CEOBENCH_VALUE_HORIZON_WEEKS="${CEOBENCH_VALUE_HORIZON_WEEKS:-26}"
# A week is credited with the discounted value changes of this many weeks
# from it on (the week that buys growth is credited with the subscribers that
# arrive after it), and reported once that many weeks have opened after it.
export CEOBENCH_CREDIT_WEEKS="${CEOBENCH_CREDIT_WEEKS:-4}"
export CEOBENCH_CREDIT_DISCOUNT="${CEOBENCH_CREDIT_DISCOUNT:-0.8}"
# Before a credit is posted it is clipped (one six-figure purchase must not
# set the scale for the episode) and divided by the running median magnitude
# of the credits so far, floored so a quiet stretch does not amplify noise.
export CEOBENCH_SCORE_CLIP="${CEOBENCH_SCORE_CLIP:-0.05}"
export CEOBENCH_SCORE_FLOOR="${CEOBENCH_SCORE_FLOOR:-0.003}"
# The agent's tools run as this unprivileged user inside the task container.
export CEOBENCH_TOOL_USER="${CEOBENCH_TOOL_USER:-agent}"
# Limits widened for a paced game: a bash command such as next-week may take
# an hour (the benchmark's limit is 1200 s; the engine gives up at 4200 s), a
# model call may wait through a pacer hold, and one simulator request is
# bounded (reef.patch reads it) so a stuck call fails instead of stalling
# the week.
export CEOBENCH_BASH_TIMEOUT_S="${CEOBENCH_BASH_TIMEOUT_S:-3600}"
export CEOBENCH_LLM_TIMEOUT_S="${CEOBENCH_LLM_TIMEOUT_S:-1800}"
export SAAS_BENCH_SIMULATOR_TIMEOUT_S="${SAAS_BENCH_SIMULATOR_TIMEOUT_S:-300}"
# The untrained baseline: 0 records the episode without posting a report, so
# the stack serves the base model throughout and trains nothing.
export CEOBENCH_REPORTS="${CEOBENCH_REPORTS:-1}"
# Pace the game to the trainer: the recipe's batch size (serve.yaml), so each
# new week starts only after the reported weeks' batches have committed.
export CEOBENCH_PACE_BATCH="${CEOBENCH_PACE_BATCH:-8}"
# The week gate reads the engine's own MRR through the task container, so the
# valuation uses the books instead of the listed-price estimate; 0 turns it off.
export CEOBENCH_ENGINE_READS="${CEOBENCH_ENGINE_READS:-1}"
# A week waits at most this long for its batch (a step takes about five
# minutes) before the game goes on without it.
export CEOBENCH_PACE_TIMEOUT_S="${CEOBENCH_PACE_TIMEOUT_S:-1200}"

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
