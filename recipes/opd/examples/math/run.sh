#!/usr/bin/env bash
# OPD on mathematical reasoning (recipes/opd/examples/math): run.py samples,
# reports and evaluates against the Reef stack this script serves. Setup
# (once): download the student and teacher checkpoints and prepare the data as
# the README describes. State goes to $OPD_RUN_DIR; the campaign's responses,
# scores to $OPD_RUN_DIR/results.
#
#   ./run.sh                                 the protocol: 55 updates, evaluated every 5
#   OPD_STEPS=1 OPD_EVAL_REPEATS=1 ./run.sh  one update and one sample per question, to check the stack
set -euo pipefail
cd "$(dirname "$0")"
export OPD_MODEL_PATH="${OPD_MODEL_PATH:?the student, the pinned Qwen3-8B-Base OpenThoughts3 SFT checkpoint}"
export OPD_TEACHER_PATH="${OPD_TEACHER_PATH:?the teacher, the pinned Qwen3-32B directory}"
export OPD_RUN_DIR="${OPD_RUN_DIR:-$PWD/work}"
OPD_DATA="${OPD_DATA:-/work/data}"               # prepare.py's output
OPD_CONFIG="${OPD_CONFIG:-$PWD/serve.yaml}"
OPD_STEPS="${OPD_STEPS:-55}"
OPD_EVAL_EVERY="${OPD_EVAL_EVERY:-5}"            # evaluate every N updates
OPD_EVAL_TOKENS="${OPD_EVAL_TOKENS:-30720}"      # the student's 32768-token window less 2048 for the prompt
REEF_PORT="${REEF_PORT:-8900}"                   # serve.yaml's reef.port
REPO_ROOT="$(cd ../../../.. && pwd)"

# Prerequisites: the Slime GPU environment (docker/README.md), the models and the prepared data.
python -c "import reef, reef_client" || { echo "run.sh: run inside the Reef GPU environment" >&2; exit 1; }
for path in "$OPD_MODEL_PATH" "$OPD_TEACHER_PATH" "$OPD_DATA/deepmath.jsonl" "$OPD_DATA/aime24.jsonl"; do
  [ -e "$path" ] || { echo "run.sh: $path not found; see README" >&2; exit 1; }
done
mkdir -p "$OPD_RUN_DIR"
[ -f "$OPD_RUN_DIR/token" ] || openssl rand -hex 16 > "$OPD_RUN_DIR/token"
export REEF_TOKEN="$(cat "$OPD_RUN_DIR/token")"
# The example scripts import recipes.opd.examples.math.*, so the repository root must be importable.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m reef serve -c "$OPD_CONFIG" > "$OPD_RUN_DIR/reef.log" 2>&1 &
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
    tail -100 "$OPD_RUN_DIR/reef.log"
    exit 1
  fi
  if curl --silent --fail --max-time 5 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "run.sh: Reef did not become ready; see $OPD_RUN_DIR/reef.log" >&2; exit 1; }

# The campaign: 64 prompts by four responses per update under the 16384-token training budget, the
# baseline and every OPD_EVAL_EVERY-th update evaluated on all 30 AIME'24 questions, 16 samples each.
python run.py --config "$OPD_CONFIG" --url "http://127.0.0.1:$REEF_PORT" \
  --scenario "${REEF_SCENARIO:-opd-math}" \
  --train-data "$OPD_DATA/deepmath.jsonl" --eval-data "$OPD_DATA/aime24.jsonl" \
  --output "$OPD_RUN_DIR/results" \
  --steps "$OPD_STEPS" --prompts-per-step 64 --samples-per-prompt 4 \
  --train-tokens 16384 --eval-tokens "$OPD_EVAL_TOKENS" --eval-repeats "${OPD_EVAL_REPEATS:-16}" \
  --eval-every "$OPD_EVAL_EVERY" --concurrency 128 "$@"
