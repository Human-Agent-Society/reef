#!/usr/bin/env bash
# The skill stream (Figure 3 of arXiv:2601.19897) under supervised fine-tuning
# through reef-eval: run.py starts the Reef stack per stage and runs each
# stage as a Harbor task. Setup (once): see README. State goes to $RUN_DIR.
#
#   ./run.sh                     Tool Use, then Science Q&A
#   SKILLS_STEPS=2 ./run.sh --stream smoke
set -euo pipefail
cd "$(dirname "$0")"
export REEF_ROOT="$(cd ../../../.. && pwd)"
export REEF_IMAGE="${REEF_IMAGE:-reef}"
export MODEL_DIR="${MODEL_DIR:-$HOME/models}"
export RUN_DIR="${RUN_DIR:-$PWD/work}"

# Prerequisites
command -v uv >/dev/null || { echo "run.sh: uv not found (pip install uv)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "run.sh: Docker is not running" >&2; exit 1; }
docker image inspect "$REEF_IMAGE" >/dev/null 2>&1 \
    || { echo "run.sh: image $REEF_IMAGE not found (build docker/Dockerfile.reef)" >&2; exit 1; }
[ -d "$MODEL_DIR/Qwen2.5-7B-Instruct" ] \
    || { echo "run.sh: $MODEL_DIR/Qwen2.5-7B-Instruct not found (hf download Qwen/Qwen2.5-7B-Instruct)" >&2; exit 1; }
mkdir -p "$RUN_DIR"
[ -f "$RUN_DIR/token" ] || openssl rand -hex 16 > "$RUN_DIR/token"
export REEF_TOKEN="$(cat "$RUN_DIR/token")"

# The stream, in an ephemeral uv environment: reef-eval with Harbor, the
# reef-client protocol, and this directory's harness package.
uv run --no-project --python 3.12 \
    --with "reef-eval[harbor]" --with reef-client --with-editable "$PWD" \
    run.py "$@"
