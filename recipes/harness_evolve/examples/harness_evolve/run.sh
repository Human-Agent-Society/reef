#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
set -e
cd "$(dirname "$0")"
mkdir -p work/recipes

# Copy serve.yaml's recipe sections where the recipe registry reads them, and
# drop the task list beside them for run.py.
export REEF_RECIPE_CONFIG_DIR="$PWD/work/recipes"
python3 materialize_recipe.py

# Start Reef, stop it again when this script exits. The -c path is absolute:
# reef resolves a relative config path against its own repo root.
PYTHONPATH=../../../.. python3 -m reef serve -c "$PWD/serve.yaml" > work/reef.log 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT

# Wait until Reef answers; a dead orchestrator fails fast with its log.
while ! curl -sf http://127.0.0.1:8900/healthz > /dev/null; do
    kill -0 "$SERVE_PID" 2>/dev/null || { cat work/reef.log >&2; exit 1; }
    sleep 1
done

python3 run.py
