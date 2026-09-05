#!/bin/bash
# Serve + run. Setup (once): see README. State and logs go to ./work.
# ./run.sh runs on pi (configs/serve.yaml); ./run.sh native runs on reef's
# native harness (configs/serve-native.yaml) through a resident reef-native
# serve process.
set -e
cd "$(dirname "$0")"
mkdir -p work/recipes work/bin

SERVE=configs/serve.yaml
if [ "${1:-}" = native ]; then
    SERVE=configs/serve-native.yaml
    # The native loop is reef's own; this launcher stands in for the
    # reef-native console script an installed reef would put on PATH.
    printf '#!/bin/sh\nexport PYTHONPATH=%s\nexec %s -m reef.harness.runners.native "$@"\n' \
        "$(cd ../.. && pwd)" "$(command -v python3)" > work/bin/reef-native
    chmod +x work/bin/reef-native
    export PATH="$PWD/work/bin:$PATH"
fi

# Copy the serve file's recipe sections where the recipe registry reads
# them, and drop the task list beside them for run.py.
export REEF_RECIPE_CONFIG_DIR="$PWD/work/recipes"
python3 harness/materialize_recipe.py "$SERVE"

# Start Reef, stop it again when this script exits. The -c path is absolute:
# reef resolves a relative config path against its own repo root.
PYTHONPATH=../.. python3 -m reef serve -c "$PWD/$SERVE" > work/reef.log 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT

# Wait until Reef answers; a dead orchestrator fails fast with its log.
while ! curl -sf http://127.0.0.1:8900/healthz > /dev/null; do
    kill -0 "$SERVE_PID" 2>/dev/null || { cat work/reef.log >&2; exit 1; }
    sleep 1
done

if [ "${1:-}" != native ]; then
    python3 run.py
    exit 0
fi

# The serve form: pull the seed tree, then one resident process follows the
# head while run.py sends the tasks to it as turns. The receipts of every
# turn spool under work/captures, where run.py's report command claims them.
export REEF_HARNESS_CAPTURES_DIR="$PWD/work/captures"
python3 run.py pull
reef-native serve --tree work/tree --scenario harness-evolve-demo --follow head --poll-interval 5 \
    > work/serve.log 2>&1 &
NATIVE_PID=$!
trap 'kill "$NATIVE_PID" "$SERVE_PID" 2>/dev/null' EXIT

# Wait until the process answers on its socket; a process that died fails fast with its log.
while ! reef-native status --tree work/tree > /dev/null 2>&1; do
    kill -0 "$NATIVE_PID" 2>/dev/null || { cat work/serve.log >&2; exit 1; }
    sleep 1
done

python3 run.py native
