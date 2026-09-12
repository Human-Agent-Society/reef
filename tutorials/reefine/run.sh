#!/bin/bash
# Reefine on reef-pi, end to end. Usage: ./run.sh bugfix | research | measure [--n N]
# Starts reef serve on configs/deployment.yaml, waits for /healthz, installs
# the served tree under work/harness, runs run.py <mode>, stops the service.
# State and logs go to ./work. Setup (once): see README.
set -e
cd "$(dirname "$0")"

MODE="${1:-}"
case "$MODE" in
    bugfix|research|measure) ;;
    *) echo "usage: ./run.sh bugfix | research | measure [--n N]" >&2; exit 2 ;;
esac
shift
mkdir -p work

# The model server and the model: ollama on this machine unless the environment says otherwise.
export REEF_UPSTREAM_URL="${REEF_UPSTREAM_URL:-http://127.0.0.1:11434}"
export REEF_UPSTREAM_MODEL="${REEF_UPSTREAM_MODEL:-gemma4:26b}"
export REEF_UPSTREAM_API_KEY="${REEF_UPSTREAM_API_KEY:-dummy}"
export REEF_TOKEN="${REEF_TOKEN:-reef-local}"
# A local 26B model answers a request in minutes; the method package's default budget is two.
export REEF_PROPOSER_TIMEOUT_S="${REEF_PROPOSER_TIMEOUT_S:-900}"
# A thinking model spends the reply budget on its reasoning first; 4096 tokens came back empty.
export REEF_PROPOSER_MAX_TOKENS="${REEF_PROPOSER_MAX_TOKENS:-16384}"

TUTORIAL="$PWD"
REPO="$(cd ../.. && pwd)"
# The install script's import check and reef serve run from this checkout; the wrapper it writes bakes this interpreter.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# deployment.yaml listens on 8901; a reef already there would answer for another deployment.
if curl -sf http://127.0.0.1:8901/healthz > /dev/null 2>&1; then
    echo "a reef already answers on 127.0.0.1:8901; stop it first" >&2
    exit 1
fi

# Start Reef from the repository root, where deployment.yaml's state directories are relative to,
# and stop it again when this script exits.
(cd "$REPO" && exec python3 -m reef serve -c "$TUTORIAL/configs/deployment.yaml") > work/reef.log 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT

# Wait until Reef answers; a dead orchestrator fails fast with its log.
while ! curl -sf http://127.0.0.1:8901/healthz > /dev/null; do
    kill -0 "$SERVE_PID" 2>/dev/null || { cat work/reef.log >&2; exit 1; }
    sleep 1
done

# The served tree, the reef-pi wrapper and the release metadata file land under work/harness.
python3 run.py install
python3 run.py "$MODE" "$@"
