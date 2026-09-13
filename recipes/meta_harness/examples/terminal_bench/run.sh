#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export REEF_WORK="${REEF_WORK:-$PWD/work}"
export REEF_MODEL="${REEF_MODEL:-openai/gpt-5.6-luna}"
export REEF_UPSTREAM_URL="${REEF_UPSTREAM_URL:-https://api.openai.com}"
export REEF_PROPOSER_MODEL="${REEF_PROPOSER_MODEL:-gpt-5.6-sol}"
export REEF_PROPOSER_URL="${REEF_PROPOSER_URL:-$REEF_UPSTREAM_URL}"
export REEF_PROPOSER_API_KEY="${REEF_PROPOSER_API_KEY:-${REEF_UPSTREAM_API_KEY:-}}"
export REEF_META_HARNESS_WORKERS="${REEF_META_HARNESS_WORKERS:-4}"
export REEF_TERMINUS_ENVIRONMENT=e2b

# The harness is installed by pip; the checkout supplies the recipe and Reef.
export PYTHONPATH="$(cd ../../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" run.py "$@"
