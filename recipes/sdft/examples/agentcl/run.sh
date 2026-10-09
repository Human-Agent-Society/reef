#!/bin/bash
# Drive an existing supervised Reef service; never allocate GPUs or mint credentials.
set -euo pipefail
cd "$(dirname "$0")"
root="$(cd ../../../.. && pwd)"
if [[ -n "${AGENTCL_PYTHON:-}" ]]; then
    python="$AGENTCL_PYTHON"
elif [[ -x "$root/.venv312/bin/python" ]]; then
    python="$root/.venv312/bin/python"
else
    python="$root/.venv/bin/python"
fi
if [[ ! -x "$python" ]]; then
    echo "Use the repository Python environment; see README.md." >&2
    exit 1
fi
exec "$python" run.py "$@"
