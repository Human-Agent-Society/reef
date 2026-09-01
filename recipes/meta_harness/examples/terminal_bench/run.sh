#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv-meta-harness"
exec uv run --locked --project "$HERE" --with-editable "$REPO_ROOT" python "$HERE/run.py" "$@"
