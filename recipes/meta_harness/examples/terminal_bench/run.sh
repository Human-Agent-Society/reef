#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
EXAMPLE="$ROOT/recipes/meta_harness/examples/terminal_bench"
cd "$ROOT"
uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.runtime
exec uv run --locked --project "$EXAMPLE" \
  python -m recipes.meta_harness.examples.terminal_bench.run "$@"
