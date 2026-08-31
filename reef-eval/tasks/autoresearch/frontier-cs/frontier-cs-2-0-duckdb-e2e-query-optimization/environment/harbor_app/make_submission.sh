#!/usr/bin/env bash
set -euo pipefail

DUCKDB_DIR="${DUCKDB_DIR:-/app/duckdb}"
OUT="${1:-/app/solution.patch}"

if [[ ! -d "$DUCKDB_DIR/.git" ]]; then
  echo "DuckDB checkout not found at $DUCKDB_DIR" >&2
  exit 2
fi

git -C "$DUCKDB_DIR" diff --binary > "$OUT"
bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "Wrote $OUT ($bytes bytes)"
