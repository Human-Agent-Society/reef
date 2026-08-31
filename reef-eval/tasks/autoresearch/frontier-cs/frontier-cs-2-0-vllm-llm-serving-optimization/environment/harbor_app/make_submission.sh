#!/usr/bin/env bash
set -euo pipefail

VLLM_DIR="${VLLM_DIR:-/app/vllm}"
OUT="${1:-/app/solution.patch}"

if [[ ! -d "$VLLM_DIR/.git" ]]; then
  echo "vLLM checkout not found at $VLLM_DIR" >&2
  exit 2
fi

# Stage everything (including new .py files) and diff against the base commit so
# the patch captures all source changes the judge will apply to a clean tree.
git -C "$VLLM_DIR" add -A
git -C "$VLLM_DIR" diff --cached --binary > "$OUT"
bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "Wrote $OUT ($bytes bytes)"
