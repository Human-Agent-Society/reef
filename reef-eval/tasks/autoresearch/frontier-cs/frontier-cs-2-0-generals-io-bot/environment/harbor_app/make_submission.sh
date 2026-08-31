#!/usr/bin/env bash
set -euo pipefail

REPO="${GENERALS_AGENT_DIR:-/app/generals_agent}"
OUT="${1:-/app/solution.patch}"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Generals agent checkout not found at $REPO" >&2
  exit 2
fi

git -C "$REPO" diff --binary -- bot.py strategy.py utils.py > "$OUT"
bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "Wrote $OUT ($bytes bytes)"
