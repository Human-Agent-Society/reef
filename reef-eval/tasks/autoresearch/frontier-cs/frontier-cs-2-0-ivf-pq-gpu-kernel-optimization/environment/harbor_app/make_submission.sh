#!/usr/bin/env bash
# Package the current ivfpqlib edits into a unified diff for submission.
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
OUT="${1:-/app/solution.patch}"

if [[ ! -d "$APP_DIR/ivfpqlib" ]]; then
  echo "ivfpqlib package not found at $APP_DIR/ivfpqlib" >&2
  exit 2
fi
if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "git repo not found at $APP_DIR (expected in the agent image)" >&2
  exit 2
fi

# Stage and diff only the package, so submission helper scripts under /app are
# never included. New kernel files under ivfpqlib/ are captured via `git add`.
git -C "$APP_DIR" add -A -- ivfpqlib
git -C "$APP_DIR" diff --cached -- ivfpqlib > "$OUT"
git -C "$APP_DIR" reset -q

bytes=$(wc -c < "$OUT" | tr -d ' ')
echo "Wrote $OUT ($bytes bytes)"
