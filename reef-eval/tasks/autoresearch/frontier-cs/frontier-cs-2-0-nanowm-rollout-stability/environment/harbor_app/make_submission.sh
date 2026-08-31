#!/usr/bin/env bash
# Write /app/solution.patch from the agent's edits to /app/nano-world-model.
set -euo pipefail
cd /app/nano-world-model
git add -A 2>/dev/null || true
git diff --staged > /app/solution.patch 2>/dev/null || git diff > /app/solution.patch
echo "wrote /app/solution.patch ($(wc -l < /app/solution.patch) lines)"
