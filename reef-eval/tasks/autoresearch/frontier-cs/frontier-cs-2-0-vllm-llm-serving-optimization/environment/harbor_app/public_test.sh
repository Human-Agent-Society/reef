#!/usr/bin/env bash
# Async public-test client. Deploys the current /app/vllm working tree to a Modal
# H100, runs the public instance subset, and reports latency + accuracy feedback
# (not merely whether the build succeeded).
#
#   bash /app/public_test.sh launch        # start an async run, prints a run id
#   bash /app/public_test.sh status <id>   # poll for latency/accuracy result
#   bash /app/public_test.sh run           # run synchronously and print result
set -euo pipefail
exec python3 /app/public_test.py "$@"
