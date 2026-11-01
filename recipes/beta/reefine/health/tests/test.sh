#!/bin/sh
set -eu

if [ -f /workspace/health.txt ] && [ "$(cat /workspace/health.txt)" = "reef-ok" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
