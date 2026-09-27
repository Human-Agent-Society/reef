#!/bin/sh
set -eu
mkdir -p /logs/verifier
if [ -f /workspace/solved.txt ] && [ "$(cat /workspace/solved.txt)" = "391" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
