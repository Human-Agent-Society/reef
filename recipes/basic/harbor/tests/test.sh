#!/bin/sh
set -eu

if [ -f /workspace/answer.txt ] && [ "$(cat /workspace/answer.txt)" = "391" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
