#!/bin/sh
set -eu

# Both answers, and no git state left in the workdir: a team's worktrees live outside it.
if [ "$(cat /workspace/sum.txt 2>/dev/null)" = "42" ] \
    && [ "$(cat /workspace/product.txt 2>/dev/null)" = "391" ] \
    && [ ! -e /workspace/.git ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
