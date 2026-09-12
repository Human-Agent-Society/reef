#!/bin/sh
# The verifier: score the finished run from its world.nmdb.
set -eu
mkdir -p /logs/verifier
cd /opt/ceobench
uv run --no-sync python /tests/score.py /workspace/ceobench-runs /logs/verifier
