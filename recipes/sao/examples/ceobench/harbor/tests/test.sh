#!/bin/sh
# The verifier: score the finished run from its world.nmdb.
set -eu
mkdir -p /logs/verifier
cd /opt/ceobench
# The harness stops the engine when the episode ends; when it could not, stop
# it here so world.nmdb is complete before it is scored.
for run in /workspace/ceobench-runs/run_*; do
    [ -f "$run/world.nmdb" ] || .venv/bin/python /opt/ceobench-engine.py stop --run-dir "$run" || true
done
uv run --no-sync python /tests/score.py /workspace/ceobench-runs /logs/verifier
