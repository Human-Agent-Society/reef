#!/bin/sh
set -eu
mkdir -p /logs/verifier
env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 /tests/verify.py --task /tests/task.json --answer /workspace/answer.py \
  --output-dir /logs/verifier --timeout-seconds 60
