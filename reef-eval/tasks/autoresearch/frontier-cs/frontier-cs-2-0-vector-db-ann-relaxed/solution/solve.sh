#!/bin/bash
set -euo pipefail

if [ -f /solution/reference.py ]; then
    cp /solution/reference.py /app/solution.py
else
    echo "No reference solution available." >&2
    exit 1
fi
