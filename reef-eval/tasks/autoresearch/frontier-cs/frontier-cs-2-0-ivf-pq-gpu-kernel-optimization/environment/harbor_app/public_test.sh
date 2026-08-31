#!/usr/bin/env bash
# Public self-test: run the patched ivfpqlib on the public shapes and print a
# correctness + rough speedup summary. Requires a GPU in the agent container.
set -euo pipefail
APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"
exec python3 "$APP_DIR/public_test.py"
