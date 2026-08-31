#!/usr/bin/env bash
# Thin wrapper so the agent can run `bash /app/submit.sh` (or just `submit.sh`)
# without thinking about Python. Forwards any args (e.g. an alt solution path).
exec python3 /app/submit.py "$@"
