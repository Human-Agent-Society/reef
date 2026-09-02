#!/bin/sh
# One invocation of the reproduction. Setup (once): pip install -e . in this
# directory, which brings GEPA at its pinned commit plus the dataset and
# LiteLLM dependencies into the environment that holds the Reef checkout, and
# Pi 0.84.2 on PATH or at REEF_PI_BINARY. The repository root on PYTHONPATH
# exposes reef itself, the sibling examples' idiom. Arguments go to run.py:
# --dry-run validates the pins without a model call.
set -eu
cd "$(dirname "$0")"
PYTHONPATH=../../../.. exec python3 run.py "$@"
