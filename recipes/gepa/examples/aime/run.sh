#!/bin/bash
# One AIME validation run of the GEPA method. Setup (once): see README - the
# Pi binary at the pinned version and an OpenAI key are required. State goes
# to ./work and a rerun resumes from it.
set -e
cd "$(dirname "$0")"
mkdir -p work

# Env defaults: the work directory gepa.yaml interpolates, the task model
# under evolution, the pinned pi binary, and the episode concurrency. run.py reads OPENAI_API_KEY
# itself, and only for a live run: --dry-run needs no key.
: "${REEF_WORK:=$PWD/work}"
: "${REEF_MODEL:=gpt-4.1-mini-2025-04-14}"
: "${REEF_PI_BINARY:=pi}"
: "${REEF_GEPA_WORKERS:=128}"
export REEF_WORK REEF_MODEL REEF_PI_BINARY REEF_GEPA_WORKERS

# The driver owns the whole run: it embeds the Reef service (gepa.yaml's
# recipe on an ephemeral port), runs the minibatch episodes against it,
# reports every score, and seals the two test passes under ./work. The
# repository root on PYTHONPATH exposes recipes.gepa.recipe, the method
# gepa.yaml names.
PYTHONPATH=../../../.. python3 run.py "$@"
