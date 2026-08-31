#!/bin/bash
# One campaign run. Setup (once): see README - docker, the WildClawBench
# dataset download, and the provider key are required before day one.
# State and logs go to ./work; REEF_SC_RUN selects the method run or the
# frozen control.
set -e
cd "$(dirname "$0")"
mkdir -p work

# Env defaults: the OpenAI-compatible endpoint (no /v1 suffix), the model
# under test (executor and evolver, the paper's qwen3-max), the pi binary
# the probe episodes run, and the provider api key (required).
: "${REEF_UPSTREAM_URL:=https://openrouter.ai/api}"
: "${REEF_MODEL:=qwen/qwen3-max}"
: "${REEF_PI_BINARY:=pi}"
: "${REEF_UPSTREAM_API_KEY:?REEF_UPSTREAM_API_KEY is required (the upstream provider key)}"
export REEF_UPSTREAM_URL REEF_MODEL REEF_PI_BINARY REEF_UPSTREAM_API_KEY

# Materialize the benchmark before day one: clone WildClawBench at the pin
# and require the task workspaces (the dataset download reef cannot fetch
# itself). This is the sealed ensure_benchmark path; a missing dataset
# fails here, not sixty tasks into the first day.
PYTHONPATH=../../../.. python3 -c "from harness import day; day.ensure_benchmark()"

# The bootstrap pool: the benchmark's shipped skills library seeds one
# skill node per SKILL.md (skillclaw.yaml's evolution.seed_skills).
REEF_SC_SKILLS="$PWD/work/wildclawbench/skills"
export REEF_SC_SKILLS

# The driver owns the whole run: it embeds the Reef service (skillclaw.yaml's
# recipe, port 8901 for the task containers), replays the day through
# docker, reports every grade, and seals rounds under ./work/campaign.
PYTHONPATH=../../../.. python3 run.py
