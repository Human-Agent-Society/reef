#!/usr/bin/env bash
# Re-pin the 72 tasks to user_sim/'s current content hash. Run after changing
# anything under user_sim/; run.sh refuses to start until the tags agree.
set -euo pipefail
cd "$(dirname "$0")"
TAG="$(cat user_sim/{Dockerfile,personas.py,pyproject.toml,student_server.py} | sha256sum | cut -c1-12)"
sed -i "s/^FROM openclawrl-user-sim:.*/FROM openclawrl-user-sim:$TAG/" harbor-tasks/gsm8k-s*/environment/Dockerfile.judge
echo "re-stamped 72 tasks to openclawrl-user-sim:$TAG"
