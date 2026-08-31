#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
python3 /tests/grade.py
