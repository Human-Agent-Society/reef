#!/bin/sh
set -eu

mkdir -p /logs/verifier
python3 /tests/grade.py
