"""Deterministic valid baseline for the BBOPlace ICCAD2015 task."""

from __future__ import annotations


def solve(info):
    return [0.0] * int(info["dim"])
