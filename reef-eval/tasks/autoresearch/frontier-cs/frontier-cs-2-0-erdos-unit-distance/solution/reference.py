"""Baseline solution: equally spaced points on a regular polygon."""

from __future__ import annotations

import math


def solve(n: int):
    radius = 1.0 / (2.0 * math.sin(math.pi / n))
    return [
        (
            radius * math.cos(2.0 * math.pi * i / n),
            radius * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]
