"""Accuracy statistics shared by the plotting scripts."""

from __future__ import annotations

import json
import math


def load_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wilson_interval(successes: float, count: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a binomial proportion; (0, 0) for an empty sample."""
    if count == 0:
        return (0.0, 0.0)
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count * count)) / denominator
    return (center - half_width, center + half_width)
