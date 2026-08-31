"""The scoring rule, run by the judge on every submission: validates the permutation exactly, recomputes both lengths.

The baseline (identity tour) is computed here from the same pinned city set,
so reward 1.0 = "no better than visiting cities in file order" and anything
above 1.0 is a genuine improvement.
"""

import json
import math
from pathlib import Path

CITIES = json.loads((Path(__file__).parent / "cities.json").read_text())["cities"]


def tour_length(order) -> float:
    n = len(order)
    return sum(
        math.dist(CITIES[order[i]], CITIES[order[(i + 1) % n]]) for i in range(n)
    )


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution.json artifact"}
    try:
        tour = json.loads(Path(artifact).read_text())["tour"]
        tour = [int(i) for i in tour]
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}
    if sorted(tour) != list(range(len(CITIES))):
        return {
            "reward": 0.0,
            "reason": f"not a permutation of 0..{len(CITIES) - 1}",
        }
    length = tour_length(tour)
    baseline = tour_length(list(range(len(CITIES))))
    return {"reward": baseline / length, "tour_length": length}
