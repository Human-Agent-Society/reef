"""The scoring rule, run by the judge on every submission: validates the packing exactly, recomputes the first-fit
baseline from the same pinned item set. Reward = first-fit bins / your bins;
1.0 = as good as first-fit, above 1.0 = genuinely better."""

import json
from pathlib import Path

_DATA = json.loads((Path(__file__).parent / "items.json").read_text())
ITEMS, CAPACITY = _DATA["items"], _DATA["capacity"]


def first_fit_bins() -> int:
    bins: list[int] = []
    for size in ITEMS:
        for i, load in enumerate(bins):
            if load + size <= CAPACITY:
                bins[i] += size
                break
        else:
            bins.append(size)
    return len(bins)


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution.json artifact"}
    try:
        bins = json.loads(Path(artifact).read_text())["bins"]
        bins = [[int(i) for i in b] for b in bins]
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}

    placed = sorted(i for b in bins for i in b)
    if placed != list(range(len(ITEMS))):
        return {"reward": 0.0, "reason": "every item must appear exactly once"}
    for k, b in enumerate(bins):
        if not b:
            return {"reward": 0.0, "reason": f"bin {k} is empty"}
        if sum(ITEMS[i] for i in b) > CAPACITY:
            return {"reward": 0.0, "reason": f"bin {k} exceeds capacity {CAPACITY}"}

    return {"reward": first_fit_bins() / len(bins), "bins_used": len(bins)}
