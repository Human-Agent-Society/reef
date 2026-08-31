"""Initial OpenEvolve candidate for reef-eval's three-circle packing task."""

import json


# EVOLVE-BLOCK-START
def solve() -> dict:
    """Return a valid baseline; evolution should improve these circles."""
    radius = 0.25
    circles = [
        [0.25, 0.25, radius],
        [0.75, 0.25, radius],
        [0.50, 0.75, radius],
    ]
    return {"circles": circles}


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    print(json.dumps(solve()))
