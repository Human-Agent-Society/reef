"""The scoring rule, run by the judge on every submission: recomputes the score from the declared artifact, from
scratch, in exact rational arithmetic.

Never reads the agent's claimed scores. Runs in a separate verifier container
that received only the declared artifacts, so nothing the agent did to its own
environment (including tampering with the public scorer) can reach this code.

Conservative by construction: exact Fraction comparisons with the documented
1e-9 tolerance, so buying radius with float-epsilon overlaps scores zero.

Protocol (uniform across reef-eval's first-party tasks): ``grade(path)`` is pure
and takes the artifact path explicitly; ``find_artifact()`` resolves the
canonical location; ``__main__`` writes a numbers-only ``reward.json`` plus a
human ``reason.txt``.
"""

import json
from fractions import Fraction
from pathlib import Path

OPTIMUM = 1.007626  # known optimum for n=3, for the normalized view
TOL = Fraction(1, 10**9)


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution.json artifact"}
    try:
        circles = json.loads(Path(artifact).read_text())["circles"]
        parsed = [(Fraction(x), Fraction(y), Fraction(r)) for x, y, r in circles]
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}
    if len(parsed) != 3:
        return {"reward": 0.0, "reason": "need exactly 3 circles"}

    one = Fraction(1)
    for x, y, r in parsed:
        if r <= 0:
            return {"reward": 0.0, "reason": "non-positive radius"}
        if x - r < -TOL or x + r > one + TOL or y - r < -TOL or y + r > one + TOL:
            return {"reward": 0.0, "reason": "circle outside unit square"}

    for i in range(3):
        for j in range(i + 1, 3):
            xi, yi, ri = parsed[i]
            xj, yj, rj = parsed[j]
            # Exact: dist^2 >= (ri + rj - tol)^2, no square roots involved.
            min_dist = ri + rj - TOL
            if min_dist > 0 and (xi - xj) ** 2 + (yi - yj) ** 2 < min_dist**2:
                return {"reward": 0.0, "reason": f"circles {i},{j} overlap"}

    total = float(sum(r for _, _, r in parsed))
    return {"reward": total, "normalized": total / OPTIMUM}
