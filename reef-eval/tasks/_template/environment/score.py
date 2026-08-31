"""The scoring rule, run by the judge on every submission. Rules:

- recompute everything from the submitted file; NEVER read agent-claimed
  scores;
- validate conservatively (exact arithmetic where floats can be gamed;
  reject, don't round, on violations);
- return {"reward": float, "reason": str}; invalid input scores 0.0 with a
  reason, never an exception.

This file ships only in the judge image. The agent never reads it and
only sees scores come back from /submit. Hidden tests go in final.py
(same signature), which runs once on the best submission at the end.

TODO(task): replace the checks in grade() with your task's.
"""

import json
from pathlib import Path


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution submitted"}
    try:
        solution = json.loads(Path(artifact).read_text())
        x = float(solution["x"])
    except (ValueError, TypeError, KeyError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e!r}"}
    if not 0.0 <= x <= 1.0:
        return {"reward": 0.0, "reason": f"x={x} outside [0, 1]"}
    return {"reward": x, "reason": "ok"}
