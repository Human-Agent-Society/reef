"""Session score: R-squared-style reward on the TRAINING points.

The judge runs this on every submission. The held-out grading lives in
final.py and runs exactly once, on the best submission, so the submission
budget cannot be spent probing the held-out set.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final import MAX_EXPR_LEN, evaluate  # noqa: E402 (same evaluator, one copy)

TRAIN = json.loads((Path(__file__).parent / "train.json").read_text())["points"]


def grade(artifact: Path | None) -> dict:
    if artifact is None or not Path(artifact).exists():
        return {"reward": 0.0, "reason": "no solution submitted"}
    try:
        expr = json.loads(Path(artifact).read_text())["expr"]
    except (KeyError, ValueError, TypeError) as e:
        return {"reward": 0.0, "reason": f"malformed solution: {e}"}
    if not isinstance(expr, str) or len(expr) > MAX_EXPR_LEN:
        return {
            "reward": 0.0,
            "reason": f"expr must be a string <= {MAX_EXPR_LEN} chars",
        }
    errors = []
    for x, y in TRAIN:
        try:
            prediction = evaluate(expr, x)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as e:
            return {"reward": 0.0, "reason": f"expr failed at x={x}: {e}"}
        if not math.isfinite(prediction):
            return {"reward": 0.0, "reason": f"non-finite prediction at x={x}"}
        errors.append((prediction - y) ** 2)
    rmse = math.sqrt(sum(errors) / len(errors))
    return {"reward": 1.0 / (1.0 + rmse), "rmse_train": rmse}
