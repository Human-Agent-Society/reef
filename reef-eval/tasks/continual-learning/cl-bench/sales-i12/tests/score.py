"""Sales-prediction scorer, run as each task's verifier.

The metric is CL-Bench's WAPE-skill, unchanged:
``1 - sum|pred - actual| / sum|actual|`` over every required
(locality, furniture, year) entry. 1.0 is a perfect forecast, 0 is as
good as predicting zero everywhere, and large over-predictions go
negative (unclipped, as upstream). Parsing mirrors the upstream runtime:
duplicates keep the first entry, and missing or unusable entries count
as predicting 0. A missing or non-JSON file scores 0.
"""

import json
import os

ANSWER_PATH = "/app/predictions.json"
TRUTH_PATH = "/tests/truth.json"
REWARD_PATH = "/logs/verifier/reward.txt"
DETAILS_PATH = "/logs/verifier/scoring.json"


def wape_skill(y_true: list[float], y_pred: list[float]) -> float:
    num = sum(abs(p - t) for p, t in zip(y_pred, y_true, strict=False))
    den = sum(abs(t) for t in y_true)
    if den == 0.0:
        return 1.0 if num == 0.0 else float("-inf")
    return 1.0 - num / den


def parse_predictions(raw_text: str) -> tuple[dict[tuple, float], str]:
    """(locality, furniture_name, year) -> predicted items, or a reason."""
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as error:
        return {}, f"invalid JSON: {error}"
    if isinstance(raw, list):
        raw = {"predictions": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("predictions"), list):
        return {}, "expected an object with a 'predictions' list"
    parsed: dict[tuple, float] = {}
    for entry in raw["predictions"]:
        try:
            key = (
                str(entry["locality"]),
                str(entry["furniture_name"]),
                int(entry["year"]),
            )
            value = float(entry["items_sold"])
        except (TypeError, KeyError, ValueError):
            continue  # unusable entry — the required key falls back to 0
        parsed.setdefault(key, value)  # duplicates: first wins, as upstream
    return parsed, "ok"


def score(raw_text: str, truth: list[dict]) -> dict:
    predictions, reason = parse_predictions(raw_text)
    if reason != "ok":
        return {"score": 0.0, "format_valid": False, "reason": reason}
    y_true, y_pred, missing = [], [], 0
    for row in truth:
        key = (row["locality"], row["furniture_name"], int(row["year"]))
        y_true.append(float(row["items_sold"]))
        value = predictions.get(key)
        if value is None:
            missing += 1
        y_pred.append(0.0 if value is None else value)
    return {
        "score": round(wape_skill(y_true, y_pred), 6),
        "format_valid": True,
        "reason": "ok",
        "n_required": len(truth),
        "n_missing_counted_as_zero": missing,
    }


def main() -> int:
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(TRUTH_PATH) as f:
        truth = json.load(f)
    raw_text = ""
    if os.path.exists(ANSWER_PATH):
        with open(ANSWER_PATH, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    result = (
        score(raw_text, truth)
        if raw_text.strip()
        else {"score": 0.0, "format_valid": False, "reason": f"no {ANSWER_PATH}"}
    )
    with open(REWARD_PATH, "w") as f:
        f.write(str(result["score"]))
    with open(DETAILS_PATH, "w") as f:
        json.dump(result, f, indent=1)
    print(f"WAPE-skill: {result['score']} ({result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
