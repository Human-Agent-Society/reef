"""One-attempt coding scores, separately labeled mean@k and explicit gain formulas."""

from __future__ import annotations

from collections import Counter
from typing import cast

if __package__:
    from .report import JsonObject
else:
    from report import JsonObject


def is_scored_episode(record: JsonObject) -> bool:
    """Accept binary scores and known zero-score evaluation limits, never infrastructure faults."""
    score = record.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or score not in (0, 1):
        return False
    if record["outcome"] == "completed":
        return True
    return (
        record.get("phase") in ("baseline", "baseline-independent", "frozen-repeat", "independent")
        and record["outcome"] == "truncated"
        and score == 0
        and record.get("fault")
        in (
            "student response or episode token window was exhausted",
            "student code tool timed out",
            "maximum student turns reached without a final submission",
        )
    )


def summarize(records: list[JsonObject]) -> JsonObject:
    """Compute accuracy on valid records; faults are counted, not scored as failures."""
    counts = Counter(str(record["outcome"]) for record in records)
    valid = [record for record in records if is_scored_episode(record)]
    scores = [float(cast(float, record["score"])) for record in valid]
    return {
        "episodes": len(records),
        "valid": len(valid),
        "faulted": counts["fault"],
        "truncated": counts["truncated"],
        "accuracy": sum(scores) / len(scores) if scores else None,
        "prompt_tokens": sum(int(cast(int, record.get("prompt_tokens", 0))) for record in records),
        "completion_tokens": sum(int(cast(int, record.get("completion_tokens", 0))) for record in records),
        "elapsed_seconds": sum(float(cast(float, record.get("elapsed_seconds", 0))) for record in records),
    }


def build_summary(records: list[JsonObject], complex_category: str = "new") -> JsonObject:
    by_phase: dict[str, list[JsonObject]] = {}
    for record in records:
        by_phase.setdefault(str(record["phase"]), []).append(record)
    result: JsonObject = {
        "score_protocol": "attempt 0 only for baseline, first-pass, frozen-repeat and independent; train mean@k separate",
        "gain_units": "percentage points",
        "formulas": {
            "PG": "100 * (first_pass_complex - baseline_complex)",
            "SG": "100 * (frozen_repeat_complex - first_pass_complex)",
            "GG": "100 * (independent_final - independent_baseline)",
        },
        "phases": {},
        "learning_curve": [],
    }
    phases = cast(JsonObject, result["phases"])
    for phase, rows in by_phase.items():
        primary = [row for row in rows if row["attempt"] == 0]
        phases[phase] = {
            "one_attempt": summarize(primary),
            "complex": summarize([row for row in primary if row["category"] == complex_category]),
            "subtasks": summarize([row for row in primary if row["category"] != complex_category]),
            "mean_at_k": summarize(rows) if phase == "train" else None,
        }
    primary_train = [row for row in by_phase.get("train", []) if row["attempt"] == 0]
    curve = cast(list, result["learning_curve"])
    for row in sorted(primary_train, key=lambda record: int(cast(int, record["position"]))):
        curve.append(
            {
                "position": row["position"],
                "category": row["category"],
                "task_id": row["task_id"],
                "score_before_update": row["score"],
                "release_id": row["release_id"],
            }
        )
    gains: JsonObject = {}
    for name, before_phase, after_phase, subset in (
        ("PG", "baseline", "train", "complex"),
        ("SG", "train", "frozen-repeat", "complex"),
        ("GG", "baseline-independent", "independent", "one_attempt"),
    ):
        before = [row for row in by_phase.get(before_phase, []) if row["attempt"] == 0]
        after = [row for row in by_phase.get(after_phase, []) if row["attempt"] == 0]
        if subset == "complex":
            before = [row for row in before if row["category"] == complex_category]
            after = [row for row in after if row["category"] == complex_category]
        before_keys = {(row["category"], row["task_id"]) for row in before}
        after_keys = {(row["category"], row["task_id"]) for row in after}
        if not before or before_keys != after_keys or len(before_keys) != len(before) or len(after_keys) != len(after):
            gains[name] = None
            continue
        if any(not is_scored_episode(row) for row in before + after):
            gains[name] = None
            continue
        gains[name] = 100 * (
            sum(float(cast(float, row["score"])) for row in after) / len(after)
            - sum(float(cast(float, row["score"])) for row in before) / len(before)
        )
    result["gains"] = gains
    result["gpu_cost"] = "not inferred from wall time; requires supervisor GPU accounting"
    return result
