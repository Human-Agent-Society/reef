"""Blind-spectrum-monitoring scorer, run as each task's verifier.

The interval math and IoU definition are ported from CL-Bench's
``blind_spectrum_monitoring/task.py`` (Apache-2.0): ground truth is the
long-run available spectrum (the complement of every persistent
transmitter footprint), the report's claims are complemented the same
way, and the reward is the interval-set IoU of the two available sets.
A malformed or missing report scores as an empty report, as upstream.
Deterministic and offline.
"""

import json
import os

ANSWER_PATH = "/app/report.json"
TRUTH_PATH = "/tests/truth.json"
REWARD_PATH = "/logs/verifier/reward.txt"
DETAILS_PATH = "/logs/verifier/scoring.json"

# ---- interval math, ported verbatim (Apache-2.0, see module docstring) ----


def _interval_bounds(center: float, bandwidth: float) -> tuple[float, float]:
    half_bw = max(0.0, bandwidth) / 2
    return center - half_bw, center + half_bw


def _normalize_intervals(intervals, *, band_start=0.0, band_end=None):
    clipped = []
    for lo, hi in intervals:
        lo = max(band_start, lo)
        hi = hi if band_end is None else min(band_end, hi)
        if hi <= lo:
            continue
        clipped.append((lo, hi))
    if not clipped:
        return []
    clipped.sort()
    merged = [clipped[0]]
    for lo, hi in clipped[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _complement_intervals(occupied_intervals, *, band_start=0.0, band_end):
    normalized = _normalize_intervals(
        occupied_intervals, band_start=band_start, band_end=band_end
    )
    if not normalized:
        return [(band_start, band_end)] if band_end > band_start else []
    available = []
    cursor = band_start
    for lo, hi in normalized:
        if lo > cursor:
            available.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < band_end:
        available.append((cursor, band_end))
    return available


def _intersect_intervals(intervals_a, intervals_b):
    intersections = []
    i = j = 0
    while i < len(intervals_a) and j < len(intervals_b):
        a_lo, a_hi = intervals_a[i]
        b_lo, b_hi = intervals_b[j]
        lo = max(a_lo, b_lo)
        hi = min(a_hi, b_hi)
        if hi > lo:
            intersections.append((lo, hi))
        if a_hi <= b_hi:
            i += 1
        else:
            j += 1
    return intersections


def _interval_set_measure(intervals) -> float:
    return sum(hi - lo for lo, hi in intervals)


# ------------------------------------------------------------------ scoring


def score_report(transmitters: list[dict], truth: dict) -> dict:
    """IoU of the reported vs true long-run available spectrum (0..1)."""
    band_width = float(truth["band_width"])
    gt_occupied = _normalize_intervals(
        [
            _interval_bounds(float(ch["center_freq"]), float(ch["bandwidth"]))
            for ch in truth["channels"]
        ],
        band_end=band_width,
    )
    gt_available = _complement_intervals(gt_occupied, band_end=band_width)
    reported_occupied = _normalize_intervals(
        [
            _interval_bounds(float(t["center_freq"]), float(t["bandwidth"]))
            for t in transmitters
        ],
        band_end=band_width,
    )
    reported_available = _complement_intervals(reported_occupied, band_end=band_width)

    overlap = _intersect_intervals(gt_available, reported_available)
    gt_bw = _interval_set_measure(gt_available)
    reported_bw = _interval_set_measure(reported_available)
    overlap_bw = _interval_set_measure(overlap)
    union_bw = gt_bw + reported_bw - overlap_bw
    score = 1.0 if union_bw == 0.0 else overlap_bw / union_bw
    return {
        "score": round(score, 4),
        "gt_available_bandwidth": round(gt_bw, 4),
        "reported_available_bandwidth": round(reported_bw, 4),
        "overlap_bandwidth": round(overlap_bw, 4),
        "unsafe_bandwidth_claimed": round(max(0.0, reported_bw - overlap_bw), 4),
        "missed_available_bandwidth": round(max(0.0, gt_bw - overlap_bw), 4),
    }


def load_transmitters(path: str) -> tuple[list[dict], str]:
    """The agent's report — malformed input degrades to an empty report."""
    if not os.path.exists(path):
        return [], f"no report at {path} (scored as empty report)"
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            report = json.load(f)
        transmitters = report["transmitters"]
        cleaned = [
            {
                "center_freq": float(t["center_freq"]),
                "bandwidth": float(t["bandwidth"]),
            }
            for t in transmitters
        ]
        return cleaned, "ok"
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        return [], f"malformed report (scored as empty report): {error}"


def main() -> int:
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(TRUTH_PATH) as f:
        truth = json.load(f)
    transmitters, reason = load_transmitters(ANSWER_PATH)
    result = {**score_report(transmitters, truth), "report_status": reason}
    with open(REWARD_PATH, "w") as f:
        f.write(str(result["score"]))
    with open(DETAILS_PATH, "w") as f:
        json.dump(result, f, indent=1)
    print(f"long-run availability IoU: {result['score']} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
