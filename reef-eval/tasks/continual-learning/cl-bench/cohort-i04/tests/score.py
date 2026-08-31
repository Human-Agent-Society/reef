"""Cohort-studies scorer, run as each task's verifier.

Ported from CL-Bench's ``cohort_studies/scorer.py`` (Apache-2.0): for
each of the 36 cohorts, the mean binary KL divergence (log base 2,
Dirichlet smoothing alpha=0.01) between the true population survival and
the prediction at 12/24/36 months. The score is the information gain in
bits over the study-wide reference:

    score = mean_c(KL_reference_c) - mean_c(KL_prediction_c)

Missing or unusable estimates are backfilled with the reference, exactly
as upstream. Negative scores are legitimate. Deterministic and offline.
"""

import json
import math
import os

ANSWER_PATH = "/app/report.json"
TRUTH_PATH = "/tests/truth.json"
REWARD_PATH = "/logs/verifier/reward.txt"
DETAILS_PATH = "/logs/verifier/scoring.json"

SMOOTHING_ALPHA = 0.01


def _smooth_dist(survival: float) -> tuple[float, float]:
    s = min(1.0, max(0.0, float(survival)))
    p = (s, 1.0 - s)
    return tuple((1.0 - SMOOTHING_ALPHA) * pk + SMOOTHING_ALPHA / 2 for pk in p)


def _binary_kl(p_true: tuple[float, float], p_pred: tuple[float, float]) -> float:
    kl = 0.0
    for pt, pp in zip(p_true, p_pred, strict=False):
        if pt > 0:
            kl += pt * math.log2(pt / pp)
    return max(0.0, kl)


def _cohort_kl(true_survs, pred_survs) -> float:
    kls = [
        _binary_kl(_smooth_dist(t), _smooth_dist(p))
        for t, p in zip(true_survs, pred_survs, strict=False)
    ]
    return sum(kls) / len(kls)


def parse_estimates(raw_text: str) -> dict[str, tuple[float, float, float]]:
    """cohort_id -> (s12, s24, s36); unusable entries are simply dropped
    (they backfill to the reference, as upstream)."""
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        return {}
    if isinstance(raw, dict):
        raw = raw.get("estimates", [])
    if not isinstance(raw, list):
        return {}
    parsed: dict[str, tuple[float, float, float]] = {}
    for entry in raw:
        try:
            parsed[str(entry["cohort_id"])] = (
                float(entry["estimated_survival_12m"]),
                float(entry["estimated_survival_24m"]),
                float(entry["estimated_survival_36m"]),
            )
        except (TypeError, KeyError, ValueError):
            continue
    return parsed


def score(raw_text: str, truth: dict) -> dict:
    reference = tuple(truth["reference_survival"])  # study-wide KM at 12/24/36
    estimates = parse_estimates(raw_text)
    kls, ref_kls, per_layer = [], [], {}
    n_estimated = 0
    for cohort in truth["cohorts"]:
        true_survs = (
            cohort["survival_12m"],
            cohort["survival_24m"],
            cohort["survival_36m"],
        )
        pred = estimates.get(cohort["cohort_id"])
        if pred is not None:
            n_estimated += 1
        # Upstream also falls back per-field on falsy values via `or`.
        pred = tuple((pred[i] if pred and pred[i] else reference[i]) for i in range(3))
        kl = _cohort_kl(true_survs, pred)
        ref_kl = _cohort_kl(true_survs, reference)
        kls.append(kl)
        ref_kls.append(ref_kl)
        per_layer.setdefault(cohort["layer"], []).append(ref_kl - kl)

    mean_kl = sum(kls) / len(kls)
    mean_ref_kl = sum(ref_kls) / len(ref_kls)
    return {
        "score": round(mean_ref_kl - mean_kl, 6),
        "mean_kl_divergence": round(mean_kl, 6),
        "mean_reference_kl": round(mean_ref_kl, 6),  # the per-instance ceiling
        "n_cohorts_total": len(truth["cohorts"]),
        "n_cohorts_estimated": n_estimated,
        "layer_scores": {
            str(layer): round(sum(v) / len(v), 6) for layer, v in per_layer.items()
        },
    }


def main() -> int:
    os.makedirs(os.path.dirname(REWARD_PATH), exist_ok=True)
    with open(TRUTH_PATH) as f:
        truth = json.load(f)
    raw_text = ""
    if os.path.exists(ANSWER_PATH):
        with open(ANSWER_PATH, encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    result = score(raw_text, truth)
    with open(REWARD_PATH, "w") as f:
        f.write(str(result["score"]))
    with open(DETAILS_PATH, "w") as f:
        json.dump(result, f, indent=1)
    print(
        f"information gain: {result['score']} bits/cohort "
        f"(ceiling {result['mean_reference_kl']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
