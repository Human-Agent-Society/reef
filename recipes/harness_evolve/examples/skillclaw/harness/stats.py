"""The preregistered gain criterion, carried verbatim from the sealed campaign.

The three functions below are copied unchanged from
``benchmarks/skill_claw/__main__.py`` at commit 0519eefb (branch
bo/skillclaw-paper, "docs: preregister the gain criterion before the
rerun", PR #175). The stats are mechanism agnostic: they read only the
sealed ``round-<n>/summary.json`` category tables, so rebuilding the
method on the harness evolution mechanism changes nothing here.

The criterion, fixed before any data is read: a category's gain counts as
real only when the method run's final day beats the control mean by more
than two control standard deviations (one sided; a million draw simulation
of the exact procedure calibrates the rule to a 5.6 percent false positive
rate per category, so about 0.33 categories of six by chance). Best day
excesses are reported as description only: the best of six days over the
same threshold fires falsely about 24 percent of the time, so it never
carries a claim.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _round_scores(run_dir: Path) -> list[dict[str, float]]:
    return [json.loads(summary.read_text())["categories"] for summary in sorted(run_dir.glob("round-*/summary.json"))]


def _mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def boost_report(frozen_dir: Path, evolve_dir: Path) -> dict[str, Any]:
    frozen = _round_scores(frozen_dir)
    evolve = _round_scores(evolve_dir)
    if not frozen or len(evolve) < 2:
        raise RuntimeError(
            f"need at least one sealed round in {frozen_dir} and two in {evolve_dir}; "
            f"found {len(frozen)} and {len(evolve)}. Run the control run first."
        )
    categories = sorted(frozen[0])
    table = {}
    for category in categories:
        # A round with every task unscored drops out of that category's mean.
        base_mean, base_sd = _mean_sd([row[category] for row in frozen if category in row])
        baseline = evolve[0].get(category)
        evolve_series = [row[category] for row in evolve[1:] if category in row]
        best = max(evolve_series)
        final = evolve_series[-1]
        band = base_sd if base_sd > 0 else 1e-9
        table[category] = {
            "frozen_mean": round(base_mean, 2),
            "frozen_sd": round(base_sd, 2),
            "evolve_baseline": round(baseline, 2) if baseline is not None else None,
            "evolve_best": round(best, 2),
            "evolve_final": round(final, 2),
            "boost_best": round(best - base_mean, 2),
            "boost_final": round(final - base_mean, 2),
            "boost_best_in_sd": round((best - base_mean) / band, 2),
            # The claim rides the final day only: best-of-six over the same
            # threshold fires falsely about 24 percent of the time per
            # category, the final day about 5.6 percent.
            "final_clears_2sd": final - base_mean > 2 * base_sd,
        }
    return {"frozen_rounds": len(frozen), "evolve_rounds": len(evolve), "categories": table}
