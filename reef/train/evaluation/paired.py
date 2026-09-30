"""The paired confidence selection: typed episode outcomes and a paired test over tasks.

An evaluation runs the candidate and the current tree as pairs on the same
task and repeat. Each episode has a label: ``valid`` (it scored),
``execution_error`` (it could not run), ``invalid`` (it ran and its scorer
found no score) or ``not_run`` (it was skipped). An episode without a score
is the harness's fault or the infrastructure's. A pair with an
infrastructure fault is void and counts as a candidate loss, so a fault a
candidate forged cannot win its pair; a harness fault takes the lowest score
of the evaluation. The repeats of a task are averaged, and the candidate is
selected only when an exact sign test and a bootstrap interval over tasks
both clear ``min_effect`` at ``confidence_level``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision, UpdateCandidate

EpisodeLabel = Literal["valid", "execution_error", "invalid", "not_run"]
EpisodeFault = Literal["harness", "infrastructure"]
PairedSelectionResult = Literal["selected", "invalid_evaluation", "insufficient_confidence"]

#: Bootstrap resamples per decision, and their fixed seed, so a decision can be derived again from its record.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class PairedConfidenceSettings:
    """The thresholds of the paired confidence selection; ``min_effect`` is in score units per task."""

    min_valid_pairs: int = 1
    min_effect: float = 0.0
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_valid_pairs, bool)
            or not isinstance(self.min_valid_pairs, int)
            or self.min_valid_pairs < 1
        ):
            raise ValueError("min_valid_pairs must be an integer of at least 1")
        if (
            isinstance(self.min_effect, bool)
            or not isinstance(self.min_effect, (int, float))
            or not math.isfinite(self.min_effect)
            or self.min_effect < 0
        ):
            raise ValueError("min_effect must be a finite number of at least 0")
        # At 0.5 the bootstrap interval would run from the median to the draw below it.
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not 0.5 < self.confidence_level < 1
        ):
            raise ValueError("confidence_level must be a number above 0.5 and below 1")


@dataclass(frozen=True)
class PairedComparison:
    """The paired test of one evaluation; ``wins``, ``losses`` and ``ties`` count tasks against ``min_effect``."""

    valid_pairs: int
    void_pairs: int
    #: Candidate minus current per task, its repeats averaged.
    task_differences: tuple[float, ...]
    wins: int
    losses: int
    ties: int
    sign_test_p_value: float
    #: The bootstrap interval of the mean task difference; ``None`` with no task.
    interval: tuple[float, float] | None

    @property
    def mean_difference(self) -> float | None:
        if not self.task_differences:
            return None
        return math.fsum(self.task_differences) / len(self.task_differences)


def side_scores_and_faults(
    metrics: Mapping[str, object], side: str
) -> tuple[tuple[float | None, ...], tuple[str | None, ...]]:
    """One side's episode scores and faults from an evaluation; each episode has a finite score or a fault."""
    vectors = []
    for key in (f"{side}_scores", f"{side}_faults"):
        value = metrics.get(key)
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ValueError(f"the paired confidence selection needs {key!r} in the evaluation")
        vectors.append(tuple(value))
    score_values, fault_values = vectors
    if len(score_values) != len(fault_values):
        raise ValueError(f"{side}_scores and {side}_faults must be of one length")
    scores: list[float | None] = []
    faults: list[str | None] = []
    for score, fault in zip(score_values, fault_values, strict=True):
        if score is None and fault in ("harness", "infrastructure"):
            scores.append(None)
            faults.append(str(fault))
        elif (
            fault is None and not isinstance(score, bool) and isinstance(score, (int, float)) and math.isfinite(score)
        ):
            scores.append(float(score))
            faults.append(None)
        else:
            raise ValueError(
                f"a {side} episode needs a finite score or a fault of 'harness' or 'infrastructure', "
                f"not score {score!r} with fault {fault!r}"
            )
    return tuple(scores), tuple(faults)


def compare_pairs(metrics: Mapping[str, object], *, min_effect: float, confidence_level: float) -> PairedComparison:
    """The paired test of an evaluation's score and fault vectors, ordered by task, then repeat."""
    candidate_scores, candidate_faults = side_scores_and_faults(metrics, "candidate")
    current_scores, current_faults = side_scores_and_faults(metrics, "current")
    repeats = metrics.get("episode_repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"the paired confidence selection needs 'episode_repeats' of at least 1, not {repeats!r}")
    pair_count = len(candidate_scores)
    if len(current_scores) != pair_count or pair_count % repeats:
        raise ValueError(
            "the paired confidence selection needs both sides evaluated on the same pairs, "
            f"a multiple of episode_repeats {repeats}"
        )
    scored = [score for score in (*candidate_scores, *current_scores) if score is not None]
    lowest, highest = (min(scored), max(scored)) if scored else (0.0, 0.0)
    differences: list[float] = []
    valid_pairs = void_pairs = 0
    for candidate_score, current_score, candidate_fault, current_fault in zip(
        candidate_scores, current_scores, candidate_faults, current_faults, strict=True
    ):
        if "infrastructure" in (candidate_fault, current_fault):
            # The worst difference the evaluation allows: a void pair is a candidate loss.
            void_pairs += 1
            differences.append(lowest - highest)
            continue
        valid_pairs += 1
        candidate_value = lowest if candidate_score is None else candidate_score
        current_value = lowest if current_score is None else current_score
        differences.append(candidate_value - current_value)
    task_differences = tuple(
        math.fsum(differences[start : start + repeats]) / repeats for start in range(0, pair_count, repeats)
    )
    wins = sum(1 for difference in task_differences if difference - min_effect > 0)
    losses = sum(1 for difference in task_differences if difference - min_effect < 0)
    untied = wins + losses
    # Exact one sided sign test: the chance of at least this many wins among the untied tasks under no effect.
    p_value = 1.0 if untied == 0 else sum(math.comb(untied, k) for k in range(wins, untied + 1)) / 2**untied
    interval = None
    if task_differences:
        generator = random.Random(BOOTSTRAP_SEED)
        task_count = len(task_differences)
        means = sorted(
            math.fsum(generator.choices(task_differences, k=task_count)) / task_count for _ in range(BOOTSTRAP_DRAWS)
        )
        lower_index = math.floor((1 - confidence_level) * BOOTSTRAP_DRAWS + 1e-9)
        interval = (means[lower_index], means[BOOTSTRAP_DRAWS - 1 - lower_index])
    return PairedComparison(
        valid_pairs=valid_pairs,
        void_pairs=void_pairs,
        task_differences=task_differences,
        wins=wins,
        losses=losses,
        ties=len(task_differences) - untied,
        sign_test_p_value=p_value,
        interval=interval,
    )


class PairedConfidenceMixin(CandidateEvaluationPlugin):
    """Give a plugin a ``decide()`` that selects when the paired test over tasks clears ``min_effect``.

    The evaluation must carry both sides' score and fault vectors; ``evaluate`` stays abstract.
    """

    def __init__(self, settings: PairedConfidenceSettings) -> None:
        if not isinstance(settings, PairedConfidenceSettings):
            raise TypeError("the paired confidence selection needs PairedConfidenceSettings")
        super().__init__()
        self.settings = settings

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        settings = self.settings
        comparison = compare_pairs(
            evaluation.metrics, min_effect=settings.min_effect, confidence_level=settings.confidence_level
        )
        valid, void, p_value = comparison.valid_pairs, comparison.void_pairs, comparison.sign_test_p_value
        mean = comparison.mean_difference
        lower, upper = (None, None) if comparison.interval is None else comparison.interval
        result: PairedSelectionResult
        if valid < settings.min_valid_pairs:
            result = "invalid_evaluation"
            reason = (
                f"invalid evaluation: {valid} valid pairs, below min_valid_pairs {settings.min_valid_pairs} "
                f"({void} void)"
            )
        elif lower is not None and p_value <= 1 - settings.confidence_level and lower > settings.min_effect:
            result = "selected"
            reason = (
                f"candidate gained {mean:.4g} per task (interval {lower:.4g} to {upper:.4g}) over {valid} valid "
                f"pairs ({void} void), sign test p {p_value:.4g}"
            )
        else:
            result = "insufficient_confidence"
            reason = (
                f"insufficient confidence: sign test p {p_value:.4g}, interval lower bound {lower:.4g}, "
                f"min_effect {settings.min_effect:g}"
            )
        return SelectionDecision(
            outcome="select" if result == "selected" else "reject",
            policy="paired_confidence",
            policy_version="1",
            reason=reason,
            evaluation=evaluation,
            metrics={
                "selection_result": result,
                "valid_pairs": valid,
                "void_pairs": void,
                "tasks": len(comparison.task_differences),
                "wins": comparison.wins,
                "losses": comparison.losses,
                "ties": comparison.ties,
                "sign_test_p_value": p_value,
                "mean_difference": mean,
                "interval_lower": lower,
                "interval_upper": upper,
                "confidence_level": settings.confidence_level,
                "min_effect": settings.min_effect,
                "min_valid_pairs": settings.min_valid_pairs,
            },
        )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "EpisodeFault",
    "EpisodeLabel",
    "PairedComparison",
    "PairedConfidenceMixin",
    "PairedConfidenceSettings",
    "PairedSelectionResult",
    "compare_pairs",
]
