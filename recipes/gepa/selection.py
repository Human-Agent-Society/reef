"""GEPA's selection: strict mean improvement over the served candidate's stored validation score.

GEPA scores every candidate on the validation set once and keeps that
number; the step's fresh run of the served tree is redundant work, not a
second opinion. Comparing against the stored mean is what keeps the served
tree equal to the archive's best candidate, GEPA's ``best_idx``. The fronts
still record everything, so a candidate that loses on the mean can go on
parenting proposals through its specialties. The selection only reads the
archive: the next step's proposer folds the scores in.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean
from typing import Any, Self

from reef.core.evaluation import CandidateEvaluator, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.evaluation.evaluators import BackendEvaluateMixin, SelectionPlugin
from reef.evaluation.harness import HarnessCandidate

from .archive import Archive
from .proposer import ARCHIVE_KEY, scores


class GEPASelection(BackendEvaluateMixin, SelectionPlugin):
    """Evaluate through the backend; select a candidate whose validation mean beats the served one's."""

    def __init__(self, candidate_backend: Any) -> None:
        super().__init__()
        self._candidate_backend = candidate_backend

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        if config:
            raise ValueError(f"the GEPA selection takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls(candidate_backend)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError(f"the GEPA selection decides a HarnessCandidate, got {type(candidate).__name__}")
        archive = Archive.from_state(candidate.method_state.get(ARCHIVE_KEY))
        candidate_scores = scores(evaluation.metrics.get("candidate_scores", ()))
        served_scores = scores(evaluation.metrics.get("current_scores", ()))
        served = archive.served
        candidate_mean = fmean(candidate_scores) if candidate_scores else 0.0
        if served is None:
            served_mean = 0.0
        elif archive.candidates[served].val_scores is None:
            # The seed's own validation arrives with the first evaluated step.
            served_mean = fmean(served_scores) if served_scores else 0.0
        else:
            served_mean = archive.mean_val(served)
        # A client's proposal and a recheck are no candidate of the archive: only the method's own is pending.
        own = not candidate.recheck and candidate.proposal_id is None
        pending = archive.pending if own else None
        selected = candidate_mean > served_mean
        # The next step folds this validation into the committed archive; the numbers report it already.
        settled = Archive.from_state(candidate.method_state.get(ARCHIVE_KEY))
        if own:
            settled.settle(candidate_scores, served_scores, selected)
        fronts = settled.fronts()
        return SelectionDecision(
            selected=selected,
            policy="gepa",
            reason=(
                f"candidate scored {candidate_mean:.4f} against {served_mean:.4f} served "
                f"over {len(candidate_scores)} validation tasks"
            ),
            evaluation=evaluation,
            metrics={
                "candidate_val_mean": candidate_mean,
                "served_val_mean": served_mean,
                "parent": None if pending is None else archive.candidates[pending].parent,
                "archive_size": len(settled.candidates),
                "front_size": len({index for front in fronts.values() for index in front}),
                "metric_calls": settled.metric_calls,
            },
        )


__all__ = ["GEPASelection"]
