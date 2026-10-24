"""Meta-Harness selection: serve a strict mean-score improvement over the committed incumbent.

Upstream Meta-Harness keeps a frontier: each candidate's average is compared
against the best average recorded so far, and the incumbent is never run
again. Reproducing that means judging against the incumbent's committed score
rather than this step's paired one, even though the step measures both. The
bar is therefore a high-water mark, which is upstream's behavior and not an
accident of this port. The selection only reads the population: the next
step's proposer records the scores.
"""

from __future__ import annotations

from collections.abc import Mapping
from statistics import fmean
from typing import Any, Self

from reef.core.evaluation import CandidateEvaluator, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.evaluation.evaluators import BackendEvaluateMixin, SelectionPlugin
from reef.evaluation.harness import HarnessCandidate

from .population import POPULATION_KEY, Population, finite_scores


class MetaHarnessSelection(BackendEvaluateMixin, SelectionPlugin):
    """Evaluate through the backend; select a candidate whose mean beats the incumbent's committed mean."""

    def __init__(self, candidate_backend: Any) -> None:
        super().__init__()
        self._candidate_backend = candidate_backend

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, candidate_backend: CandidateEvaluator) -> Self:
        if config:
            raise ValueError(f"the Meta-Harness selection takes no config, got {', '.join(sorted(map(str, config)))}")
        return cls(candidate_backend)

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        if not isinstance(candidate, HarnessCandidate):
            raise TypeError(f"the Meta-Harness selection decides a HarnessCandidate, got {type(candidate).__name__}")
        population = Population.from_dict(candidate.method_state.get(POPULATION_KEY))
        candidate_scores = finite_scores(evaluation.metrics.get("candidate_scores", ()))
        current_scores = finite_scores(evaluation.metrics.get("current_scores", ()))
        if len(candidate_scores) != len(current_scores):
            raise ValueError("Meta-Harness evaluation score vectors must have equal lengths")
        served = None if population.served_id is None else population.served
        incumbent_scores = (None if served is None else served.scores) or current_scores
        candidate_mean = fmean(candidate_scores)
        incumbent_mean = fmean(incumbent_scores)
        metrics: dict[str, Any] = {
            "candidate_mean": candidate_mean,
            "incumbent_mean": incumbent_mean,
            "population_size": len(population.candidates),
            "proposer_calls": population.proposer_calls,
        }
        # A client's proposal and a recheck are no candidate of the population: only the method's own is pending.
        if not candidate.recheck and candidate.proposal_id is None and population.pending_id is not None:
            proposed = population.pending
            metrics["population_candidate_id"] = proposed.candidate_id
            metrics["population_parent_id"] = proposed.parent_id
            # The next step records this evaluation; the count includes it already.
            metrics["target_episode_calls"] = population.episode_calls + len(candidate_scores) + len(current_scores)
        else:
            metrics["target_episode_calls"] = population.episode_calls
        return SelectionDecision(
            selected=candidate_mean > incumbent_mean,
            policy="meta_harness",
            reason=(
                f"candidate scored {candidate_mean:.4f} against committed incumbent {incumbent_mean:.4f} "
                f"over {len(candidate_scores)} episode results"
            ),
            evaluation=evaluation,
            metrics=metrics,
        )


__all__ = ["MetaHarnessSelection"]
