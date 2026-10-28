"""Harness candidate evaluation: run evaluation tasks as episodes, keep every score, and decide by the scores.

``scoring``
    ``EpisodeScorer``, the scorer a deployment names in
    ``evolution.evaluation.scorer``, and what Reef keeps of one scored
    episode.
``runner``
    The worker pool that runs and scores episodes.
``history``
    Every tree, episode and decision of a scenario; the score cache.
``evaluator``
    ``HarnessEpisodeEvaluator``: scores any tree on any tasks through the
    cache, and scores a ``HarnessCandidate`` against the head tree.
``policy``
    The publication policies ``compare``, ``floor`` and ``none``.
``tasks``
    The tasks each step evaluates, and the collection of traffic prompts.
``settings``
    The ``evolution.evaluation`` section.
"""

from reef.evaluation.harness.evaluator import (
    EVALUATION_SIDES,
    EvaluationContext,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessEpisodeEvaluator,
)
from reef.evaluation.harness.history import CandidateRecord, DecisionRecord, EpisodeRecord, History
from reef.evaluation.harness.policy import ComparePolicy, EpisodePolicy, FloorPolicy, NoEvaluationPolicy
from reef.evaluation.harness.runner import EpisodeEvaluationWorker, EvaluationWorkerPool, evaluation_selection
from reef.evaluation.harness.scoring import (
    EpisodeScorer,
    FailureObservation,
    ScoredEpisode,
    VerifierRewardScorer,
    verifier_reward,
)
from reef.evaluation.harness.settings import EvaluationSettings
from reef.evaluation.harness.tasks import EvaluationTasks, StepTasks, TaskCollector

__all__ = [
    "EVALUATION_SIDES",
    "CandidateRecord",
    "ComparePolicy",
    "DecisionRecord",
    "EpisodeEvaluationWorker",
    "EpisodePolicy",
    "EpisodeRecord",
    "EpisodeScorer",
    "EvaluationContext",
    "EvaluationSettings",
    "EvaluationTasks",
    "EvaluationWorkerPool",
    "FailureObservation",
    "FloorPolicy",
    "HarnessCandidate",
    "HarnessCandidateEvaluator",
    "HarnessEpisodeEvaluator",
    "History",
    "NoEvaluationPolicy",
    "ScoredEpisode",
    "StepTasks",
    "TaskCollector",
    "VerifierRewardScorer",
    "evaluation_selection",
    "verifier_reward",
]
