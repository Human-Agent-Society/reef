"""Harness candidate evaluation: run evaluation tasks as rollouts, keep every score, and decide by the scores.

``scoring``
    ``RolloutScorer``, the scorer a deployment names in
    ``evolution.evaluation.scorer``, and what Reef keeps of one scored
    rollout.
``runner``
    The worker pool that runs and scores rollouts.
``history``
    Every tree, rollout and decision of a scenario; the score cache.
``evaluator``
    ``HarnessRolloutEvaluator``: scores any tree on any tasks through the
    cache, and scores a ``HarnessCandidate`` against the head tree.
``policy``
    The publication policies ``compare``, ``floor`` and ``none``.
``tasks``
    The tasks each step evaluates, and the promotion of traffic prompts.
``settings``
    The ``evolution.evaluation`` section.
"""

from reef.evaluation.harness.evaluator import (
    EVALUATION_SIDES,
    EvaluationContext,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessRolloutEvaluator,
)
from reef.evaluation.harness.history import CandidateRecord, DecisionRecord, History, RolloutRecord
from reef.evaluation.harness.policy import ComparePolicy, FloorPolicy, NoEvaluationPolicy, RolloutPolicy
from reef.evaluation.harness.runner import EvaluationWorkerPool, RolloutEvaluationWorker, evaluation_selection
from reef.evaluation.harness.scoring import (
    FailureObservation,
    RolloutScorer,
    ScoredRollout,
    VerifierRewardScorer,
    verifier_reward,
)
from reef.evaluation.harness.settings import EvaluationSettings
from reef.evaluation.harness.tasks import EvaluationTasks, Promoter, StepTasks

__all__ = [
    "EVALUATION_SIDES",
    "CandidateRecord",
    "ComparePolicy",
    "DecisionRecord",
    "EvaluationContext",
    "EvaluationSettings",
    "EvaluationTasks",
    "EvaluationWorkerPool",
    "FailureObservation",
    "FloorPolicy",
    "HarnessCandidate",
    "HarnessCandidateEvaluator",
    "HarnessRolloutEvaluator",
    "History",
    "NoEvaluationPolicy",
    "Promoter",
    "RolloutEvaluationWorker",
    "RolloutPolicy",
    "RolloutRecord",
    "RolloutScorer",
    "ScoredRollout",
    "StepTasks",
    "VerifierRewardScorer",
    "evaluation_selection",
    "verifier_reward",
]
