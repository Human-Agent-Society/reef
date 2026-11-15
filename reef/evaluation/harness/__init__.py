"""Harness candidate evaluation: run evaluation tasks as runs, keep every score, and decide by the scores.

``scoring``
    ``TaskScorer``, the scorer a deployment names in
    ``evolution.evaluation.scorer``, and what Reef keeps of one scored
    run.
``runner``
    The worker pool that runs and scores runs.
``history``
    Every tree, run and decision of a scenario; the score cache.
``evaluator``
    ``HarnessTaskEvaluator``: scores any tree on any tasks through the
    cache, and scores a ``HarnessCandidate`` against the head tree.
``policy``
    The publication policies ``compare``, ``floor`` and ``none``.
``tasks``
    The tasks each job evaluates, and the collection of traffic prompts.
``settings``
    The ``evolution.evaluation`` section.
"""

from reef.evaluation.harness.evaluator import (
    EVALUATION_SIDES,
    EvaluationContext,
    HarnessCandidate,
    HarnessCandidateEvaluator,
    HarnessTaskEvaluator,
)
from reef.evaluation.harness.history import CandidateRecord, DecisionRecord, History, RunRecord
from reef.evaluation.harness.policy import ComparePolicy, FloorPolicy, NoEvaluationPolicy, TaskScorePolicy
from reef.evaluation.harness.runner import EvaluationWorkerPool, TaskEvaluationWorker, evaluation_selection
from reef.evaluation.harness.scoring import (
    FailureObservation,
    ScoredRun,
    TaskScorer,
    VerifierRewardScorer,
    verifier_reward,
)
from reef.evaluation.harness.settings import EvaluationSettings
from reef.evaluation.harness.tasks import EvaluationTasks, JobTasks, TaskCollector

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
    "HarnessTaskEvaluator",
    "History",
    "JobTasks",
    "NoEvaluationPolicy",
    "RunRecord",
    "ScoredRun",
    "TaskCollector",
    "TaskEvaluationWorker",
    "TaskScorePolicy",
    "TaskScorer",
    "VerifierRewardScorer",
    "evaluation_selection",
    "verifier_reward",
]
