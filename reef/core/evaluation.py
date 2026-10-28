"""Artifact-agnostic contracts for evaluating and selecting candidate updates.

An evaluation runs rollouts and gives their scores (``RolloutScores``). A publication policy reads the scores
and decides whether a release is published; training on tasks assembles the same rollouts into samples.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The JSON value of a failed rollout's score (``RolloutScore.to_json``).
FAILED = "failed"


@dataclass(frozen=True)
class RolloutScore:
    """The score of one rollout: one run of a task under one policy in one environment.

    Values: ``value`` is a finite number when the rollout ran: the scorer's value, or 0 when the policy gave no
    answer. ``value`` is None when the rollout failed: the environment could not run it, so the policy did not
    cause the result. Purpose: a rollout runner makes it; publication policies and sample assembly read it
    through ``RolloutScores``.
    """

    value: float | None

    def __post_init__(self) -> None:
        value = self.value
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("a rollout score is a finite number, or None for a failed rollout")
        object.__setattr__(self, "value", float(value))

    @classmethod
    def scored(cls, value: float) -> RolloutScore:
        """The score of a rollout that ran; ``ValueError`` unless ``value`` is a finite number."""
        if value is None:
            raise ValueError("a scored rollout has a finite value")
        return cls(value)

    @classmethod
    def failed(cls) -> RolloutScore:
        return cls(None)

    @property
    def is_failed(self) -> bool:
        return self.value is None

    def to_json(self) -> float | str:
        """The value, or ``FAILED`` for a failed rollout."""
        return FAILED if self.value is None else self.value

    @classmethod
    def from_json(cls, value: object) -> RolloutScore:
        """Read ``to_json``'s value; ``ValueError`` for another value."""
        if value == FAILED:
            return cls.failed()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"a rollout score is a number or {FAILED!r}, not {value!r}")
        return cls.scored(value)


@dataclass(frozen=True)
class RolloutScores:
    """The scores of an evaluation's rollouts, by task.

    Values: ``tasks`` is a non-empty tuple of ``(task, scores)`` pairs, in run order. ``task`` is a non-empty
    string; a task may appear more than once. ``scores`` is a non-empty tuple of ``RolloutScore`` values, one
    for each run of the task, and every task has the same number of them (``count``). Purpose: an evaluation
    gives it; a publication policy reads it, such as ``compare_scores`` and ``check_floor``.
    """

    tasks: tuple[tuple[str, tuple[RolloutScore, ...]], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, tuple) or not self.tasks:
            raise ValueError("RolloutScores holds one or more tasks")
        counts = set()
        for entry in self.tasks:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("RolloutScores.tasks holds (task, scores) pairs")
            task, scores = entry
            if not isinstance(task, str) or not task:
                raise ValueError("a RolloutScores task is a non-empty string")
            if not isinstance(scores, tuple) or not scores or not all(isinstance(s, RolloutScore) for s in scores):
                raise TypeError("a RolloutScores task has a non-empty tuple of RolloutScore values")
            counts.add(len(scores))
        if len(counts) != 1:
            raise ValueError("every task of a RolloutScores has the same number of rollouts")

    @classmethod
    def by_task(cls, tasks: Sequence[str], scores: Sequence[RolloutScore]) -> RolloutScores:
        """The scores of ``scores``, given task by task: the first ``len(scores) // len(tasks)`` belong to the first
        task, and so on. ``ValueError`` when ``scores`` does not divide evenly."""
        if not tasks or len(scores) % len(tasks):
            raise ValueError("the rollout scores divide evenly among one or more tasks")
        count = len(scores) // len(tasks)
        return cls(
            tuple((task, tuple(scores[index * count : (index + 1) * count])) for index, task in enumerate(tasks))
        )

    @property
    def count(self) -> int:
        """The rollouts of each task."""
        return len(self.tasks[0][1])

    def rollouts(self) -> tuple[RolloutScore, ...]:
        """Every rollout's score, task by task, in run order."""
        return tuple(score for _, scores in self.tasks for score in scores)

    def failed_count(self) -> int:
        """The rollouts that failed."""
        return sum(score.is_failed for score in self.rollouts())

    def to_json(self) -> list[dict[str, Any]]:
        """The value as JSON: ``[{"task": ..., "scores": [...]}, ...]``; ``from_json`` reads it back."""
        return [{"task": task, "scores": [score.to_json() for score in scores]} for task, scores in self.tasks]

    @classmethod
    def from_json(cls, value: object) -> RolloutScores:
        """Read ``to_json``'s value; ``ValueError`` for another value."""
        if not isinstance(value, list):
            raise ValueError("rollout scores are a list of tasks")
        tasks = []
        for entry in value:
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"task", "scores"}
                or not isinstance(entry["scores"], list)
            ):
                raise ValueError("each task of rollout scores is an object of task and scores")
            tasks.append((entry["task"], tuple(RolloutScore.from_json(score) for score in entry["scores"])))
        return cls(tuple(tasks))


def mean_score(scores: Iterable[RolloutScore]) -> float | None:
    """The mean of the rollouts that did not fail; None when every rollout failed."""
    values = [score.value for score in scores if score.value is not None]
    return math.fsum(values) / len(values) if values else None


@dataclass(frozen=True)
class UpdateCandidate:
    """One update that has been produced but not selected for serving.

    ``candidate_id`` is the stable idempotency and audit key for the attempt.
    """

    candidate_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be a non-empty string")


@dataclass(frozen=True, kw_only=True)
class CheckpointCandidate(UpdateCandidate):
    """A checkpoint that a weight trainer's job saved and that Reef has not decided to serve.

    Values: ``training_job_id`` (non-empty) and ``step`` (a non-negative int) name the checkpoint within its
    job, and ``candidate_id`` is ``"<training_job_id>:<step>"``. ``release_path`` is the non-empty directory
    that Reef publishes when it serves the checkpoint (``SavedCheckpoint.release_path``). ``training_metrics``
    is the training component's telemetry for it. Purpose: ``JobLoop.decide`` makes the value, and the
    recipe's candidate evaluation reads it.
    """

    training_job_id: str
    step: int
    release_path: str
    training_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.training_job_id, str) or not self.training_job_id:
            raise ValueError("CheckpointCandidate.training_job_id must be a non-empty string")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("CheckpointCandidate.step must be a non-negative int")
        if not isinstance(self.release_path, str) or not self.release_path:
            raise ValueError("CheckpointCandidate.release_path must be a non-empty string")
        if self.candidate_id != f"{self.training_job_id}:{self.step}":
            raise ValueError("CheckpointCandidate.candidate_id must be '<training_job_id>:<step>'")
        if not isinstance(self.training_metrics, Mapping):
            raise TypeError("CheckpointCandidate.training_metrics must be a mapping")


@dataclass(frozen=True)
class EvaluationResult:
    """Measurements produced by evaluating one candidate against the incumbent."""

    evaluator: str
    evaluator_version: str
    metrics: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, str) or not self.evaluator:
            raise ValueError("evaluator must be a non-empty string")
        if not isinstance(self.evaluator_version, str) or not self.evaluator_version:
            raise ValueError("evaluator_version must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "evaluator_version": self.evaluator_version,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SelectionDecision:
    """The durable, explainable decision for one evaluated candidate."""

    selected: bool
    policy: str
    reason: str
    evaluation: EvaluationResult
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.selected, bool):
            raise ValueError("selected must be a boolean")
        if not isinstance(self.policy, str) or not self.policy:
            raise ValueError("policy must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "policy": self.policy,
            "reason": self.reason,
            "evaluation": self.evaluation.to_dict(),
            "metrics": dict(self.metrics),
        }


class CandidateSelector(ABC):
    """Turn an evaluation result into a select-or-reject decision.

    Implementations inherit this contract explicitly. Evaluation and selection
    remain independent capabilities; a complete plugin implements both.
    """

    @abstractmethod
    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision: ...


class CandidateEvaluator(ABC):
    """Measure a candidate without deciding whether to publish it."""

    @abstractmethod
    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult: ...


class CandidateEvaluationPlugin(CandidateEvaluator, CandidateSelector):
    """Measure a candidate and decide whether Reef serves it."""

    def evaluate_and_decide(self, candidate: UpdateCandidate) -> SelectionDecision:
        """Evaluate ``candidate`` and decide on that evaluation.

        Raise ``ValueError`` when the decision does not carry the evaluation
        that ``evaluate`` returned, since Reef records the two together.
        """
        evaluation = self.evaluate(candidate)
        decision = self.decide(candidate, evaluation)
        if decision.evaluation is not evaluation:
            raise ValueError("candidate evaluator must retain the evaluation result supplied by Reef")
        return decision


__all__ = [
    "FAILED",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "CheckpointCandidate",
    "EvaluationResult",
    "RolloutScore",
    "RolloutScores",
    "SelectionDecision",
    "UpdateCandidate",
    "mean_score",
]
