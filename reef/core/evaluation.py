"""Artifact-agnostic contracts for evaluating and selecting candidate updates.

An evaluation runs rollouts and gives their scores (``RolloutScores``). A publication policy reads the scores
and decides whether a release is published; training on tasks assembles the same rollouts into samples.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

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


#: The kind of an artifact. A ``harness`` is a whole agent harness that Reef runs as it is, such as a Harbor agent
#: or an adapter's binary. A ``harness_plugin`` installs into a base harness, such as an evolved tree, a skill
#: tree, or request defaults. A ``model`` is a whole model, such as a provider's model or full weights. A ``model_adapter`` is a LoRA
#: adapter on a base model.
ArtifactKind = Literal["harness", "harness_plugin", "model", "model_adapter"]
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = ("harness", "harness_plugin", "model", "model_adapter")


@dataclass(frozen=True)
class PolicyHarness:
    """The harness side of a policy: a base harness, with the plugin that installs into it.

    Values: ``base`` is the non-empty name of the base harness: an adapter name, such as ``pi`` or ``native``, or a
    Harbor agent name, such as ``terminus-2``. ``plugin`` maps each relative path of the plugin's files to the
    file's text. An empty ``plugin`` runs the base harness as it is.
    Purpose: a ``RolloutRequest`` names the harness of its rollout, and a ``RolloutRunner`` runs it.
    ``kind`` is ``harness_plugin`` when ``plugin`` has files, else ``harness``.
    """

    base: str
    plugin: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.base, str) or not self.base:
            raise ValueError("a policy harness names its base harness")
        if not isinstance(self.plugin, Mapping) or any(
            not isinstance(path, str) or not path or not isinstance(text, str) for path, text in self.plugin.items()
        ):
            raise ValueError("a policy harness plugin maps non-empty file paths to text")
        object.__setattr__(self, "plugin", MappingProxyType(dict(self.plugin)))

    @property
    def kind(self) -> ArtifactKind:
        return "harness_plugin" if self.plugin else "harness"

    def to_json(self) -> dict[str, Any]:
        return {"base": self.base, "plugin": dict(self.plugin)}

    @classmethod
    def from_json(cls, value: object) -> PolicyHarness:
        """The harness that ``to_json`` gave; raise ``ValueError`` for any other value."""
        if not isinstance(value, Mapping) or set(value) != {"base", "plugin"}:
            raise ValueError("a policy harness is an object of base and plugin")
        return cls(value["base"], value["plugin"])


@dataclass(frozen=True)
class PolicyModel:
    """The model side of a policy: a model, with at most one LoRA adapter on it.

    Values: ``name`` is the non-empty name that an endpoint serves the model under: a provider's model name, or the
    name that a deployment serves its weights under. ``adapter`` is None, or the non-empty location of a LoRA
    adapter on that model, such as a Tinker sampler path.
    Purpose: a ``RolloutRequest`` names the model of its rollout, and a ``RolloutRunner`` serves it.
    ``kind`` is ``model_adapter`` when ``adapter`` is set, else ``model``.
    """

    name: str
    adapter: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a policy model names the model")
        if self.adapter is not None and (not isinstance(self.adapter, str) or not self.adapter):
            raise ValueError("a policy model adapter is None or a non-empty location")

    @property
    def kind(self) -> ArtifactKind:
        return "model" if self.adapter is None else "model_adapter"

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "adapter": self.adapter}

    @classmethod
    def from_json(cls, value: object) -> PolicyModel:
        """The model that ``to_json`` gave; raise ``ValueError`` for any other value."""
        if not isinstance(value, Mapping) or set(value) != {"name", "adapter"}:
            raise ValueError("a policy model is an object of name and adapter")
        return cls(value["name"], value["adapter"])


@dataclass(frozen=True)
class PolicyEvaluation:
    """One evaluation of a policy: the policy, and the score of each of its rollouts.

    Values: ``harness`` and ``model`` are the policy that every rollout ran. ``scores`` holds the score of each
    rollout, by task, in run order. ``mean_score`` is ``mean_score(scores.rollouts())``.
    Purpose: an evaluation of an artifact gives it; ``to_json`` and ``from_json`` keep it with the evaluation.
    """

    harness: PolicyHarness
    model: PolicyModel
    scores: RolloutScores

    def __post_init__(self) -> None:
        if not isinstance(self.harness, PolicyHarness) or not isinstance(self.model, PolicyModel):
            raise TypeError("a policy evaluation holds a PolicyHarness and a PolicyModel")
        if not isinstance(self.scores, RolloutScores):
            raise TypeError("a policy evaluation holds RolloutScores")

    @property
    def mean_score(self) -> float | None:
        return mean_score(self.scores.rollouts())

    def to_json(self) -> dict[str, Any]:
        return {"harness": self.harness.to_json(), "model": self.model.to_json(), "scores": self.scores.to_json()}

    @classmethod
    def from_json(cls, value: object) -> PolicyEvaluation:
        """The evaluation that ``to_json`` gave; raise ``ValueError`` for any other value."""
        if not isinstance(value, Mapping) or set(value) != {"harness", "model", "scores"}:
            raise ValueError("a policy evaluation is an object of harness, model and scores")
        return cls(
            PolicyHarness.from_json(value["harness"]),
            PolicyModel.from_json(value["model"]),
            RolloutScores.from_json(value["scores"]),
        )


@dataclass(frozen=True)
class RolloutRequest:
    """One rollout to run: a task, under a harness and a model.

    Values: ``harness`` and ``model`` are the policy of the rollout. ``task`` is a non-empty task: a task prompt,
    or the path of a Harbor task directory. ``record_dir`` is None, or a directory that does not exist yet, where
    the runner keeps the rollout's trajectory.
    Purpose: a caller gives requests to ``RolloutRunner.run_rollouts``, which runs one rollout for each.
    """

    harness: PolicyHarness
    model: PolicyModel
    task: str
    record_dir: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.harness, PolicyHarness) or not isinstance(self.model, PolicyModel):
            raise TypeError("a rollout request holds a PolicyHarness and a PolicyModel")
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("a rollout request names a non-empty task")
        if self.record_dir is not None and not isinstance(self.record_dir, Path):
            raise TypeError("a rollout request's record_dir is None or a Path")


class RolloutRunner[RolloutT](ABC):
    """Runs rollouts in one environment: one rollout for each ``RolloutRequest``.

    Values: an implementation holds what all its rollouts share, such as the environment and the model endpoint.
    ``RolloutT`` is the result of one rollout, with its trajectory. The Harbor agent runner
    (``TaskRolloutRunner``) and the harness tree runner (``HarnessRolloutEvaluator``) implement it.
    Purpose: a harness evaluation, a checkpoint evaluation and training on tasks run their rollouts through it.
    To run a task ``count`` times, a caller repeats its request ``count`` times.
    """

    @abstractmethod
    def run_rollouts(self, requests: Sequence[RolloutRequest]) -> list[RolloutT]:
        """Run one rollout of each request, and give one rollout per request, in request order.

        Raise ``ValueError`` before any rollout runs when a request names a harness, a model or a record
        directory that this runner cannot use.
        """

    @abstractmethod
    def rollout_score(self, rollout: RolloutT) -> RolloutScore:
        """The score of one rollout that ``run_rollouts`` gave."""


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
    "ARTIFACT_KINDS",
    "FAILED",
    "ArtifactKind",
    "CandidateEvaluationPlugin",
    "CandidateEvaluator",
    "CandidateSelector",
    "CheckpointCandidate",
    "EvaluationResult",
    "PolicyEvaluation",
    "PolicyHarness",
    "PolicyModel",
    "RolloutRequest",
    "RolloutRunner",
    "RolloutScore",
    "RolloutScores",
    "SelectionDecision",
    "UpdateCandidate",
    "mean_score",
]
