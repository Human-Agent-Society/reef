"""How one harness rollout is scored, and what Reef keeps of a scored rollout.

A deployment names its :class:`RolloutScorer` in ``evolution.evaluation.scorer``.
:class:`VerifierRewardScorer` scores a Harbor task directory by its verifier's
reward. :class:`ScoredRollout` is one rollout as the evaluation keeps it, and
:class:`FailureObservation` is why a rollout failed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.evaluation import RolloutScore
from reef.core.plugins import ConfiguredPlugin
from reef.core.tasks import primary_reward
from reef.harness.rollouts.model_binding import ModelBindings
from reef.harness.rollouts.run import RolloutResult
from reef.runtime.executor.requirements import ExecutionRequirements


class RolloutScorer(ConfiguredPlugin, ABC):
    """Base class for scoring one harness-evolution rollout.

    Implement:
        ``score`` — given a task name and its rollout result, return a
        float score. Higher is better; non-finite values raise.
    A model-based judge also overrides ``score_with_models``, which receives
    the step's model bindings. ``score`` stays abstract, so a judge
    implements it too, for example to raise because it needs the bindings.
    """

    @abstractmethod
    def score(self, task: str, result: RolloutResult) -> float:
        """Score one rollout result for a task."""

    def execution_requirements(self) -> ExecutionRequirements:
        """Override when scoring loads a local GPU model or needs cluster placement.

        API calls do not request local GPUs. GPU scorers must initialize their
        model lazily in the allocated worker, not while the recipe is built.
        """
        return ExecutionRequirements()

    def score_with_models(self, task: str, result: RolloutResult, models: ModelBindings) -> float:
        """The score that the evaluation records for one rollout: ``score(task, result)`` unless a judge overrides
        it to call ``models``, the step's bindings."""
        return self.score(task, result)


def verifier_reward(task: str, result: RolloutResult) -> float:
    """The Harbor verifier's reward for a task directory rollout; :class:`VerifierRewardScorer` scores with it.

    A task directory rollout's trajectory holds one ``verifier`` row with the task it ran and the rewards
    its verifier wrote (``reef.harness.rollouts.harbor_task`` writes it, and so may an adapter's own runner);
    Harbor's primary reward is the ``reward`` entry, else the sole entry. A failed
    rollout or a verifier that wrote nothing scores 0 (the evaluation has already marked a rollout whose
    trial never ran as failed); a rollout that exited without a row scores 0; a row for another task, several
    rewards without a ``reward`` entry, or a reward that is not a finite number is an error.
    """
    rows = [event for event in result.trajectory if event.get("type") == "verifier"]
    if not rows and result.exit_code:
        return 0.0
    if len(rows) != 1:
        raise ValueError(f"expected one verifier record for {task!r}, found {len(rows)}")
    row = rows[0]
    if row.get("task") != task:
        raise ValueError(f"the verifier record names {row.get('task')!r}, not {task!r}")
    if row.get("failed"):
        return 0.0
    rewards = row.get("rewards")
    reward = primary_reward(rewards) if isinstance(rewards, Mapping) else row.get("reward")
    if reward is None:
        if isinstance(rewards, Mapping) and rewards:
            raise ValueError(f"the verifier for {task!r} wrote {sorted(rewards)} and no 'reward' entry")
        return 0.0
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
        raise ValueError(f"the verifier reward for {task!r} must be a finite number, not {reward!r}")
    return float(reward)


class VerifierRewardScorer(RolloutScorer):
    """Score a rollout by its Harbor verifier's reward (:func:`verifier_reward`); for a task manifest evaluation."""

    def score(self, task: str, result: RolloutResult) -> float:
        return verifier_reward(task, result)


@dataclass(frozen=True)
class FailureObservation:
    """Why one rollout failed, as the evaluation saw it.

    Values: ``task`` is the task the rollout ran; ``stage`` names where it
    failed (``launch``, ``trajectory``, ``residue``, ``trial``, ``graph``,
    ``loop`` or ``exit``); ``cause`` is the error text.
    Purpose: the evaluation makes one per failed rollout; the history keeps
    it with the rollout's record.
    """

    task: str
    stage: str
    cause: str

    def to_dict(self) -> dict[str, str]:
        return {"task": self.task, "stage": self.stage, "cause": self.cause}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FailureObservation:
        return cls(task=str(value["task"]), stage=str(value["stage"]), cause=str(value["cause"]))


@dataclass(frozen=True)
class ScoredRollout:
    """One rollout as the evaluation keeps it: its rollout score, why it failed, and what the trajectory showed.

    Values: ``score`` is a ``RolloutScore``. It is failed when the rollout
    could not launch, its trial never ran or its trajectory could not be read:
    the environment, not the tree, decided it. It is 0 when the tree gave no
    answer: an error ended its turn, or ``forbid_residue`` refused the files it
    left. ``failure`` says why in each of these cases, and is also set for a
    scored rollout that exited nonzero (stage ``exit``). ``residue`` ≥ 0 counts files the
    rollout left outside the cleanup whitelist. ``agents`` sums the work of
    each agent; ``path`` is the root's stage path and end reason, ``None``
    when no trajectory was read. ``reply`` is the final assistant text, the
    reply a text grader reads, ``None`` when the trajectory holds none.
    ``transcript_read`` says whether a session log was found.
    ``record_archive`` is the kept record of a remote worker, as a gzip tar.
    Purpose: an evaluation worker makes one per rollout; the evaluator
    stores it in the history and reads its scores.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.score, RolloutScore):
            raise TypeError("ScoredRollout.score must be a RolloutScore")
        if self.score.is_failed and self.failure is None:
            raise ValueError("a failed ScoredRollout must say why it failed")

    score: RolloutScore
    failure: FailureObservation | None
    residue: int = 0
    agents: dict[str, dict[str, int]] = field(default_factory=dict)
    path: dict[str, Any] | None = None
    reply: str | None = None
    transcript_read: bool = True
    record_archive: bytes | None = field(default=None, repr=False)


__all__ = ["FailureObservation", "RolloutScorer", "ScoredRollout", "VerifierRewardScorer", "verifier_reward"]
