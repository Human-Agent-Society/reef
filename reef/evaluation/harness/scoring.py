"""How one harness episode is scored, and what Reef keeps of a scored episode.

A deployment names its :class:`EpisodeScorer` in ``evolution.evaluation.scorer``.
:class:`VerifierRewardScorer` scores a Harbor task directory by its verifier's
reward. :class:`ScoredEpisode` is one episode as the evaluation keeps it, and
:class:`FailureObservation` is why an episode failed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from reef.core.plugins import ConfiguredPlugin
from reef.core.tasks import primary_reward
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.runtime.executor.requirements import ExecutionRequirements


class EpisodeScorer(ConfiguredPlugin, ABC):
    """Base class for scoring one harness-evolution episode.

    Implement:
        ``score`` — given a task name and its episode result, return a
        float score. Higher is better; non-finite values raise.
    A model-based judge also overrides ``score_with_models``, which receives
    the step's model bindings. ``score`` stays abstract, so a judge
    implements it too, for example to raise because it needs the bindings.
    """

    @abstractmethod
    def score(self, task: str, result: EpisodeResult) -> float:
        """Score one episode result for a task."""

    def execution_requirements(self) -> ExecutionRequirements:
        """Override when scoring loads a local GPU model or needs cluster placement.

        API calls do not request local GPUs. GPU scorers must initialize their
        model lazily in the allocated worker, not while the recipe is built.
        """
        return ExecutionRequirements()

    def score_with_models(self, task: str, result: EpisodeResult, models: ModelBindings) -> float:
        """The score that the evaluation records for one episode: ``score(task, result)`` unless a judge overrides
        it to call ``models``, the step's bindings."""
        return self.score(task, result)


def verifier_reward(task: str, result: EpisodeResult) -> float:
    """The Harbor verifier's reward for a task directory episode; :class:`VerifierRewardScorer` scores with it.

    The terminus runner writes one ``verifier`` row per episode with the task it ran and the rewards
    its verifier wrote; Harbor's primary reward is the ``reward`` entry, else the sole entry. A failed
    episode or a verifier that wrote nothing scores 0 (the evaluation has already set aside an episode whose
    trial never ran); an episode that exited without a row scores 0; a row for another task, several
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


class VerifierRewardScorer(EpisodeScorer):
    """Score an episode by its Harbor verifier's reward (:func:`verifier_reward`); for a task manifest evaluation."""

    def score(self, task: str, result: EpisodeResult) -> float:
        return verifier_reward(task, result)


@dataclass(frozen=True)
class FailureObservation:
    """Why one episode failed, as the evaluation saw it.

    Values: ``task`` is the task the episode ran; ``stage`` names where it
    failed (``launch``, ``trajectory``, ``residue``, ``trial``, ``graph``,
    ``loop`` or ``exit``); ``cause`` is the error text.
    Purpose: the evaluation makes one per failed episode; the history keeps
    it with the episode's record.
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
class ScoredEpisode:
    """One episode as the evaluation keeps it: the score, or why it has none, and what the trajectory showed.

    Values: ``score`` is a finite number, or ``None`` for an episode that
    could not run, and then ``failure`` says why. A scored episode may carry a
    ``failure`` too (a nonzero exit). ``residue`` ≥ 0 counts files the
    episode left outside the cleanup whitelist. ``agents`` sums the work of
    each agent; ``path`` is the root's stage path and end reason, ``None``
    when no trajectory was read. ``reply`` is the final assistant text, the
    reply a text grader reads, ``None`` when the trajectory holds none.
    ``transcript_read`` says whether a session log was found.
    ``record_archive`` is the kept record of a remote worker, as a gzip tar.
    Purpose: an evaluation worker makes one per episode; the evaluator
    stores it in the history and reads its scores.
    """

    score: float | None
    failure: FailureObservation | None
    residue: int = 0
    agents: dict[str, dict[str, int]] = field(default_factory=dict)
    path: dict[str, Any] | None = None
    reply: str | None = None
    transcript_read: bool = True
    record_archive: bytes | None = field(default=None, repr=False)


__all__ = ["EpisodeScorer", "FailureObservation", "ScoredEpisode", "VerifierRewardScorer", "verifier_reward"]
