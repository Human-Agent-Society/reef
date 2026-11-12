"""The trial observer of the evaluation service (``evaluation.trial_observer``).

The evaluation worker runs each attempt of a rollout as a Harbor trial in a temporary directory, and removes the
directory when the attempt ends. A deployment that keeps more of a trial than Reef keeps, such as the logs of the
agent and the verifier, names a ``TrialObserver``. The worker gives it each trial before it removes the directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from reef.core.plugins import ConfiguredPlugin


@dataclass(frozen=True)
class EvaluationTrial:
    """One attempt of one rollout, as Harbor ran it.

    Values: ``scenario`` and ``evaluation_id`` name the evaluation; ``task_index``, ``task_name`` and ``repeat`` name
    the rollout, as in ``EvaluationRollout``; ``attempt`` is the 1-based attempt of the rollout's work item;
    ``directory`` is the trial directory on this host, or None when Harbor made none; ``error`` is the error text of
    the trial, or "" when the trial had no error.

    Purpose: the evaluation worker gives it to the deployment's ``TrialObserver``.
    """

    scenario: str
    evaluation_id: str
    task_index: int
    task_name: str
    repeat: int
    attempt: int
    directory: Path | None
    error: str


class TrialObserver(ConfiguredPlugin, ABC):
    """Reads each trial of the evaluation service before the worker removes its directory.

    ``observe`` runs on the worker's thread, after the trial and before the score. It must not keep a reference to
    the directory, because the worker removes it when ``observe`` returns. The worker logs an exception of
    ``observe``, and the rollout and its score do not change.
    """

    @abstractmethod
    def observe(self, trial: EvaluationTrial) -> None: ...


__all__ = ["EvaluationTrial", "TrialObserver"]
