"""Storage contract for the artifact evaluations that clients ask a scenario for.

A client asks Reef to evaluate one artifact of a scenario
(``POST /reef/scenarios/{scenario}/evaluations``), or Reef evaluates each new
artifact with the scenario's settings. The evaluation store keeps each request,
its state and its result, so a client reads them after the evaluation ends and
after a restart. Reef does not resume an evaluation that was queued or running
when its service stopped: a later read reports it failed. Archiving the
scenario retires its evaluations with its records.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactEvaluationRequest, PolicyEvaluation

#: The state of an evaluation: it waits for its turn, it runs, it finished with a result, or it failed with an error.
EvaluationState = Literal["queued", "running", "finished", "failed"]
EVALUATION_STATES: tuple[EvaluationState, ...] = ("queued", "running", "finished", "failed")


class EvaluationStateConflict(ReefError):
    """An evaluation is not in the state that a change of state needs, or does not exist."""


@dataclass(frozen=True)
class StoredEvaluation:
    """One artifact evaluation in an evaluation store.

    Values: ``evaluation_id`` and ``owner`` are non-empty strings: ``owner``
    names the service instance that runs the evaluation. ``request`` is the
    ``ArtifactEvaluationRequest``. ``created_at`` is a finite time in seconds,
    and ``finished_at`` is None exactly when ``state`` is ``queued`` or
    ``running``, else a finite time not before ``created_at``. ``result`` is set exactly when
    ``state`` is ``finished``, and ``error``, a non-empty string, exactly when
    ``state`` is ``failed``.

    Purpose: an ``EvaluationStore`` makes it; the evaluation service reads it
    and answers the evaluation routes with ``to_json``.
    """

    evaluation_id: str
    owner: str
    request: ArtifactEvaluationRequest
    state: EvaluationState
    created_at: float
    finished_at: float | None = None
    result: PolicyEvaluation | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id:
            raise ValueError("a stored evaluation needs a non-empty evaluation_id")
        if not isinstance(self.owner, str) or not self.owner:
            raise ValueError("a stored evaluation needs a non-empty owner")
        if not isinstance(self.request, ArtifactEvaluationRequest):
            raise TypeError("a stored evaluation's request is an ArtifactEvaluationRequest")
        if self.state not in EVALUATION_STATES:
            raise ValueError(f"a stored evaluation's state is one of {EVALUATION_STATES}, not {self.state!r}")
        if not math.isfinite(self.created_at):
            raise ValueError("a stored evaluation's created_at is a finite time")
        if (self.finished_at is None) != (self.state in ("queued", "running")):
            raise ValueError("a stored evaluation has finished_at exactly when it has ended")
        if self.finished_at is not None and (
            not math.isfinite(self.finished_at) or self.finished_at < self.created_at
        ):
            raise ValueError("a stored evaluation's finished_at is a finite time not before created_at")
        if (self.result is not None) != (self.state == "finished"):
            raise ValueError("a stored evaluation has a result exactly when it finished")
        if self.result is not None and not isinstance(self.result, PolicyEvaluation):
            raise TypeError("a stored evaluation's result is a PolicyEvaluation")
        if (self.error is not None) != (self.state == "failed"):
            raise ValueError("a stored evaluation has an error exactly when it failed")
        if self.error is not None and (not isinstance(self.error, str) or not self.error):
            raise ValueError("a stored evaluation's error is a non-empty string")

    def to_json(self) -> dict[str, object]:
        """The evaluation as an evaluation route answers it: the request, the state, and the result or error."""
        return {
            "evaluation_id": self.evaluation_id,
            "state": self.state,
            "request": self.request.to_json(),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "policy_evaluation": None if self.result is None else self.result.to_json(),
            "mean_score": None if self.result is None else self.result.mean_score,
            "error": self.error,
        }


class EvaluationStore(ABC):
    """The artifact evaluations of one scenario.

    Values: the state of a store is a finite set of stored evaluations with
    distinct IDs. An evaluation is created ``queued`` or ``running``; a queued
    one starts running once (``begin``). It ends once, as ``finished`` from
    ``running``, or as ``failed`` from either; an ended evaluation does not
    change.

    Purpose: the evaluation service creates an evaluation when it accepts a
    request (``create``), starts a queued one when a run is free (``begin``),
    ends it when its rollouts end (``finish``, ``fail``), and reads the store
    for the evaluation routes (``get``, ``newest``).
    """

    @abstractmethod
    def create(
        self,
        *,
        evaluation_id: str,
        owner: str,
        request: ArtifactEvaluationRequest,
        created_at: float,
        state: Literal["queued", "running"] = "running",
    ) -> StoredEvaluation:
        """Save a new evaluation in ``state`` and return it.

        Raises ``ValueError`` when the store holds ``evaluation_id`` already.
        """

    @abstractmethod
    def begin(self, evaluation_id: str) -> StoredEvaluation:
        """Start a queued evaluation and return it, ``running``; ``EvaluationStateConflict`` for any other."""

    @abstractmethod
    def finish(self, evaluation_id: str, *, result: PolicyEvaluation, finished_at: float) -> StoredEvaluation:
        """End a running evaluation with ``result`` and return it; ``EvaluationStateConflict`` for any other."""

    @abstractmethod
    def fail(self, evaluation_id: str, *, error: str, finished_at: float) -> StoredEvaluation:
        """End a queued or running evaluation with ``error`` and return it; ``EvaluationStateConflict`` for any
        other."""

    @abstractmethod
    def get(self, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id``, or None when the store holds none."""

    @abstractmethod
    def newest(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations, newest ``created_at`` first; ``ValueError`` unless ``limit`` is positive."""


__all__ = ["EVALUATION_STATES", "EvaluationState", "EvaluationStateConflict", "EvaluationStore", "StoredEvaluation"]
