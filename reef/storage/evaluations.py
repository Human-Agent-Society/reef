"""Storage contract for the artifact evaluations that clients ask a scenario for.

A client asks Reef to evaluate one artifact of a scenario
(``POST /reef/scenarios/{scenario}/evaluations``), or Reef evaluates each new
artifact with the scenario's settings. The evaluation store keeps each request,
its state and its result, so a client reads them after the evaluation ends and
after a restart. Reef does not resume an evaluation that was queued or running
when its service stopped: a later read reports it failed. A client can run the
rollouts of an ended evaluation that have no score again; the evaluation then
runs and ends again (``reopen``). Archiving the
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


class EvaluationStateConflictError(ReefError):
    """An evaluation is not in the state that a change of state needs, or does not exist."""


class EvaluationStartConflictError(ReefError):
    """A start ID is already in use, or its evaluation has a different request."""


@dataclass(frozen=True)
class EvaluationStart:
    """The start ID, request fingerprint and saved admission data of one evaluation.

    The ID is a non-blank client string of at most 256 characters. It contains no NUL or surrogate code point.
    The fingerprint is a 64-digit lowercase SHA-256 hex string.
    ``first_work_json`` is a non-empty string that this store treats as opaque.
    The evaluation service writes and reads its JSON to complete admission after an interruption.
    """

    start_id: str
    fingerprint: str
    first_work_json: str

    def __post_init__(self) -> None:
        check_evaluation_start_id(self.start_id)
        if len(self.fingerprint) != 64 or any(digit not in "0123456789abcdef" for digit in self.fingerprint):
            raise ValueError("the request fingerprint must be a SHA-256 hex string")
        if not isinstance(self.first_work_json, str) or not self.first_work_json:
            raise ValueError("the saved admission data must be a non-empty string")


def check_evaluation_start_id(start_id: object) -> str:
    """Return a valid evaluation start ID or raise ValueError."""
    if (
        not isinstance(start_id, str)
        or not start_id.strip()
        or len(start_id) > 256
        or any(char == "\0" or 0xD800 <= ord(char) <= 0xDFFF for char in start_id)
    ):
        raise ValueError("start_id must be a non-empty string of at most 256 characters without NUL or surrogates")
    return start_id


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

    @property
    def mean_score(self) -> float | None:
        """The result's mean score by the request's weighting, or None without a score."""
        if self.result is None:
            return None
        if self.request.collection is None:
            return self.result.mean_score
        return self.request.collection.mean_score(self.result.scores)

    def to_json(self) -> dict[str, object]:
        """The evaluation as an evaluation route answers it: the request, the state, and the result or error.

        ``mean_score`` and ``benchmark_scores`` follow the request's collection (``EvaluationCollection``). Without
        a collection, ``mean_score`` is the mean of every rollout and ``benchmark_scores`` is empty.
        """
        collection = self.request.collection
        return {
            "evaluation_id": self.evaluation_id,
            "state": self.state,
            "request": self.request.to_json(include_task_digests=False),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "policy_evaluation": None if self.result is None else self.result.to_json(),
            "mean_score": self.mean_score,
            "benchmark_scores": (
                [] if self.result is None or collection is None else collection.benchmark_scores(self.result.scores)
            ),
            "error": self.error,
        }


class EvaluationStore(ABC):
    """The artifact evaluations of one scenario.

    Values: the state of a store is a finite set of stored evaluations with
    distinct IDs. An evaluation is created ``queued`` or ``running``; a queued
    one starts running once (``begin``). It ends as ``finished`` from
    ``running``, or as ``failed`` from either. An ended evaluation does not
    change until ``reopen`` makes it ``running`` again, after which it ends
    again.

    Purpose: the evaluation service creates an evaluation when it accepts a
    request (``create``), starts a queued one when a run is free (``begin``),
    ends it when its rollouts end (``finish``, ``fail``), runs an ended one
    again when a client retries its rollouts (``reopen``), changes how many of
    its rollouts run at once when a client runs them again (``limit_rollouts``), and reads the store
    for the evaluation routes (``read``, ``newest``).
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
        start: EvaluationStart | None = None,
    ) -> StoredEvaluation:
        """Save a new evaluation in ``state`` and return it.

        With ``start``, reserve its ID and save the first work item in the same transaction.
        Raise ``ValueError`` when the store holds ``evaluation_id``.
        Raise ``EvaluationStartConflictError`` when the start ID is already in use.
        """

    @abstractmethod
    def read_started(self, start_id: str, fingerprint: str) -> tuple[StoredEvaluation, str | None] | None:
        """Read the evaluation and its pending admission data, or return None for an unused start ID.

        Raise ``EvaluationStartConflictError`` if the stored fingerprint differs.
        """

    @abstractmethod
    def pending_starts(self) -> tuple[tuple[str, str], ...]:
        """Return the evaluation ID and saved admission data of each pending start."""

    @abstractmethod
    def pending_start(self, evaluation_id: str) -> str | None:
        """Return the saved admission data of this evaluation's pending start, else None."""

    @abstractmethod
    def complete_start(self, evaluation_id: str) -> None:
        """Clear the pending first work item after the work queue stores it."""

    @abstractmethod
    def begin(self, evaluation_id: str) -> StoredEvaluation:
        """Start a queued evaluation and return it, ``running``; ``EvaluationStateConflictError`` for any other."""

    @abstractmethod
    def finish(self, evaluation_id: str, *, result: PolicyEvaluation, finished_at: float) -> StoredEvaluation:
        """End a running evaluation with ``result`` and return it; ``EvaluationStateConflictError`` for any other."""

    @abstractmethod
    def fail(self, evaluation_id: str, *, error: str, finished_at: float) -> StoredEvaluation:
        """End a queued or running evaluation with ``error`` and return it; ``EvaluationStateConflictError`` for any
        other."""

    @abstractmethod
    def reopen(self, evaluation_id: str) -> StoredEvaluation:
        """Make a finished or failed evaluation ``running`` again and return it, without its result, its error and
        its ``finished_at``; ``EvaluationStateConflictError`` for any other."""

    @abstractmethod
    def limit_rollouts(self, evaluation_id: str, max_concurrent_rollouts: int | None) -> StoredEvaluation:
        """Store ``max_concurrent_rollouts`` as the most rollouts of the evaluation that run at once, and return the
        evaluation; ``EvaluationStateConflictError`` when the store holds no such evaluation, and ``ValueError`` for
        a value that ``ArtifactEvaluationRequest`` refuses. The rest of the evaluation stays as it is."""

    @abstractmethod
    def read(self, evaluation_id: str) -> StoredEvaluation | None:
        """The evaluation ``evaluation_id``, or None when the store holds none."""

    @abstractmethod
    def newest(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations, newest ``created_at`` first; ``ValueError`` unless ``limit`` is positive."""

    def newest_for_list(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
        """At most ``limit`` evaluations for a list. Stores may leave task digests out of these requests."""
        return self.newest(limit=limit)


__all__ = [
    "EVALUATION_STATES",
    "EvaluationState",
    "EvaluationStateConflictError",
    "EvaluationStore",
    "StoredEvaluation",
]
