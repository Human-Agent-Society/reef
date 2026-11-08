"""Storage contract for the artifact evaluations that clients ask a scenario for.

A client asks Reef to evaluate one artifact of a scenario
(``POST /reef/scenarios/{scenario}/evaluations``), or Reef evaluates each new
artifact with the scenario's settings. The evaluation store keeps each request,
its state and its result, so a client reads them after the evaluation ends and
after a restart. Reef does not resume an evaluation that was queued or running
when its service stopped: a later read reports it failed. Archiving the
scenario retires its evaluations with its records.

The model calls of each rollout are not in this store: each is a record item of
the scenario's owner in the data store, whose origin source is ``evaluation``
(``reef.storage.record_items``). This module defines their values.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from reef.core.errors import ReefError
from reef.core.evaluation import ArtifactEvaluationRequest, PolicyEvaluation
from reef.core.values import JsonValue, require_text

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


def require_integer(value: object, name: str, *, minimum: int = 0) -> None:
    """Raise ``ValueError`` naming ``name`` unless ``value`` is an integer from ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} is an integer from {minimum}")


@dataclass(frozen=True)
class EvaluationCallSummary:
    """One model call of an evaluation rollout, without its bodies.

    Values: ``task_index`` and ``repeat`` name the rollout in its evaluation, as ``EvaluationRollout`` does.
    ``attempt`` is the 1-based attempt of the rollout's work item that made the call. ``index`` is the 0-based
    position of the call in that attempt, in the order in which the calls ended. ``path`` is the non-empty path that
    the agent called, and ``status`` the HTTP status that it got, an integer from 100. ``started_at`` is a finite time in seconds, and
    ``duration_s`` a finite number of seconds from 0. ``prompt_tokens`` and ``completion_tokens`` are integers from 0,
    or None when the reply does not tell them. ``finish_reason`` is the non-empty reason that the reply gives, or
    None. ``tool_calls`` is the number of tool calls in the reply, an integer from 0. A call whose ``status`` is not
    200 has no ``finish_reason`` and no tool call. ``(task_index, repeat, attempt, index)`` is the key of the call in
    its evaluation.

    Purpose: the evaluation worker makes it for each call that a rollout's capture proxy captures; the evaluation
    routes answer it with ``to_json``.
    """

    task_index: int
    repeat: int
    attempt: int
    index: int
    path: str
    status: int
    started_at: float
    duration_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    tool_calls: int

    def __post_init__(self) -> None:
        for count, name in (
            (self.task_index, "task_index"),
            (self.repeat, "repeat"),
            (self.index, "index"),
            (self.tool_calls, "tool_calls"),
        ):
            require_integer(count, name)
        require_integer(self.attempt, "attempt", minimum=1)
        require_integer(self.status, "status", minimum=100)
        for tokens, name in ((self.prompt_tokens, "prompt_tokens"), (self.completion_tokens, "completion_tokens")):
            if tokens is not None:
                require_integer(tokens, name)
        require_text(self.path, "path")
        if not math.isfinite(self.started_at):
            raise ValueError("a call's started_at is a finite time")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError("a call's duration_s is a finite number of seconds from 0")
        if self.finish_reason is not None and (not isinstance(self.finish_reason, str) or not self.finish_reason):
            raise ValueError("a call's finish_reason is a non-empty string or None")
        if self.status != 200 and (self.finish_reason is not None or self.tool_calls):
            raise ValueError("a failed call has no finish_reason and no tool call")

    def to_json(self) -> dict[str, JsonValue]:
        """The call as the call routes answer it in its rollout: every field but ``task_index`` and ``repeat``."""
        return {
            "attempt": self.attempt,
            "index": self.index,
            "path": self.path,
            "status": self.status,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
        }


@dataclass(frozen=True)
class EvaluationCall:
    """One model call of an evaluation rollout, with its bodies.

    Values: ``summary`` is the call's ``EvaluationCallSummary``. ``request`` is the JSON object that the agent sent,
    and ``response`` the JSON object that it got: the reply, or the error body of a failed call. Each is None when the
    capture proxy kept no JSON object.

    Purpose: the evaluation worker makes it for each captured call, and the data store keeps it as a record item
    (``reef.storage.record_items.evaluation_call_item``). The call route answers it with ``to_json``.
    """

    summary: EvaluationCallSummary
    request: Mapping[str, JsonValue] | None
    response: Mapping[str, JsonValue] | None

    def __post_init__(self) -> None:
        if not isinstance(self.summary, EvaluationCallSummary):
            raise TypeError("a call's summary is an EvaluationCallSummary")
        for body, name in ((self.request, "request"), (self.response, "response")):
            if body is not None and not isinstance(body, Mapping):
                raise TypeError(f"a call's {name} is a JSON object or None")

    def to_json(self) -> dict[str, JsonValue]:
        """The summary's ``to_json`` with ``request`` and ``response``."""
        return {
            **self.summary.to_json(),
            "request": None if self.request is None else dict(self.request),
            "response": None if self.response is None else dict(self.response),
        }


@dataclass(frozen=True)
class EvaluationCallTotals:
    """The totals of the kept calls of one evaluation rollout, over all its attempts.

    Values: ``count`` is the number of calls, an integer from 1. ``failed`` is the number of calls whose status is
    not 200, from 0 to ``count``. ``prompt_tokens`` and ``completion_tokens`` are the sums of the known token counts
    of the calls, integers from 0; a call whose count is None adds nothing.

    Purpose: ``ArtifactEvaluations.rollouts_json`` adds up the calls of each rollout into one; the rollouts route
    answers it with ``to_json``.
    """

    count: int
    failed: int
    prompt_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        require_integer(self.count, "count", minimum=1)
        require_integer(self.failed, "failed")
        require_integer(self.prompt_tokens, "prompt_tokens")
        require_integer(self.completion_tokens, "completion_tokens")
        if self.failed > self.count:
            raise ValueError("a rollout has no more failed calls than calls")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "count": self.count,
            "failed": self.failed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
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


__all__ = [
    "EVALUATION_STATES",
    "EvaluationCall",
    "EvaluationCallSummary",
    "EvaluationCallTotals",
    "EvaluationState",
    "EvaluationStateConflict",
    "EvaluationStore",
    "StoredEvaluation",
]
