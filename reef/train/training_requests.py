"""Training requests in the training request queue, not in the record store (RFC #1054).

A training request (``POST /reef/train``) is control's work, not an observed record. Each request is one item
of the ``requests`` work queue (``reef.storage.work_queue``):

- ``id`` is the request ID: the client's ``agent_record_id``, or a new one;
- ``scenario`` and ``serial_key`` are its scenario, so the requests of one scenario settle in the order they came;
- ``kind`` is ``training_request_kind(scenario)``, so control takes only the items of the scenario that it settles;
- ``payload`` is the normalized request (``normalize_training_request_payload``).

A request is queued until a training commit consumes it; then control finishes its item. A commit records the
request in its ``metrics["training_request"]``, so the history of a request outlives its queue item, which
control deletes after the work retention.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from reef.core.training_request import TrainingRequest
from reef.storage.records import RecordConflictError
from reef.storage.work_queue import JsonValue, NewWork, WorkConflictError, WorkItem, WorkQueue
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)

#: The queue of training requests.
TRAINING_REQUESTS_QUEUE = "training_requests"
#: The ``WorkItem.metadata`` key that holds what the failed step of a request said.
REQUEST_FAILURE_KEY = "failure"
#: The failure of a request that a client cancelled (``ScenarioTrainingRequests.cancel``).
REQUEST_CANCELLED_ERROR = "cancelled by request"


def training_request_kind(scenario: str) -> str:
    """The work-item kind of the requests of ``scenario``."""
    return f"training_request:{scenario}"


@dataclass(frozen=True)
class RequestCancel:
    """The result of one cancel of a training request.

    Values: ``queued`` is whether no commit had consumed the request, so the cancel made it fail and it trains no
    job. ``job`` is the newest training job of the request after the cancel, as ``job_json`` gives it, or None for a
    request with no job.
    Purpose: ``Dispatcher.cancel_request`` makes it, and the cancel route answers with it.
    """

    queued: bool
    job: dict[str, JsonValue] | None


@dataclass(frozen=True)
class TrainingRequestQueue:
    """The training request queue of every scenario of one deployment.

    Values: ``queue`` is the work queue store that holds ``TRAINING_REQUESTS_QUEUE``.
    Purpose: the dispatcher makes it from its coordination store and gives it to the scenario factory, which gives
    each scenario its own requests (``of``).
    """

    queue: WorkQueue

    def of(self, scenario: str) -> ScenarioTrainingRequests:
        """The requests of ``scenario``."""
        return ScenarioTrainingRequests(self.queue, scenario)


@dataclass(frozen=True)
class ScenarioTrainingRequests:
    """The training requests of one scenario, as items of the training request queue.

    Values: ``queue`` is the work queue store; ``scenario`` a non-empty scenario name; ``worker_id`` the worker that
    takes an item to finish it. The items are those of ``TRAINING_REQUESTS_QUEUE`` with this scenario, as the module
    docstring defines them.
    Purpose: the dispatcher files requests (``file``) and cancels them (``cancel``); a scenario gives the queued ones
    to its trainers and finishes those that a commit consumed (``give``); the request page reads one (``get``).
    """

    queue: WorkQueue
    scenario: str
    worker_id: str = "control"

    def file(self, request_id: str, payload: Mapping[str, Any]) -> WorkItem:
        """Queue the request ``request_id`` with the normalized ``payload``; return its item.

        The same ID with the same payload returns the item that exists; with another payload it raises
        ``RecordConflictError``, as a record with the same ID and other content does.
        """
        work = NewWork(
            TRAINING_REQUESTS_QUEUE,
            request_id,
            self.scenario,
            training_request_kind(self.scenario),
            cast(dict[str, JsonValue], dict(payload)),
            serial_key=self.scenario,
        )
        try:
            return self.queue.append(work)
        except WorkConflictError as exc:
            raise RecordConflictError(f"request {request_id!r} already has different content") from exc

    def get(self, request_id: str) -> WorkItem | None:
        """The item of the request ``request_id`` of this scenario, in any state, or None."""
        item = self.queue.read(TRAINING_REQUESTS_QUEUE, request_id)
        return item if item is not None and item.scenario == self.scenario else None

    def queued(self) -> tuple[TrainingRequest, ...]:
        """The requests that no commit consumed yet, oldest first."""
        return tuple(
            TrainingRequest.from_dict(item.payload, request_id=item.id)
            for item in self.queue.items(
                TRAINING_REQUESTS_QUEUE, scenario=self.scenario, kind=training_request_kind(self.scenario)
            )
            if not item.final
        )

    def mark_failed(self, request_id: str, error: str) -> None:
        """Store that the step of the request ``request_id`` failed with ``error``, so that a reload or a restart
        still consumes the request with a skip row."""
        self.queue.write_metadata(TRAINING_REQUESTS_QUEUE, request_id, REQUEST_FAILURE_KEY, error)

    def cancel(self, request_id: str) -> WorkItem | None:
        """Cancel the request ``request_id``: a queued request fails with ``REQUEST_CANCELLED_ERROR``
        (``mark_failed``), unless its step failed before, so a trainer consumes it with a skip row and no job. A
        request that a commit consumed does not change. Return the item as it was before the call, or None when the
        scenario has no such request."""
        item = self.get(request_id)
        if item is not None and not item.final and REQUEST_FAILURE_KEY not in item.metadata:
            self.mark_failed(request_id, REQUEST_CANCELLED_ERROR)
        return item

    def failures(self) -> dict[str, str]:
        """What the failed step of each queued request said, by ID; a request whose step did not fail has none."""
        return {
            item.id: str(item.metadata[REQUEST_FAILURE_KEY])
            for item in self.queue.items(
                TRAINING_REQUESTS_QUEUE, scenario=self.scenario, kind=training_request_kind(self.scenario)
            )
            if not item.final and REQUEST_FAILURE_KEY in item.metadata
        }

    def settle(self, consumed: Iterable[str]) -> None:
        """Finish the items of the queued requests whose IDs are in ``consumed``, oldest first.

        Requests train in the order they came, so the consumed ones are the oldest queued; settling stops at
        the first queued request that is not consumed.
        """
        consumed_ids = set(consumed)
        for request in self.queued():
            if request.id not in consumed_ids:
                return
            item = self.queue.take(TRAINING_REQUESTS_QUEUE, self.worker_id, (training_request_kind(self.scenario),))
            if item is None or item.id != request.id:
                # Another control holds it; its lease returns it, and the next settle finishes it.
                logger.warning(
                    "scenario %r: request %s is held elsewhere; it settles later", self.scenario, request.id
                )
                return
            self.queue.finish(TRAINING_REQUESTS_QUEUE, item.id, self.worker_id, {"consumed": True})

    def give(self, trainers: Iterable[Trainer]) -> None:
        """Give every trainer of ``trainers`` the queued requests that none of them consumed, and finish the items
        of those that one consumed (``settle``). A trainer's commit, or its recovery, consumes a request."""
        trainers = tuple(trainers)
        queued = self.queued()
        ids = [request.id for request in queued]
        consumed: set[str] = set()
        for trainer in trainers:
            consumed |= trainer.consumed_among(ids)
        self.settle(consumed)
        pending = tuple(request for request in queued if request.id not in consumed)
        failures = self.failures()
        for trainer in trainers:
            trainer.queue_requests(pending, failures)


__all__ = [
    "REQUEST_CANCELLED_ERROR",
    "REQUEST_FAILURE_KEY",
    "TRAINING_REQUESTS_QUEUE",
    "RequestCancel",
    "ScenarioTrainingRequests",
    "TrainingRequestQueue",
    "training_request_kind",
]
