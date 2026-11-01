"""Requested task-generation rounds in the samples service (``reef.record2sample.requested_generation``).

Control files each request of a client as one ``task_generation`` item of the ``samples`` queue
(``file_task_generation``):

- ``id`` is ``task_generation:<scenario>:<n>``, new for each request, where ``n`` is a random hexadecimal string: two
  requests of the same reports are two rounds;
- ``serial_key`` is ``task_generation:<scenario>:<component>``, so the requests of one trainer run one at a time, in
  the order they came;
- ``payload`` is ``{"scenario", "component", "report_ids"}``, where ``report_ids`` is a list of distinct report IDs.

A ``TaskGenerationWorker`` takes the item and gives the round to the trainer's record-to-sample stage through a
``TaskGenerations``. It holds the item while the round waits and runs, as the training service holds a job, and
looks for the result every ``RESULT_POLL_SECONDS``. The samples worker's selections advance the stage meanwhile and
store the round's tasks. A complete generation finishes the item with ``{"generation", "train_tasks"}``; a round that
ran out of its attempts, or did not start, fails it with the stage's error. When the worker stops, or the process
dies and the item's lease expires, another worker takes the item, and the stage continues the same generation.
"""

from __future__ import annotations

import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import cast

from reef.execution.workers import ItemFailed, QueueWorker, WorkerStopped
from reef.record2sample.requested_generation import RequestedRoundResult
from reef.record2sample.service.worker import SAMPLES_QUEUE
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLost, WorkQueue

#: The kind of the item of one requested task-generation round.
TASK_GENERATION_KIND = "task_generation"
#: The seconds between two looks of a worker for the result of the round that it holds.
RESULT_POLL_SECONDS = 1.0


def task_generation_key(scenario: str, component: str) -> str:
    """The ``serial_key`` of the requested rounds of one trainer."""
    return f"{TASK_GENERATION_KIND}:{scenario}:{component}"


def file_task_generation(queue: WorkQueue, scenario: str, component: str, report_ids: tuple[str, ...]) -> WorkItem:
    """Queue a requested round of the trainer of ``component`` in ``scenario`` from the reports ``report_ids``;
    return its item."""
    payload = cast(
        dict[str, JsonValue], {"scenario": scenario, "component": component, "report_ids": list(report_ids)}
    )
    return queue.add(
        NewWork(
            SAMPLES_QUEUE,
            f"{TASK_GENERATION_KIND}:{scenario}:{uuid.uuid4().hex}",
            scenario,
            TASK_GENERATION_KIND,
            payload,
            serial_key=task_generation_key(scenario, component),
        )
    )


def task_generation_item(queue: WorkQueue, scenario: str, request_id: str) -> WorkItem | None:
    """The item of the requested round ``request_id`` of ``scenario``, or None."""
    item = queue.get(SAMPLES_QUEUE, request_id)
    if item is None or item.kind != TASK_GENERATION_KIND or item.scenario != scenario:
        return None
    return item


def requested_reports(item: WorkItem) -> tuple[str, tuple[str, ...]]:
    """The component and the report IDs of a ``task_generation`` item; ``ValueError`` for another payload."""
    component, report_ids = item.payload.get("component"), item.payload.get("report_ids")
    if not isinstance(component, str) or not component:
        raise ValueError("a task generation names its component")
    if not isinstance(report_ids, list) or any(
        not isinstance(report_id, str) or not report_id for report_id in report_ids
    ):
        raise ValueError("a task generation names its report_ids, a list of non-empty strings")
    return component, tuple(cast(list[str], report_ids))


class TaskGenerations(ABC):
    """Gives a trainer the requested rounds of a samples worker, and reads their results.

    Purpose: a ``TaskGenerationWorker`` calls ``advance_request`` for the item it holds, once and then at each look.
    """

    @abstractmethod
    def advance_request(
        self, scenario: str, component: str, request_id: str, report_ids: tuple[str, ...]
    ) -> RequestedRoundResult | None:
        """Give the trainer of ``component`` in ``scenario`` the round of the request ``request_id`` from the
        reports ``report_ids``, unless the trainer holds the request; return the round's result once it ended,
        and None while it waits or runs. Raise ``ValueError`` for a request that cannot run, such as one of a
        scenario that does not exist."""


class TaskGenerationWorker(QueueWorker):
    """One worker of the samples service for requested task-generation rounds.

    Values: as ``QueueWorker``, on the ``task_generation`` items of the samples queue; ``generations`` gives each
    round to its trainer; ``result_poll_seconds`` is the wait between two looks for a round's result, a positive
    number.

    Purpose: the samples service starts it beside its ``SamplesWorker`` (``start``) and stops it (``stop``). A worker
    runs one round at a time; more samples processes run the rounds of more trainers at once.
    """

    def __init__(
        self,
        queue: WorkQueue,
        generations: TaskGenerations,
        *,
        worker_id: str,
        poll_seconds: float = 0.2,
        result_poll_seconds: float = RESULT_POLL_SECONDS,
    ) -> None:
        super().__init__(queue, SAMPLES_QUEUE, [TASK_GENERATION_KIND], worker_id=worker_id, poll_seconds=poll_seconds)
        if result_poll_seconds <= 0:
            raise ValueError("result_poll_seconds is positive")
        self.generations = generations
        self.result_poll_seconds = result_poll_seconds

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        component, report_ids = requested_reports(item)
        while (result := self.generations.advance_request(item.scenario, component, item.id, report_ids)) is None:
            if self.stopped.wait(self.result_poll_seconds):
                raise WorkerStopped
            if lease_lost.is_set():
                raise WorkLeaseLost(f"worker {self.worker_id} no longer holds {item.id}")
        if result.error:
            raise ItemFailed(result.error)
        return {"generation": result.generation, "train_tasks": list(result.train_tasks)}


__all__ = [
    "RESULT_POLL_SECONDS",
    "TASK_GENERATION_KIND",
    "TaskGenerationWorker",
    "TaskGenerations",
    "file_task_generation",
    "requested_reports",
    "task_generation_item",
    "task_generation_key",
]
