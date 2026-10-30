"""The samples service: it turns a scenario's new records into samples, apart from training.

Control does not make samples on the training path. When a scenario accepts records, control adds a
``record_selection`` item for each trainer that trains to the ``samples`` queue. A ``SamplesWorker`` takes the
item and has a ``SampleProducers`` make the samples: the trainer's record-to-record stage selects and groups the
records, its record-to-sample stage makes samples of the formed record sets, and the samples land in the sample
store. Control's trainer reads the sample store and builds jobs from it. Slow stages (judges, generated tasks and
their rollouts) never hold the training path.

A ``record_selection`` item:

- ``id`` is ``record_selection:<scenario>:<component>:<number>``, where ``number`` counts the earlier selections of
  the same trainer; two requests at once compute the same ID, so they add one item;
- ``serial_key`` is ``record_selection:<scenario>:<component>``, so one worker at a time selects the records of one
  trainer, in order;
- ``payload`` is ``{"scenario", "component"}``.

A selection reads every record that its trainer has not used yet when it runs. So a request adds nothing while a
selection of the same trainer waits in the queue; it adds one when the trainer's selections are final or running.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import cast

from reef.execution.workers import ServiceWorker
from reef.storage.work_queue import JsonValue, NewWork, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The queue of sample work.
SAMPLES_QUEUE = "samples"
#: The kind of the item that makes samples of a trainer's new records.
RECORD_SELECTION_KIND = "record_selection"
#: The seconds until a worker looks again at a trainer whose stages still run background work.
BACKGROUND_POLL_SECONDS = 1.0


def selection_key(scenario: str, component: str) -> str:
    """The ``serial_key`` of the record selection of one trainer."""
    return f"{RECORD_SELECTION_KIND}:{scenario}:{component}"


def request_selection(
    queue: WorkQueue, scenario: str, component: str, *, available_at: float | None = None
) -> WorkItem:
    """Ask for a selection of the new records of the trainer of ``component`` in ``scenario``; return the item that
    will read them: a queued one, or a new one."""
    key = selection_key(scenario, component)
    earlier = [
        item
        for item in queue.items(SAMPLES_QUEUE, scenario=scenario, kind=RECORD_SELECTION_KIND)
        if item.serial_key == key
    ]
    waiting = [item for item in earlier if item.state == "queued"]
    if waiting:
        return waiting[0]
    payload = cast(dict[str, JsonValue], {"scenario": scenario, "component": component})
    return queue.add(
        NewWork(
            SAMPLES_QUEUE,
            f"{key}:{len(earlier)}",
            scenario,
            RECORD_SELECTION_KIND,
            payload,
            serial_key=key,
            available_at=available_at,
        )
    )


class SampleProducers(ABC):
    """Makes the samples of one trainer from its new records.

    Purpose: a ``SamplesWorker`` calls ``produce`` for each ``record_selection`` item it takes.
    """

    @abstractmethod
    def produce(self, scenario: str, component: str) -> bool:
        """Make samples of the records that the trainer of ``component`` in ``scenario`` has not used yet, and store
        them; return whether background work of the stages still runs, so the worker looks again later. A
        scenario that is gone, or a trainer that trains nothing, makes nothing and returns False."""


class SamplesWorker(ServiceWorker):
    """One worker of the samples service.

    Values: ``queue`` is the work queue; ``producers`` makes the samples; ``worker_id`` names this worker in
    leases; ``poll_seconds`` is the wait between two looks at the queue.

    Purpose: the samples service starts one or more (``start``) and stops them (``stop``).
    """

    def __init__(
        self, queue: WorkQueue, producers: SampleProducers, *, worker_id: str, poll_seconds: float = 0.2
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds is positive")
        self.queue = queue
        self.producers = producers
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name=f"reef-samples-{self.worker_id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            if not self.run_once():
                self.stopped.wait(self.poll_seconds)

    def run_once(self) -> bool:
        """Take one record selection and make its samples; return whether there was one."""
        self.queue.requeue_expired()
        item = self.queue.take(SAMPLES_QUEUE, self.worker_id, [RECORD_SELECTION_KIND])
        if item is None:
            return False
        scenario = cast(str, item.payload["scenario"])
        component = cast(str, item.payload["component"])
        try:
            self.queue.start(SAMPLES_QUEUE, item.id, self.worker_id)
            background = self.producers.produce(scenario, component)
            self.queue.finish(SAMPLES_QUEUE, item.id, self.worker_id, {"background": background})
        except WorkLeaseLost:
            logger.warning("samples worker %s lost %s", self.worker_id, item.id)
            return True
        except Exception as exc:
            logger.exception("samples worker %s could not make the samples of %s", self.worker_id, item.id)
            try:
                self.queue.fail(SAMPLES_QUEUE, item.id, self.worker_id, f"{type(exc).__name__}: {exc}", retryable=True)
            except WorkLeaseLost:
                logger.warning("samples worker %s lost %s", self.worker_id, item.id)
            return True
        if background:
            request_selection(self.queue, scenario, component, available_at=time.time() + BACKGROUND_POLL_SECONDS)
        return True


__all__ = [
    "BACKGROUND_POLL_SECONDS",
    "RECORD_SELECTION_KIND",
    "SAMPLES_QUEUE",
    "SampleProducers",
    "SamplesWorker",
    "request_selection",
    "selection_key",
]
