"""The samples service: it turns a scenario's new records into samples, apart from training.

Control does not make samples on the training path. When a scenario accepts records, control adds a
``record_selection`` item for each trainer that trains to the ``samples`` queue. A ``SamplesWorker`` takes the
item and has a ``SampleProducers`` make the samples: the trainer's record-to-sample stage selects and groups the
records and makes samples of the record sets that it forms, and the samples land in the sample store. Control's trainer reads the sample store and builds jobs from it. Slow stages (judges, generated tasks and
their rollouts) never hold the training path.

A ``record_selection`` item:

- ``id`` is ``record_selection:<scenario>:<component>:<number>``, where ``number`` counts the earlier selections of
  the same trainer; two requests at once compute the same ID, so they add one item;
- ``serial_key`` is ``record_selection:<scenario>:<component>``, so one worker at a time selects the records of one
  trainer, in order;
- ``payload`` is ``{"scenario", "component"}``.

A selection reads every record that its trainer has not used yet when it runs. So a request adds nothing while a
selection of the same trainer waits in the queue; it adds one when the trainer's selections are final or running.
A stage can also make samples without a new record: when background work ends, or when a set that waits for time
is due. The worker then asks for the next selection at that time.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from reef.execution.workers import QueueWorker
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue

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
    payload = cast(dict[str, JsonValue], {"scenario": scenario, "component": component})
    return queue.request(
        SAMPLES_QUEUE,
        scenario,
        RECORD_SELECTION_KIND,
        selection_key(scenario, component),
        payload,
        available_at=available_at,
    )


@dataclass(frozen=True)
class ProducedSamples:
    """What one record selection did.

    Values: ``changed`` is whether it gave the trainer something new to build a job from: a new stored sample or a
    new training instruction; ``background`` is whether background work of the stages still runs, so the samples
    service looks again later. ``due_in_seconds`` is a non-negative finite number: the seconds until a set that waits
    for time alone is due, such as a trajectory at the end of its idle time. None means that no set waits for time.

    Counts describe this completed attempt only, not earlier failed attempts. None means unavailable.
    ``records_read`` counts replayed records, including ignored records; ``samples_created`` counts new inserts.

    Purpose: ``SampleProducers.produce`` returns it; the samples worker writes it as the selection's result, and
    control wakes the trainer's worker only when ``changed`` is true.
    """

    changed: bool
    background: bool
    records_read: int | None = None
    samples_created: int | None = None
    due_in_seconds: float | None = None

    def __post_init__(self) -> None:
        for count in (self.records_read, self.samples_created):
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError("sample processing counts must be non-negative integers or None")
        due = self.due_in_seconds
        if due is not None and (isinstance(due, bool) or not isinstance(due, (int, float)) or not 0 <= due < math.inf):
            raise ValueError("due_in_seconds must be a non-negative finite number or None")
        if not isinstance(self.changed, bool) or not isinstance(self.background, bool):
            raise TypeError("changed and background are booleans")
        if self.samples_created and not self.changed:
            raise ValueError("new samples must change the trainer's input")


class SampleProducers(ABC):
    """Makes the samples of one trainer from its new records.

    Purpose: a ``SamplesWorker`` calls ``produce`` for each ``record_selection`` item it takes.
    """

    @abstractmethod
    def produce(self, scenario: str, component: str) -> ProducedSamples:
        """Make samples of the records that the trainer of ``component`` in ``scenario`` has not used yet, and store
        them; return what the selection did. A scenario that is gone, or a trainer that trains nothing, makes
        nothing."""


class SamplesWorker(QueueWorker):
    """One worker of the samples service.

    Values: as ``QueueWorker``, on the record selections of the samples queue; ``producers`` makes the samples.
    A selection that names no component fails for good.

    Purpose: the samples service starts one or more (``start``) and stops them (``stop``). A selection that left
    work in the background asks for the next selection after ``BACKGROUND_POLL_SECONDS``. A selection that left
    only a set that waits for time asks for the next selection after ``due_in_seconds``.
    """

    def __init__(
        self, queue: WorkQueue, producers: SampleProducers, *, worker_id: str, poll_seconds: float = 0.2
    ) -> None:
        super().__init__(queue, SAMPLES_QUEUE, [RECORD_SELECTION_KIND], worker_id=worker_id, poll_seconds=poll_seconds)
        self.producers = producers

    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        component = item.payload.get("component")
        if not isinstance(component, str) or not component:
            raise ValueError("a record selection names its component")
        produced = self.producers.produce(item.scenario, component)
        return {
            "changed": produced.changed,
            "background": produced.background,
            "records_read": produced.records_read,
            "samples_created": produced.samples_created,
            "due_in_seconds": produced.due_in_seconds,
        }

    def item_finished(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        # The next selection is a new item: the finished one keeps its ID until it ends, and an add of the same ID
        # changes nothing.
        if result["background"]:
            delay = BACKGROUND_POLL_SECONDS
        elif result["due_in_seconds"] is not None:
            delay = cast(float, result["due_in_seconds"])
        else:
            return
        component = cast(str, item.payload["component"])
        request_selection(self.queue, item.scenario, component, available_at=time.time() + delay)


__all__ = [
    "BACKGROUND_POLL_SECONDS",
    "RECORD_SELECTION_KIND",
    "SAMPLES_QUEUE",
    "ProducedSamples",
    "SampleProducers",
    "SamplesWorker",
    "request_selection",
    "selection_key",
]
