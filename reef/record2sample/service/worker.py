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

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from reef.execution.workers import QueueWorker
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue, request_waiting

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
    return request_waiting(
        queue,
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
    service looks again later.

    Purpose: ``SampleProducers.produce`` returns it; the samples worker writes it as the selection's result, and
    control wakes the trainer's worker only when ``changed`` is true.
    """

    changed: bool
    background: bool

    def __post_init__(self) -> None:
        if not isinstance(self.changed, bool) or not isinstance(self.background, bool):
            raise TypeError("changed and background are booleans")


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
    work in the background asks for the next selection after ``BACKGROUND_POLL_SECONDS``.
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
        return {"changed": produced.changed, "background": produced.background}

    def item_finished(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        # The next selection is a new item: the finished one keeps its ID until it ends, and an add of the same ID
        # changes nothing.
        if result["background"]:
            component = cast(str, item.payload["component"])
            request_selection(self.queue, item.scenario, component, available_at=time.time() + BACKGROUND_POLL_SECONDS)


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
