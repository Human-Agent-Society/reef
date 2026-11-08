"""The samples service: it turns a scenario's new records into samples, apart from training.

Control does not make samples on the training path. When a scenario accepts records, and after each commit, control
adds a ``record_selection`` item for each trainer that trains to the ``samples`` queue. A ``SamplesWorker`` takes the
item and has a ``SampleProducers`` make the samples: the trainer's record-to-sample stage selects and groups the
records and makes samples of the record sets that it forms, and the samples become data items that the scenario uses. Control's trainer reads them and builds jobs from them. Slow stages (judges, generated tasks and
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

Each request names its trigger (``TRIGGERS``). The item records the time of the first request of each trigger in
``metadata["trigger:<name>"]``, so a reader can tell why the selection ran (RFC #1267). While a selection runs, its
counts so far are in ``metadata["progress"]``.

While only background work runs, the next selection waits twice as long as the last one, from
``BACKGROUND_POLL_SECONDS`` up to ``MAX_BACKGROUND_POLL_SECONDS``. A selection that changed something resets the wait.
The wait is in ``metadata["background_poll_seconds"]`` of the next selection.
"""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from reef.execution.workers import QueueWorker
from reef.storage.work_queue import JsonValue, WorkItem, WorkQueue

#: The queue of sample work.
SAMPLES_QUEUE = "samples"
#: The kind of the item that makes samples of a trainer's new records.
RECORD_SELECTION_KIND = "record_selection"
#: The first wait, in seconds, until a worker looks again at a trainer whose stages still run background work.
BACKGROUND_POLL_SECONDS = 1.0
#: The longest wait, in seconds, between two looks at background work that changed nothing.
MAX_BACKGROUND_POLL_SECONDS = 30.0
#: The sources that request a selection: accepted records, a data sync, a commit, a scenario reload, background
#: work that still runs, and a set that waits for time.
TRIGGERS = ("records", "data", "commit", "reload", "background", "due")
#: The shortest time, in seconds, between two progress writes of one selection.
PROGRESS_INTERVAL_SECONDS = 1.0

#: Reports the counts of a running selection so far: the records read and the samples created.
SampleProgress = Callable[[int, int], None]


def selection_key(scenario: str, component: str) -> str:
    """The ``serial_key`` of the record selection of one trainer."""
    return f"{RECORD_SELECTION_KIND}:{scenario}:{component}"


def request_selection(
    queue: WorkQueue, scenario: str, component: str, *, trigger: str, available_at: float | None = None
) -> WorkItem:
    """Ask for a selection of the new records of the trainer of ``component`` in ``scenario`` because of ``trigger``
    (one of ``TRIGGERS``); return the item that will read them: a queued one, or a new one."""
    if trigger not in TRIGGERS:
        raise ValueError(f"a selection trigger is one of {TRIGGERS}, not {trigger!r}")
    payload = cast(dict[str, JsonValue], {"scenario": scenario, "component": component})
    item = queue.request(
        SAMPLES_QUEUE,
        scenario,
        RECORD_SELECTION_KIND,
        selection_key(scenario, component),
        payload,
        available_at=available_at,
    )
    key = f"trigger:{trigger}"
    if key in item.metadata:
        return item
    return queue.set_metadata(SAMPLES_QUEUE, item.id, key, time.time()) or item


def selection_triggers(item: WorkItem) -> list[JsonValue]:
    """The triggers of the selection ``item``, each ``{"name", "at"}``, oldest first; empty for an older item."""
    found = sorted(
        (at, name.removeprefix("trigger:"))
        for name, at in item.metadata.items()
        if name.startswith("trigger:") and isinstance(at, (int, float))
    )
    return [{"name": name, "at": at} for at, name in found]


@dataclass(frozen=True)
class ProducedSamples:
    """What one record selection did.

    Values: ``changed`` is whether it gave the trainer something new to build a job from: a new stored sample or a
    new training request; ``background`` is whether background work of the stages still runs, so the samples
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
    def produce(self, scenario: str, component: str, progress: SampleProgress) -> ProducedSamples:
        """Make samples of the records that the trainer of ``component`` in ``scenario`` has not used yet, and store
        them; return what the selection did. Call ``progress`` with the counts so far while the work runs. A
        scenario that is gone, or a trainer that trains nothing, makes nothing."""


class SamplesWorker(QueueWorker):
    """One worker of the samples service.

    Values: as ``QueueWorker``, on the record selections of the samples queue; ``producers`` makes the samples.
    A selection that names no component fails for good.

    Purpose: the samples service starts one or more (``start``) and stops them (``stop``). A selection that left
    work in the background asks for the next selection after the backoff wait (``background_poll_seconds``). A
    selection that left only a set that waits for time asks for the next selection after ``due_in_seconds``.
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
        produced = self.producers.produce(item.scenario, component, self.progress_writer(item))
        return {
            "changed": produced.changed,
            "background": produced.background,
            "records_read": produced.records_read,
            "samples_created": produced.samples_created,
            "due_in_seconds": produced.due_in_seconds,
        }

    def progress_writer(self, item: WorkItem) -> SampleProgress:
        """A ``SampleProgress`` that writes the counts of this attempt of ``item`` to its metadata, at most once in
        ``PROGRESS_INTERVAL_SECONDS``. The result holds the final counts."""
        last_write = -math.inf

        def write(records_read: int, samples_created: int) -> None:
            nonlocal last_write
            now = time.monotonic()
            if now - last_write < PROGRESS_INTERVAL_SECONDS:
                return
            last_write = now
            progress: dict[str, JsonValue] = {
                "attempt": item.attempt_count,
                "records_read": records_read,
                "samples_created": samples_created,
                "at": time.time(),
            }
            self.queue.set_metadata(SAMPLES_QUEUE, item.id, "progress", progress)

        return write

    def item_finished(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        # The next selection is a new item: the finished one keeps its ID until it ends, and an add of the same ID
        # changes nothing.
        component = cast(str, item.payload["component"])
        if result["background"]:
            delay = background_poll_seconds(item, changed=bool(result["changed"]))
            next_item = request_selection(
                self.queue, item.scenario, component, trigger="background", available_at=time.time() + delay
            )
            self.queue.set_metadata(SAMPLES_QUEUE, next_item.id, "background_poll_seconds", delay)
        elif result["due_in_seconds"] is not None:
            delay = cast(float, result["due_in_seconds"])
            request_selection(self.queue, item.scenario, component, trigger="due", available_at=time.time() + delay)


def background_poll_seconds(item: WorkItem, *, changed: bool) -> float:
    """The wait before the next look at background work after the selection ``item``: ``BACKGROUND_POLL_SECONDS``
    after a change, else twice the wait before ``item``, at most ``MAX_BACKGROUND_POLL_SECONDS``."""
    last = item.metadata.get("background_poll_seconds")
    if changed or not isinstance(last, (int, float)):
        return BACKGROUND_POLL_SECONDS
    return min(2 * float(last), MAX_BACKGROUND_POLL_SECONDS)


__all__ = [
    "BACKGROUND_POLL_SECONDS",
    "MAX_BACKGROUND_POLL_SECONDS",
    "RECORD_SELECTION_KIND",
    "SAMPLES_QUEUE",
    "TRIGGERS",
    "ProducedSamples",
    "SampleProducers",
    "SampleProgress",
    "SamplesWorker",
    "background_poll_seconds",
    "request_selection",
    "selection_key",
    "selection_triggers",
]
