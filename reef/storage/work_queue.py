"""Work queues: the one protocol through which control gives work to the samples, training and evaluation services.

A producer adds a ``WorkItem``. A worker takes the oldest item it can take, holds a lease on it while it works,
extends the lease, adds progress, and finishes or fails it. An item whose lease expires goes back to the queue,
so another worker continues it. Readers find changes by ``changed_sequence``. The item ID is a function of the
item's inputs, so work that runs again writes the same result.

Two fields order the work:

- Items with the same ``serial_key`` run one at a time, in ``created_sequence`` order: a worker can take one only
  when every earlier item with that key is final. The training jobs of one trainer use this.
- A worker can take an item only at or after its ``available_at`` time. A polling item uses this to wait.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from reef.core.errors import ReefError
from reef.core.served_models import require_text

logger = logging.getLogger(__name__)

#: The state of a work item. ``finished`` and ``failed`` are final.
WorkState = Literal["queued", "assigned", "running", "finished", "failed"]
WORK_STATES: tuple[WorkState, ...] = ("queued", "assigned", "running", "finished", "failed")
FINAL_STATES: frozenset[str] = frozenset({"finished", "failed"})
HELD_STATES: frozenset[str] = frozenset({"assigned", "running"})

#: JSON-compatible values that a work item carries.
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class WorkConflict(ReefError):
    """An item with the same ID exists with a different payload."""


class WorkLeaseLost(ReefError):
    """The worker no longer holds the lease of the item, so it must stop work on it."""


@dataclass(frozen=True)
class WorkItem:
    """One item of work in a queue.

    Values:

    - ``queue`` and ``id`` are non-empty strings; ``id`` is unique in its queue and is a function of the inputs.
    - ``scenario`` names the scenario the work belongs to. ``kind`` is a non-empty string that workers filter on.
    - ``payload`` is a JSON object, the input of the work.
    - ``serial_key`` is None or a non-empty string (see the module docstring).
    - ``available_at`` and ``created_at`` are finite times in seconds.
    - ``state`` is one of ``WORK_STATES``. ``worker_id`` and ``lease_expires_at`` are set exactly when ``state`` is
      ``assigned`` or ``running``.
    - ``attempt_count`` is the number of times a worker took the item, an integer from 0.
    - ``progress`` is a tuple of JSON objects in the order a worker added them.
    - ``result`` is set exactly when ``state`` is ``finished``; ``error`` is a non-empty string set exactly when
      ``state`` is ``failed``, or the last retryable error of a ``queued`` item.
    - ``created_sequence`` and ``changed_sequence`` are positive integers; ``changed_sequence`` increases at each
      change.

    Purpose: a ``WorkQueue`` makes it; workers read the payload and progress, and readers read the result.
    """

    queue: str
    id: str
    scenario: str
    kind: str
    payload: Mapping[str, JsonValue]
    state: WorkState = "queued"
    serial_key: str | None = None
    available_at: float = 0.0
    created_at: float = 0.0
    attempt_count: int = 0
    worker_id: str | None = None
    lease_expires_at: float | None = None
    progress: tuple[Mapping[str, JsonValue], ...] = ()
    result: Mapping[str, JsonValue] | None = None
    error: str | None = None
    created_sequence: int = 1
    changed_sequence: int = 1

    def __post_init__(self) -> None:
        for value, name in ((self.queue, "queue"), (self.id, "id"), (self.scenario, "scenario"), (self.kind, "kind")):
            require_text(value, name)
        if not isinstance(self.payload, Mapping):
            raise TypeError("a work item's payload is a JSON object")
        if self.state not in WORK_STATES:
            raise ValueError(f"a work item's state is one of {WORK_STATES}, not {self.state!r}")
        if self.serial_key is not None:
            require_text(self.serial_key, "serial_key")
        for time_value, name in ((self.available_at, "available_at"), (self.created_at, "created_at")):
            if (
                isinstance(time_value, bool)
                or not isinstance(time_value, (int, float))
                or not math.isfinite(time_value)
            ):
                raise ValueError(f"{name} is a finite time")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int) or self.attempt_count < 0:
            raise ValueError("attempt_count is an integer from 0")
        held = self.state in HELD_STATES
        if (self.worker_id is not None) != held or (self.lease_expires_at is not None) != held:
            raise ValueError("a work item has a worker and a lease exactly while it is assigned or running")
        if (self.result is not None) != (self.state == "finished"):
            raise ValueError("a work item has a result exactly when it finished")
        if self.state == "failed" and not self.error:
            raise ValueError("a failed work item has an error")
        if self.state in ("finished", "assigned", "running") and self.error is not None:
            raise ValueError("only a queued or failed work item carries an error")
        if self.created_sequence < 1 or self.changed_sequence < self.created_sequence:
            raise ValueError("sequences are positive and changed_sequence is not before created_sequence")

    @property
    def final(self) -> bool:
        """Whether the item finished or failed, after which it does not change."""
        return self.state in FINAL_STATES


@dataclass(frozen=True)
class NewWork:
    """The inputs of ``WorkQueue.add``: an item that is not in a queue yet.

    Values: as the same fields of ``WorkItem``; ``available_at`` None means at once.

    Purpose: producers make it; ``add`` turns it into a queued ``WorkItem``.
    """

    queue: str
    id: str
    scenario: str
    kind: str
    payload: Mapping[str, JsonValue] = field(default_factory=dict)
    serial_key: str | None = None
    available_at: float | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.queue, "queue"), (self.id, "id"), (self.scenario, "scenario"), (self.kind, "kind")):
            require_text(value, name)
        if not isinstance(self.payload, Mapping):
            raise TypeError("a work item's payload is a JSON object")
        if self.serial_key is not None:
            require_text(self.serial_key, "serial_key")
        if self.available_at is not None and (
            isinstance(self.available_at, bool)
            or not isinstance(self.available_at, (int, float))
            or not math.isfinite(self.available_at)
        ):
            raise ValueError("available_at is a finite time or None")


class WorkQueue(ABC):
    """A set of work queues in one store.

    Values: the state is a finite set of ``WorkItem`` values with distinct ``(queue, id)`` pairs, and a sequence
    counter that numbers every change across the store. ``lease_seconds`` is the lease length that ``take`` and
    ``extend_lease`` give, and ``max_attempts`` the number of times a worker can take one item.

    Purpose: control and the services add work (``add``); workers take, extend, progress, finish and fail it;
    any service returns expired work (``requeue_expired``); readers follow it (``read_changes``, ``get``).
    """

    def __init__(self, *, lease_seconds: float = 60.0, max_attempts: Mapping[str, int] | None = None) -> None:
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds is a positive number")
        self.lease_seconds = float(lease_seconds)
        self.max_attempts: dict[str, int] = dict(max_attempts or {})
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.max_attempts.values()):
            raise ValueError("max_attempts maps queue names to positive integers")

    def attempts_allowed(self, queue: str) -> int:
        """How many times a worker can take one item of ``queue``; three unless configured."""
        return self.max_attempts.get(queue, 3)

    @abstractmethod
    def add(self, work: NewWork, *, now: float | None = None) -> WorkItem:
        """Add ``work`` as a queued item and return it. When an item with the same queue and ID exists, return it
        unchanged if its payload equals ``work.payload``, else raise ``WorkConflict``."""

    @abstractmethod
    def take(self, queue: str, worker_id: str, kinds: Sequence[str], *, now: float | None = None) -> WorkItem | None:
        """Assign the queued item of ``queue`` that ``worker_id`` can take with the lowest ``created_sequence`` and
        return it, or None when there is none.

        A worker can take an item when its ``kind`` is in ``kinds``, its ``available_at`` is not after ``now``,
        and every item with the same ``serial_key`` and a lower ``created_sequence`` is final. The item becomes
        ``assigned``, with ``worker_id``, ``lease_expires_at = now + lease_seconds``, and one more attempt.
        """

    @abstractmethod
    def extend_lease(self, queue: str, item_id: str, worker_id: str, *, now: float | None = None) -> WorkItem:
        """Set ``lease_expires_at = now + lease_seconds`` when ``worker_id`` holds the item; else raise
        ``WorkLeaseLost``."""

    @abstractmethod
    def start(self, queue: str, item_id: str, worker_id: str, *, now: float | None = None) -> WorkItem:
        """Change the held item from ``assigned`` to ``running``; ``WorkLeaseLost`` when ``worker_id`` does not hold
        it. A running item stays running."""

    @abstractmethod
    def add_progress(
        self,
        queue: str,
        item_id: str,
        worker_id: str,
        progress: Mapping[str, JsonValue],
        *,
        number: int,
        now: float | None = None,
    ) -> WorkItem:
        """Append ``progress`` as progress number ``number`` (0-based) of the held item. Adding a number that the
        item already has changes nothing, so a worker that continues the item can report again safely; a number
        beyond the next one raises ``ValueError``. ``WorkLeaseLost`` when ``worker_id`` does not hold the item."""

    @abstractmethod
    def finish(
        self, queue: str, item_id: str, worker_id: str, result: Mapping[str, JsonValue], *, now: float | None = None
    ) -> WorkItem:
        """Make the held item ``finished`` with ``result``; ``WorkLeaseLost`` when ``worker_id`` does not hold it."""

    @abstractmethod
    def fail(
        self,
        queue: str,
        item_id: str,
        worker_id: str,
        error: str,
        *,
        retryable: bool,
        now: float | None = None,
    ) -> WorkItem:
        """End the held attempt with ``error``: the item becomes ``queued`` when ``retryable`` and it has attempts
        left, else ``failed``. ``WorkLeaseLost`` when ``worker_id`` does not hold it."""

    @abstractmethod
    def cancel(self, queue: str, item_id: str, error: str) -> WorkItem | None:
        """Make an item that is not final ``failed`` with ``error``, for the producer that added it. A worker that
        held it gets ``WorkLeaseLost`` at its next call and stops. Return the item, or None when it does not exist;
        a final item does not change."""

    @abstractmethod
    def requeue_expired(self, *, now: float | None = None) -> tuple[WorkItem, ...]:
        """Return each held item whose lease expired to ``queued``, or make it ``failed`` when it has no attempts
        left; return the changed items."""

    @abstractmethod
    def get(self, queue: str, item_id: str) -> WorkItem | None:
        """The item, or None."""

    @abstractmethod
    def read_changes(self, queue: str, after_sequence: int, *, limit: int = 1000) -> tuple[WorkItem, ...]:
        """The items of ``queue`` with ``changed_sequence > after_sequence``, in that order, at most ``limit``."""

    @abstractmethod
    def items(self, queue: str, *, scenario: str | None = None, kind: str | None = None) -> tuple[WorkItem, ...]:
        """The items of ``queue``, optionally of one scenario and kind, in ``created_sequence`` order."""


def clock(now: float | None) -> float:
    """``now``, or the current time when it is None."""
    return time.time() if now is None else float(now)


def request_waiting(
    queue: WorkQueue,
    queue_name: str,
    scenario: str,
    kind: str,
    key: str,
    payload: Mapping[str, JsonValue],
    *,
    available_at: float | None = None,
) -> WorkItem:
    """Ask for one run of the work that ``key`` names; return the item that will do it: a queued one, or a new one.

    The items of ``key`` share it as their ``serial_key``, so they run one at a time, in order. An item reads the
    state that it works on when it runs, so a queued item covers every request made while it waits: a request adds
    nothing then. Otherwise it adds ``<key>:<n>``, where ``n`` counts the earlier items of ``key``; two requests at
    once compute the same ID, so they add one item.
    """
    earlier = [item for item in queue.items(queue_name, scenario=scenario, kind=kind) if item.serial_key == key]
    waiting = [item for item in earlier if item.state == "queued"]
    if waiting:
        return waiting[0]
    return queue.add(
        NewWork(
            queue_name, f"{key}:{len(earlier)}", scenario, kind, payload, serial_key=key, available_at=available_at
        )
    )


class LeaseExtender:
    """Extends the lease of one held item every third of the lease, on a background thread, while its work runs.

    Values: ``queue`` holds the item; ``item`` is the item that ``worker_id`` holds; ``done`` ends the extension.
    The thread ends at ``stop`` or when the worker no longer holds the item, and then sets ``lease_lost``; a failed
    extension that is no lost lease is logged and tried again at the next third.

    Purpose: a worker opens one around work that can take longer than the lease (``with LeaseExtender(...)``).
    """

    def __init__(self, queue: WorkQueue, item: WorkItem, worker_id: str) -> None:
        self.queue = queue
        self.item = item
        self.worker_id = worker_id
        self.done = threading.Event()
        #: Set when the worker no longer holds the item: another worker took it, or control cancelled it.
        self.lease_lost = threading.Event()
        self.thread = threading.Thread(target=self.run, name=f"reef-lease-{item.id}", daemon=True)

    def __enter__(self) -> LeaseExtender:
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        self.done.set()
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join()

    def run(self) -> None:
        while not self.done.wait(self.queue.lease_seconds / 3):
            try:
                self.queue.extend_lease(self.item.queue, self.item.id, self.worker_id)
            except WorkLeaseLost:
                self.lease_lost.set()
                return
            except Exception:
                logger.exception("could not extend the lease of %s", self.item.id)


__all__ = [
    "FINAL_STATES",
    "HELD_STATES",
    "WORK_STATES",
    "JsonValue",
    "LeaseExtender",
    "NewWork",
    "WorkConflict",
    "WorkItem",
    "WorkLeaseLost",
    "WorkQueue",
    "WorkState",
    "clock",
    "request_waiting",
]
