"""Background workers that a service runs, and the group that starts and stops them with the process.

A service that takes work from a queue (the training, sample and evaluation services) runs one or more workers
on background threads. A process registers the workers of the services it runs in one ``ServiceWorkers`` group,
which starts them with the process and stops them, in reverse order, before the process closes what they use.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from reef.storage.work_queue import JsonValue, LeaseExtender, WorkItem, WorkLeaseLost, WorkQueue

logger = logging.getLogger(__name__)

#: The services of a deployment. ``reef serve --component NAME`` runs one in a process; the one-process layout runs
#: all of them in one process, on the same interfaces.
SERVICE_COMPONENTS = ("control", "inference", "samples", "training", "evaluation", "data")


def service_components(names: object) -> frozenset[str]:
    """``names`` as a set of service names; raise ``ValueError`` for an empty set or an unknown name."""
    if isinstance(names, str) or not isinstance(names, (set, frozenset, tuple, list)) or not names:
        raise ValueError(f"name one or more services of {SERVICE_COMPONENTS}")
    unknown = sorted(str(name) for name in names if name not in SERVICE_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown services {unknown}; the services are {SERVICE_COMPONENTS}")
    return frozenset(names)


@dataclass(frozen=True)
class ServiceSet:
    """The services that one process runs.

    Values: ``names`` is a non-empty subset of ``SERVICE_COMPONENTS`` (``service_components``). Each ``runs_*``
    property is whether ``names`` holds that service.

    Purpose: assembly makes it for a process (``ServiceSet.of``); the dispatcher, the routes and the apps read the
    ``runs_*`` properties instead of the service names.
    """

    names: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", service_components(self.names))

    @classmethod
    def of(cls, names: object) -> ServiceSet:
        """The set of ``names``; ``ValueError`` as ``service_components``."""
        return cls(service_components(names))

    @classmethod
    def every(cls) -> ServiceSet:
        """Every service: the one-process layout."""
        return cls(frozenset(SERVICE_COMPONENTS))

    @property
    def runs_control(self) -> bool:
        return "control" in self.names

    @property
    def runs_inference(self) -> bool:
        return "inference" in self.names

    @property
    def runs_samples(self) -> bool:
        return "samples" in self.names

    @property
    def runs_training(self) -> bool:
        return "training" in self.names

    @property
    def runs_evaluation(self) -> bool:
        return "evaluation" in self.names

    @property
    def runs_data(self) -> bool:
        return "data" in self.names


class ServiceWorker(ABC):
    """A worker that a service runs in the background until it stops.

    Purpose: ``ServiceWorkers`` starts it with the process (``start``) and stops it before the process ends
    (``stop``). ``stop`` waits for the worker's thread and releases what the worker owns.
    """

    @abstractmethod
    def start(self) -> None:
        """Start working on a background thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop, wait for the thread, and release what the worker owns."""


class WorkerStopped(Exception):
    """``run_item`` stopped because its worker stops: the item goes back to the queue, and another worker runs it."""


class ItemFailed(Exception):
    """The work of an item ran and failed, so a retry fails again: the item fails for good with this message."""


class QueueWorker(ServiceWorker):
    """A worker that takes the items of some kinds from one queue and runs each one, until it stops.

    Values: ``queue`` holds the items; ``queue_name`` and ``kinds`` name the items that it takes; ``worker_id``
    names it in leases; ``poll_seconds`` is the wait after a look that found no item, a positive number.
    ``permanent_errors`` are the errors of ``run_item`` that a retry cannot fix, since the item's input stays the
    same: ``ValueError`` by default, and a subclass names its own.

    Purpose: a service subclasses it and defines ``run_item``. Each round requeues the expired items, takes one
    item, starts it, extends its lease while ``run_item`` runs, finishes the item with what ``run_item`` returns,
    and then calls ``item_finished``. ``run_item`` gets the event that the lease extension sets when the worker no
    longer holds the item, so that long work can stop. Its errors end the item as follows:

    - ``ItemFailed``: the item fails for good, with the error's message;
    - one of ``permanent_errors``: the item fails for good;
    - ``WorkerStopped``: the item goes back to the queue at once;
    - ``WorkLeaseLost``: the item stays with the worker that holds it now;
    - any other error: the item fails as retryable, until it uses up its attempts.

    A store error ends the round, and the worker looks again after ``poll_seconds``.
    """

    permanent_errors: tuple[type[Exception], ...] = (ValueError,)

    def __init__(
        self,
        queue: WorkQueue,
        queue_name: str,
        kinds: Sequence[str],
        *,
        worker_id: str,
        poll_seconds: float = 0.2,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds is positive")
        if not kinds:
            raise ValueError("a queue worker takes one or more kinds")
        self.queue = queue
        self.queue_name = queue_name
        self.kinds = tuple(kinds)
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    @abstractmethod
    def run_item(self, item: WorkItem, lease_lost: threading.Event) -> Mapping[str, JsonValue]:
        """Do the work of ``item`` and return its result; ``lease_lost`` is set when the worker no longer holds
        the item."""

    def item_finished(self, item: WorkItem, result: Mapping[str, JsonValue]) -> None:
        """React to ``item``, which finished with ``result``; the default does nothing."""
        return

    def start(self) -> None:
        if self.thread is not None:
            return
        self.stopped.clear()
        self.thread = threading.Thread(target=self.run, name=f"reef-{self.worker_id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                took = self.run_once()
            except Exception:
                logger.exception("worker %s could not read the %s queue", self.worker_id, self.queue_name)
                took = False
            if not took:
                self.stopped.wait(self.poll_seconds)

    def run_once(self) -> bool:
        """Take one item and run it; return whether there was one."""
        self.queue.requeue_expired()
        item = self.queue.take(self.queue_name, self.worker_id, list(self.kinds))
        if item is None:
            return False
        try:
            with LeaseExtender(self.queue, item, self.worker_id) as extender:
                self.queue.start(self.queue_name, item.id, self.worker_id)
                result = self.run_item(item, extender.lease_lost)
            self.queue.finish(self.queue_name, item.id, self.worker_id, dict(result))
        except WorkLeaseLost:
            logger.warning("worker %s lost %s", self.worker_id, item.id)
            return True
        except WorkerStopped:
            self.fail(item, "the worker stopped", retryable=True)
            return True
        except ItemFailed as exc:
            self.fail(item, str(exc), retryable=False)
            return True
        except Exception as exc:
            logger.exception("worker %s could not run %s", self.worker_id, item.id)
            self.fail(item, f"{type(exc).__name__}: {exc}", retryable=not isinstance(exc, self.permanent_errors))
            return True
        self.item_finished(item, result)
        return True

    def fail(self, item: WorkItem, error: str, *, retryable: bool) -> None:
        """Fail ``item`` with ``error``; nothing happens when another worker holds it now."""
        try:
            self.queue.fail(self.queue_name, item.id, self.worker_id, error, retryable=retryable)
        except WorkLeaseLost:
            logger.warning("worker %s lost %s", self.worker_id, item.id)


@dataclass
class ServiceWorkers:
    """The background workers of the services in one process.

    Values: ``workers`` in the order they were added; ``started`` is whether the group runs.

    Purpose: assembly adds the workers of the services this process runs; the process starts the group once it
    can serve and stops it before it closes the stores the workers use.
    """

    workers: list[ServiceWorker] = field(default_factory=list)
    started: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, worker: ServiceWorker) -> None:
        """Add ``worker``; start it at once when the group runs."""
        with self.lock:
            self.workers.append(worker)
            if self.started:
                worker.start()

    def start(self) -> None:
        """Start every worker that is not running; a second call does nothing."""
        with self.lock:
            if self.started:
                return
            self.started = True
            for worker in self.workers:
                worker.start()

    def stop(self) -> None:
        """Stop every worker, the last added first, and empty the group."""
        with self.lock:
            workers, self.workers, self.started = list(self.workers), [], False
        for worker in reversed(workers):
            worker.stop()


__all__ = [
    "SERVICE_COMPONENTS",
    "ItemFailed",
    "QueueWorker",
    "ServiceSet",
    "ServiceWorker",
    "ServiceWorkers",
    "WorkerStopped",
    "service_components",
]
