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


class QueueWorker(ServiceWorker):
    """A worker that takes the items of some kinds from one queue and runs each one, until it stops.

    Values: ``queue`` holds the items; ``queue_name`` and ``kinds`` name the items that it takes; ``worker_id``
    names it in leases; ``poll_seconds`` is the wait after a look that found no item, a positive number.

    Purpose: a service subclasses it and defines ``run_item``. Each round requeues the expired items, takes one
    item, starts it, extends its lease while ``run_item`` runs, and finishes the item with what ``run_item``
    returns. A ``ValueError`` that ``run_item`` raises fails the item for good, since its input does not change on a
    retry; any other error fails it as retryable. A lost lease leaves the item to the worker that holds it now. A
    store error ends the round, and the worker looks again after ``poll_seconds``.
    """

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
    def run_item(self, item: WorkItem) -> Mapping[str, JsonValue]:
        """Do the work of ``item`` and return its result."""

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
            with LeaseExtender(self.queue, item, self.worker_id):
                self.queue.start(self.queue_name, item.id, self.worker_id)
                result = self.run_item(item)
            self.queue.finish(self.queue_name, item.id, self.worker_id, dict(result))
        except WorkLeaseLost:
            logger.warning("worker %s lost %s", self.worker_id, item.id)
        except Exception as exc:
            logger.exception("worker %s could not run %s", self.worker_id, item.id)
            try:
                self.queue.fail(
                    self.queue_name,
                    item.id,
                    self.worker_id,
                    f"{type(exc).__name__}: {exc}",
                    retryable=not isinstance(exc, ValueError),
                )
            except WorkLeaseLost:
                logger.warning("worker %s lost %s", self.worker_id, item.id)
        return True


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
    "QueueWorker",
    "ServiceWorker",
    "ServiceWorkers",
    "service_components",
]
