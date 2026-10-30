"""Background workers that a service runs, and the group that starts and stops them with the process.

A service that takes work from a queue (the training, sample and evaluation services) runs one or more workers
on background threads. A process registers the workers of the services it runs in one ``ServiceWorkers`` group,
which starts them with the process and stops them, in reverse order, before the process closes what they use.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

#: The services of a deployment. ``reef serve --component NAME`` runs one in a process; the one-process layout runs
#: all of them in one process, on the same interfaces.
SERVICE_COMPONENTS = ("control", "inference", "samples", "training", "evaluation")


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
    "ServiceWorker",
    "ServiceWorkers",
    "service_components",
]
