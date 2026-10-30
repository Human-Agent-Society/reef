"""What assembly gives the services of one process: the coordination store and the background workers."""

from __future__ import annotations

from dataclasses import dataclass

from reef.execution.workers import ServiceWorkers
from reef.storage.coordination import Coordination


@dataclass(frozen=True)
class ServiceSetup:
    """What assembly gives the parts of a service that use the coordination store.

    Values: ``coordination`` is the process's coordination store; ``workers`` is the group that runs the
    process's background workers; ``runs_training`` is whether this process runs the training service's workers.

    Purpose: assembly makes it once per process and passes it to the builders of the services.
    """

    coordination: Coordination
    workers: ServiceWorkers
    runs_training: bool = True


__all__ = ["ServiceSetup"]
