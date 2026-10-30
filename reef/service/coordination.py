"""What assembly gives the services of one process: the coordination store and the background workers."""

from __future__ import annotations

from dataclasses import dataclass, field

from reef.execution.workers import SERVICE_COMPONENTS, ServiceWorkers, service_components
from reef.storage.coordination import Coordination


@dataclass(frozen=True)
class ServiceSetup:
    """What assembly gives the parts of a service that use the coordination store.

    Values: ``coordination`` is the process's coordination store; ``workers`` is the group that runs the
    process's background workers; ``components`` is the non-empty set of services (``SERVICE_COMPONENTS``) that
    this process runs.

    Purpose: assembly makes it once per process and passes it to the builders of the services.
    """

    coordination: Coordination
    workers: ServiceWorkers
    components: frozenset[str] = field(default_factory=lambda: frozenset(SERVICE_COMPONENTS))

    def __post_init__(self) -> None:
        service_components(self.components)


__all__ = ["ServiceSetup"]
