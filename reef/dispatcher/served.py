"""Control's writer of ``served_models``: what each loaded scenario serves, for the inference service to read.

The active control instance (the holder of the control lease) runs one ``ServedModelsPublisher``. Each round, it
writes the ``ServedArtifact`` of each loaded scenario's active artifact under ``ScenarioModel(scenario, ACTIVE)``,
when the scenario's runtime reaches the artifact from outside the process (``InferenceRuntime.served_artifact``).
When a round replaces or removes a row, the old value keeps serving for ``stop_delay_seconds`` (a
``ScheduledStop``): a request that read the old row before the change ends on it. A standby instance only tries to
take the lease.
"""

from __future__ import annotations

import logging
import threading
import time

from reef.core.served_models import ACTIVE, ScenarioModel, ScheduledStop, ServedArtifact
from reef.execution.workers import ServiceWorker
from reef.scenario.registry import ScenarioRegistry
from reef.storage.coordination import Coordination

logger = logging.getLogger(__name__)

#: The seconds an inference replica caches ``served_models`` at most.
CACHE_SECONDS = 5.0


class ServedModelsPublisher(ServiceWorker):
    """Writes ``served_models`` for the scenarios that this control instance holds.

    Values: ``registry`` holds the loaded scenarios; ``coordination`` holds ``served_models`` and the control lease;
    ``instance_id`` names this control instance in the lease; ``stop_delay_seconds`` is how long a replaced value
    keeps serving, at least ``CACHE_SECONDS`` plus the longest request; ``interval_seconds`` is the time between
    rounds.

    Purpose: the dispatcher of a control instance runs it; the inference service reads what it writes.
    """

    def __init__(
        self,
        registry: ScenarioRegistry,
        coordination: Coordination,
        *,
        instance_id: str,
        stop_delay_seconds: float,
        interval_seconds: float = 1.0,
    ) -> None:
        if stop_delay_seconds < CACHE_SECONDS:
            raise ValueError(f"stop_delay_seconds is at least the replica cache time, {CACHE_SECONDS} s")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds is positive")
        self.registry = registry
        self.coordination = coordination
        self.instance_id = instance_id
        self.stop_delay_seconds = stop_delay_seconds
        self.interval_seconds = interval_seconds
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name="reef-served-models", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("could not write served_models")
            self.stopped.wait(self.interval_seconds)

    def desired(self, scenario: str) -> dict[str, ServedArtifact]:
        """The rows that ``scenario`` should have now, by name; empty when it is not loaded here."""
        current = self.registry.get_loaded(scenario)
        if current is None or current.runtime is None:
            return {}
        artifact_id = current.current_artifact_ref().release_id
        served = current.runtime.served_artifact(artifact_id, scenario)
        return {} if served is None else {ACTIVE: served}

    def run_once(self, *, now: float | None = None) -> bool:
        """One round; return whether this instance holds the lease and wrote the table."""
        moment = time.time() if now is None else now
        state = self.coordination.control_lease.acquire(self.instance_id, now=moment)
        if state is None:
            return False
        table = self.coordination.served_models
        number = state.number
        for current in self.registry.loaded_scenarios():
            wanted = self.desired(current.name)
            present = table.read_scenario(current.name)
            for name in sorted(set(present) | set(wanted)):
                old, new = present.get(name), wanted.get(name)
                if old == new:
                    continue
                key = ScenarioModel(current.name, name)
                if new is None:
                    table.remove(key, control_lease_number=number)
                else:
                    table.write(key, new, control_lease_number=number)
                if old is not None and not table.is_referenced(old):
                    table.schedule_stop(
                        ScheduledStop(old, moment + self.stop_delay_seconds), control_lease_number=number
                    )
        for stop in table.due_stops(moment):
            # A runtime frees the weights it replaced when it loads new ones, so the stop only clears the row.
            table.delete_stop(stop.served, control_lease_number=number)
        return True


__all__ = ["CACHE_SECONDS", "ServedModelsPublisher"]
