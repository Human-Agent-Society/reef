"""Control's writer of ``served_models``: what each loaded scenario serves, for the inference service to read.

Control runs as one active instance and standby instances. A control instance starts only once it holds the
control lease (``wait_for_control_lease``); until then it serves nothing, so a load balancer sends requests to the
active instance alone. The active instance runs one ``ServedModelsPublisher``, which also extends the lease. Each round, it
writes the ``ServedArtifact`` of each loaded scenario's active artifact under ``ScenarioModel(scenario, ACTIVE)``,
when the scenario's runtime reaches the artifact from outside the process (``InferenceRuntime.served_artifact``).
The artifact is the active ``artifact_id`` of the component that the runtime serves: the weight trainer's
component, or the scenario's only component. When a round replaces or removes a row, the old value keeps serving for ``stop_delay_seconds`` (a
``ScheduledStop``): a request that read the old row before the change ends on it. A standby instance only tries to
take the lease. An instance that loses the lease stops its training work (``ControlInstance``); a stopped instance
releases the lease, so that a standby takes it at once.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

from reef.core.served_models import ACTIVE, ScenarioModel, ScheduledStop, ServedArtifact
from reef.execution.workers import ServiceWorker
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.storage.control_lease import ControlLease, ControlLeaseState
from reef.storage.coordination import Coordination

logger = logging.getLogger(__name__)

#: The seconds an inference replica caches ``served_models`` at most.
CACHE_SECONDS = 5.0


def wait_for_control_lease(
    lease: ControlLease, instance_id: str, *, poll_seconds: float | None = None
) -> ControlLeaseState:
    """Take the control lease for ``instance_id``, waiting while another instance holds a lease that has not
    expired; return the state. ``poll_seconds`` is the time between tries, a third of the lease by default."""
    interval = lease.lease_seconds / 3 if poll_seconds is None else poll_seconds
    reported = False
    while (state := lease.acquire(instance_id)) is None:
        if not reported:
            holder = lease.current()
            logger.info(
                "standby: control instance %s waits for the control lease held by %s",
                instance_id,
                None if holder is None else holder.instance_id,
            )
            reported = True
        time.sleep(interval)
    logger.info("control instance %s holds the control lease, number %d", instance_id, state.number)
    return state


class ControlInstance(ABC):
    """What a control instance does when it loses the control lease.

    Purpose: the dispatcher implements it; ``ServedModelsPublisher`` calls it.
    """

    @abstractmethod
    def control_lease_lost(self) -> None:
        """Another instance took the lease: stop the work that only the active instance does."""


class ServedModelsPublisher(ServiceWorker):
    """Writes ``served_models`` for the scenarios that this control instance holds.

    Values: ``registry`` holds the loaded scenarios; ``coordination`` holds ``served_models`` and the control lease;
    ``instance_id`` names this control instance in the lease, and ``lease`` is the lease that it took
    (``wait_for_control_lease``); ``instance`` is told when the lease is lost;
    ``stop_delay_seconds`` is how long a replaced value keeps serving, at least ``CACHE_SECONDS`` plus the longest
    request; ``interval_seconds`` is the time between rounds, less than the lease. ``held`` is whether this
    instance still holds the lease, under ``lease_number``; it starts true.

    Purpose: the dispatcher of a control instance starts it as soon as it takes the lease, so the lease is extended
    while the rest of the instance starts; the inference service reads what it writes.
    """

    def __init__(
        self,
        registry: ScenarioRegistry,
        coordination: Coordination,
        instance: ControlInstance,
        lease: ControlLeaseState,
        *,
        instance_id: str,
        stop_delay_seconds: float,
        interval_seconds: float = 1.0,
    ) -> None:
        if stop_delay_seconds < CACHE_SECONDS:
            raise ValueError(f"stop_delay_seconds is at least the replica cache time, {CACHE_SECONDS} s")
        if not 0 < interval_seconds < coordination.control_lease.lease_seconds:
            raise ValueError("interval_seconds is positive and less than the control lease")
        self.registry = registry
        self.coordination = coordination
        self.instance = instance
        if lease.instance_id != instance_id:
            raise ValueError(f"the control lease given to {instance_id} is held by {lease.instance_id}")
        self.held = True
        self.lease_number = lease.number
        self.instance_id = instance_id
        self.stop_delay_seconds = stop_delay_seconds
        self.interval_seconds = interval_seconds
        #: Serializes the writes of a commit and of a round.
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.run, name="reef-served-models", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the rounds and release the lease, so that a standby instance takes it at once."""
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.held:
            self.coordination.control_lease.release(self.instance_id)
            self.held = False

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
        status = current.artifact_status()
        # The runtime serves the weights: the component of the weight trainer, or the only component.
        weights = [item.component for item in current.trainers if item.trainer.job_loop is not None]
        names = weights or (list(status) if len(status) == 1 else [])
        artifact_id = status[names[0]]["active"] if names else None
        if artifact_id is None:
            return {}
        served = current.runtime.served_artifact(artifact_id, scenario)
        return {} if served is None else {ACTIVE: served}

    def publish(self, current: Scenario, *, now: float | None = None) -> None:
        """Write the rows of ``current`` now, as an activation does in the same step as its commit, so a request
        after the commit reads the new value. Nothing happens when this instance does not hold the lease; a lease
        that another instance took meanwhile refuses the write (``ControlLeaseConflict``)."""
        with self.lock:
            if self.held:
                self.write_rows(current, time.time() if now is None else now)

    def write_rows(self, current: Scenario, moment: float) -> None:
        """Write the difference between ``current``'s rows and what it serves; the caller holds ``lock``."""
        table = self.coordination.served_models
        wanted = self.desired(current.name)
        present = table.read_scenario(current.name)
        for name in sorted(set(present) | set(wanted)):
            old, new = present.get(name), wanted.get(name)
            if old == new:
                continue
            key = ScenarioModel(current.name, name)
            if new is None:
                table.remove(key, control_lease_number=self.lease_number)
            else:
                table.write(key, new, control_lease_number=self.lease_number)
            if old is not None and not table.is_referenced(old):
                table.schedule_stop(
                    ScheduledStop(old, moment + self.stop_delay_seconds), control_lease_number=self.lease_number
                )

    def run_once(self, *, now: float | None = None) -> bool:
        """One round: extend the lease, write the rows of every loaded scenario that a commit has not written yet,
        and clear the due stops; return whether this instance holds the lease."""
        moment = time.time() if now is None else now
        state = self.coordination.control_lease.acquire(self.instance_id, now=moment)
        if state is None:
            if self.held:
                self.held = False
                logger.error("control instance %s lost the control lease", self.instance_id)
                self.instance.control_lease_lost()
            return False
        self.held = True
        self.lease_number = state.number
        for current in self.registry.loaded_scenarios():
            self.publish(current, now=moment)
        table = self.coordination.served_models
        for stop in table.due_stops(moment):
            # A runtime frees the weights it replaced when it loads new ones, so the stop only clears the row.
            table.delete_stop(stop.served, control_lease_number=self.lease_number)
        return True


__all__ = ["CACHE_SECONDS", "ControlInstance", "ServedModelsPublisher", "wait_for_control_lease"]
