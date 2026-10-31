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

from reef.artifact.artifact import Artifact
from reef.core.served_models import ACTIVE, CACHE_SECONDS, ScenarioModel, ScheduledStop, ServedArtifact
from reef.execution.workers import ServiceWorker
from reef.runtime.interfaces import EngineClients, InferenceHandler, InferenceRuntime
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.storage.control_lease import ControlLease, ControlLeaseState
from reef.storage.coordination import Coordination

logger = logging.getLogger(__name__)


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
    (``wait_for_control_lease``); ``instance`` is told when the lease is lost; ``runtime`` is the deployment's
    serving runtime, or ``None`` when it has none: a stop that a former control instance left in ``served_models``
    is due on it before any scenario loads;
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
        runtime: InferenceRuntime | None,
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
        #: The handler of the runtime that serves each row's engine, by URL, for the inference routes of this process.
        self.engines: dict[str, InferenceHandler] = {}
        #: The runtime that serves each row's engine, by URL: it stops serving a value when its stop is due.
        self.runtimes: dict[str, InferenceRuntime] = {} if runtime is None else {runtime.base_url: runtime}
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
        """The rows that ``scenario`` should have now, by name; empty when it is not loaded here or serves nothing.

        ``active`` is the artifact that the runtime serves: the active artifact of the weights component (the
        weight trainer's component, or the only component), or, for a scenario whose model is no Reef artifact,
        such as a provider's, the active release. Each artifact served beside it (#997) gets a row under its own
        ``artifact_id``: a weights artifact with the runtime's value for that artifact on request
        (``InferenceRuntime.served_artifact_on_request``), any other artifact with the active model, which answers
        the calls of that artifact's candidate. Each component's active artifact also has a
        row under its ``artifact_id`` with the active model, so it answers at its own URL.
        """
        current = self.registry.get_loaded(scenario)
        if current is None or current.runtime is None:
            return {}
        runtime = current.runtime
        status = current.artifact_status()
        component = current.served_component()
        active = runtime.served_artifact(current.serving_artifact_id(), scenario)
        if active is None:
            return {}
        rows = {ACTIVE: active}
        # Each component's active artifact also answers at its own URL (#997), with the active model.
        for component_status in status.values():
            if component_status["active"] is not None:
                rows[component_status["active"]] = active
        for artifact_id, beside in current.served_beside().items():
            if beside.component == component:
                release = Artifact(beside.release, current.repository).materialize()
                served = runtime.served_artifact_on_request(current.surface.component_artifact(release, component))
            else:
                served = active
            if served is not None:
                rows[artifact_id] = served
        return rows

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
        if current.runtime is not None:
            # In one process, the inference routes reach each row's engine through the runtime that serves it.
            for served in wanted.values():
                self.engines[served.url] = current.runtime.inference_handler
                self.runtimes[served.url] = current.runtime
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

    def withdraw(self, scenario: str, *, now: float | None = None) -> None:
        """Remove every row of ``scenario``, which is deleted, so the inference service answers it 404; each removed
        value stops after the stop delay, as a replaced one does."""
        with self.lock:
            if not self.held:
                return
            moment = time.time() if now is None else now
            table = self.coordination.served_models
            for name, old in table.read_scenario(scenario).items():
                table.remove(ScenarioModel(scenario, name), control_lease_number=self.lease_number)
                if not table.is_referenced(old):
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
            runtime = self.runtimes.get(stop.served.url)
            if runtime is not None and not table.is_referenced(stop.served):
                runtime.stop_serving(stop.served)
            table.delete_stop(stop.served, control_lease_number=self.lease_number)
        return True


class PublishedEngines(EngineClients):
    """The engines of the runtimes that this control process serves with, by the URL that ``served_models`` names.

    Values: ``publisher`` keeps the handler of each published row's runtime. Purpose: in one process, the inference
    routes reach each row's engine through it.
    """

    def __init__(self, publisher: ServedModelsPublisher) -> None:
        self.publisher = publisher

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        with self.publisher.lock:
            found = self.publisher.engines.get(served.url)
        if found is None:
            raise LookupError(f"no engine of this process serves {served.url}")
        return found


__all__ = ["ControlInstance", "PublishedEngines", "ServedModelsPublisher", "wait_for_control_lease"]
