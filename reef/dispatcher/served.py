"""Control queues artifact startup and renews its lease. Inference publishes the resulting routes."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import cast

from reef.core.errors import ReefError
from reef.core.served_models import CACHE_SECONDS
from reef.execution.workers import ServiceWorker
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.storage.control_lease import ControlLease, ControlLeaseState
from reef.storage.coordination import Coordination
from reef.storage.served_models import AUTOMATIC_RETRIES
from reef.storage.work_queue import JsonValue, WorkItem, clock

logger = logging.getLogger(__name__)

#: The wait after the first failure of a serving request before control retries it by itself. It equals the default
#: work lease and is twice the default control lease, so the leases of the services that a restart ended have expired
#: when the retry starts.
FIRST_RETRY_DELAY_SECONDS = 60.0
#: The longest wait between automatic retries. A failure that every retry repeats costs at most 24 startups a day,
#: and a cause that goes away heals within an hour.
MAX_RETRY_DELAY_SECONDS = 3600.0


def automatic_retry_count(request: WorkItem) -> int:
    """The number of automatic retries in a row that ``request``, a serving request, continues:
    ``metadata[AUTOMATIC_RETRIES]``, or 0 without the key. Raise ``ReefError`` for a value that is no non-negative
    int."""
    count = request.metadata.get(AUTOMATIC_RETRIES, 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ReefError(f"serving request {request.id} has an invalid automatic retry count {count!r}")
    return count


def retry_delay_seconds(retries: int) -> float:
    """The wait after the failure of a serving request with ``retries`` automatic retries before control retries it:
    ``min(FIRST_RETRY_DELAY_SECONDS * 2 ** retries, MAX_RETRY_DELAY_SECONDS)`` for an int ``retries`` from 0."""
    # The exponent stops at 64, far beyond the cap, so that the float stays finite.
    return min(FIRST_RETRY_DELAY_SECONDS * 2.0 ** min(retries, 64), MAX_RETRY_DELAY_SECONDS)


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


def take_control_lease(lease: ControlLease, instance_id: str) -> ControlLeaseState:
    """Take the control lease for ``instance_id`` now; raise ``ReefError`` when another instance holds it."""
    state = lease.acquire(instance_id)
    if state is None:
        holder = lease.current()
        raise ReefError(
            f"control instance {None if holder is None else holder.instance_id} holds the control lease: a standby "
            "waits for it (wait_for_control_lease) and gives the dispatcher its state"
        )
    return state


class ControlInstance(ABC):
    """What a control instance does when it loses the control lease.

    Purpose: the dispatcher implements it; ``ServedModelsPublisher`` calls it.
    """

    @abstractmethod
    def control_lease_lost(self) -> None:
        """Another instance took the lease: stop the work that only the active instance does."""


class ServedModelsPublisher(ServiceWorker):
    """Queue each loaded scenario's selection while this instance holds the control lease.

    ``registry`` owns the scenarios; ``coordination`` stores requests and the lease.
    ``instance`` stops control work after lease loss. ``stop_delay_seconds`` bounds route draining.
    The positive ``interval_seconds`` is shorter than the control lease.

    A serving request states the selection that a scenario should serve, so its failure is not final: each round
    retries a failed current request of the desired payload once ``retry_delay_seconds`` of its automatic retry count
    passed after it ended (``publish``).
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
            raise ValueError("stop_delay_seconds must cover the inference route cache")
        if not 0 < interval_seconds < coordination.control_lease.lease_seconds:
            raise ValueError("interval_seconds must be positive and shorter than the control lease")
        if lease.instance_id != instance_id:
            raise ValueError("the control lease belongs to another instance")
        self.registry = registry
        self.coordination = coordination
        self.instance = instance
        self.instance_id = instance_id
        self.lease_number = lease.number
        self.stop_delay_seconds = stop_delay_seconds
        self.interval_seconds = interval_seconds
        self.held = True
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, name="reef-serving-requests", daemon=True)
            self.thread.start()

    def stop(self) -> None:
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
                logger.exception("could not queue serving requests")
            self.stopped.wait(self.interval_seconds)

    def desired(self, scenario: str) -> dict[str, JsonValue]:
        """The selected snapshot, the route names and the serving config of the served component, without opening
        or calling an engine."""
        current = self.registry.get_loaded(scenario)
        if current is None or current.runtime is None or not current.runtime.serves_requests:
            return {}
        aliases: list[JsonValue] = [
            status["active"] for status in current.artifact_status().values() if status["active"] is not None
        ]
        beside: dict[str, JsonValue] = {
            artifact_id: {
                "selection": cast(dict[str, JsonValue], served.selection.to_dict()),
                "component": served.component,
            }
            for artifact_id, served in current.served_beside().items()
        }
        return {
            "artifact_id": current.serving_artifact_id(),
            "selection": cast(dict[str, JsonValue], current.artifact_selection().to_dict()),
            "aliases": aliases,
            "beside": beside,
            "serving_config": current.served_serving_config(),
        }

    def write_traffic_weights(self, scenario: str, weights: Mapping[str, int]) -> None:
        """Make ``weights`` the traffic split of ``scenario`` (``ServedModelsWriter.write_traffic_weights``)."""
        self.coordination.served_models.write_traffic_weights(
            scenario, weights, control_lease_number=self.lease_number
        )

    def publish(self, current: Scenario, *, now: float | None = None, retry_failed: bool = False) -> None:
        """Queue the desired payload of ``current`` at ``now`` while this instance holds the lease.

        Let ``failed`` be the scenario's current request when it failed with the desired payload, ``n`` its
        ``automatic_retry_count`` and ``t`` its ``ended_at``. Then:

        - ``retry_failed`` true (a person activates or restores): a new request retries ``failed`` at once, with
          the count 0.
        - ``retry_failed`` false and ``now >= t + retry_delay_seconds(n)``: a new request retries ``failed``, with
          the count ``n + 1``.
        - otherwise ``failed`` stays the current request.

        Without ``failed``, a desired payload that equals the current request's changes nothing, and another
        payload replaces it with a new request.
        """
        moment = clock(now)
        with self.lock:
            if not self.held:
                return
            desired = self.desired(current.name)
            retries = 0
            if not retry_failed:
                failed = self.coordination.served_models.current_request(current.name)
                if failed is not None and failed.state == "failed" and failed.ended_at is not None:
                    count = automatic_retry_count(failed)
                    if moment >= failed.ended_at + retry_delay_seconds(count):
                        retry_failed, retries = True, count + 1
            request = self.coordination.served_models.request(
                current.name,
                desired,
                control_lease_number=self.lease_number,
                stop_delay_seconds=self.stop_delay_seconds,
                retry_failed=retry_failed,
                automatic_retries=retries,
                now=moment,
            )
            if retries and automatic_retry_count(request) == retries:
                logger.info(
                    "scenario %s: control retries its failed serving request as %s, automatic retry %d",
                    current.name,
                    request.id,
                    retries,
                )

    def withdraw(self, scenario: str, *, now: float | None = None) -> None:
        """Withdraw the scenario and its pending startup before archiving its files."""
        with self.lock:
            if self.held:
                self.coordination.served_models.request(
                    scenario,
                    {},
                    control_lease_number=self.lease_number,
                    stop_delay_seconds=self.stop_delay_seconds,
                    now=now,
                )

    def run_once(self, *, now: float | None = None) -> bool:
        """Renew the lease and publish every loaded scenario, which retries the failed requests that are due;
        return whether this instance holds the lease."""
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
        return True


__all__ = [
    "FIRST_RETRY_DELAY_SECONDS",
    "MAX_RETRY_DELAY_SECONDS",
    "ControlInstance",
    "ServedModelsPublisher",
    "automatic_retry_count",
    "retry_delay_seconds",
    "take_control_lease",
    "wait_for_control_lease",
]
