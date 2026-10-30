"""Control's writer of ``served_models``: the active artifact of each loaded scenario, the lease, and old values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.dispatcher.served import CACHE_SECONDS, ControlInstance, ServedModelsPublisher, wait_for_control_lease
from reef.storage.control_lease import MemoryControlLease
from reef.storage.coordination import Coordination, CoordinationResource
from reef.storage.served_models import MemoryServedModels
from reef.storage.work_queue import MemoryWorkQueue


class NoDatabase(CoordinationResource):
    def close(self) -> None:
        return None


class Runtime:
    """Serves ``math`` at one URL, under the model name of the release that serves it, through one handler."""

    def __init__(self) -> None:
        self.release = "release-1"
        self.inference_handler = object()
        self.stopped: list[ServedArtifact] = []

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        return ServedArtifact(artifact_id, "http://engine:8000", f"{scenario}-{artifact_id}")

    def stop_serving(self, served: ServedArtifact) -> None:
        self.stopped.append(served)


class Registry:
    def __init__(self, runtime: Runtime) -> None:
        self.scenario = SimpleNamespace(
            name="math",
            runtime=runtime,
            artifact_status=lambda: {"weights": {"active": runtime.release, "training_start": None}},
            served_component=lambda: "weights",
            serving_artifact_id=lambda: runtime.release,
            served_beside=dict,
        )

    def loaded_scenarios(self) -> tuple[SimpleNamespace, ...]:
        return (self.scenario,)

    def get_loaded(self, scenario: str) -> SimpleNamespace | None:
        return self.scenario if scenario == "math" else None


class Instance(ControlInstance):
    def __init__(self) -> None:
        self.lost = 0

    def control_lease_lost(self) -> None:
        self.lost += 1


def coordination() -> Coordination:
    lease = MemoryControlLease(lease_seconds=30.0)
    return Coordination(NoDatabase(), MemoryWorkQueue(), MemoryServedModels(lease), lease)


@pytest.mark.unit
def test_the_active_artifact_is_published_and_a_replaced_value_is_stopped_after_the_delay() -> None:
    runtime = Runtime()
    store = coordination()
    lease = store.control_lease.acquire("control-1", now=99.0)
    publisher = ServedModelsPublisher(
        Registry(runtime), store, Instance(), lease, instance_id="control-1", stop_delay_seconds=CACHE_SECONDS + 10
    )
    key = ScenarioModel("math", ACTIVE)
    first = ServedArtifact("release-1", "http://engine:8000", "math-release-1")

    assert publisher.run_once(now=100.0)
    assert store.served_models.read(key) == first
    assert publisher.engines == {"http://engine:8000": runtime.inference_handler}

    runtime.release = "release-2"
    assert publisher.run_once(now=101.0)
    assert store.served_models.read(key).artifact_id == "release-2"
    assert [stop.served for stop in store.served_models.due_stops(116.0)] == [first]
    assert store.served_models.due_stops(110.0) == ()
    publisher.run_once(now=116.0)
    assert store.served_models.due_stops(116.0) == ()
    # The engine stops serving the replaced value when its stop is due.
    assert runtime.stopped == [first]


@pytest.mark.unit
def test_a_standby_takes_over_when_the_lease_expires_and_the_old_instance_stops_its_work() -> None:
    store = coordination()
    registry = Registry(Runtime())
    first, second = Instance(), Instance()
    active = ServedModelsPublisher(
        registry,
        store,
        first,
        store.control_lease.acquire("control-1", now=100.0),
        instance_id="control-1",
        stop_delay_seconds=60,
    )
    assert active.run_once(now=100.0)
    assert store.control_lease.acquire("control-2", now=110.0) is None
    taken = store.control_lease.acquire("control-2", now=131.0)
    standby = ServedModelsPublisher(registry, store, second, taken, instance_id="control-2", stop_delay_seconds=60)
    assert standby.run_once(now=131.0)
    assert store.control_lease.current().instance_id == "control-2"
    assert not active.run_once(now=132.0)
    assert (first.lost, second.lost) == (1, 0)


@pytest.mark.unit
def test_a_stopped_instance_releases_the_lease_so_the_next_one_starts_at_once() -> None:
    store = coordination()
    lease = wait_for_control_lease(store.control_lease, "control-1", poll_seconds=0.01)
    active = ServedModelsPublisher(
        Registry(Runtime()), store, Instance(), lease, instance_id="control-1", stop_delay_seconds=60
    )
    assert active.run_once()
    active.stop()
    state = wait_for_control_lease(store.control_lease, "control-2", poll_seconds=0.01)
    assert (state.instance_id, state.number) == ("control-2", 2)


@pytest.mark.unit
def test_an_instance_whose_lease_expired_while_it_started_stops_at_its_first_round() -> None:
    store = coordination()
    instance = Instance()
    lease = store.control_lease.acquire("control-1", now=100.0)
    # A slow start-up: a standby takes the expired lease before the first round.
    assert store.control_lease.acquire("control-2", now=131.0) is not None
    publisher = ServedModelsPublisher(
        Registry(Runtime()), store, instance, lease, instance_id="control-1", stop_delay_seconds=60
    )
    assert not publisher.run_once(now=132.0)
    assert instance.lost == 1
