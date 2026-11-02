"""Control queues artifact selections and renews its lease without changing an engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.core.artifact_ref import ArtifactRef
from reef.core.artifact_selection import ArtifactSelection, SelectedArtifact
from reef.core.served_models import CACHE_SECONDS, ScheduledStop, ServedArtifact
from reef.dispatcher.served import ControlInstance, ServedModelsPublisher, wait_for_control_lease
from reef.storage.coordination import Coordination
from reef.storage.sql_coordination import (
    SQLControlLease,
    SQLiteCoordinationDatabase,
    SQLMarks,
    SQLServedModels,
    SQLWorkQueue,
)


class Runtime:
    """A runtime that records stops and refuses engine access from control."""

    serves_requests = True

    def __init__(self) -> None:
        self.release = "release-1"
        self.base_url = "http://engine:8000"
        self.stopped: list[ServedArtifact] = []

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        raise AssertionError("control must not ask the runtime to serve an artifact")

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
            current_artifact_ref=lambda: SimpleNamespace(release_id=runtime.release),
            artifact_selection=lambda ref=None: ArtifactSelection(
                {
                    "weights": SelectedArtifact(
                        runtime.release if ref is None else "older-weights",
                        ArtifactRef("content", runtime.release if ref is None else ref.release_id, None),
                    )
                }
            ),
            served_beside=dict,
        )

    def loaded_scenarios(self) -> tuple[SimpleNamespace, ...]:
        return (self.scenario,)

    def get_loaded(self, scenario: str) -> SimpleNamespace | None:
        return self.scenario if scenario == "math" else None


class EmptyRegistry(Registry):
    """A registry that has loaded no scenario yet."""

    def loaded_scenarios(self) -> tuple[SimpleNamespace, ...]:
        return ()


class Instance(ControlInstance):
    def __init__(self) -> None:
        self.lost = 0

    def control_lease_lost(self) -> None:
        self.lost += 1


def coordination() -> Coordination:
    database = SQLiteCoordinationDatabase(None)
    lease = SQLControlLease(database)
    return Coordination(database, SQLWorkQueue(database), SQLServedModels(database, lease), lease, SQLMarks(database))


@pytest.mark.unit
def test_control_queues_the_selected_artifact_without_publishing_a_ready_route() -> None:
    runtime = Runtime()
    store = coordination()
    lease = store.control_lease.acquire("control-1", now=99.0)
    publisher = ServedModelsPublisher(
        Registry(runtime),
        store,
        Instance(),
        lease,
        instance_id="control-1",
        stop_delay_seconds=CACHE_SECONDS + 10,
    )
    assert publisher.run_once(now=100.0)
    first = store.served_models.current_request("math")
    assert first.state == "queued" and first.started_at is None
    assert first.payload == {
        "artifact_id": "release-1",
        "selection": Registry(runtime).scenario.artifact_selection().to_dict(),
        "aliases": ["release-1"],
        "beside": {},
    }
    assert store.served_models.read_scenario("math") == {}
    assert publisher.run_once(now=100.5)
    assert store.served_models.current_request("math").id == first.id

    runtime.release = "release-2"
    assert publisher.run_once(now=101.0)
    second = store.served_models.current_request("math")
    assert second.id != first.id and second.payload["artifact_id"] == "release-2"
    assert store.work_queue.get(first.queue, first.id).state == "failed"
    assert store.served_models.read_scenario("math") == {}
    assert runtime.stopped == []


@pytest.mark.unit
def test_control_queues_beside_artifacts_as_references_and_withdraws_the_current_request() -> None:
    runtime = Runtime()
    registry = Registry(runtime)
    registry.scenario.served_beside = lambda: {
        "older-weights": SimpleNamespace(release=SimpleNamespace(release_id="release-0"), component="weights")
    }
    store = coordination()
    lease = store.control_lease.acquire("control", now=0)
    publisher = ServedModelsPublisher(registry, store, Instance(), lease, instance_id="control", stop_delay_seconds=60)
    publisher.run_once(now=1)
    request = store.served_models.current_request("math")
    assert request.payload["beside"] == {
        "older-weights": {
            "selection": registry.scenario.artifact_selection(SimpleNamespace(release_id="release-0")).to_dict(),
            "component": "weights",
        }
    }
    publisher.withdraw("math", now=2)
    assert store.served_models.current_request("math").payload == {}
    assert store.work_queue.get(request.queue, request.id).state == "failed"


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


@pytest.mark.unit
def test_control_leaves_scheduled_engine_stops_to_inference() -> None:
    runtime = Runtime()
    store = coordination()
    old = ServedArtifact("release-0", runtime.base_url, "math-release-0")
    former = store.control_lease.acquire("control-1", now=100.0)
    store.served_models.schedule_stop(ScheduledStop(old, 110.0), control_lease_number=former.number)
    lease = store.control_lease.acquire("control-2", now=131.0)
    publisher = ServedModelsPublisher(
        EmptyRegistry(runtime),
        store,
        Instance(),
        lease,
        instance_id="control-2",
        stop_delay_seconds=60,
    )
    assert publisher.run_once(now=131.0)
    assert runtime.stopped == []
    assert store.served_models.due_stops(131.0) == (ScheduledStop(old, 110.0),)
