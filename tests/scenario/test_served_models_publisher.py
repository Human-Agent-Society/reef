"""Control queues artifact selections and renews its lease without changing an engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.core.artifact_selection import ArtifactSelection
from reef.core.served_models import CACHE_SECONDS, ScheduledStop, ServedArtifact
from reef.dispatcher.served import ControlInstance, ServedModelsPublisher, retry_delay_seconds, wait_for_control_lease
from reef.storage.coordination import Coordination
from reef.storage.served_models import AUTOMATIC_RETRIES, SERVING_KIND, SERVING_QUEUE
from reef.storage.sql_coordination import SQLControlLease, SQLiteCoordinationDatabase, SQLServedModels, SQLWorkQueue


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
            current_snapshot=lambda: SimpleNamespace(release_id=runtime.release),
            artifact_selection=lambda ref=None: ArtifactSelection(
                {"weights": runtime.release if ref is None else "older-weights"}
            ),
            served_beside=dict,
            served_serving_config=lambda: {"temperature": 0.6},
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
    return Coordination(database, SQLWorkQueue(database), SQLServedModels(database, lease), lease)


def fail_current_request(store: Coordination, *, now: float) -> str:
    """Let an inference worker take the current serving request and fail it for good; return its ID."""
    item = store.work_queue.take(SERVING_QUEUE, "inference", [SERVING_KIND], now=now)
    store.work_queue.fail(SERVING_QUEUE, item.id, "inference", "the engine is not ready", retryable=False, now=now)
    return item.id


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
        "serving_config": {"temperature": 0.6},
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
        "older-weights": SimpleNamespace(
            selection=ArtifactSelection({"weights": "older-weights"}), component="weights"
        )
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


@pytest.mark.unit
def test_control_retries_a_failed_request_after_a_delay_that_doubles_up_to_an_hour() -> None:
    runtime = Runtime()
    store = coordination()
    lease = store.control_lease.acquire("control", now=0.0)
    publisher = ServedModelsPublisher(
        Registry(runtime), store, Instance(), lease, instance_id="control", stop_delay_seconds=60
    )
    publisher.run_once(now=0.0)
    delays = [60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3600.0, 3600.0]
    assert [retry_delay_seconds(retries) for retries in range(len(delays))] == delays
    assert retry_delay_seconds(10_000) == 3600.0
    moment = 0.0
    for retries, delay in enumerate(delays):
        failed = fail_current_request(store, now=moment)
        assert publisher.run_once(now=moment + delay - 1)
        assert store.served_models.current_request("math").id == failed
        moment += delay
        assert publisher.run_once(now=moment)
        retried = store.served_models.current_request("math")
        assert retried.id != failed and retried.state == "queued"
        assert retried.metadata == {AUTOMATIC_RETRIES: retries + 1}

    # A failed request whose payload is no longer desired is replaced at once, with no retry count.
    fail_current_request(store, now=moment)
    runtime.release = "release-2"
    assert publisher.run_once(now=moment + 1)
    replaced = store.served_models.current_request("math")
    assert replaced.payload["artifact_id"] == "release-2" and replaced.metadata == {}


@pytest.mark.unit
def test_a_manual_retry_starts_at_once_and_resets_the_delay() -> None:
    store = coordination()
    registry = Registry(Runtime())
    lease = store.control_lease.acquire("control", now=0.0)
    publisher = ServedModelsPublisher(registry, store, Instance(), lease, instance_id="control", stop_delay_seconds=60)
    publisher.run_once(now=0.0)
    fail_current_request(store, now=0.0)
    publisher.run_once(now=60.0)
    fail_current_request(store, now=60.0)
    assert store.served_models.current_request("math").metadata == {AUTOMATIC_RETRIES: 1}

    # Activate and restore publish with retry_failed: the retry starts before the 120 s delay ends.
    publisher.publish(registry.scenario, retry_failed=True, now=61.0)
    manual = store.served_models.current_request("math")
    assert manual.state == "queued" and manual.metadata == {}
    fail_current_request(store, now=61.0)
    publisher.run_once(now=120.0)
    assert store.served_models.current_request("math").id == manual.id
    publisher.run_once(now=121.0)
    assert store.served_models.current_request("math").metadata == {AUTOMATIC_RETRIES: 1}
