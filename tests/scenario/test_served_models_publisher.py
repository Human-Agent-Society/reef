"""Control's writer of ``served_models``: the active artifact of each loaded scenario, the lease, and old values."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.dispatcher.served import CACHE_SECONDS, ServedModelsPublisher
from reef.storage.control_lease import MemoryControlLease
from reef.storage.coordination import Coordination, CoordinationResource
from reef.storage.served_models import MemoryServedModels
from reef.storage.work_queue import MemoryWorkQueue


class NoDatabase(CoordinationResource):
    def close(self) -> None:
        return None


class Runtime:
    """Serves ``math`` at one URL, under the model name of the release that serves it."""

    def __init__(self) -> None:
        self.release = "release-1"

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        return ServedArtifact(artifact_id, "http://engine:8000", f"{scenario}-{artifact_id}")


class Registry:
    def __init__(self, runtime: Runtime) -> None:
        self.scenario = SimpleNamespace(
            name="math",
            runtime=runtime,
            trainers=(),
            artifact_status=lambda: {"weights": {"active": runtime.release, "training_start": None}},
        )

    def loaded_scenarios(self) -> tuple[SimpleNamespace, ...]:
        return (self.scenario,)

    def get_loaded(self, scenario: str) -> SimpleNamespace | None:
        return self.scenario if scenario == "math" else None


def coordination() -> Coordination:
    lease = MemoryControlLease(lease_seconds=30.0)
    return Coordination(NoDatabase(), MemoryWorkQueue(), MemoryServedModels(lease), lease)


@pytest.mark.unit
def test_the_active_artifact_is_published_and_a_replaced_value_is_stopped_after_the_delay() -> None:
    runtime = Runtime()
    store = coordination()
    publisher = ServedModelsPublisher(
        Registry(runtime), store, instance_id="control-1", stop_delay_seconds=CACHE_SECONDS + 10
    )
    key = ScenarioModel("math", ACTIVE)
    first = ServedArtifact("release-1", "http://engine:8000", "math-release-1")

    assert publisher.run_once(now=100.0)
    assert store.served_models.read(key) == first

    runtime.release = "release-2"
    assert publisher.run_once(now=101.0)
    assert store.served_models.read(key).artifact_id == "release-2"
    assert [stop.served for stop in store.served_models.due_stops(116.0)] == [first]
    assert store.served_models.due_stops(110.0) == ()
    publisher.run_once(now=116.0)
    assert store.served_models.due_stops(116.0) == ()


@pytest.mark.unit
def test_a_standby_instance_writes_nothing_until_the_lease_expires() -> None:
    store = coordination()
    registry = Registry(Runtime())
    active = ServedModelsPublisher(registry, store, instance_id="control-1", stop_delay_seconds=60)
    standby = ServedModelsPublisher(registry, store, instance_id="control-2", stop_delay_seconds=60)

    assert active.run_once(now=100.0)
    assert not standby.run_once(now=110.0)
    assert standby.run_once(now=131.0)
    assert store.control_lease.current().instance_id == "control-2"
