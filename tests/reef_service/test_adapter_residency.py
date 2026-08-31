"""Engine-global adapter residency shared by every scenario on one engine."""

from __future__ import annotations

from typing import Any

import pytest

from reef.runtime.adapter_residency import (
    AdapterCapacityExhausted,
    AdapterNotActive,
    AdapterResidencyError,
    AdapterResidencyManager,
)
from reef.surface import adapter_name


class FakeEngine:
    """Records loads/unloads and can be told to fail either."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.payloads: list[Any] = []
        self.unloaded: list[str] = []
        self.fail_load: set[str] = set()
        self.fail_unload: set[str] = set()

    def load_adapter(self, name: str, payload: Any) -> None:
        if name in self.fail_load:
            raise RuntimeError(f"engine refused {name}")
        self.loaded.append(name)
        self.payloads.append(payload)

    def unload_adapter(self, name: str) -> None:
        if name in self.fail_unload:
            raise RuntimeError(f"engine keeps {name}")
        self.unloaded.append(name)

    @property
    def resident(self) -> set[str]:
        return set(self.loaded) - set(self.unloaded)


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def test_activation_loads_then_advances_the_scenario_current(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=4)
    name = manager.activate("a", "v1", engine, payload="bytes-of-v1")
    assert name == adapter_name("a", "v1")
    assert engine.loaded == [name] and engine.payloads == ["bytes-of-v1"]
    current = manager.current("a")
    assert current is not None and current.version == "v1" and current.current and current.protected


def test_two_scenarios_with_the_same_revision_label_do_not_collide(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=4)
    a = manager.activate("a", "v1", engine)
    b = manager.activate("b", "v1", engine)
    assert a != b
    assert engine.resident == {a, b}
    assert manager.resolve("a", "v1").name == a
    assert manager.resolve("b", "v1").name == b


def test_load_failure_leaves_the_incumbent_active_and_no_phantom_slot(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "v1", engine)
    engine.fail_load.add(adapter_name("a", "v2"))
    with pytest.raises(AdapterResidencyError, match="could not load"):
        manager.activate("a", "v2", engine)
    assert manager.current("a").version == "v1"
    assert [entry.version for entry in manager.resident()] == ["v1"]
    assert manager.status()["counters"]["load_failures"] == 1


def test_eviction_is_oldest_unprotected_first_across_scenarios(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=3)
    manager.activate("a", "a1", engine)  # order 0, becomes non-current after a2
    manager.activate("b", "b1", engine)  # order 1, current for b
    manager.activate("a", "a2", engine)  # order 2, current for a
    # Full. Activating b2 must evict a1 (oldest unprotected), never b1 or a2.
    manager.activate("b", "b2", engine)
    assert engine.unloaded == [adapter_name("a", "a1")]
    assert manager.current("a").version == "a2"
    assert manager.current("b").version == "b2"
    assert manager.status()["counters"]["evictions"] == 1


def test_capacity_exhaustion_fails_closed_when_every_slot_is_protected(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.activate("b", "b1", engine)
    with pytest.raises(AdapterCapacityExhausted, match="exhausted"):
        manager.activate("c", "c1", engine)
    assert engine.unloaded == []
    assert manager.current("c") is None
    assert manager.current("a").version == "a1" and manager.current("b").version == "b1"
    assert manager.status()["counters"]["capacity_rejections"] == 1


def test_supersede_lets_a_paused_scenario_evict_its_own_current_revision(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.activate("b", "b1", engine)
    # Without supersede a's own current revision is protected like any other.
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("a", "a2", engine)
    manager.activate("a", "a2", engine, supersede=True)
    assert engine.unloaded == [adapter_name("a", "a1")]
    assert manager.current("a").version == "a2" and manager.current("b").version == "b1"
    # A peer's current revision is still never the victim.
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("c", "c1", engine, supersede=True)


def test_supersede_does_not_override_a_pin_or_an_in_flight_lease(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    manager.activate("a", "a1", engine)
    manager.pin("a", "a1")
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("a", "a2", engine, supersede=True)
    manager.unpin("a", "a1")
    lease = manager.lease("a", "a1")
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("a", "a2", engine, supersede=True)
    lease.release()
    manager.activate("a", "a2", engine, supersede=True)
    assert engine.unloaded == [adapter_name("a", "a1")]


def test_make_room_then_register_accounts_for_an_out_of_band_load(engine: FakeEngine) -> None:
    """The training path loads through its own transport and registers afterwards."""
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.activate("b", "b1", engine)
    manager.make_room("a", engine, supersede=True)
    assert engine.unloaded == [adapter_name("a", "a1")]
    assert manager.current("a") is None
    name = manager.register("a", "a2")
    assert name == adapter_name("a", "a2")
    assert engine.loaded == [adapter_name("a", "a1"), adapter_name("b", "b1")], "register loads nothing"
    assert manager.current("a").version == "a2"
    assert manager.resolve("a", "a2").name == name
    status = manager.status()
    assert status["resident"] == 2 and status["counters"]["loads"] == 3
    # Registering the resident revision again is idempotent.
    assert manager.register("a", "a2") == name and manager.status()["resident"] == 2


def test_make_room_refuses_when_every_slot_is_protected(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    manager.activate("a", "a1", engine)
    with pytest.raises(AdapterCapacityExhausted):
        manager.make_room("b", engine, supersede=True)
    assert engine.unloaded == [] and manager.current("a").version == "a1"


def test_register_reclaims_a_leaked_slot_the_engine_reloaded(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.activate("a", "a2", engine)
    engine.fail_unload.add(adapter_name("a", "a1"))
    with pytest.raises(AdapterCapacityExhausted, match="leaked"):
        manager.make_room("a", engine, supersede=True)
    assert manager.status()["leaked"] == 1
    # The trainer republished a1 in place (its transport upserts by name).
    manager.register("a", "a1")
    assert manager.status()["leaked"] == 0 and manager.current("a").version == "a1"
    assert manager.status()["recent_actions"][-1]["action"] == "reclaimed"


def test_register_beyond_capacity_is_counted_not_hidden(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    manager.activate("a", "a1", engine)
    manager.register("b", "b1")
    assert manager.status()["resident"] == 2


def test_pinned_revisions_survive_eviction(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.pin("a", "a1")
    manager.activate("a", "a2", engine)
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("a", "a3", engine)
    manager.unpin("a", "a1")
    manager.activate("a", "a3", engine)
    assert engine.unloaded == [adapter_name("a", "a1")]


def test_in_flight_requests_protect_a_superseded_revision(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    lease = manager.lease("a", "a1")
    manager.activate("a", "a2", engine)
    with pytest.raises(AdapterCapacityExhausted):
        manager.activate("a", "a3", engine)
    lease.release()
    lease.release()  # idempotent
    manager.activate("a", "a3", engine)
    assert engine.unloaded == [adapter_name("a", "a1")]


def test_lease_fails_closed_for_a_revision_that_is_not_active(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    with pytest.raises(AdapterNotActive, match="no active adapter"):
        manager.lease("a", "a1")
    manager.activate("a", "a1", engine)
    with pytest.raises(AdapterNotActive, match="not the requested"):
        manager.lease("a", "a0")
    with pytest.raises(AdapterNotActive):
        manager.lease("b", "a1")


def test_failed_unload_leaks_the_slot_visibly_and_is_retried_first(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "a1", engine)
    manager.activate("a", "a2", engine)
    engine.fail_unload.add(adapter_name("a", "a1"))
    with pytest.raises(AdapterCapacityExhausted, match="leaked"):
        manager.activate("a", "a3", engine)
    status = manager.status()
    assert status["leaked"] == 1 and status["resident"] == 2
    assert status["counters"]["unload_failures"] == 1
    assert manager.current("a").version == "a2"
    # Once the engine lets go, the leaked slot is reclaimed before any live one.
    engine.fail_unload.clear()
    manager.activate("a", "a3", engine)
    assert engine.unloaded == [adapter_name("a", "a1")]
    assert manager.status()["leaked"] == 0


def test_alias_routes_a_republished_version_through_the_resident_adapter(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    name = manager.activate("a", "a1", engine)
    assert manager.alias("a", "rollback-1") == name
    assert manager.resolve("a", "rollback-1").name == name
    assert manager.lease("a", "a1").adapter_name == name
    assert engine.loaded == [name]
    with pytest.raises(AdapterNotActive):
        manager.alias("b", "x")


def test_reactivating_the_resident_revision_is_idempotent(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    manager.activate("a", "a1", engine)
    manager.activate("a", "a1", engine)
    assert engine.loaded == [adapter_name("a", "a1")]


def test_reconcile_drops_lost_adapters_and_unloads_strays(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=4)
    kept = manager.activate("a", "a1", engine)
    lost = manager.activate("b", "b1", engine)
    dropped = manager.reconcile([kept, "reef-adapter-stale.v0"], engine)
    assert dropped == (lost,)
    assert manager.current("b") is None
    assert manager.current("a").name == kept
    assert "reef-adapter-stale.v0" in engine.unloaded
    with pytest.raises(AdapterNotActive):
        manager.lease("b", "b1")


def test_release_scenario_makes_its_adapter_evictable(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=1)
    manager.activate("a", "a1", engine)
    manager.release_scenario("a")
    manager.activate("b", "b1", engine)
    assert engine.unloaded == [adapter_name("a", "a1")]


def test_unbounded_manager_never_evicts(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager()
    for index in range(5):
        manager.activate("a", f"v{index}", engine)
    assert engine.unloaded == [] and len(engine.resident) == 5


def test_status_is_bounded_and_per_scenario(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=3)
    manager.activate("a", "a1", engine)
    manager.activate("a", "a2", engine)
    manager.activate("b", "b1", engine)
    status = manager.status()
    assert status["capacity"] == 3 and status["resident"] == 3 and status["protected"] == 2
    assert status["scenarios"]["a"]["current"] == {"version": "a2", "adapter": adapter_name("a", "a2")}
    assert status["scenarios"]["a"]["resident"] == ["a1", "a2"]
    assert status["scenarios"]["b"]["resident"] == ["b1"]


def test_capacity_and_names_must_be_well_formed(engine: FakeEngine) -> None:
    with pytest.raises(ValueError):
        AdapterResidencyManager(0)
    with pytest.raises(ValueError):
        AdapterResidencyManager(True)  # type: ignore[arg-type]
    manager = AdapterResidencyManager()
    with pytest.raises(ValueError, match="scenario"):
        manager.activate("", "v1", engine)
    with pytest.raises(ValueError, match="version"):
        manager.register("a", "")


class AmbiguousEngine(FakeEngine):
    """Loads the adapter, then the reply is lost: Reef sees a timeout."""

    def __init__(self) -> None:
        super().__init__()
        self.ambiguous: set[str] = set()
        self.held: set[str] = set()

    def load_adapter(self, name: str, payload: Any) -> None:
        super().load_adapter(name, payload)
        self.held.add(name)
        if name in self.ambiguous:
            raise TimeoutError("reply lost after the engine loaded the adapter")

    def unload_adapter(self, name: str) -> None:
        super().unload_adapter(name)
        self.held.discard(name)

    @property
    def resident(self) -> set[str]:
        return set(self.held)


def test_ambiguous_load_timeout_is_cleaned_up_and_the_retry_succeeds() -> None:
    engine = AmbiguousEngine()
    manager = AdapterResidencyManager(capacity=2)
    manager.activate("a", "v1", engine)
    name = adapter_name("a", "v2")
    engine.ambiguous.add(name)
    with pytest.raises(AdapterResidencyError, match="could not load"):
        manager.activate("a", "v2", engine)
    # Deterministic outcome: Reef counts no slot, and the engine was told to
    # drop the name, so neither side holds v2 after the ambiguous attempt.
    assert manager.current("a").version == "v1"
    assert name not in engine.resident
    counters = manager.status()["counters"]
    assert counters["load_failures"] == 1 and counters["load_cleanups"] == 1
    engine.ambiguous.clear()
    assert manager.activate("a", "v2", engine) == name
    assert manager.current("a").version == "v2" and name in engine.resident
    assert [entry["action"] for entry in manager.status()["recent_actions"]] == ["load_failed", "load_cleanup"]


def test_cleanup_failure_after_an_ambiguous_load_is_visible() -> None:
    engine = AmbiguousEngine()
    manager = AdapterResidencyManager(capacity=2)
    name = adapter_name("a", "v1")
    engine.ambiguous.add(name)
    engine.fail_unload.add(name)
    with pytest.raises(AdapterResidencyError):
        manager.activate("a", "v1", engine)
    status = manager.status()
    assert status["resident"] == 0
    assert status["counters"]["cleanup_failures"] == 1 and status["counters"]["load_cleanups"] == 0
    assert status["recent_actions"][-1]["action"] == "cleanup_failed"


def test_recent_actions_are_bounded(engine: FakeEngine) -> None:
    manager = AdapterResidencyManager(capacity=2)
    for index in range(50):
        manager.activate("a", f"v{index}", engine)
    actions = manager.status()["recent_actions"]
    assert len(actions) == 32 and all(entry["action"] == "evicted" for entry in actions)
