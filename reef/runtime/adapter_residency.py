"""Runtime-global residency of scenario-owned adapters on one shared base model.

One serving engine holds one base model and a finite number of loaded LoRA
adapters. Several scenarios may each evolve an independent adapter runtime load chain on
that engine, so the capacity accounting cannot live with any single scenario:
two scenarios with private residency windows would overcommit the same slots
and evict each other's current revision. :class:`AdapterResidencyManager` is
the one accounting point for every adapter an engine holds — the training
bridge owns one per engine, and every scenario's publication, restart
recovery, and eviction goes through it.

The manager owns names, slots, and protection; the engine owns bytes. It
asks an :class:`AdapterEngine` to load or unload by name and otherwise treats
the engine as a black box: the payload it hands to ``load_adapter`` is
whatever the caller supplied (a materialized PEFT directory, a callable that
pushes trained tensors) and is never interpreted here.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal, Protocol, runtime_checkable

from reef.core.errors import ReefError
from reef.surface.adapter import adapter_name

logger = logging.getLogger(__name__)

ResidentState = Literal["active", "leaked"]


class AdapterResidencyError(ReefError):
    """The shared engine cannot make a scenario's adapter revision servable."""


class AdapterCapacityExhausted(AdapterResidencyError):
    """Every loaded adapter is protected, so nothing can be evicted for a new one."""


class AdapterNotActive(AdapterResidencyError):
    """A request resolved to an adapter revision the engine does not hold."""


@runtime_checkable
class AdapterEngine(Protocol):
    """Engine-side adapter operations the residency manager drives.

    ``load_adapter`` must return only once the engine can serve requests that
    name ``name``; raising means the adapter never became resident.
    ``payload`` is the caller's opaque description of the bytes to load.
    ``unload_adapter`` raising means the slot may still be occupied.
    """

    def load_adapter(self, name: str, payload: Any) -> None: ...

    def unload_adapter(self, name: str) -> None: ...


#: How many recovery actions ``status()`` keeps; enough to explain the most
#: recent restart or capacity incident without growing with uptime.
RECENT_ACTIONS = 32


@dataclass(frozen=True)
class ResidentAdapter:
    """One adapter the engine holds, with everything that protects it."""

    name: str
    scenario: str
    runtime_load_id: str
    #: Every runtime load served by this adapter. A rollback republishes
    #: the same bytes under a new runtime_load_id, which aliases here instead of
    #: consuming a second slot.
    runtime_load_ids: tuple[str, ...]
    order: int
    state: ResidentState
    current: bool
    pinned: bool
    in_flight: int

    @property
    def protected(self) -> bool:
        return self.current or self.pinned or self.in_flight > 0


class _Slot:
    __slots__ = ("in_flight", "name", "order", "pinned", "runtime_load_id", "runtime_load_ids", "scenario", "state")

    def __init__(self, name: str, scenario: str, runtime_load_id: str, order: int) -> None:
        self.name = name
        self.scenario = scenario
        self.runtime_load_id = runtime_load_id
        self.runtime_load_ids: list[str] = [runtime_load_id]
        self.order = order
        self.state: ResidentState = "active"
        self.pinned = False
        self.in_flight = 0


class AdapterLease:
    """Protects one resident adapter for the lifetime of one inference attempt."""

    def __init__(self, manager: AdapterResidencyManager, name: str) -> None:
        self._manager = manager
        self._name = name
        self._released = False

    @property
    def adapter_name(self) -> str:
        return self._name

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._release(self._name)


class AdapterResidencyManager:
    """Bounded, scenario-aware bookkeeping of the adapters one engine holds.

    Invariants:

    - one engine adapter name maps to exactly one (scenario, revision), so a
      recorded ``lora_path`` proves which durable adapter answered;
    - a scenario's current revision, any explicitly pinned revision, and any
      revision with an in-flight request are never evicted — except that a
      caller which has paused serving may supersede its own current revision
      when nothing else fits (``supersede=True``);
    - eviction is deterministic: the oldest unprotected activation goes
      first, regardless of which scenario owns it;
    - a failed load never occupies a slot; a failed unload keeps occupying
      one as ``leaked`` until a later unload succeeds, so capacity
      degradation is visible rather than hidden.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0):
            raise ValueError("adapter capacity must be a positive integer or None")
        self._capacity = capacity
        self._lock = RLock()
        self._slots: dict[str, _Slot] = {}
        self._current: dict[str, str] = {}
        self._next_order = 0
        self._counters = {
            "loads": 0,
            "load_failures": 0,
            "load_cleanups": 0,
            "cleanup_failures": 0,
            "unloads": 0,
            "unload_failures": 0,
            "evictions": 0,
            "capacity_rejections": 0,
            "strays_unloaded": 0,
            "lost_dropped": 0,
        }
        self._actions: deque[dict[str, str]] = deque(maxlen=RECENT_ACTIONS)

    @property
    def capacity(self) -> int | None:
        return self._capacity

    # -- Activation ------------------------------------------------------

    def activate(
        self,
        scenario: str,
        runtime_load_id: str,
        engine: AdapterEngine | None,
        *,
        payload: Any = None,
        supersede: bool = False,
    ) -> str:
        """Make ``runtime_load_id`` the revision ``scenario`` serves; return its engine name.

        Loads the adapter first and advances the scenario's current pointer
        only after the engine confirms the load, so a failure leaves the
        incumbent revision active and touches no other scenario. ``payload``
        is handed to the engine untouched. ``supersede`` lets the scenario's
        own current revision be evicted to make room — only safe while no
        request can reach it (generation paused).
        """
        _require_scenario(scenario)
        _require_runtime_load_id(runtime_load_id)
        name = adapter_name(scenario, runtime_load_id)
        with self._lock:
            slot = self._slots.get(name)
            if slot is not None and slot.state == "active":
                self._current[scenario] = name
                return name
            # A leaked slot still holds the engine's copy of exactly this
            # revision; reloading it in place reclaims the slot.
            if slot is not None and (unload_error := self._unload(slot, engine)) is not None:
                raise AdapterResidencyError(
                    f"adapter {name!r} leaked in the engine and cannot be reloaded until the unload "
                    f"succeeds: {unload_error}"
                ) from unload_error
            self._make_room(scenario, engine, supersede=supersede)
            slot = _Slot(name, scenario, runtime_load_id, self._next_order)
            self._next_order += 1
            self._slots[name] = slot
            try:
                if engine is not None:
                    engine.load_adapter(name, payload)
            except Exception as exc:
                # Never leave a phantom slot: it would evict a live peer later
                # to make room for an adapter the engine never held.
                self._slots.pop(name, None)
                self._counters["load_failures"] += 1
                self._note("load_failed", name, scenario, str(exc))
                self._cleanup_ambiguous_load(name, scenario, engine)
                raise AdapterResidencyError(
                    f"engine could not load adapter {name!r} for scenario {scenario!r}: {exc}"
                ) from exc
            self._counters["loads"] += 1
            self._current[scenario] = name
            return name

    def make_room(self, scenario: str, engine: AdapterEngine | None, *, supersede: bool = False) -> None:
        """Free one slot for an adapter ``scenario`` is about to load out of band.

        The training path publishes through its own transport (tensors
        pushed from the trainer, under a runtime_load_id it only knows once the
        publication completes) and then :meth:`register` the result. The
        eviction policy is :meth:`activate`'s.
        """
        _require_scenario(scenario)
        with self._lock:
            self._make_room(scenario, engine, supersede=supersede)

    def register(self, scenario: str, runtime_load_id: str) -> str:
        """Record an adapter the engine already holds as ``scenario``'s current revision.

        Pairs with :meth:`make_room` for loads the manager did not drive.
        Registering the resident revision again is idempotent; a slot that
        leaked under this name is reclaimed in place, since the engine just
        confirmed it holds exactly these bytes.
        """
        _require_scenario(scenario)
        _require_runtime_load_id(runtime_load_id)
        name = adapter_name(scenario, runtime_load_id)
        with self._lock:
            slot = self._slots.get(name)
            if slot is None:
                if self._capacity is not None and len(self._slots) >= self._capacity:
                    # The caller skipped make_room, or the engine let more in
                    # than the cap; account for it rather than hide it.
                    logger.warning(
                        "adapter %r registered beyond capacity %d; the engine holds %d adapters",
                        name,
                        self._capacity,
                        len(self._slots) + 1,
                    )
                slot = _Slot(name, scenario, runtime_load_id, self._next_order)
                self._next_order += 1
                self._slots[name] = slot
                self._counters["loads"] += 1
            elif slot.state == "leaked":
                slot.state = "active"
                self._note("reclaimed", name, scenario)
            self._current[scenario] = name
            return name

    def _cleanup_ambiguous_load(self, name: str, scenario: str, engine: AdapterEngine | None) -> None:
        """Make a failed load deterministic: the engine must not hold ``name``.

        A timeout or dropped connection cannot tell an engine that rejected
        the adapter from one that loaded it after Reef stopped waiting. Reef
        never counts the slot either way, so the engine is told to drop the
        name; a refusal for an adapter it never held is harmless, and one
        for an adapter it does hold is recorded so the leak is visible. The
        caller's retry then reloads under the same name with nothing hidden.
        """
        if engine is None:
            return
        try:
            engine.unload_adapter(name)
        except Exception as exc:
            self._counters["cleanup_failures"] += 1
            self._note("cleanup_failed", name, scenario, str(exc))
            logger.info("engine did not drop adapter %r after its failed load: %s", name, exc)
            return
        self._counters["load_cleanups"] += 1
        self._note("load_cleanup", name, scenario)

    def alias(self, scenario: str, runtime_load_id: str) -> str:
        """Serve ``runtime_load_id`` through the adapter ``scenario`` currently holds.

        Rollback republishes an older revision's bytes as a new artifact
        runtime_load_id. The engine already holds those bytes under the source name,
        so the new runtime_load_id routes there instead of loading a duplicate.
        """
        _require_scenario(scenario)
        if not runtime_load_id:
            raise ValueError("adapter alias requires a non-empty runtime load")
        with self._lock:
            name = self._current.get(scenario)
            slot = self._slots.get(name) if name is not None else None
            if slot is None or slot.state != "active":
                raise AdapterNotActive(
                    f"scenario {scenario!r} has no active adapter to alias runtime load {runtime_load_id!r} to"
                )
            if runtime_load_id not in slot.runtime_load_ids:
                slot.runtime_load_ids.append(runtime_load_id)
            return slot.name

    def release_scenario(self, scenario: str) -> None:
        """Forget which revision ``scenario`` serves; its adapters become evictable."""
        with self._lock:
            self._current.pop(scenario, None)

    # -- Routing ---------------------------------------------------------

    def resolve(self, scenario: str, runtime_load_id: str) -> ResidentAdapter:
        """The active adapter serving ``runtime_load_id`` for ``scenario``; fail closed otherwise."""
        with self._lock:
            slot = self._slot_for(scenario, runtime_load_id)
            return self._describe(slot)

    def lease(self, scenario: str, runtime_load_id: str) -> AdapterLease:
        """Resolve and protect the adapter for one in-flight request."""
        with self._lock:
            slot = self._slot_for(scenario, runtime_load_id)
            slot.in_flight += 1
            return AdapterLease(self, slot.name)

    def _slot_for(self, scenario: str, runtime_load_id: str) -> _Slot:
        current_name = self._current.get(scenario)
        if current_name is None:
            raise AdapterNotActive(
                f"scenario {scenario!r} has no active adapter; its committed revision was never activated"
            )
        slot = self._slots.get(current_name)
        if slot is None or slot.state != "active":
            raise AdapterNotActive(f"adapter {current_name!r} for scenario {scenario!r} is not active in the engine")
        if runtime_load_id not in slot.runtime_load_ids:
            raise AdapterNotActive(
                f"scenario {scenario!r} serves adapter revision {slot.runtime_load_id!r}, not the requested {runtime_load_id!r}"
            )
        return slot

    def _release(self, name: str) -> None:
        with self._lock:
            slot = self._slots.get(name)
            if slot is not None and slot.in_flight > 0:
                slot.in_flight -= 1

    # -- Pinning ---------------------------------------------------------

    def pin(self, scenario: str, runtime_load_id: str) -> None:
        """Protect a resident revision from eviction until :meth:`unpin`."""
        with self._lock:
            slot = self._resident_slot(scenario, runtime_load_id)
            slot.pinned = True

    def unpin(self, scenario: str, runtime_load_id: str) -> None:
        with self._lock:
            slot = self._resident_slot(scenario, runtime_load_id)
            slot.pinned = False

    def _resident_slot(self, scenario: str, runtime_load_id: str) -> _Slot:
        for slot in self._slots.values():
            if slot.scenario == scenario and runtime_load_id in slot.runtime_load_ids:
                return slot
        raise AdapterNotActive(
            f"scenario {scenario!r} has no resident adapter for runtime_load_id {runtime_load_id!r}"
        )

    # -- Capacity --------------------------------------------------------

    def _make_room(self, scenario: str, engine: AdapterEngine | None, *, supersede: bool) -> None:
        if self._capacity is None:
            return
        superseding = self._current.get(scenario) if supersede else None
        while len(self._slots) >= self._capacity:
            victim = self._eviction_candidate(superseding)
            if victim is None:
                self._counters["capacity_rejections"] += 1
                protected = sorted(slot.name for slot in self._slots.values())
                raise AdapterCapacityExhausted(
                    f"engine adapter capacity {self._capacity} is exhausted and every resident adapter is "
                    f"protected (current, pinned, or in flight): {protected}; scenario {scenario!r} cannot activate"
                )
            if (unload_error := self._unload(victim, engine)) is not None:
                # The engine refused to let go; the slot stays occupied and
                # the caller sees exhaustion rather than a silent overcommit.
                # The unload failure rides along: "capacity exhausted" alone
                # would hide that the real event may be a dead engine.
                self._counters["capacity_rejections"] += 1
                raise AdapterCapacityExhausted(
                    f"engine adapter capacity {self._capacity} is exhausted and evicting {victim.name!r} "
                    f"failed ({unload_error}); scenario {scenario!r} cannot activate until the leaked slot "
                    f"is reclaimed"
                ) from unload_error
            self._counters["evictions"] += 1
            self._note("evicted", victim.name, victim.scenario, f"for scenario {scenario!r}")

    def _eviction_candidate(self, superseding: str | None) -> _Slot | None:
        candidates = [
            slot for slot in self._slots.values() if not self._protected(slot, exempt_current=slot.name == superseding)
        ]
        if not candidates:
            return None
        # Leaked slots are retried first: reclaiming one costs no live adapter.
        candidates.sort(key=lambda slot: (slot.state != "leaked", slot.order))
        return candidates[0]

    def _unload(self, slot: _Slot, engine: AdapterEngine | None) -> Exception | None:
        """Drop ``slot`` from the engine and the table; a returned failure leaves it ``leaked``."""
        try:
            if engine is not None:
                engine.unload_adapter(slot.name)
        except Exception as exc:
            self._counters["unload_failures"] += 1
            slot.state = "leaked"
            self._note("leaked", slot.name, slot.scenario, str(exc))
            logger.warning("could not unload adapter %r; its engine slot leaks", slot.name, exc_info=True)
            return exc
        self._counters["unloads"] += 1
        self._slots.pop(slot.name, None)
        for scenario, name in list(self._current.items()):
            if name == slot.name:
                self._current.pop(scenario, None)
        return None

    # -- Reconciliation --------------------------------------------------

    def reconcile(self, engine_names: Iterable[str], engine: AdapterEngine | None) -> tuple[str, ...]:
        """Align bookkeeping with what the engine reports after a restart.

        Names the engine holds but Reef never activated are unloaded (they
        belong to a dead process); names Reef tracks but the engine lost are
        dropped so the scenario's next activation reloads them. Returns the
        names dropped from tracking.
        """
        held = set(engine_names)
        with self._lock:
            for name in sorted(held - set(self._slots)):
                stray = _Slot(name, scenario="", runtime_load_id="", order=-1)
                self._slots[name] = stray
                if self._unload(stray, engine) is None:
                    self._counters["strays_unloaded"] += 1
                    self._note("stray_unloaded", name, "")
            lost = tuple(sorted(name for name in self._slots if name not in held and self._slots[name].order >= 0))
            for name in lost:
                slot = self._slots.pop(name)
                self._current.pop(slot.scenario, None)
                self._counters["lost_dropped"] += 1
                self._note("lost_dropped", name, slot.scenario)
            return lost

    def _note(self, action: str, adapter: str, scenario: str, detail: str | None = None) -> None:
        entry = {"action": action, "adapter": adapter, "scenario": scenario}
        if detail:
            entry["detail"] = detail[:200]
        self._actions.append(entry)

    # -- Observation -----------------------------------------------------

    def resident(self) -> tuple[ResidentAdapter, ...]:
        with self._lock:
            return tuple(self._describe(slot) for slot in sorted(self._slots.values(), key=lambda s: s.order))

    def current(self, scenario: str) -> ResidentAdapter | None:
        with self._lock:
            name = self._current.get(scenario)
            slot = self._slots.get(name) if name is not None else None
            return None if slot is None else self._describe(slot)

    def status(self) -> dict[str, Any]:
        """A bounded status block: global capacity plus per-scenario residency."""
        with self._lock:
            resident = [self._describe(slot) for slot in sorted(self._slots.values(), key=lambda s: s.order)]
            scenarios: dict[str, dict[str, Any]] = {}
            for entry in resident:
                if not entry.scenario:
                    continue
                block = scenarios.setdefault(entry.scenario, {"current": None, "resident": []})
                block["resident"].append(entry.runtime_load_id)
                if entry.current:
                    # The served runtime_load_id is the newest alias; the adapter name
                    # still says which revision's bytes the engine holds.
                    block["current"] = {"runtime_load_id": entry.runtime_load_ids[-1], "adapter": entry.name}
            return {
                "capacity": self._capacity,
                "resident": len(resident),
                "leaked": sum(1 for entry in resident if entry.state == "leaked"),
                "protected": sum(1 for entry in resident if entry.protected),
                "in_flight": sum(entry.in_flight for entry in resident),
                "counters": dict(self._counters),
                "recent_actions": list(self._actions),
                "scenarios": scenarios,
            }

    def _protected(self, slot: _Slot, *, exempt_current: bool = False) -> bool:
        if slot.pinned or slot.in_flight > 0:
            return True
        return not exempt_current and self._current.get(slot.scenario) == slot.name

    def _describe(self, slot: _Slot) -> ResidentAdapter:
        return ResidentAdapter(
            name=slot.name,
            scenario=slot.scenario,
            runtime_load_id=slot.runtime_load_id,
            runtime_load_ids=tuple(slot.runtime_load_ids),
            order=slot.order,
            state=slot.state,
            current=self._current.get(slot.scenario) == slot.name,
            pinned=slot.pinned,
            in_flight=slot.in_flight,
        )


def _require_scenario(scenario: str) -> None:
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("adapter residency requires a non-empty scenario name")


def _require_runtime_load_id(runtime_load_id: str) -> None:
    if not isinstance(runtime_load_id, str) or not runtime_load_id:
        raise ValueError("adapter residency requires a non-empty runtime_load_id")


__all__ = [
    "AdapterCapacityExhausted",
    "AdapterEngine",
    "AdapterLease",
    "AdapterNotActive",
    "AdapterResidencyError",
    "AdapterResidencyManager",
    "ResidentAdapter",
]
