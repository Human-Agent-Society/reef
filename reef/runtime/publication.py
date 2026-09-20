"""Weight publication, commit gating, version allocation, and LoRA residency.

Three objects cooperate here, bottom-up:

- :class:`AdapterResidencyManager` accounts for the LoRA adapters one shared
  engine holds, so several scenarios cannot overcommit its slots.
- :class:`BackendWeightPublisher` moves weights from a training sender to an
  inference receiver, allocating each transfer's identity and verifying
  every engine afterwards.
- :class:`TrainingPublication` owns the durable job-marker transitions and
  the barrier that keeps a published version unserved until Reef commits it.

:class:`WeightUpdateLock` fences the native transport those objects drive.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal

from reef.core.errors import ReefError
from reef.runtime.interfaces import (
    AdapterEngine,
    InferenceBackend,
    RuntimeLoadId,
    ScenarioHistoryStore,
    TrainingBackend,
    TrainingJobState,
    TrainingJobStore,
)
from reef.surface.adapter import adapter_name, parse_adapter_name
from reef.surface.base import InferenceLease

logger = logging.getLogger(__name__)

# -- Adapter residency --------------------------------------------------------

ResidentState = Literal["active", "leaked"]

#: How many recovery actions ``status()`` keeps; enough to explain the most
#: recent restart or capacity incident without growing with uptime.
RECENT_ACTIONS = 32

#: Order of a slot that only mirrors a stray engine name; never a real activation.
STRAY_ORDER = -1


class AdapterResidencyError(ReefError):
    """The shared engine cannot make a scenario's adapter revision servable."""


class AdapterCapacityExhausted(AdapterResidencyError):
    """Every loaded adapter is protected, so nothing can be evicted for a new one.

    Raised before anything is unloaded, so the engine still holds exactly what
    it held and every scenario keeps serving: the publication was refused, not
    half-applied.
    """


class AdapterEvictionFailed(AdapterCapacityExhausted):
    """The engine refused to release the slot chosen for eviction.

    Still a capacity failure for the caller, but a different one: a plain
    :class:`AdapterCapacityExhausted` proves the engine is fine, while this
    says it would not let go and may be wedged. Callers that recover engines
    must branch on this subclass *before* the base class.
    """


class AdapterNotActive(AdapterResidencyError):
    """A request resolved to an adapter revision the engine does not hold."""


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


@dataclass(eq=False)
class _Slot:
    """Mutable bookkeeping for one engine adapter name."""

    name: str
    scenario: str
    runtime_load_id: str
    order: int
    runtime_load_ids: list[str] = field(init=False)
    state: ResidentState = "active"
    pinned: bool = False
    in_flight: int = 0

    def __post_init__(self) -> None:
        self.runtime_load_ids = [self.runtime_load_id]


class AdapterLease(InferenceLease):
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
        self._manager.release_lease(self._name)


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

    Residency bounds what the engine can serve *now*; it says nothing about
    which recorded samples a training job may still use. Those are separate
    by construction — a sample carries its own log probabilities and producing
    runtime load ID — so evicting a revision never narrows the staleness
    window, and widening the staleness window never demands more slots.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0):
            raise ValueError("adapter capacity must be a positive integer or None")
        self._capacity = capacity
        self._lock = RLock()
        self._slots: dict[str, _Slot] = {}
        #: Scenario name -> engine adapter name of the revision it serves.
        self._current: dict[str, str] = {}
        self._next_order = 0
        self._counters = dict.fromkeys(
            (
                "loads",
                "load_failures",
                "load_cleanups",
                "cleanup_failures",
                "unloads",
                "unload_failures",
                "evictions",
                "capacity_rejections",
                "strays_unloaded",
                "lost_dropped",
            ),
            0,
        )
        self._actions: deque[dict[str, str]] = deque(maxlen=RECENT_ACTIONS)

    @property
    def capacity(self) -> int | None:
        return self._capacity

    # -- Activation

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
            self._add_slot(name, scenario, runtime_load_id)
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
                self._add_slot(name, scenario, runtime_load_id)
                self._counters["loads"] += 1
            elif slot.state == "leaked":
                slot.state = "active"
                self._note("reclaimed", name, scenario)
            self._current[scenario] = name
            return name

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
            slot = self._current_slot(scenario)
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

    def _add_slot(self, name: str, scenario: str, runtime_load_id: str) -> _Slot:
        slot = _Slot(name, scenario, runtime_load_id, self._next_order)
        self._next_order += 1
        self._slots[name] = slot
        return slot

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

    # -- Routing

    def resolve(self, scenario: str, runtime_load_id: str) -> ResidentAdapter:
        """The active adapter serving ``runtime_load_id`` for ``scenario``; fail closed otherwise."""
        with self._lock:
            return self._describe(self._serving_slot(scenario, runtime_load_id))

    def lease(self, scenario: str, runtime_load_id: str) -> AdapterLease:
        """Resolve and protect the adapter for one in-flight request."""
        with self._lock:
            slot = self._serving_slot(scenario, runtime_load_id)
            slot.in_flight += 1
            return AdapterLease(self, slot.name)

    def release_lease(self, name: str) -> None:
        """Drop one in-flight protection; :class:`AdapterLease` calls this once."""
        with self._lock:
            slot = self._slots.get(name)
            if slot is not None and slot.in_flight > 0:
                slot.in_flight -= 1

    def _serving_slot(self, scenario: str, runtime_load_id: str) -> _Slot:
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

    def _current_slot(self, scenario: str) -> _Slot | None:
        name = self._current.get(scenario)
        return None if name is None else self._slots.get(name)

    # -- Pinning

    def pin(self, scenario: str, runtime_load_id: str) -> None:
        """Protect a resident revision from eviction until :meth:`unpin`."""
        with self._lock:
            self._resident_slot(scenario, runtime_load_id).pinned = True

    def unpin(self, scenario: str, runtime_load_id: str) -> None:
        with self._lock:
            self._resident_slot(scenario, runtime_load_id).pinned = False

    def _resident_slot(self, scenario: str, runtime_load_id: str) -> _Slot:
        for slot in self._slots.values():
            if slot.scenario == scenario and runtime_load_id in slot.runtime_load_ids:
                return slot
        raise AdapterNotActive(
            f"scenario {scenario!r} has no resident adapter for runtime_load_id {runtime_load_id!r}"
        )

    # -- Capacity

    def _make_room(self, scenario: str, engine: AdapterEngine | None, *, supersede: bool) -> None:
        if self._capacity is None:
            return
        superseding = self._current.get(scenario) if supersede else None
        while len(self._slots) >= self._capacity:
            victim = self._eviction_candidate(superseding)
            if victim is None:
                self._counters["capacity_rejections"] += 1
                raise self._capacity_exhausted(scenario)
            if (unload_error := self._unload(victim, engine)) is not None:
                # The engine refused to let go; the slot stays occupied and
                # the caller sees exhaustion rather than a silent overcommit.
                # The unload failure rides along: "capacity exhausted" alone
                # would hide that the real event may be a dead engine.
                self._counters["capacity_rejections"] += 1
                raise AdapterEvictionFailed(
                    f"engine adapter capacity {self._capacity} is exhausted and evicting {victim.name!r} "
                    f"failed ({unload_error}); scenario {scenario!r} cannot activate until the leaked slot "
                    f"is reclaimed"
                ) from unload_error
            self._counters["evictions"] += 1
            self._note("evicted", victim.name, victim.scenario, f"for scenario {scenario!r}")

    def _capacity_exhausted(self, scenario: str) -> AdapterCapacityExhausted:
        protected = sorted(self._slots)
        # Name the remedy: the usual cause is a capacity sized for one
        # scenario on an engine that several now share, and the operator
        # cannot infer the needed slot count from the names.
        sharing = {slot.scenario for slot in self._slots.values()} | {scenario}
        return AdapterCapacityExhausted(
            f"engine adapter capacity {self._capacity} is exhausted and every resident adapter is "
            f"protected (current, pinned, or in flight): {protected}; scenario {scenario!r} cannot "
            f"activate. {len(sharing)} scenarios share this engine and each keeps its current "
            f"revision resident, so it needs at least {len(sharing) + 1} slots: raise "
            f"--max-loaded-loras. Nothing was unloaded; every scenario keeps serving."
        )

    def _eviction_candidate(self, superseding: str | None) -> _Slot | None:
        candidates = [
            slot for slot in self._slots.values() if not self._protected(slot, exempt_current=slot.name == superseding)
        ]
        if not candidates:
            return None
        # Leaked slots are retried first: reclaiming one costs no live adapter.
        return min(candidates, key=lambda slot: (slot.state != "leaked", slot.order))

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
        if self._current.get(slot.scenario) == slot.name:
            self._current.pop(slot.scenario)
        return None

    # -- Reconciliation

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
                stray = _Slot(name, scenario="", runtime_load_id="", order=STRAY_ORDER)
                self._slots[name] = stray
                if self._unload(stray, engine) is None:
                    self._counters["strays_unloaded"] += 1
                    self._note("stray_unloaded", name, "")
            lost = tuple(
                sorted(name for name, slot in self._slots.items() if name not in held and slot.order != STRAY_ORDER)
            )
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

    # -- Observation

    def resident(self) -> tuple[ResidentAdapter, ...]:
        with self._lock:
            return tuple(self._describe(slot) for slot in self._ordered_slots())

    def current(self, scenario: str) -> ResidentAdapter | None:
        with self._lock:
            slot = self._current_slot(scenario)
            return None if slot is None else self._describe(slot)

    def status(self) -> dict[str, Any]:
        """A bounded status block: global capacity plus per-scenario residency."""
        with self._lock:
            resident = [self._describe(slot) for slot in self._ordered_slots()]
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

    def _ordered_slots(self) -> list[_Slot]:
        return sorted(self._slots.values(), key=lambda slot: slot.order)

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


# -- Transport fencing --------------------------------------------------------


class WeightUpdateLock:
    """Serialize fan-out and fence an uncertain transport.

    Host this object in a serial actor; it does not manage a thread or process.
    """

    def __init__(self) -> None:
        self._locked = False
        self._poisoned = False
        self._completed_phases: dict[str, dict[str, str | None]] = {}

    def acquire(self) -> bool:
        if self._poisoned:
            raise RuntimeError("rollout engine lock is poisoned; reconnect weight-update transports")
        if self._locked:
            return False
        self._locked = True
        return True

    def release(self) -> None:
        if self._poisoned:
            raise RuntimeError("poisoned rollout engine lock cannot be released")
        if not self._locked:
            raise RuntimeError("rollout engine lock is not acquired")
        self._locked = False

    def poison(self) -> None:
        self._locked = True
        self._poisoned = True

    def status(self) -> dict[str, bool]:
        return {"locked": self._locked, "poisoned": self._poisoned}

    def complete_phase(self, phase_id: str, error: str | None = None) -> None:
        if not isinstance(phase_id, str) or not phase_id:
            raise ValueError("phase_id must be a non-empty string")
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError("phase error must be a non-empty string or None")
        self._completed_phases[phase_id] = {"error": error}

    def phase_status(self, phase_id: str) -> dict[str, str | None] | None:
        return self._completed_phases.get(phase_id)

    def clear_phases(self) -> None:
        if self._poisoned or self._locked:
            raise RuntimeError("cannot clear phases while the rollout transport is active")
        self._completed_phases.clear()


# -- Durable publication ------------------------------------------------------

#: Marker states whose publication already happened; a repeat is a replay.
PUBLISHED_STATES = frozenset({"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"})


class WeightPublisher(ABC):
    """Model operations needed by Reef's durable publication transaction.

    Transport stays in the backend: ``publish`` must verify that all engines
    received the returned runtime load ID, without resuming generation.
    ``recover`` replaces uncertain engines and restores other committed adapters;
    its marker is ``None`` when no durable training job exists.
    ``pause`` and ``resume`` are idempotent barriers across every serving engine.
    ``abort`` prevents inference after an uncertain or partial update.
    """

    @abstractmethod
    def recover(self, marker: Mapping[str, Any] | None) -> None: ...

    @abstractmethod
    def pause(self) -> None: ...

    @abstractmethod
    def publish(self, marker: Mapping[str, Any], *, force_full: bool) -> str: ...

    @abstractmethod
    def republish(self, runtime_load_id: str, marker: Mapping[str, Any] | None) -> str:
        """Resend unchanged trainer weights with a full transfer and the same identity."""

    @abstractmethod
    def resume(self) -> None: ...

    @abstractmethod
    def restore_incumbent(self) -> None: ...

    @abstractmethod
    def abort(self) -> None: ...


@dataclass(frozen=True)
class PublicationResult:
    """A durable result and whether this call performed its publication."""

    marker: Mapping[str, Any]
    published: bool


class TrainingPublication:
    """Own marker transitions and the barrier that gates serving on Reef's commit.

    The enclosing training coordinator must serialize these operations with
    training and shutdown. A single shared ``phase`` also lets its training and
    checkpoint operations report progress through the same health endpoint.
    No tensor data passes through this object. A missing marker path supports
    legacy coordinators that never execute durable training jobs.
    """

    def __init__(
        self, store: TrainingJobStore | None, publisher: WeightPublisher, state: TrainingJobState | None = None
    ) -> None:
        self._store = store
        self._publisher = publisher
        self.state = TrainingJobState() if state is None else state

    @property
    def phase(self) -> str:
        return self.state.phase

    @phase.setter
    def phase(self, phase: str) -> None:
        self.state.phase = phase

    def _require_store(self) -> TrainingJobStore:
        if self._store is None:
            raise RuntimeError("training job checkpoint path is not configured")
        return self._store

    def _job_marker(self, job_id: str) -> dict[str, Any]:
        if not job_id:
            raise ValueError("training_job_id must be non-empty")
        marker = self._require_store().read()
        if marker is None or marker["job_id"] != job_id:
            raise RuntimeError(f"unknown training job {job_id!r}")
        return marker

    def _abort(self) -> None:
        self.phase = "weight_sync_failed"
        with suppress(Exception):
            self._publisher.abort()

    @contextmanager
    def _aborting_on_failure(self) -> Iterator[None]:
        """Mark the publication failed and fence inference if the body raises."""
        try:
            yield
        except BaseException:
            self._abort()
            raise

    def publish(self, job_id: str) -> PublicationResult:
        """Publish one checkpoint, leaving every engine paused."""
        marker = self._job_marker(job_id)
        status = marker["status"]
        if status in PUBLISHED_STATES:
            return PublicationResult(marker, published=False)
        if status not in {"CHECKPOINT", "UPDATING_WEIGHTS"}:
            raise RuntimeError(f"training job is {status}; operator recovery required")
        recovering = status == "UPDATING_WEIGHTS"
        # Whether the marker already says the engines need rebuilding. Every
        # failure below retires them, so the answer has to be yes before this
        # call returns unsuccessfully.
        engines_declared_uncertain = recovering
        try:
            if recovering:
                self._publisher.recover(marker)
            self._publisher.pause()
            # Persist uncertainty once every engine has crossed the pause
            # barrier, so a publication that fails from here on replays as a
            # recovery rather than as a fresh checkpoint.
            if not recovering:
                self._require_store().transition(marker, "UPDATING_WEIGHTS")
                engines_declared_uncertain = True
            self.phase = "publishing"
            version = self._publisher.publish(marker, force_full=recovering)
            if not isinstance(version, str) or not version:
                raise RuntimeError("weight publisher returned an empty runtime load ID")
            self._require_store().transition(marker, "READY_TO_COMMIT", runtime_load_id=version)
            self.phase = "awaiting_commit"
        except AdapterEvictionFailed:
            self._abort()
            raise
        except AdapterCapacityExhausted:
            # No weights left the trainer; terminating unrelated adapters would
            # not resolve capacity pressure. The pending marker still gates commit.
            self.phase = "serving"
            raise
        except BaseException:
            self._abort()
            if not engines_declared_uncertain:
                self._declare_engines_uncertain(marker)
            raise
        return PublicationResult(marker, published=True)

    def _declare_engines_uncertain(self, marker: dict[str, Any]) -> None:
        """Record that a failed barrier retired the engines before the marker moved.

        Retiring them made CHECKPOINT unreplayable: a retry from it would pause
        handles that no longer exist. UPDATING_WEIGHTS is the status whose
        replay rebuilds the engines and forces a full publication.
        """
        with suppress(Exception):
            self._require_store().transition(marker, "UPDATING_WEIGHTS")

    def republish(self, runtime_load_id: str) -> str:
        """Restore replaced engines without bypassing the durable commit gate.

        A trainer with an unfinished or rejected candidate cannot represent the
        incumbent. Such jobs must use their normal publication/recovery path.
        The owner must retain the last verified identity across failed attempts.
        """
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise ValueError("republication requires a non-empty runtime load ID")
        if self.phase == "stopped":
            raise RuntimeError("training coordinator is stopped")
        marker = self._store.read() if self._store is not None else None
        if marker is not None:
            if marker["status"] not in PUBLISHED_STATES:
                raise RuntimeError(f"cannot republish serving from {marker['status']}; use training job recovery")
            if marker["runtime_load_id"] != runtime_load_id:
                raise RuntimeError("serving runtime load ID does not match the training marker")
        with self._aborting_on_failure():
            # Fence before recovery: replacement engines must inherit pause
            # intent, and monitoring must not restart before verified transfer.
            self._publisher.pause()
            self.phase = "publishing"
            self._publisher.recover(marker)
            published = self._publisher.republish(runtime_load_id, marker)
            if published != runtime_load_id:
                raise RuntimeError(
                    f"serving republication changed runtime load ID {runtime_load_id!r} to {published!r}"
                )
        self.finish_recovery(marker, published)
        return published

    def reject(self, job_id: str) -> Mapping[str, Any]:
        """Durably reject before restoring the incumbent's released resources."""
        marker = self._job_marker(job_id)
        status = marker["status"]
        if status == "REJECTED":
            return marker
        if status == "CHECKPOINT":
            self._require_store().transition(marker, "REJECTING")
        elif status != "REJECTING":
            raise RuntimeError(f"cannot reject training job {job_id!r} from {status}")
        self._publisher.restore_incumbent()
        self._require_store().transition(marker, "REJECTED")
        self.phase = "serving"
        return marker

    def acknowledge(self, job_id: str) -> None:
        """Persist Reef's commit acknowledgement before allowing requests again."""
        marker = self._job_marker(job_id)
        if marker["status"] == "COMPLETE":
            if marker.get("commit_acknowledged") is not True:
                self._require_store().write({**marker, "commit_acknowledged": True})
            return
        if marker["status"] == "READY_TO_COMMIT":
            self._require_store().transition(marker, "HEAD_COMMITTED", commit_acknowledged=True)
        if marker["status"] != "HEAD_COMMITTED":
            raise RuntimeError(f"cannot acknowledge training job {job_id!r} from {marker['status']}")
        self._publisher.resume()
        self._require_store().transition(marker, "COMPLETE", commit_acknowledged=True)
        self.phase = "serving"

    @contextmanager
    def recovery(self, marker: dict[str, Any] | None) -> Iterator[None]:
        """Fence startup restoration and abort any failed backend reconstruction.

        The backend restores checkpoint state and calls ``finish_recovery``
        inside this scope. A successful transfer alone cannot resume serving.
        """
        with self._aborting_on_failure():
            self.prepare_recovery(marker)
            yield

    def prepare_recovery(self, marker: dict[str, Any] | None) -> None:
        """Reassert startup pause, including committed and marker-free restarts."""
        if marker is not None and marker["status"] == "RUNNING":
            raise RuntimeError(f"ambiguous training job {marker['job_id']}")
        try:
            with self._aborting_on_failure():
                self._publisher.pause()
                self.phase = "recovering"
                if marker is not None and marker["status"] == "CHECKPOINT":
                    self._require_store().transition(marker, "UPDATING_WEIGHTS")
        except BaseException:
            if marker is not None and marker["status"] == "CHECKPOINT":
                # Startup retires the engines the same way a publication does,
                # so the next boot must rebuild them instead of replaying a
                # CHECKPOINT whose engines are gone.
                self._declare_engines_uncertain(marker)
            raise

    def finish_recovery(self, marker: dict[str, Any] | None, runtime_load_id: str) -> None:
        """Record recovered publication while preserving the durable commit gate.

        The backend has already restored checkpoint tensors and verified every
        engine. Previously published jobs must retain their serving identity.
        """
        status = None if marker is None else marker["status"]
        with self._aborting_on_failure():
            if not isinstance(runtime_load_id, str) or not runtime_load_id:
                raise RuntimeError("weight publisher returned an empty runtime load ID")
            if marker is not None and status in PUBLISHED_STATES and runtime_load_id != marker["runtime_load_id"]:
                raise RuntimeError(
                    "checkpoint republication changed runtime load ID "
                    f"{marker['runtime_load_id']!r} to {runtime_load_id!r}"
                )
            if marker is not None and status == "UPDATING_WEIGHTS":
                self._require_store().transition(marker, "READY_TO_COMMIT", runtime_load_id=runtime_load_id)
                self.phase = "awaiting_commit"
            elif status == "READY_TO_COMMIT":
                self.phase = "awaiting_commit"
            elif marker is not None and status == "HEAD_COMMITTED":
                self.acknowledge(str(marker["job_id"]))
            elif status in {None, "COMPLETE", "REJECTED"}:
                self._publisher.resume()
                self.phase = "serving"
            else:
                raise RuntimeError(f"training marker is {status}; operator recovery required")


# -- Backend weight transfer --------------------------------------------------


class BackendWeightPublisher(WeightPublisher, AdapterEngine):
    """Publish weights across independently owned training and inference backends.

    This object owns transfer identities, colocated memory handoff, adapter
    residency and receiver verification. TrainingPublication supplies the
    durable commit barrier; the scheduler serializes both with optimizer work.
    As the :class:`AdapterEngine` behind its residency manager, it loads an
    adapter by asking the sender for it and unloads one at the receiver.
    """

    def __init__(
        self,
        training: TrainingBackend,
        inference: InferenceBackend,
        state: TrainingJobState,
        store: TrainingJobStore | None,
    ) -> None:
        self._store = store
        self.training = training
        self.inference = inference
        self.context = training.context
        self.config = training.config
        self.state = state
        self.history = self.context.history
        self.residency = AdapterResidencyManager(self.config.adapter_capacity) if self.config.lora else None
        # With the LoRA base kept resident, a colocated step releases only the
        # KV cache and CUDA graphs; ``None`` releases everything.
        self.release_tags = (
            ("kv_cache", "cuda_graph")
            if (self.config.lora and self.config.colocate and self.config.keep_lora_base_resident)
            else None
        )
        self.generation_paused = False

    @property
    def runtime_load_id(self) -> str:
        return self.context.runtime_load_id

    @runtime_load_id.setter
    def runtime_load_id(self, value: str) -> None:
        self.context.runtime_load_id = value

    def require_history(self) -> ScenarioHistoryStore:
        """The per-scenario history; only LoRA runs with a checkpoint save path keep one."""
        if self.history is None:
            raise RuntimeError("scenario bookkeeping requires LoRA training with a checkpoint save path")
        return self.history

    def require_residency(self) -> AdapterResidencyManager:
        if self.residency is None:
            raise RuntimeError("adapter residency requires a LoRA bridge")
        return self.residency

    def marker_scenario(self, marker: Mapping[str, Any] | None) -> str | None:
        """The scenario a marker's publication belongs to, when the bridge trains per scenario."""
        if self.history is None or marker is None:
            return None
        scenario = marker.get("scenario")
        return str(scenario) if isinstance(scenario, str) and scenario else None

    # -- Adapter engine

    def load_adapter(self, name: str, payload: Any) -> None:
        scenario, version = parse_adapter_name(name)
        files = self.training.adapter_files(scenario, version)
        if files is not None:
            self.inference.load_adapter_files(name, files, None)
            return
        self.training.send_adapter(scenario, name)

    def unload_adapter(self, name: str) -> None:
        self.inference.unload_adapter(name)

    # -- Generation barrier

    def pause_generation(self, *, reconcile: bool = False) -> None:
        if self.generation_paused and not reconcile:
            return
        self.inference.pause()
        self.generation_paused = True

    def pause(self) -> None:
        # Reassert the owner barrier even if the bridge cached a prior pause;
        # replacement controllers/engines may not have observed that RPC.
        self.pause_generation(reconcile=True)

    def resume(self) -> None:
        if not self.generation_paused:
            return
        self.inference.resume()
        self.generation_paused = False

    def abort(self) -> None:
        self.inference.abort()

    # -- Colocated memory handoff

    def prepare_training(self) -> None:
        if self.config.colocate:
            self.pause_generation()
            self.inference.offload(self.release_tags)

    def restore_incumbent(self) -> None:
        if not self.config.colocate:
            return
        # Pairs with the training step's offload: resuming a region that was
        # never released fails, because the receiver resumes by removing the tag
        # from the set release added it to.
        if self.release_tags is None:
            self.inference.onload_weights()
        self.inference.onload_kv()
        self.resume()

    # -- Version identity

    def initialize_version(self) -> None:
        self.training.initialize_version(self.runtime_load_id)
        self.inference.initialize_version(self.runtime_load_id)
        self._verify_engines_serve(self.runtime_load_id, "after version sync")

    def next_runtime_load_id(self) -> str:
        """Allocate the next serving identity independently of a sender attempt."""
        current = RuntimeLoadId.parse(self.runtime_load_id)
        return str(RuntimeLoadId(current.incarnation, current.sequence + 1))

    def publication_target(self, marker: Mapping[str, Any] | None) -> str:
        if marker is not None:
            target = marker.get("target_runtime_load_id")
            if isinstance(target, str) and target:
                return target
        target = self.next_runtime_load_id()
        if marker is not None:
            # Persist intent before any bytes leave the sender. An uncertain
            # partial transfer and process restart must reuse the same target.
            if self._store is None:
                raise RuntimeError("training checkpoint path is not configured")
            self._store.write({**marker, "target_runtime_load_id": target})
        return target

    def _verify_engines_serve(self, runtime_load_id: str, moment: str) -> None:
        observed = [str(value) for value in self.inference.runtime_load_ids()]
        if not observed or set(observed) != {runtime_load_id}:
            raise RuntimeError(f"serving engines disagree {moment}: {observed!r}")

    # -- Weight transfer

    def update_serving(
        self, *, force_full: bool = False, scenario: str | None = None, runtime_load_id: str | None = None
    ) -> str:
        """Publish the group's weights; ``scenario`` names the adapter a LoRA publication belongs to.

        A per-scenario adapter publication loads a new versioned name into
        every engine, so the residency manager frees a slot first (evicting
        the publishing scenario's own current revision when nothing else
        fits: generation is paused, so no request observes the gap) and
        records the published revision afterwards.

        Admission runs before any weight leaves the trainer. A capacity
        rejection therefore means nothing was published and every engine still
        serves what it served, so it must not terminate them — that took down
        scenarios which were never part of the publication (#65). An eviction
        the engine refused is the opposite: its state is uncertain, so the
        terminate-and-recover path stays (#61).
        """
        residency = self.residency if scenario is not None else None
        try:
            self.state.phase = "publishing"
            target = runtime_load_id or self.next_runtime_load_id()
            if residency is not None and scenario is not None:
                residency.make_room(scenario, self, supersede=True)
            version = self._transfer(target, force_full=force_full, scenario=scenario)
            if residency is not None and scenario is not None:
                residency.register(scenario, version)
        except AdapterEvictionFailed:
            self._fail_transfer()
            raise
        except AdapterCapacityExhausted:
            # Admission was refused before any weight left the trainer.
            self.state.phase = "serving"
            raise
        except BaseException:
            self._fail_transfer()
            raise
        self.runtime_load_id = version
        return version

    def _transfer(self, target: str, *, force_full: bool, scenario: str | None = None) -> str:
        """Send the trainer's weights as ``target`` and verify every engine received them."""
        if scenario is not None:
            files = self.training.adapter_files(scenario, target)
            if files is not None:
                # A trainer that delivers files never sends: Reef's receiver loads the adapter.
                self.inference.load_adapter_files(adapter_name(scenario, target), files, target)
                self._verify_engines_serve(target, "after update")
                return target
        if self.config.colocate:
            # Repeatable release also covers retry after a partial receive:
            # a released sender may need to reconstruct GPU workers.
            self.inference.offload(self.release_tags)
        self.training.prepare_weights(target, force_full=force_full)
        if self.config.colocate and self.release_tags is None:
            self.inference.onload_weights()
        version = self.training.send_weights(target, force_full=force_full)
        if version != target:
            raise RuntimeError(f"weight sender returned runtime load ID {version!r}; expected {target!r}")
        if self.config.colocate:
            self.inference.onload_kv()
        self._verify_engines_serve(version, "after update")
        return version

    def _fail_transfer(self) -> None:
        """Weights may be half-applied: keep inference fenced until recovery."""
        self.state.phase = "weight_sync_failed"
        with suppress(Exception):
            self.inference.abort()

    def publish(self, marker: Mapping[str, Any], *, force_full: bool) -> str:
        if self.history is not None:
            self.training.activate_scenario(str(marker["scenario"]))
        published = self.update_serving(
            force_full=force_full,
            scenario=self.marker_scenario(marker),
            runtime_load_id=self.publication_target(marker),
        )
        if self.history is not None:
            scenario = str(marker["scenario"])
            self.history.record_publication(scenario, published, adapter_name(scenario, published))
        return published

    def republish(self, runtime_load_id: str, marker: Mapping[str, Any] | None) -> str:
        try:
            return self.update_serving(
                force_full=True, scenario=self.marker_scenario(marker), runtime_load_id=runtime_load_id
            )
        finally:
            # A failed/mismatched transfer must not replace the retry identity.
            self.runtime_load_id = runtime_load_id

    # -- Recovery

    def recover(self, marker: Mapping[str, Any] | None) -> None:
        self.inference.recover()
        if self.history is not None:
            # Replacement engines boot without adapters. Restore other scenarios
            # before this job's complete transfer, and release dead residency slots.
            self.require_residency().reconcile((), self)
            self.restore_scenario_adapters(marker)

    def restore_scenario_adapters(self, marker: Mapping[str, Any] | None) -> None:
        """Re-register every scenario's committed adapter after a restart.

        The Megatron checkpoint restores only the slot's last occupant; the
        other scenarios come back from their persisted slot snapshots. Each
        is loaded under the name its last publication recorded, so Reef's
        routing for that scenario keeps resolving. The marker's scenario is
        activated last: the regular startup republication then publishes it
        under the recovered runtime load ID.
        """
        history = self.require_history()
        residency = self.require_residency()
        active = marker.get("scenario") if marker is not None else None
        pending = [
            (scenario, adapter)
            for scenario in history.scenarios
            if (adapter := history.adapter(scenario)) is not None and scenario != active
        ]
        if not pending and active is None:
            return
        self.pause_generation()
        if self.config.colocate:
            self.inference.onload_weights()
        for scenario, adapter in pending:
            _, version = parse_adapter_name(adapter)
            residency.activate(scenario, version, self)
        if active is not None:
            self.training.activate_scenario(active)
