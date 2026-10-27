"""LoRA residency: which adapters one serving engine holds, for which scenario.

:class:`AdapterResidencyManager` accounts for the LoRA adapters one shared
engine holds, so several scenarios cannot overcommit its slots. The serving
runtime (``reef.runtime.serving.AdapterServingRuntime``) drives it.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

from reef.core.errors import ReefError
from reef.runtime.interfaces import AdapterEngine, WeightFiles
from reef.surface.adapter import adapter_name

logger = logging.getLogger(__name__)

ResidentState = Literal["active", "leaked"]

#: How many residency actions ``status()`` keeps; enough to explain the most
#: recent capacity incident without growing with uptime.
RECENT_ACTIONS = 32


class AdapterResidencyError(ReefError):
    """The shared engine cannot make a scenario's adapter revision servable."""


class AdapterCapacityExhausted(AdapterResidencyError):
    """Every loaded adapter is protected, so nothing can be evicted for a new one.

    Raised before anything is unloaded, so the engine still holds exactly what
    it held and every scenario keeps serving: the load was refused, not
    half-applied.
    """


class AdapterEvictionFailed(AdapterCapacityExhausted):
    """The engine refused to release the slot chosen for eviction.

    Still a capacity failure for the caller, but a different one: a plain
    :class:`AdapterCapacityExhausted` proves the engine is fine, while this
    says it would not let go and may be wedged.
    """


@dataclass(eq=False)
class ResidentAdapter:
    """One adapter name the engine holds, and the manager's bookkeeping for it.

    ``name`` is ``adapter_name(scenario, runtime_load_id)``; ``scenario`` and
    ``runtime_load_id`` are non-empty. ``files`` is the adapter directory the
    engine loaded it from. ``order`` is the activation order, a non-negative
    int unique within one manager, which eviction follows. ``state`` is
    ``active``, or ``leaked`` after the engine refused to unload it.
    ``AdapterResidencyManager`` makes the values; it and the serving runtime
    read them.
    """

    name: str
    scenario: str
    runtime_load_id: str
    files: WeightFiles
    order: int
    state: ResidentState = "active"


class AdapterResidencyManager:
    """Bounded, scenario-aware bookkeeping of the adapters one engine holds.

    ``capacity`` is the positive number of adapters the engine can hold, or
    ``None`` for no limit. Invariants:

    - one engine adapter name maps to exactly one (scenario, revision), so a
      recorded adapter name proves which durable adapter answered;
    - a scenario's current revision is never evicted, except that a caller
      which has paused serving may supersede its own current revision when
      nothing else fits (``supersede=True``);
    - eviction is deterministic: a leaked adapter goes first, then the
      oldest-activated adapter that is no scenario's current revision,
      regardless of which scenario owns it; activating a resident adapter
      again does not change its order;
    - a failed load never occupies a slot; a failed unload keeps occupying
      one as ``leaked`` until a later unload succeeds, so capacity
      degradation is visible rather than hidden.
    """

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and (not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0):
            raise ValueError("adapter capacity must be a positive integer or None")
        self.capacity = capacity
        self.lock = RLock()
        self.adapters: dict[str, ResidentAdapter] = {}
        #: Scenario name -> engine adapter name of the revision it serves.
        self.current: dict[str, str] = {}
        self.next_order = 0
        self.counters = dict.fromkeys(
            (
                "loads",
                "load_failures",
                "load_cleanups",
                "cleanup_failures",
                "unloads",
                "unload_failures",
                "evictions",
                "capacity_rejections",
            ),
            0,
        )
        self.actions: deque[dict[str, str]] = deque(maxlen=RECENT_ACTIONS)

    def activate(
        self,
        scenario: str,
        runtime_load_id: str,
        engine: AdapterEngine,
        *,
        files: WeightFiles,
        supersede: bool = False,
    ) -> str:
        """Make ``runtime_load_id`` the revision ``scenario`` serves; return its engine name.

        Loads the adapter from ``files`` first and advances the scenario's
        current pointer only after the engine confirms the load. A failure
        touches no other scenario. It leaves the incumbent revision current,
        unless ``supersede`` evicted it to make room: the scenario then serves
        no adapter. ``supersede`` is only safe while no request can reach the
        incumbent (generation paused).
        """
        require_name(scenario, "scenario")
        require_name(runtime_load_id, "runtime_load_id")
        name = adapter_name(scenario, runtime_load_id)
        with self.lock:
            adapter = self.adapters.get(name)
            if adapter is not None and adapter.state == "active":
                self.current[scenario] = name
                return name
            # A leaked slot still holds the engine's copy of exactly this revision; reloading it reclaims the slot.
            if adapter is not None and (unload_error := self.unload(adapter, engine)) is not None:
                raise AdapterResidencyError(
                    f"adapter {name!r} leaked in the engine and cannot be reloaded until the unload "
                    f"succeeds: {unload_error}"
                ) from unload_error
            self.make_room(scenario, engine, supersede=supersede)
            self.adapters[name] = ResidentAdapter(name, scenario, runtime_load_id, files, self.next_order)
            self.next_order += 1
            try:
                engine.load_adapter(name, files)
            except Exception as exc:
                # Never leave a phantom slot: it would evict a live peer later for an adapter the engine never held.
                self.adapters.pop(name, None)
                self.counters["load_failures"] += 1
                self.note("load_failed", name, scenario, str(exc))
                self.drop_after_failed_load(name, scenario, engine)
                raise AdapterResidencyError(
                    f"engine could not load adapter {name!r} for scenario {scenario!r}: {exc}"
                ) from exc
            self.counters["loads"] += 1
            self.current[scenario] = name
            return name

    def current_adapter(self, scenario: str) -> ResidentAdapter | None:
        """The resident adapter ``scenario`` serves, or None when it serves none."""
        with self.lock:
            name = self.current.get(scenario)
            return None if name is None else self.adapters[name]

    def current_adapters(self) -> tuple[ResidentAdapter, ...]:
        """Every scenario's current adapter, in activation order."""
        with self.lock:
            return tuple(
                sorted((self.adapters[name] for name in self.current.values()), key=lambda adapter: adapter.order)
            )

    def release(self, scenario: str, engine: AdapterEngine) -> None:
        """Unload the revision ``scenario`` serves, after which the scenario serves no adapter.

        Nothing happens when the scenario serves none. Raise
        ``AdapterResidencyError`` when the engine refuses the unload; the
        adapter then stays resident as ``leaked`` and no longer protected.
        """
        require_name(scenario, "scenario")
        with self.lock:
            name = self.current.pop(scenario, None)
            adapter = None if name is None else self.adapters.get(name)
            if adapter is not None and (unload_error := self.unload(adapter, engine)) is not None:
                raise AdapterResidencyError(
                    f"engine could not unload adapter {adapter.name!r} of scenario {scenario!r}: {unload_error}"
                ) from unload_error

    def drop_after_failed_load(self, name: str, scenario: str, engine: AdapterEngine) -> None:
        """Make a failed load deterministic: the engine must not hold ``name``.

        A timeout or dropped connection cannot tell an engine that rejected
        the adapter from one that loaded it after Reef stopped waiting. Reef
        never counts the slot either way, so the engine is told to drop the
        name; a refusal for an adapter it never held is harmless, and one for
        an adapter it does hold is recorded so the leak is visible.
        """
        try:
            engine.unload_adapter(name)
        except Exception as exc:
            self.counters["cleanup_failures"] += 1
            self.note("cleanup_failed", name, scenario, str(exc))
            logger.info("engine did not drop adapter %r after its failed load: %s", name, exc)
            return
        self.counters["load_cleanups"] += 1
        self.note("load_cleanup", name, scenario)

    def make_room(self, scenario: str, engine: AdapterEngine, *, supersede: bool) -> None:
        """Evict until one more adapter fits; raise ``AdapterCapacityExhausted`` when every adapter is protected."""
        if self.capacity is None:
            return
        superseding = self.current.get(scenario) if supersede else None
        while len(self.adapters) >= self.capacity:
            victim = self.eviction_candidate(superseding)
            if victim is None:
                self.counters["capacity_rejections"] += 1
                raise self.capacity_exhausted(scenario)
            if (unload_error := self.unload(victim, engine)) is not None:
                # The engine refused to let go: the caller sees exhaustion rather than a silent overcommit.
                self.counters["capacity_rejections"] += 1
                raise AdapterEvictionFailed(
                    f"engine adapter capacity {self.capacity} is exhausted and evicting {victim.name!r} "
                    f"failed ({unload_error}); scenario {scenario!r} cannot activate until the leaked slot "
                    f"is reclaimed"
                ) from unload_error
            self.counters["evictions"] += 1
            self.note("evicted", victim.name, victim.scenario, f"for scenario {scenario!r}")

    def capacity_exhausted(self, scenario: str) -> AdapterCapacityExhausted:
        """The error for a load that no eviction can make room for; it names the slots needed."""
        sharing = {adapter.scenario for adapter in self.adapters.values()} | {scenario}
        return AdapterCapacityExhausted(
            f"engine adapter capacity {self.capacity} is exhausted and every resident adapter is "
            f"its scenario's current revision: {sorted(self.adapters)}; scenario {scenario!r} cannot "
            f"activate. {len(sharing)} scenarios share this engine and each keeps its current "
            f"revision resident, so it needs at least {len(sharing) + 1} slots: raise "
            f"--max-loaded-loras. Nothing was unloaded; every scenario keeps serving."
        )

    def eviction_candidate(self, superseding: str | None) -> ResidentAdapter | None:
        """The adapter to evict next (leaked ones first, then the oldest), or ``None`` when all are protected."""
        candidates = [
            adapter
            for adapter in self.adapters.values()
            if adapter.name == superseding or not self.is_current(adapter)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda adapter: (adapter.state != "leaked", adapter.order))

    def unload(self, adapter: ResidentAdapter, engine: AdapterEngine) -> Exception | None:
        """Drop ``adapter`` from the engine and the table; a returned failure leaves it ``leaked``."""
        try:
            engine.unload_adapter(adapter.name)
        except Exception as exc:
            self.counters["unload_failures"] += 1
            adapter.state = "leaked"
            self.note("leaked", adapter.name, adapter.scenario, str(exc))
            logger.warning("could not unload adapter %r; its engine slot leaks", adapter.name, exc_info=True)
            return exc
        self.counters["unloads"] += 1
        self.adapters.pop(adapter.name, None)
        if self.current.get(adapter.scenario) == adapter.name:
            self.current.pop(adapter.scenario)
        return None

    def note(self, action: str, adapter: str, scenario: str, detail: str | None = None) -> None:
        """Keep one residency action for ``status()``; the oldest drops out after ``RECENT_ACTIONS``."""
        entry = {"action": action, "adapter": adapter, "scenario": scenario}
        if detail:
            entry["detail"] = detail[:200]
        self.actions.append(entry)

    def status(self) -> dict[str, Any]:
        """A bounded status block: global capacity plus per-scenario residency."""
        with self.lock:
            adapters = sorted(self.adapters.values(), key=lambda adapter: adapter.order)
            scenarios: dict[str, dict[str, Any]] = {}
            for adapter in adapters:
                block = scenarios.setdefault(adapter.scenario, {"current": None, "resident": []})
                block["resident"].append(adapter.runtime_load_id)
                if self.is_current(adapter):
                    block["current"] = {"runtime_load_id": adapter.runtime_load_id, "adapter": adapter.name}
            return {
                "capacity": self.capacity,
                "resident": len(adapters),
                "leaked": sum(1 for adapter in adapters if adapter.state == "leaked"),
                "protected": sum(1 for adapter in adapters if self.is_current(adapter)),
                "counters": dict(self.counters),
                "recent_actions": list(self.actions),
                "scenarios": scenarios,
            }

    def is_current(self, adapter: ResidentAdapter) -> bool:
        return self.current.get(adapter.scenario) == adapter.name


def require_name(value: str, name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"adapter residency requires a non-empty {name}")
