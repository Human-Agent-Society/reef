"""Scenario admission: whether a service creates a scenario, and what it unloads to load one.

The registry asks a :class:`ScenarioAdmission` plugin before it creates a
scenario and before it loads one. The plugin sets the policy: how many
scenarios the service holds, and which loaded scenarios unload first. Reef
keeps the data safe: it unloads only idle scenarios. If too few scenarios are
idle, Reef refuses the load.

:class:`ScenarioLimits` is the built-in plugin. It applies
``reef.max_loaded_scenarios`` and ``reef.max_scenarios``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from reef.core.errors import ScenarioLimitReached
from reef.core.plugins import ConfiguredPlugin


@dataclass(frozen=True)
class LoadedScenario:
    """A scenario that holds a load slot."""

    name: str
    #: The ``time.monotonic()`` value of the last request, record or local cycle of the scenario.
    last_used_at: float
    #: True while the scenario loads. Reef cannot unload a scenario that loads.
    loading: bool = False


@dataclass(frozen=True)
class LoadAdmission:
    """The unloads that must occur before a scenario loads."""

    #: The number of loaded scenarios that Reef must unload first.
    unload_count: int = 0
    #: The scenarios that Reef tries to unload, in this order. Reef skips a scenario that is not idle.
    candidates: tuple[str, ...] = ()


class ScenarioAdmission(ConfiguredPlugin, ABC):
    """Decides whether a service creates a scenario, and what it unloads to load one.

    Reef calls the methods for a scenario while it holds the lock of that
    scenario. The calls for different scenarios occur one at a time. A method
    must return quickly, and it must not call the dispatcher.
    """

    @abstractmethod
    def check_creation(self, scenario: str, *, registered_count: int) -> None:
        """Check whether the new ``scenario`` may be created.

        ``registered_count`` is the number of registered scenarios, without
        ``scenario``. To refuse the scenario, raise ``ScenarioLimitReached``.
        """

    @abstractmethod
    def admit_load(self, scenario: str, *, loaded: tuple[LoadedScenario, ...]) -> LoadAdmission:
        """Tell what Reef must unload before it loads ``scenario``.

        ``loaded`` contains every scenario with a load slot, also the
        scenarios that still load. If fewer candidates are idle than
        ``unload_count``, Reef refuses the load with ``ScenarioCapacityFull``.
        To refuse the load directly, raise ``ScenarioCapacityFull``.
        """


class ScenarioLimits(ScenarioAdmission):
    """The built-in admission. It limits the loaded scenarios and the registered scenarios.

    ``None`` means no limit. At the load limit, Reef unloads the idle
    scenario with the oldest use first.
    """

    def __init__(self, *, max_loaded_scenarios: int | None = None, max_scenarios: int | None = None) -> None:
        for setting, value in (("max_loaded_scenarios", max_loaded_scenarios), ("max_scenarios", max_scenarios)):
            if value is not None and value < 1:
                raise ValueError(f"reef.{setting} must be positive, got {value}")
        self.max_loaded_scenarios = max_loaded_scenarios
        self.max_scenarios = max_scenarios

    def check_creation(self, scenario: str, *, registered_count: int) -> None:
        if self.max_scenarios is not None and registered_count >= self.max_scenarios:
            raise ScenarioLimitReached(
                f"cannot create scenario {scenario!r}: the service holds its limit of {self.max_scenarios} "
                "scenarios (reef.max_scenarios); delete a scenario first"
            )

    def admit_load(self, scenario: str, *, loaded: tuple[LoadedScenario, ...]) -> LoadAdmission:
        if self.max_loaded_scenarios is None or len(loaded) < self.max_loaded_scenarios:
            return LoadAdmission()
        least_recent_first = sorted(
            (entry for entry in loaded if not entry.loading), key=lambda entry: entry.last_used_at
        )
        return LoadAdmission(
            unload_count=len(loaded) - self.max_loaded_scenarios + 1,
            candidates=tuple(entry.name for entry in least_recent_first),
        )


__all__ = ["LoadAdmission", "LoadedScenario", "ScenarioAdmission", "ScenarioLimits"]
