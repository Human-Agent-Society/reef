"""In-memory scenario table with factory, per-scenario locks, and the
single-training-scenario invariant.

Owns the scenario dict, per-scenario RLocks, the scenario factory, and the
training-binding invariant (at most one training scenario per process). All
table access goes through this registry; the dispatcher delegates scenario
resolution and uses the registry for lookups.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackendFactory
from reef.core.errors import ReefError, UnknownScenario
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.recipe.base import Recipe
from reef.runtime.base import TrainingRuntime
from reef.scenario.factory import ScenarioFactory
from reef.scenario.scenario import Scenario


class ScenarioRegistry:
    """In-memory scenario table with factory, per-scenario locks, and the
    single-training-scenario invariant.

    Owns the scenario dict, per-scenario RLocks, the scenario factory, and the
    training-binding invariant (at most one training scenario per process).
    All table access goes through this registry; the dispatcher delegates
    scenario resolution and uses the registry for lookups.
    """

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        agent_record_dir: Path | None = None,
        allow_implicit_creation: bool = True,
        experiment_tracker: ExperimentTracker | None = None,
    ) -> None:
        self._scenario_factory = ScenarioFactory(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            agent_record_dir=agent_record_dir,
            experiment_tracker=(experiment_tracker if experiment_tracker is not None else NullExperimentTracker()),
        )
        self._backend_factory = backend_factory
        self._recipe = recipe
        self._scenarios: dict[str, Scenario] = {}
        self._scenario_locks: dict[str, RLock] = {}
        self._lock = Lock()
        self._training_scenario: str | None = None
        self._training_scenarios: list[str] = []
        self._preload_errors: dict[str, str] = {}
        self._allow_implicit_creation = allow_implicit_creation
        self._on_training_scenario_resolved: Callable[[Scenario], None] | None = None

    def recipe_has_files(self) -> bool:
        """Whether the served recipe creates a file-serving surface."""
        # Capability probe only: file serving does not depend on the scenario.
        return self._recipe.build_surface("").files is not None

    @property
    def training_scenario_name(self) -> str | None:
        """The first-bound training scenario (the only one on a single-scenario runtime)."""
        with self._lock:
            return self._training_scenario

    @property
    def training_scenario_names(self) -> tuple[str, ...]:
        """Every scenario the training thread drives, in binding order."""
        with self._lock:
            return tuple(self._training_scenarios)

    @property
    def training_status_scenario_names(self) -> tuple[str, ...]:
        """Every loaded scenario with a local or dispatched training backend."""
        with self._lock:
            scenarios = tuple(self._scenarios.values())
            dispatched = tuple(self._training_scenarios)
        local = tuple(
            scenario.name
            for scenario in scenarios
            if scenario.name not in dispatched and scenario.trainer.training_backend is not None
        )
        return (*dispatched, *local)

    @property
    def preload_errors(self) -> dict[str, str]:
        with self._lock:
            return dict(self._preload_errors)

    def record_preload_error(self, scenario: str, error: str) -> None:
        with self._lock:
            self._preload_errors[scenario] = error

    def set_training_scenario_callback(self, callback: Callable[[Scenario], None]) -> None:
        self._on_training_scenario_resolved = callback

    def has(self, scenario: str) -> bool:
        """True when the scenario exists in memory or in durable registration."""
        with self._lock:
            loaded = scenario in self._scenarios
        return loaded or self._scenario_factory.has_registration(scenario)

    def has_loaded(self, scenario: str) -> bool:
        """True when the scenario is in the in-memory table (not just durable)."""
        with self._lock:
            return scenario in self._scenarios

    def get(self, scenario: str) -> Scenario:
        """Get a loaded scenario by name (must exist in memory)."""
        with self._lock:
            return self._scenarios[scenario]

    def get_optional(self, scenario: str | None) -> Scenario | None:
        """Get a loaded scenario, or None if not loaded / name is None."""
        if scenario is None:
            return None
        with self._lock:
            return self._scenarios.get(scenario)

    def get_or_create(
        self,
        scenario: str,
        release_id: str | None = None,
        *,
        allow_implicit_creation: bool | None = None,
    ) -> Scenario | None:
        """Resolve a scenario, creating it when allowed.

        A missing scenario is created when ``allow_implicit_creation`` is true
        (defaulting to the registry's deployment flag); otherwise ``None`` is
        returned so the caller can surface an unknown-scenario error.
        """
        if allow_implicit_creation is None:
            allow_implicit_creation = self._allow_implicit_creation
        with self.lock_for(scenario):
            if not allow_implicit_creation and not self.has(scenario):
                return None
            return self._resolve(scenario, release_id)

    def require(self, scenario: str) -> Scenario:
        """Resolve an existing scenario; raise UnknownScenario if not found."""
        if not self.has(scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        return self._resolve(scenario, None)

    def list(self) -> tuple[dict[str, Any], ...]:
        """Known scenarios: loaded ones with their binding, durable ones by name."""
        registered: tuple[str, ...] = ()
        if isinstance(self._backend_factory, EnumerableRepositoryBackendFactory):
            registered = self._backend_factory.list_registrations()
        with self._lock:
            loaded = dict(self._scenarios)
        rows = []
        for name in sorted(set(registered) | set(loaded)):
            current = loaded.get(name)
            row: dict[str, Any] = {"scenario": name, "loaded": current is not None}
            if current is not None:
                ref = current.repository.require_current_artifact()
                row["release_id"] = ref.release_id
                row["content_id"] = ref.content_id
            rows.append(row)
        return tuple(rows)

    def lock_for(self, scenario: str) -> RLock:
        with self._lock:
            lock = self._scenario_locks.get(scenario)
            if lock is None:
                lock = RLock()
                self._scenario_locks[scenario] = lock
            return lock

    def reload(self, scenario: str) -> Scenario:
        """Rebuild a scenario from durable state after a training failure."""
        with self.lock_for(scenario):
            recovered = self._scenario_factory.load_or_create(scenario, None)
            with self._lock:
                dropped = self._scenarios.get(scenario)
                self._scenarios[scenario] = recovered
            if dropped is not None:
                # Close outside the state lock: teardown may join processor
                # worker threads. The per-scenario lock still excludes accepts,
                # so nothing observes the dropped instance mid-close.
                dropped.close()
            return recovered

    def close_all(self) -> tuple[Scenario, ...]:
        with self._lock:
            return tuple(self._scenarios.values())

    def _resolve(
        self,
        scenario: str,
        release_id: str | None,
    ) -> Scenario:
        with self._lock:
            current = self._scenarios.get(scenario)
        if current is not None:
            self._scenario_factory.validate_existing(current, release_id)
            return current
        current = self._scenario_factory.load_or_create(scenario, release_id)
        training = isinstance(current.runtime, TrainingRuntime)
        with self._lock:
            shared_runtime = training and getattr(current.runtime, "concurrent_training_scenarios", False)
            if training and not shared_runtime and self._training_scenario not in (None, scenario):
                current.close()
                raise ReefError(
                    f"training is already bound to scenario {self._training_scenario!r}: a reef process "
                    f"trains one scenario for its lifetime, so {scenario!r} needs its own stack "
                    f"(restart this one, or run a second stack on other ports)"
                )
            if training:
                if self._training_scenario is None:
                    self._training_scenario = scenario
                if scenario not in self._training_scenarios:
                    self._training_scenarios.append(scenario)
            self._scenarios[scenario] = current
        if training and self._on_training_scenario_resolved is not None:
            self._on_training_scenario_resolved(current)
        return current
