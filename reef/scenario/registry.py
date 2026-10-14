"""In-memory scenario table with factory, per-scenario locks, and the
single-training-scenario invariant.

Owns the scenario dict, per-scenario RLocks, the scenario factory, and the
training-binding invariant (at most one training scenario per process). All
table access goes through this registry; the dispatcher delegates scenario
resolution and uses the registry for lookups. Before the registry creates or
loads a scenario, it asks its ``ScenarioAdmission``. To free a load slot, it
asks its ``ScenarioUnloader`` (the dispatcher) to unload an idle scenario.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any

from reef.artifact.repository import (
    CachedRepositoryBackendFactory,
    EnumerableRepositoryBackendFactory,
    RepositoryBackendFactory,
)
from reef.core.errors import ReefError, ScenarioCapacityFull, UnknownScenario
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.recipe.base import Recipe
from reef.scenario.admission import LoadedScenario, ScenarioAdmission, ScenarioLimits
from reef.scenario.factory import ScenarioFactory
from reef.scenario.scenario import Scenario
from reef.storage.scenario import ScenarioStorage

logger = logging.getLogger(__name__)


class ReplacedScenarioCloser(ABC):
    """Closes the instance a reload replaced, now or once nothing runs on it any more."""

    @abstractmethod
    def close_replaced(self, instance: Scenario) -> None: ...


class CloseReplacedAtOnce(ReplacedScenarioCloser):
    """The registry's own closer: the replaced instance closes as the reload returns."""

    def close_replaced(self, instance: Scenario) -> None:
        instance.close()


class ScenarioUnloader(ABC):
    """Unloads an idle scenario to free its load slot."""

    @abstractmethod
    def unload_if_idle(self, scenario: str) -> bool:
        """Unload ``scenario`` if it is idle now, and return whether it was unloaded.

        The method must not wait for a lock of the scenario.
        """


class NeverUnload(ScenarioUnloader):
    """The default unloader of the registry. Without a dispatcher, the registry cannot know that a scenario is idle."""

    def unload_if_idle(self, scenario: str) -> bool:
        return False


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
        allow_implicit_creation: bool = True,
        experiment_tracker: ExperimentTracker | None = None,
        scenario_storage: ScenarioStorage,
        admission: ScenarioAdmission | None = None,
    ) -> None:
        self._model_configs: dict[str, ModelConfig] = {}
        self._storage = scenario_storage
        self._scenario_factory = ScenarioFactory(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            scenario_storage=scenario_storage,
            experiment_tracker=(experiment_tracker if experiment_tracker is not None else NullExperimentTracker()),
        )
        self._backend_factory = backend_factory
        self._recipe = recipe
        self._scenarios: dict[str, Scenario] = {}
        self._scenario_locks: dict[str, RLock] = {}
        self._lock = Lock()
        #: Every training scenario in binding order; the first is the one a single-scenario runtime binds.
        self._training_scenarios: list[str] = []
        self._preload_errors: dict[str, str] = {}
        self.replaced_closer: ReplacedScenarioCloser = CloseReplacedAtOnce()
        self._allow_implicit_creation = allow_implicit_creation
        self._on_training_scenario_resolved: Callable[[Scenario], None] | None = None
        self.admission = admission if admission is not None else ScenarioLimits()
        #: Makes admission serial, so that two loads cannot both take the last load slot.
        self.admission_lock = Lock()
        self.unloader: ScenarioUnloader = NeverUnload()
        #: The ``time.monotonic()`` value of the last request, record or local cycle of each loaded scenario.
        self.last_used_at: dict[str, float] = {}
        #: The scenarios that admission let load and that are not loaded yet. Each one holds a load slot.
        self.loading: set[str] = set()

    @property
    def training_scenario_name(self) -> str | None:
        """The first-bound training scenario (the only one on a single-scenario runtime)."""
        with self._lock:
            return self._training_scenarios[0] if self._training_scenarios else None

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
            if scenario.name not in dispatched
            and any(bound.trainer.candidate_backend is not None for bound in scenario.trainers)
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

    def set_replaced_closer(self, closer: ReplacedScenarioCloser) -> None:
        """Who closes the instance a reload replaced: a local cycle may still evaluate on it."""
        self.replaced_closer = closer

    def set_unloader(self, unloader: ScenarioUnloader) -> None:
        """Set the unloader that frees a load slot for admission."""
        self.unloader = unloader

    def touch(self, scenario: str) -> None:
        """Record a use of a loaded scenario. A scenario that is not loaded is ignored."""
        with self._lock:
            if scenario in self._scenarios:
                self.last_used_at[scenario] = time.monotonic()

    def unused_since(self, monotonic_seconds: float) -> tuple[str, ...]:
        """The loaded scenarios with no use after the ``time.monotonic()`` value ``monotonic_seconds``."""
        with self._lock:
            return tuple(name for name in self._scenarios if self.last_used_at[name] < monotonic_seconds)

    def has(self, scenario: str) -> bool:
        """True when the scenario exists in memory or in durable registration."""
        with self._lock:
            loaded = scenario in self._scenarios
        return loaded or self._scenario_factory.has_registration(scenario)

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

    def get_loaded(self, scenario: str, release_id: str | None = None) -> Scenario | None:
        """The loaded instance, or None: nothing is created or recovered, and the scenario's lock is not taken.

        An evaluation episode's call resolves this way. A delete or a
        reload holds the scenario's lock while it closes the instance, which
        waits for the episode in flight, and an episode of a deleted
        scenario must not bring the scenario back.
        """
        current = self.get_optional(scenario)
        if current is not None:
            self._scenario_factory.validate_existing(current, release_id)
            self.touch(scenario)
        return current

    def require(self, scenario: str) -> Scenario:
        """Resolve an existing scenario; raise UnknownScenario if not found."""
        if not self.has(scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        return self._resolve(scenario, None)

    def list(self) -> tuple[dict[str, Any], ...]:
        """Known scenarios: loaded ones with their binding, durable ones by name, each with the harness adapter.

        The adapter is the recipe's, so a scenario still loading or one whose
        preload failed names it too; a recipe with no harness names none.
        """
        registered: tuple[str, ...] = ()
        if isinstance(self._backend_factory, EnumerableRepositoryBackendFactory):
            registered = self._backend_factory.list_registrations()
        with self._lock:
            loaded = dict(self._scenarios)
        adapter = self._recipe.harness_adapter
        rows = []
        for name in sorted(set(registered) | set(loaded)):
            current = loaded.get(name)
            row: dict[str, Any] = {"scenario": name, "loaded": current is not None}
            if current is not None:
                ref = current.repository.require_current_artifact()
                row["release_id"] = ref.release_id
                row["content_id"] = ref.content_id
            if adapter is not None:
                row["adapter"] = adapter
            rows.append(row)
        return tuple(rows)

    def lock_for(self, scenario: str) -> RLock:
        with self._lock:
            lock = self._scenario_locks.get(scenario)
            if lock is None:
                lock = RLock()
                self._scenario_locks[scenario] = lock
            return lock

    def set_training_mode(self, scenario: str, training_mode: str) -> Scenario:
        """Select an existing scenario's mode and store it, so reloads and restarts apply it again."""
        with self.lock_for(scenario):
            current = self.require(scenario)
            current.set_training_mode(training_mode)
            settings = self._storage.read_settings(scenario)
            self._storage.write_settings(scenario, replace(settings, training_mode=training_mode))
            return current

    def configure_model(
        self, scenario: str, value: object, *, create: bool = False, release_id: str | None = None
    ) -> Scenario:
        with self.lock_for(scenario):
            exists = self.has(scenario)
            if not create and not exists:
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            if not exists:
                # Check before the settings are stored, so that a refused scenario leaves nothing behind.
                self.check_creation(scenario)
            # Creating an existing scenario never overwrites its configuration.
            if not create or not exists:
                candidate = ModelConfig.from_value(value)
                self._recipe.with_model_config(candidate)
                current = self._model_config(scenario)
                # A validated value is an object or null.
                stored = value if isinstance(value, Mapping) else None
                settings = self._storage.read_settings(scenario)
                self._storage.write_settings(scenario, replace(settings, model_config=stored))
                current.runtime = candidate.runtime
            return self._resolve(scenario, release_id)

    def reload(self, scenario: str) -> Scenario:
        """Rebuild a scenario from durable state after a training failure.

        Accepts and inference reach the new instance at once; the replaced
        instance goes to the registry's closer (see ``set_replaced_closer``).
        """
        with self.lock_for(scenario):
            recovered = self._scenario_factory.load_or_create(
                scenario, None, model_config=self._model_config(scenario)
            )
            self._apply_stored_training_mode(recovered)
            with self._lock:
                dropped = self._scenarios.get(scenario)
                self._scenarios[scenario] = recovered
                self.last_used_at[scenario] = time.monotonic()
            if dropped is not None:
                # Close outside the state lock: teardown may join processor
                # worker threads. Accepts reach the recovered instance, so
                # nothing observes the dropped instance mid-close.
                self.replaced_closer.close_replaced(dropped)
            return recovered

    def remove(self, scenario: str) -> Scenario | None:
        """Drop every in-memory hold on the scenario; the instance, for the caller to close outside the state lock.

        The training thread's list loses the name first, so a step already
        in flight finds no scenario to commit to and no durable state to
        reload; any preload error goes with it. The stored settings stay
        until ``archive_store``. Call under ``lock_for(scenario)``.
        """
        with self._lock:
            dropped = self._scenarios.pop(scenario, None)
            self.last_used_at.pop(scenario, None)
            self._training_scenarios = [name for name in self._training_scenarios if name != scenario]
            self._model_configs.pop(scenario, None)
            self._preload_errors.pop(scenario, None)
        return dropped

    def unload(self, scenario: str) -> Scenario | None:
        """Remove the loaded instance from the table and return it. The caller closes it.

        All durable state stays, and the next request loads the scenario
        again. The cached model configuration also stays, so the scenario
        keeps its runtime. Call this method under ``lock_for(scenario)``.
        """
        with self._lock:
            dropped = self._scenarios.pop(scenario, None)
            self.last_used_at.pop(scenario, None)
        return dropped

    def archive_registration(self, scenario: str) -> tuple[str, ...]:
        """Move the scenario's durable registration aside and forget its cached backend; what was archived."""
        if not isinstance(self._backend_factory, CachedRepositoryBackendFactory):
            raise NotImplementedError("this repository backend cannot archive a scenario")
        return self._backend_factory.archive_registration(scenario)

    def forget_lock(self, scenario: str) -> None:
        """Release the per-scenario lock's slot once nothing holds it; a later create makes a fresh one."""
        with self._lock:
            self._scenario_locks.pop(scenario, None)

    def archive_store(self, scenario: str) -> tuple[str, ...]:
        return self._storage.archive(scenario)

    def loaded_scenarios(self) -> tuple[Scenario, ...]:
        """Return the currently loaded scenario instances as a tuple."""
        with self._lock:
            return tuple(self._scenarios.values())

    def _model_config(self, scenario: str) -> ModelConfig:
        if scenario not in self._model_configs:
            value = self._storage.read_settings(scenario).model_config
            self._model_configs[scenario] = ModelConfig.from_value(value)
        return self._model_configs[scenario]

    def _apply_stored_training_mode(self, instance: Scenario) -> None:
        """Select the mode a person stored for the scenario on a newly built instance.

        A mode the recipe no longer supports keeps the recipe's default, so the
        scenario still loads and a person can select another mode. Any other
        failure closes the instance.
        """
        training_mode = self._storage.read_settings(instance.name).training_mode
        if training_mode is None or instance.training_mode == training_mode:
            return
        try:
            instance.set_training_mode(training_mode)
        except NotImplementedError as exc:
            logger.warning(
                "Scenario %r keeps its default training mode: the stored mode %r is not supported (%s)",
                instance.name,
                training_mode,
                exc,
            )
        except BaseException:
            instance.close()
            raise

    def _resolve(
        self,
        scenario: str,
        release_id: str | None,
    ) -> Scenario:
        with self._lock:
            current = self._scenarios.get(scenario)
        if current is not None:
            self._scenario_factory.validate_existing(current, release_id)
            self.touch(scenario)
            return current
        if not self._scenario_factory.has_registration(scenario):
            self.check_creation(scenario)
        self.take_load_slot(scenario)
        try:
            current = self._scenario_factory.load_or_create(
                scenario, release_id, model_config=self._model_config(scenario)
            )
            self._apply_stored_training_mode(current)
        finally:
            with self._lock:
                self.loading.discard(scenario)
        training_runtime = current.training_runtime
        with self._lock:
            shared_runtime = training_runtime is not None and training_runtime.concurrent_training_scenarios
            bound_scenario = self._training_scenarios[0] if self._training_scenarios else None
            if training_runtime is not None and not shared_runtime and bound_scenario not in (None, scenario):
                current.close()
                raise ReefError(
                    f"training is already bound to scenario {bound_scenario!r}: a reef process "
                    f"trains one scenario for its lifetime, so {scenario!r} needs its own stack "
                    f"(restart this one, or run a second stack on other ports)"
                )
            if training_runtime is not None and scenario not in self._training_scenarios:
                self._training_scenarios.append(scenario)
            self._scenarios[scenario] = current
            self.last_used_at[scenario] = time.monotonic()
        if training_runtime is not None and self._on_training_scenario_resolved is not None:
            self._on_training_scenario_resolved(current)
        return current

    def check_creation(self, scenario: str) -> None:
        """Ask admission whether the new ``scenario`` may be created. If admission refuses it, the error propagates."""
        if isinstance(self._backend_factory, EnumerableRepositoryBackendFactory):
            registered_count = len(self._backend_factory.list_registrations())
        else:
            with self._lock:
                registered_count = len(self._scenarios)
        try:
            self.admission.check_creation(scenario, registered_count=registered_count)
        except ReefError as exc:
            logger.warning("refused to create scenario %r: %s", scenario, exc)
            raise

    def take_load_slot(self, scenario: str) -> None:
        """Take a load slot for ``scenario`` before it loads. ``_resolve`` releases the slot after the load.

        Admission tells how many loaded scenarios must unload first. This
        method unloads idle candidates until there are enough. If there are
        too few, it raises ``ScenarioCapacityFull``.

        Lock order: the lock of ``scenario``, then the admission lock. The
        unloader only tries the locks of other scenarios and never waits for
        them, so no deadlock can occur.
        """
        with self.admission_lock:
            now = time.monotonic()
            with self._lock:
                loaded = (
                    *(LoadedScenario(name, self.last_used_at[name]) for name in self._scenarios),
                    *(LoadedScenario(name, now, loading=True) for name in self.loading),
                )
            try:
                load_admission = self.admission.admit_load(scenario, loaded=loaded)
            except ReefError as exc:
                logger.warning("refused to load scenario %r: %s", scenario, exc)
                raise
            unloaded_count = 0
            for candidate in load_admission.candidates:
                if unloaded_count == load_admission.unload_count:
                    break
                if self.unloader.unload_if_idle(candidate):
                    unloaded_count += 1
            if unloaded_count < load_admission.unload_count:
                logger.warning("refused to load scenario %r: too few loaded scenarios are idle", scenario)
                raise ScenarioCapacityFull(
                    f"cannot load scenario {scenario!r}: {len(loaded)} scenarios are loaded, and too few of them "
                    "are idle to free a load slot; retry later"
                )
            with self._lock:
                self.loading.add(scenario)
