"""Process-wide coordinator that owns every scenario and its serving state.

The dispatcher delegates scenario table management (creation, resolution,
per-scenario locking) to :class:`ScenarioRegistry`, then layers record
acceptance, background training drains, and publication on top. Request
handling lives in ``reef.service``; the dispatcher is transport-free.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Literal

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackendFactory
from reef.core.errors import ScenarioBusy, ScenarioCapacityFull, UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportValidationError, validate_report_payload
from reef.core.training_request import normalize_training_request_payload
from reef.harness.tree.nodes import directive_shaped, secret_shaped
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.recipe.base import Recipe
from reef.recipe.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.runtime.interfaces import RuntimeContractError, TrainingRuntime
from reef.runtime.recovery import marker_in_flight
from reef.scenario.admission import ScenarioAdmission
from reef.scenario.registry import ReplacedScenarioCloser, ScenarioRegistry, ScenarioUnloader
from reef.scenario.scenario import Scenario, SettledTrainingResultError, StaleTrainingResultError
from reef.storage.records import RecordConflict, RecordRetention
from reef.storage.scenario import ScenarioStorage
from reef.train.backend import CandidateBackend
from reef.train.types import TrainingBatch, TrainStepResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalBackendWorker:
    ready: Event
    thread: Thread


class _ScenarioTrainingError(Exception):
    """One scenario's training turn failed; the others keep their state."""

    def __init__(self, scenario: str, cause: Exception) -> None:
        super().__init__(f"{scenario}: {cause}")
        self.scenario = scenario
        self.cause = cause


#: How soon a local worker that stood aside for closed admission looks again.
STOOD_ASIDE_RETRY_SECONDS = 5.0
#: The training thread's name in the error record; local workers are named by their component.
TRAINING_THREAD_SOURCE = "training"


def local_error_source(component: str | None) -> str:
    return "local" if component is None else f"local:{component}"


@dataclass
class _TrainingState:
    lock: Lock = field(default_factory=Lock)
    ready: Event = field(default_factory=Event)
    #: The last error per (scenario, worker that recorded it): only that worker clears its own.
    errors: dict[tuple[str, str], str] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    status_build_error: str | None = None
    storage_status: Mapping[str, Any] | None = None
    last_drain: float | None = None
    undrained_warned: bool = False
    thread: Thread | None = None
    #: Results refused because another trainer's commit replaced their base, per (scenario, component).
    stale_refusals_in_a_row: dict[tuple[str, str | None], int] = field(default_factory=dict)
    stale_refusals_total: dict[tuple[str, str | None], int] = field(default_factory=dict)


@dataclass
class _LifecycleState:
    preload_thread: Thread | None = None
    metrics_thread: Thread | None = None
    idle_unload_thread: Thread | None = None


def training_request_refusal(text: str, requires: Sequence[Mapping[str, Any]] = ()) -> str | None:
    """Why admission refuses an instruction; the reason names the rule, never the text or the item.

    The text becomes proposer input and a catalog row, so it meets the
    screens a promoted task prompt meets; a ``requires`` name, check or
    prompt is shown to the person and recorded in the commit, so it meets
    them too."""
    if secret_shaped(text):
        return "the request text carries a credential shaped literal; a request never holds secrets"
    if directive_shaped(text):
        return "the request text carries an instruction override phrasing or a chat template control token"
    for item in requires:
        for value in (str(item.get("name", "")), str(item.get("check") or ""), str(item.get("prompt") or "")):
            if secret_shaped(value):
                return "a requires item carries a credential shaped literal; a request never holds secrets"
            if directive_shaped(value):
                return "a requires item carries an instruction override phrasing or a chat template control token"
    return None


@dataclass
class ScenarioCycles:
    lock: Lock = field(default_factory=Lock)
    waiting_for_admission: set[str | None] = field(default_factory=set)
    reload_pending: bool = False
    replaced: list[Scenario] = field(default_factory=list)


class WorkerCoordination(ReplacedScenarioCloser):
    """Own local worker lifetime, dispatched turns, and scenario replacement.

    Service starts held or running. Opening releases held workers; stopping and closed never reopen.
    Each scenario serializes local cycles with one lock. A colocated job takes all
    cycle locks in name order; workers yield while it waits. A failed local cycle
    defers reload while a dispatched batch is reserved. Replaced instances close
    only after the cycle or dispatched turn using them releases its lock.

    Lock order: cycle locks -> registry scenario lock -> coordination lock.
    The coordination lock also protects Dispatcher training metrics. Never wait
    for a cycle lock, join a worker, or close a scenario while holding it.
    Registry reloads can call close_replaced while holding their scenario lock:
    closure therefore tries the cycle lock without waiting.
    """

    def __init__(self, registry: ScenarioRegistry, lock: Lock, *, held: bool) -> None:
        self.registry = registry
        self.lock = lock
        self.phase: Literal["held", "running", "stopping", "closed"] = "held" if held else "running"
        self.closed = Event()
        self.turn_waiting = Event()
        self.workers: dict[tuple[str, str | None], LocalBackendWorker] = {}
        self.scenarios: dict[str, ScenarioCycles] = {}

    def start(self, scenario: str, component: str | None, worker: LocalBackendWorker) -> None:
        with self.lock:
            if self.phase == "closed":
                return
            key = (scenario, component)
            existing = self.workers.get(key)
            if existing is None:
                self.workers[key] = worker
                worker.thread.start()
            else:
                worker = existing
        worker.ready.set()

    def remove(self, scenario: str) -> None:
        """Unregister workers without joining them under the registry lock."""
        with self.lock:
            workers = [self.workers.pop(key) for key in tuple(self.workers) if key[0] == scenario]
            state = self.scenarios.get(scenario)
            if state is not None:
                state.waiting_for_admission.clear()
                state.reload_pending = False
        for worker in workers:
            worker.ready.set()

    def wait(self, scenario: str, component: str | None, ready: Event) -> bool:
        with self.lock:
            state = self.scenarios.get(scenario)
            waiting = state is not None and component in state.waiting_for_admission
        ready.wait(timeout=STOOD_ASIDE_RETRY_SECONDS if waiting else None)
        ready.clear()
        with self.lock:
            # A worker for a reloaded instance with the same key replaces this worker. Then this worker stops.
            worker = self.workers.get((scenario, component))
            return self.phase != "closed" and worker is not None and worker.ready is ready

    def wake_all(self) -> None:
        with self.lock:
            workers = tuple(self.workers.values())
        for worker in workers:
            worker.ready.set()

    def open(self) -> None:
        with self.lock:
            if self.phase == "held":
                self.phase = "running"

    def stop(self) -> None:
        with self.lock:
            if self.phase != "closed":
                self.phase = "stopping"

    def begin_close(self) -> tuple[LocalBackendWorker, ...]:
        """Freeze registration and wake the exact workers shutdown must join."""
        with self.lock:
            self.phase = "closed"
            self.closed.set()
            workers = tuple(self.workers.values())
        for worker in workers:
            worker.ready.set()
        return workers

    def cycles_run(self) -> bool:
        with self.lock:
            return self.phase == "running"

    def check_admission(self, current: Scenario, component: str | None) -> bool:
        runtime = current.runtime
        admission_open = runtime is None or runtime.inference_admission_status.get("open", True)
        with self.lock:
            state = self.scenarios.setdefault(current.name, ScenarioCycles())
            if admission_open:
                state.waiting_for_admission.discard(component)
            else:
                state.waiting_for_admission.add(component)
        return admission_open

    def cycle_lock(self, scenario: str) -> Lock:
        with self.lock:
            return self.scenarios.setdefault(scenario, ScenarioCycles()).lock

    @contextmanager
    def cycle(self, scenario: str) -> Iterator[None]:
        try:
            with self.cycle_lock(scenario):
                yield
        finally:
            # A cycle is a use of the scenario. Its idle time starts when its last cycle ends.
            self.registry.touch(scenario)
            self.close_replaced_instances(scenario)

    def dispatched_turn(self, current: Scenario, component: str | None) -> ExitStack:
        """Hold colocated cycle locks through execution, commit, and acknowledgement."""
        backend = current.trainer_for(component).candidate_backend
        if backend is None or not backend.colocated:
            return ExitStack()
        names = {loaded.name for loaded in self.registry.loaded_scenarios()}
        names.add(current.name)
        with self.lock:
            names.update(self.scenarios)
        names_in_order = sorted(names)
        self.turn_waiting.set()
        try:
            with ExitStack() as turn:
                # Callbacks run after every lock is released, including on partial acquisition failure.
                for name in names_in_order:
                    turn.callback(self.close_replaced_instances, name)
                for name in names_in_order:
                    turn.enter_context(self.cycle_lock(name))
                return turn.pop_all()
        finally:
            self.turn_waiting.clear()

    def close_replaced(self, instance: Scenario) -> None:
        with self.lock:
            self.scenarios.setdefault(instance.name, ScenarioCycles()).replaced.append(instance)
        self.close_replaced_instances(instance.name)

    def close_replaced_instances(self, scenario: str) -> None:
        lock = self.cycle_lock(scenario)
        while True:
            if not lock.acquire(blocking=False):
                return  # The holder drains replacements when it releases the cycle lock.
            try:
                with self.lock:
                    state = self.scenarios[scenario]
                    replaced, state.replaced = state.replaced, []
                for instance in replaced:
                    instance.close()
            finally:
                lock.release()
            with self.lock:
                if not state.replaced:
                    return

    def instance_change_pending(self, scenario: str) -> bool:
        """Whether a deferred reload of ``scenario`` waits, or a replaced instance of it waits to close."""
        with self.lock:
            state = self.scenarios.get(scenario)
            return state is not None and (state.reload_pending or bool(state.replaced))

    def take_replaced(self) -> list[Scenario]:
        """Collect remaining replacements after all workers have joined."""
        with self.lock:
            replaced = [instance for state in self.scenarios.values() for instance in state.replaced]
            for state in self.scenarios.values():
                state.replaced.clear()
            return replaced

    def reload_after_local_failure(self, current: Scenario) -> None:
        """Reload durable state now, or after the reserved job; do not wake the failing worker."""
        if not current.store.durable:
            return
        with self.registry.lock_for(current.name):
            if self.registry.get_optional(current.name) is not current:
                return
            if current.is_job_reserved:
                with self.lock:
                    self.scenarios.setdefault(current.name, ScenarioCycles()).reload_pending = True
                logger.info("scenario %r reloads once its training job has landed", current.name)
                return
            self.reload(current.name, current)

    def reload_deferred(self, current: Scenario) -> bool:
        """Called under the cycle lock: reload only after the reserved job has landed."""
        with self.lock:
            state = self.scenarios.get(current.name)
            if state is None or not state.reload_pending:
                return False
        with self.registry.lock_for(current.name):
            if self.registry.get_optional(current.name) is not current or current.is_job_reserved:
                return False
            self.reload(current.name, current)
        return True

    def reload(self, scenario: str, current: Scenario | None) -> Scenario:
        """Rebuild durable state, retaining failed instructions that still need skip rows."""
        with self.lock:
            state = self.scenarios.get(scenario)
            if state is not None:
                state.reload_pending = False
        failures = (
            {bound.component: bound.trainer.instruction_failures() for bound in current.trainers}
            if current is not None
            else {}
        )
        recovered = self.registry.reload(scenario)
        for component, failed in failures.items():
            if failed:
                recovered.trainer_for(component).set_instruction_failures(failed)
        return recovered


class Dispatcher(ScenarioUnloader):
    """Coordinate scenario creation, inference state, training, and model commits.

    Invariant: at most one weight-training scenario per process. Its serial
    thread is bound to the first scenario using the deployment's
    ``TrainingRuntime``; resolving a second raises (enforced in
    :class:`ScenarioRegistry`).
    Local training backends are unlimited and drain on per-scenario threads.
    The dispatcher owns its recipe's runtime and closes it after all scenarios.
    It unloads an idle scenario (see :meth:`scenario_is_idle`) after
    ``scenario_idle_unload_seconds``, and when scenario admission needs a
    load slot.
    """

    storage_retry_seconds: float = 5.0
    # Poll interval while a pipeline has background jobs that run (see
    # TrainingPipeline.background_jobs_running). A finished job, such as a
    # judgment, can make a batch ready, but no new record sets the ready
    # event. Thus the thread checks for a batch at this interval.
    background_job_poll_seconds: float = 1.0
    # One drain, plus one more after reloading the scenario from durable state.
    drain_attempts: int = 2
    # A local result refused this many times in a row, each time because a
    # dispatched commit replaced its base, stops spinning: the worker keeps
    # the batch, reports the refusals, and prepares it again on its next wake.
    stale_refusal_limit: int = 3
    # A ready batch should be reserved by the next drain; one that sits longer
    # means the training thread is not waking. Status reads perform the check,
    # so the alarm rides the health polling that already watches the service.
    undrained_warning_seconds: float = 60.0
    operational_metrics_interval_seconds: float = 10.0
    # The longest time between two checks for scenarios that are idle for long enough to unload.
    idle_unload_check_seconds: float = 30.0

    def __init__(
        self,
        recipe: Recipe,
        backend_factory: RepositoryBackendFactory,
        *,
        local_artifact_dir: Path | None = None,
        allow_implicit_creation: bool = True,
        experiment_tracker: ExperimentTracker | None = None,
        scenario_storage: ScenarioStorage,
        hold_local_cycles: bool = False,
        scenario_idle_unload_seconds: float | None = None,
        scenario_admission: ScenarioAdmission | None = None,
    ) -> None:
        """``hold_local_cycles``: run no local cycle until :meth:`open_local_cycles`, for a service whose recipe
        calls this same service, which does not answer yet while the dispatcher starts.

        ``scenario_idle_unload_seconds``: unload a scenario that is idle for
        this time. With ``None``, every scenario stays loaded.

        ``scenario_admission``: decides whether a scenario is created or
        loaded. The default admits all scenarios.
        """
        if scenario_idle_unload_seconds is not None and scenario_idle_unload_seconds <= 0:
            raise ValueError("scenario_idle_unload_seconds must be positive")
        self._recipe = recipe
        self._storage = scenario_storage
        self._record_retention_lock = Lock()
        self._experiment_tracker = experiment_tracker if experiment_tracker is not None else NullExperimentTracker()
        self._registry = ScenarioRegistry(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            scenario_storage=scenario_storage,
            allow_implicit_creation=allow_implicit_creation,
            experiment_tracker=self._experiment_tracker,
            admission=scenario_admission,
        )
        self._registry.set_training_scenario_callback(self._start_training)
        self._registry.set_unloader(self)
        self._training = _TrainingState()
        self._lifecycle = _LifecycleState()
        self.workers = WorkerCoordination(self._registry, self._training.lock, held=hold_local_cycles)
        self._registry.set_replaced_closer(self.workers)
        if isinstance(backend_factory, EnumerableRepositoryBackendFactory):
            self._lifecycle.preload_thread = Thread(
                target=self._preload_scenarios,
                args=(backend_factory.list_registrations(),),
                name="reef-preload",
                daemon=True,
            )
            self._lifecycle.preload_thread.start()
        if self._experiment_tracker.operational_metrics_enabled:
            self._lifecycle.metrics_thread = Thread(
                target=self.run_operational_metrics,
                name="reef-operational-metrics",
                daemon=True,
            )
            self._lifecycle.metrics_thread.start()
        if scenario_idle_unload_seconds is not None:
            self._lifecycle.idle_unload_thread = Thread(
                target=self.run_idle_unload,
                args=(scenario_idle_unload_seconds,),
                name="reef-idle-unload",
                daemon=True,
            )
            self._lifecycle.idle_unload_thread.start()

    @property
    def recipe(self) -> Recipe:
        """The deployment's recipe, before any scenario's own model settings."""
        return self._recipe

    # -- Scenario API (delegates to registry) ----------------------------

    def has_scenario(self, scenario: str) -> bool:
        return self._registry.has(scenario)

    def get_or_create_scenario(
        self,
        scenario: str,
        *,
        release_id: str | None = None,
        allow_implicit_creation: bool | None = None,
    ) -> Scenario | None:
        return self._registry.get_or_create(
            scenario,
            release_id,
            allow_implicit_creation=allow_implicit_creation,
        )

    def loaded_scenario(self, scenario: str, *, release_id: str | None = None) -> Scenario | None:
        """The loaded instance of ``scenario``, or None; see :meth:`ScenarioRegistry.get_loaded`."""
        return self._registry.get_loaded(scenario, release_id)

    def configure_scenario_model(
        self, scenario: str, model: object, *, create: bool = False, release_id: str | None = None
    ) -> Scenario:
        return self._registry.configure_model(scenario, model, create=create, release_id=release_id)

    def set_training_mode(self, scenario: str, training_mode: str) -> dict[str, Any]:
        with self._registry.lock_for(scenario):
            current = self._registry.set_training_mode(scenario, training_mode)
            self._wake_training(current)
            return {"scenario": scenario, "training_mode": current.training_mode}

    def _wake_training(self, current: Scenario) -> None:
        if current.training_runtime is not None:
            self._training.ready.set()
        for bound in current.trainers:
            backend = bound.trainer.candidate_backend
            if backend is not None and not backend.dispatched:
                self._start_local_backend_worker(current.name, bound.component)

    def list_scenarios(self) -> tuple[dict[str, Any], ...]:
        return self._registry.list()

    def prune_record_archives(self, retention: RecordRetention) -> int:
        """Apply deployment-wide retention without racing scenario file moves."""
        with self._record_retention_lock:
            return self._storage.prune(max_bytes=retention.max_bytes)

    def delete_scenario(self, scenario: str) -> dict[str, Any]:
        """Remove a scenario from this deployment and move its own state aside.

        Under the scenario's lock, so no accept or commit interleaves: the
        training thread and the local worker lose the name, the loaded
        instance closes, the storage archives the records, commits and
        settings, the recipe's own directories move
        beside themselves, and the repository registration is archived so
        the name is free. Artifacts other scenarios share stay.
        """
        if "/" in scenario or scenario in ("", ".", ".."):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        with self._record_retention_lock, self._registry.lock_for(scenario):
            if not self._registry.has(scenario):
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            loaded = self._registry.get_optional(scenario)
            if (loaded is not None and loaded.is_job_reserved) or self.training_job_in_flight(scenario):
                # The backend's job would outlive its scenario: nothing could commit or acknowledge it, and
                # its marker keeps inference admission closed for every scenario on the runtime.
                raise ScenarioBusy(
                    f"cannot delete scenario {scenario!r} while its training job is out; "
                    "retry once the job has committed or been rejected"
                )
            dropped = self._registry.remove(scenario)
            self._stop_local_backend_worker(scenario)
            self._record_training_error(scenario, None, source=None)
            with self._training.lock:
                self._training.failure_counts.pop(scenario, None)
            if dropped is not None:
                for bound in dropped.trainers:
                    backend = bound.trainer.candidate_backend
                    if backend is not None:
                        backend.retire_scenario(scenario)
                dropped.close()
            self.workers.close_replaced_instances(scenario)
            archived = [*self._archive_scenario_state(scenario), *self._registry.archive_registration(scenario)]
        self._registry.forget_lock(scenario)
        return {"scenario": scenario, "archived": archived}

    def scenario_is_idle(self, current: Scenario) -> bool:
        """Whether Reef can unload ``current`` with no loss of data and no stop of work.

        A scenario is idle when all of these are true:

        - Its storage is durable.
        - No request runs on it.
        - No trainer has records that it would still train on.
        - No release exists only in memory.
        - No reload or close of an instance waits.

        A weight-training scenario is never idle, because only a delete
        releases its runtime binding. The caller holds the lock and the
        cycle lock of the scenario, so no record or cycle starts meanwhile.
        """
        if current.training_runtime is not None or not current.store.durable:
            return False
        if current.operations.active_count() > 0 or current.repository.holds_local_releases:
            return False
        if self.workers.instance_change_pending(current.name):
            return False
        return not any(bound.trainer.holds_untrained_records() for bound in current.trainers)

    def unload_if_idle(self, scenario: str) -> bool:
        """Unload ``scenario`` if it is idle, and return whether it was unloaded.

        Unload closes the loaded instance and stops its local workers. It is
        a delete without the archive: the storage, the repository
        registration and the settings stay, and the next request loads the
        scenario again. Counters in memory start again from zero. If another
        thread holds the lock or the cycle lock of the scenario, the scenario
        is busy, so this method never waits for them.
        """
        with self._registry.lock_for(scenario, blocking=False) as acquired:
            if not acquired:
                return False
            cycle_lock = self.workers.cycle_lock(scenario)
            if not cycle_lock.acquire(blocking=False):
                return False
            try:
                current = self._registry.get_optional(scenario)
                if current is None or not self.scenario_is_idle(current):
                    return False
                self._registry.unload(scenario)
                self._stop_local_backend_worker(scenario)
                current.close()
            finally:
                cycle_lock.release()
            self.workers.close_replaced_instances(scenario)
        logger.info("unloaded idle scenario %r; its next request loads it again", scenario)
        return True

    def run_idle_unload(self, idle_seconds: float) -> None:
        """Until the dispatcher closes, unload each scenario that has no use for ``idle_seconds`` and is idle."""
        while not self.workers.closed.wait(min(idle_seconds, self.idle_unload_check_seconds)):
            for name in self._registry.unused_since(time.monotonic() - idle_seconds):
                self.unload_if_idle(name)

    def training_job_in_flight(self, scenario: str) -> bool:
        """Whether the training runtime's durable job marker names a job of ``scenario`` still out.

        A loaded instance knows its reserved batch; the marker also covers a
        scenario whose turn failed and was rebuilt, or one not loaded yet,
        whose job the backend still holds. Every job's marker names the
        scenario that owns it.
        """
        try:
            marker = self.in_flight_job_marker()
        except Exception as exc:
            # A runtime that does not answer cannot say whether a job is out; a blind delete could orphan one.
            raise ScenarioBusy(
                f"cannot delete scenario {scenario!r}: the training runtime does not say whether its job is out "
                f"({self._error_text(exc)}); retry once it answers"
            ) from exc
        return marker is not None and marker.get("scenario") == scenario

    def in_flight_job_marker(self) -> Mapping[str, Any] | None:
        """The training runtime's durable job marker while it names a job still out; runtime failures propagate."""
        runtime = self._recipe.training_runtime
        if runtime is None:
            return None
        marker = runtime.training_job_status()
        return marker if marker_in_flight(marker) else None

    def _archive_scenario_state(self, scenario: str) -> list[str]:
        """Move the scenario's own files and directories under an ``archived`` sibling, stamped so a name can be deleted twice."""
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        moved = list(self._registry.archive_store(scenario))
        for directory in self._recipe.scenario_state_dirs(scenario):
            if directory.exists():
                destination = directory.parent / "archived" / f"{directory.name}-{stamp}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(directory), str(destination))
                moved.append(str(destination))
        return moved

    def recipe_has_files(self) -> bool:
        """Whether the served recipe creates a file-serving surface."""
        # Capability probe only: file serving does not depend on the scenario.
        return self._recipe.build_surface("").files is not None

    def list_releases(self, scenario: str) -> tuple[dict[str, Any], ...]:
        with self._registry.lock_for(scenario):
            return self._registry.require(scenario).releases()

    def read_records(
        self, scenario: str, *, after_sequence: int = 0, limit: int = 50, request_type: RequestType | None = None
    ) -> dict[str, Any]:
        """Read retained summaries, optionally of one request type, without changing the training queue."""
        from reef.scenario.history import read_records

        with self._registry.lock_for(scenario):
            return read_records(
                self._registry.require(scenario),
                after_sequence=after_sequence,
                limit=limit,
                request_type=request_type,
            )

    def read_record(self, scenario: str, record_id: str) -> dict[str, Any] | None:
        """Read a retained trace within its scenario, including consumed records."""
        from reef.scenario.history import read_record

        with self._registry.lock_for(scenario):
            return read_record(self._registry.require(scenario), record_id)

    def read_commits(
        self, scenario: str, *, after_step: int = 0, limit: int = 50, record_ids: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        from reef.scenario.history import read_commits

        with self._registry.lock_for(scenario):
            return read_commits(
                self._registry.require(scenario), after_step=after_step, limit=limit, record_ids=record_ids
            )

    def scenario_contract(self, scenario: str) -> dict[str, Any]:
        with self._registry.lock_for(scenario):
            current = self._registry.require(scenario)
            pipeline = current.trainer.pipeline
            return {
                "scenario": scenario,
                # The key keeps its name for callers; the value names the recipe's task-to-batch stage.
                "processor": type(pipeline.task2batch).__name__,
                "training_mode": current.trainer.pipeline.training_mode,
                "status": dict(current.trainer.pipeline_status()),
                "required_request_types": sorted(rt.value for rt in pipeline.required_request_types),
            }

    def promote(self, scenario: str, release_id: str) -> ArtifactRef:
        """Serve a pending release: the rollback path under its own record operation."""
        return self.rollback(scenario, release_id, operation="promote")

    def rollback(self, scenario: str, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        while True:
            with self._registry.lock_for(scenario):
                current = self._registry.require(scenario)
            # Wait for earlier reports without blocking scenario resolution.
            with current.experiment_reporting(), self._registry.lock_for(scenario):
                if self._registry.get_optional(scenario) is not current:
                    # Reload or deletion won the gap before the reporting scope.
                    continue
                return current.rollback(release_id, operation=operation)

    def current_artifact(
        self,
        scenario: str,
        *,
        release_id: str | None = None,
    ) -> Artifact:
        current = self.get_or_create_scenario(scenario, release_id=release_id)
        if current is None:
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        return Artifact(current.repository.require_current_artifact(), current.repository)

    # -- Record acceptance -----------------------------------------------

    def accept_record(
        self,
        item: AgentRecord,
        *,
        release_id: str | None = None,
    ) -> AgentRecord:
        with self._registry.lock_for(item.scenario):
            # An instruction names a scenario that exists; inference and reports keep implicit creation.
            current = self.get_or_create_scenario(
                item.scenario,
                release_id=release_id,
                allow_implicit_creation=False if item.request_type is RequestType.TRAIN else None,
            )
            if current is None:
                raise UnknownScenario(f"unknown scenario {item.scenario!r}")
            return self._accept_record(current, item)

    def accept_records(
        self, items: Sequence[AgentRecord], *, release_id: str | None = None
    ) -> tuple[AgentRecord, ...]:
        """Validate an ordered import batch, commit it, then wake its consumer."""
        if not items:
            raise ValueError("a record batch must not be empty")
        scenario = items[0].scenario
        if any(item.scenario != scenario or item.request_type is RequestType.TRAIN for item in items):
            raise ValueError("a record batch must contain inferences or reports from one scenario")
        with self._registry.lock_for(scenario):
            current = self.get_or_create_scenario(scenario, release_id=release_id)
            if current is None:
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            preceding: dict[str, AgentRecord] = {}
            for item in items:
                self.validate_record(current, item, preceding)
                preceding[item.agent_record_id] = item
            try:
                with current.operations.measure("ingest/write"):
                    appended = current.records.append_many(items)
            except RecordConflict:
                current.operations.increment("ingest/rejected_conflict_total")
                raise
            for result in appended:
                metric = "ingest/accepted_total" if result.inserted else "ingest/duplicates_total"
                current.operations.increment(metric)
            if any(result.inserted for result in appended):
                self.process_accepted_records(current)
            return tuple(result.item for result in appended)

    def validate_record(
        self, current: Scenario, item: AgentRecord, preceding: Mapping[str, AgentRecord]
    ) -> AgentRecord | None:
        try:
            if item.request_type is RequestType.TRAIN:
                if (existing := current.records.existing_receipt(item)) is not None:
                    return existing
                if current.training_mode == "auto":
                    raise ValueError("explicit training requests require training_mode='manual' or 'hybrid'")
                if all(bound.trainer.candidate_backend is None for bound in current.trainers):
                    raise ValueError("explicit training requests require a training backend")
                request = normalize_training_request_payload(item.payload)
                if item.references:
                    raise ValueError("training instructions do not reference inference receipts")
                refusal = training_request_refusal(request["text"], request["requires"])
                if refusal is not None:
                    raise ValueError(refusal)
            # Schema enforcement: reject a malformed report before it is durably
            # appended, so the producer's POST fails with the violation naming
            # the broken field instead of the record dying silently at training
            # time. An undeclared schema keeps open ingress; a scenario of several
            # components admits what any of them accepts, and each trainer releases
            # a report shaped for another.
            if item.request_type is RequestType.REPORT:
                # An identical retry remains valid after capacity eviction.
                if (existing := current.records.existing_receipt(item)) is not None:
                    return existing
                validate_report_payload(item.payload)
                if (report_type := current.report_type) is not None:
                    report_type.from_dict(item.payload)
                if len(set(item.references)) != len(item.references):
                    raise ReportValidationError("report references must be unique")
                for reference in item.references:
                    source = preceding.get(reference)
                    if source is None:
                        stored_reference = current.records.get_for_audit(item.scenario, reference)
                        source = stored_reference.item if stored_reference is not None else None
                    if source is None or source.request_type is not RequestType.INFERENCE:
                        raise ReportValidationError(
                            f"report reference {reference!r} must identify an existing inference in scenario {item.scenario!r}"
                        )
        except ReportValidationError:
            current.operations.increment("ingest/rejected_report_total")
            raise
        except RecordConflict:
            current.operations.increment("ingest/rejected_conflict_total")
            raise
        except ValueError:
            current.operations.increment("ingest/rejected_request_total")
            raise
        return None

    def _accept_record(self, current: Scenario, item: AgentRecord) -> AgentRecord:
        if (existing := self.validate_record(current, item, {})) is not None:
            current.operations.increment("ingest/duplicates_total")
            return existing
        try:
            with current.operations.measure("ingest/write"):
                appended = current.records.append_result(item)
        except RecordConflict:
            current.operations.increment("ingest/rejected_conflict_total")
            raise
        stored = appended.item
        if not appended.inserted:
            current.operations.increment("ingest/duplicates_total")
            return stored
        current.operations.increment("ingest/accepted_total")
        self.process_accepted_records(current)
        return stored

    def process_accepted_records(self, current: Scenario) -> None:
        """Start consumption only after the accepted records are durable."""
        # Every component's trainer sees the record: a dispatched backend wakes
        # the training thread, a local backend its own worker, and a trainer
        # without a backend consumes inline.
        self._wake_training(current)
        if current.training_runtime is not None:
            return
        for bound in current.trainers:
            if bound.trainer.candidate_backend is not None:
                continue
            result = current.prepare_training_step(bound.component)
            if result is not None:
                # Commit consumption durably with the model update so recovery
                # never repeats a batch whose update was already published.
                self._commit_result(current.name, result, bound.component)

    # -- Commit & publication --------------------------------------------

    def _commit_result(self, scenario: str, result: TrainStepResult, component: str | None = None) -> None:
        current = self._registry.get(scenario)
        if current.trainer_for(component).result_on_record(result):
            # Another trainer's commit settled this trainer's durable record since the caller read the result.
            raise SettledTrainingResultError(f"scenario {scenario!r} component {component!r}: its result is on record")
        current.commit(result, component=component)

    # -- Preload ---------------------------------------------------------

    def _preload_scenarios(self, scenarios: tuple[str, ...]) -> None:
        # Preload failures land in their own training_status field: a scenario
        # that cannot be re-loaded at boot is a different failure domain from
        # a training step that cannot commit.
        for scenario in scenarios:
            try:
                current = self.get_or_create_scenario(scenario)
                if current is not None:
                    # Rows left unread before the restart train now, not once the next record arrives.
                    self._wake_training(current)
            except ScenarioCapacityFull:
                # This is not a preload failure. The scenario loads on its first request, when a load slot is free.
                logger.info("scenario %r is not preloaded: the service holds its limit of loaded scenarios", scenario)
            except Exception as exc:
                logger.exception("failed to preload scenario %r", scenario)
                self._registry.record_preload_error(scenario, f"{type(exc).__name__}: {exc}")

    # -- Background workers ---------------------------------------------

    def run_operational_metrics(self) -> None:
        """Sample while training is idle, blocked, or failing, without status polling."""
        while not self.workers.closed.wait(self.operational_metrics_interval_seconds):
            self.record_operational_metrics()

    def record_operational_metrics(self) -> None:
        """Publish numeric state only; one unavailable scenario must not hide others."""
        for current in self._registry.loaded_scenarios():
            self.record_scenario_operational_metrics(current)

    def record_scenario_operational_metrics(self, current: Scenario) -> None:
        """Keep recipe, storage, and provider sampling failures local to one scenario."""
        try:
            metrics = current.trainer.operational_metrics()
            metrics.update(current.operations.snapshot())
            with self._training.lock:
                metrics["training/failed_attempts_total"] = self._training.failure_counts.get(current.name, 0)
                metrics["training/error"] = int(any(key[0] == current.name for key in self._training.errors))
                if current.training_runtime is not None:
                    metrics["training/checkpoint_storage_blocked"] = int(self._training.storage_status is not None)
            current.trainer.pipeline.experiment_logger.log(metrics, namespace="operations")
        except Exception as exc:
            logger.warning("operational metrics unavailable for scenario %r (%s)", current.name, type(exc).__name__)

    def _start_training(self, scenario: Scenario) -> None:
        """Start the training drain thread if it hasn't been started yet.

        Called by the registry whenever it resolves a training scenario — the
        single notification point, so every resolution path (get_or_create,
        require, preload) triggers it without the dispatcher remembering to
        call from each entry point.
        """
        if self.workers.closed.is_set():
            return
        with self._training.lock:
            if self._training.thread is not None:
                return
            self._training.thread = Thread(
                target=self._run_training,
                name="reef-training",
                daemon=True,
            )
            self._training.thread.start()
        self._training.ready.set()

    def _start_local_backend_worker(self, scenario: str, component: str | None = None) -> None:
        ready = Event()
        thread = Thread(
            target=self._run_local_backend_worker,
            args=(scenario, component, ready),
            name=(
                f"reef-local-backend-{scenario}" if component is None else f"reef-local-backend-{scenario}/{component}"
            ),
            daemon=True,
        )
        self.workers.start(scenario, component, LocalBackendWorker(ready, thread))

    def _stop_local_backend_worker(self, scenario: str) -> None:
        self.workers.remove(scenario)
        with self._training.lock:
            for counts in (self._training.stale_refusals_in_a_row, self._training.stale_refusals_total):
                for key in tuple(counts):
                    if key[0] == scenario:
                        counts.pop(key)

    def stale_refusals_total(self, scenario: str, component: str | None) -> int:
        with self._training.lock:
            return self._training.stale_refusals_total.get((scenario, component), 0)

    def record_local_cycle_error(self, scenario: str, component: str | None, error: str) -> None:
        """Record a local cycle's failure, unless the scenario was deleted under it: nothing would ever clear that."""
        with self._training.lock:
            if (scenario, component) in self.workers.workers:
                self.note_training_error(scenario, error, local_error_source(component))

    def _run_local_backend_worker(self, scenario: str, component: str | None, ready: Event) -> None:
        try:
            while self.workers.wait(scenario, component, ready):
                self._drain_local_backend(scenario, component)
        except Exception as exc:
            logger.exception("local backend worker stopped unexpectedly for scenario %r", scenario)
            self._record_training_error(scenario, self._error_text(exc), source=local_error_source(component))

    def _drain_local_backend(self, scenario: str, component: str | None) -> None:
        try:
            # A dispatched job waiting for its turn goes before the next cycle; it wakes the workers after.
            while (
                not self.workers.turn_waiting.is_set()
                and self.workers.cycles_run()
                and self._process_local_backend_step(scenario, component)
            ):
                pass
        except Exception as exc:
            logger.exception("local backend failed to commit for scenario %r", scenario)
            self.record_local_cycle_error(scenario, component, self._error_text(exc))

    def _fail_instruction(self, current: Scenario, cause: Exception, component: str | None = None) -> None:
        """A failed instruction step consumes the instruction with a skip row on the next step; wake for it."""
        if not current.trainer_for(component).fail_pending_instruction(self._error_text(cause)):
            return
        with self._registry.lock_for(current.name):
            # Wake the worker so failed instructions settle even when no new records arrive; a scenario
            # deleted or reloaded meanwhile starts no worker for an instance that is gone.
            if self._registry.get_optional(current.name) is current:
                self._wake_training(current)

    def _reload_after_training_failure(self, scenario: str, cause: Exception) -> None:
        current = self._registry.get_optional(scenario)
        if current is None:
            if not self._registry.has(scenario):
                # Deleted while its step was in flight: nothing durable to reload.
                return
            self.workers.reload(scenario, None)
            return
        self._fail_instruction(current, cause, current.dispatched_component)
        self.workers.reload(scenario, current)
        # The rebuilt instance holds the local components' rows unread; their workers look again.
        self.workers.wake_all()

    def _process_local_backend_step(self, scenario: str, component: str | None = None) -> bool:
        current = self._registry.get_optional(scenario)
        if current is None:
            # Deleted while a harness step was in flight: the step ends without
            # a commit and without an error, as documented in http-api.rst.
            return False
        if current.trainer_for(component).candidate_backend is None:
            raise RuntimeContractError(f"scenario {scenario!r} has no local backend")
        if not self.workers.check_admission(current, component):
            return False
        self._record_training_error(scenario, None, source=local_error_source(component))
        # Local workers of one scenario take turns for a whole cycle, prepare
        # and commit together, under this lock alone: letting their commits
        # race only had the slower worker refused as stale on every cycle, and
        # each refusal threw away a full candidate evaluation. A dispatched
        # commit still lands meanwhile; the stale policy answers it.
        with self.workers.cycle(scenario):
            if not self.workers.cycles_run():
                return False
            loaded = self._registry.get_optional(scenario)
            if loaded is None:
                # Deleted while this worker waited for its turn.
                return False
            if loaded is not current:
                # Reloaded while this worker waited: look again on the rebuilt instance, whose rows are unread.
                return True
            # The job whose turn this worker waited for may have left admission closed.
            if not self.workers.check_admission(current, component):
                return False
            if self.workers.reload_deferred(current):
                return True  # The rebuilt instance holds this worker's rows unread.
            try:
                result = current.prepare_training_step(component)
            except Exception as exc:
                self._fail_instruction(current, exc, component)
                self.workers.reload_after_local_failure(current)
                raise
            if result is None:
                return False
            # Keep only the short commit and recovery window under the scenario
            # registry lock. Candidate generation above can take minutes and must
            # not block record acceptance for this scenario.
            with current.experiment_reporting(), self._registry.lock_for(scenario):
                if not self.workers.cycles_run():
                    # The service stopped under this cycle: its model calls may have failed as the routes went away.
                    logger.info(
                        "scenario %r component %r: the service is stopping; the step waits", scenario, component
                    )
                    return False
                loaded = self._registry.get_optional(scenario)
                if loaded is not current:
                    # Rebuilt under this cycle by a sibling trainer's failure, or deleted: the result belongs to the
                    # instance that prepared it, and the step ends without a commit. A rebuilt instance holds these
                    # rows unread: look again on it.
                    return loaded is not None
                try:
                    self._commit_result(scenario, result, component)
                except StaleTrainingResultError as stale:
                    return self.retry_stale_result(scenario, current, component, stale)
                except SettledTrainingResultError:
                    # The step this worker would have made is in the log already; look again for the next one.
                    logger.info("scenario %r component %r: another commit settled its result", scenario, component)
                    return True
                except Exception:
                    # A record may already have crossed the fsync commit point.
                    # Reload before rollback or acceptance can observe the stale
                    # in-memory step and append the same step number again.
                    self.workers.reload_after_local_failure(current)
                    raise
        with self._training.lock:
            self._training.stale_refusals_in_a_row.pop((scenario, component), None)
        return True

    def retry_stale_result(
        self, scenario: str, current: Scenario, component: str | None, stale: StaleTrainingResultError
    ) -> bool:
        """Keep the refused batch for another preparation; after a few refusals in a row, wait for the next wake.

        Only a dispatched commit can still overtake a local result: the local
        workers take turns. A weights job that lands during every local cycle
        would otherwise keep the local worker preparing and discarding forever
        with nothing in the status to show for it.
        """
        key = (scenario, component)
        with self._training.lock:
            in_a_row = self._training.stale_refusals_in_a_row.get(key, 0) + 1
            self._training.stale_refusals_in_a_row[key] = in_a_row
            self._training.stale_refusals_total[key] = self._training.stale_refusals_total.get(key, 0) + 1
        reevaluate = stale.policy == "reevaluate"
        current.retry_pending(component, keep_candidate=reevaluate)
        if in_a_row < self.stale_refusal_limit:
            logger.info(
                "scenario %r component %r: %s; %s",
                scenario,
                component,
                stale,
                "evaluating the candidate again" if reevaluate else "preparing the batch again",
            )
            return True
        message = (
            f"StaleTrainingResultError: component {component!r} was refused {in_a_row} times in a row; "
            "its batch is kept and prepared again on the next wake"
        )
        logger.warning("scenario %r: %s", scenario, message)
        self._record_training_error(scenario, message, source=local_error_source(component))
        return False

    def _run_training(self) -> None:
        # Keep prepare, remote execution, and the trainer/version-chain commit
        # in one serial thread. Accepts may signal readiness while a commit is
        # slow; serial draining prevents two wake-ups from observing and
        # committing the same pending result as separate steps.
        try:
            while True:
                self._training.ready.wait(self._training_wait_timeout())
                self._training.ready.clear()
                if self.workers.closed.is_set():
                    return
                self._drain_training()
                self._record_training_drain()
        except Exception as exc:
            name = self._registry.training_scenario_name or "<unbound>"
            logger.exception("training thread stopped unexpectedly for scenario %r", name)
            self._record_training_error(name, self._error_text(exc))

    def _training_wait_timeout(self) -> float | None:
        """How long the training thread may sleep before re-checking.

        Blocked checkpoint storage retries on its own interval. While a
        pipeline has background jobs that run, a finished job can make a
        batch ready without a new record. Thus the thread polls at
        ``background_job_poll_seconds``. Otherwise it sleeps until the next
        accepted record.
        """
        timeout: float | None = self.storage_retry_seconds if self._training.storage_status is not None else None
        for name in self._training_scenario_names():
            current = self._registry.get_optional(name)
            if current is not None and any(
                bound.trainer.pipeline.background_jobs_running() for bound in current.trainers
            ):
                timeout = (
                    self.background_job_poll_seconds
                    if timeout is None
                    else min(timeout, self.background_job_poll_seconds)
                )
        return timeout

    def _drain_training(self) -> None:
        # One recovery attempt, not a retry loop: if draining fails we reload
        # the scenario from durable state and drain once more, so a crash
        # between a durable commit and processor cleanup is healed on the spot. A
        # second failure still reloads (leaving a clean scenario for the next
        # wake-up) but is not spun on; the cause is reported through
        # training_status.
        for _ in range(self.drain_attempts):
            try:
                while self._process_training():
                    pass
                return
            except _ScenarioTrainingError as failure:
                name = failure.scenario
                logger.exception("training thread failed to commit for scenario %r", name)
                self._record_training_error(name, self._error_text(failure.cause))
                self._reload_after_training_failure(name, failure.cause)
            except Exception as exc:
                name = self._registry.training_scenario_name or "<unbound>"
                logger.exception("training thread failed to commit for scenario %r", name)
                self._record_training_error(name, self._error_text(exc))
                self._reload_after_training_failure(name, exc)

    def _training_scenario_names(self) -> tuple[str, ...]:
        return self._registry.training_scenario_names

    def _process_training(self) -> bool:
        """Give every training scenario one turn; True when any of them progressed.

        Several scenarios share one training runtime only when it time-slices
        its adapter slot between them; the round-robin keeps the schedule
        deterministic and fair, and a failure in one scenario's turn reloads
        that scenario alone.
        """
        names = self._training_scenario_names()
        if not names:
            # The last training scenario was deleted: the thread idles until a scenario binds again.
            return False
        progressed = False
        for name in names:
            try:
                progressed = self._process_training_scenario(name) or progressed
            except Exception as exc:
                raise _ScenarioTrainingError(name, exc) from exc
        return progressed

    def _process_training_scenario(self, name: str) -> bool:
        current = self._registry.get_optional(name)
        if current is None:
            raise RuntimeContractError(f"training thread is not bound to scenario {name!r}")
        runtime = current.training_runtime
        if not isinstance(runtime, TrainingRuntime):
            raise RuntimeContractError(
                f"training thread requires a TrainingRuntime for scenario {current.name!r}, got {type(runtime).__name__}"
            )
        self._record_training_error(current.name, None)
        # A crash may leave remote serving updated but paused after Reef's
        # commit, or checkpointed before the weight update. Recover that
        # pending step before deciding whether another batch is available.
        component = current.dispatched_component
        backend = current.trainer_for(component).candidate_backend
        if backend is None or not backend.dispatched:
            raise RuntimeContractError(f"training scenario {current.name!r} has no dispatched training backend")
        backend.recover_pending_step(
            current.scenario_step,
            committed_training_job_id=current.committed_training_job_id,
            committed_training_without_job_id=current.committed_training_without_job_id,
        )
        with self._registry.lock_for(name):
            if self._registry.get_optional(name) is not current:
                # Reloaded since the turn began: the next turn reserves on the rebuilt instance.
                return True
            # Reserved under the registry lock: a local cycle's failure from here on defers its reload.
            batch = current.reserve_training_batch(component)
        if batch is None:
            return False
        try:
            return self.run_dispatched_turn(current, component, backend, batch)
        finally:
            # Local cycles that stood aside or yielded for the job run now, on every scenario, whatever the
            # outcome; a cycle whose reload waited for the job rebuilds the scenario as its first act.
            self.workers.wake_all()

    def run_dispatched_turn(
        self, current: Scenario, component: str | None, backend: CandidateBackend, batch: TrainingBatch
    ) -> bool:
        with self.workers.dispatched_turn(current, component):
            execution = current.execute_reserved_training_step(component)
            if execution.outcome == "retry":
                if execution.storage is None:
                    raise RuntimeContractError("retry execution must carry storage status")
                self._set_training_storage_status(dict(execution.storage))
                return False
            self._set_training_storage_status(None)
            if execution.outcome not in ("drop", "commit"):
                raise RuntimeContractError(f"training backend returned unsupported outcome: {execution.outcome!r}")
            # The registry lock keeps a local worker's reload out between the instance check and the commit.
            with current.experiment_reporting(), self._registry.lock_for(current.name):
                loaded = self._registry.get_optional(current.name)
                if loaded is None:
                    raise RuntimeContractError(
                        f"scenario {current.name!r} was deleted under its training job; the backend's job needs "
                        "operator recovery"
                    )
                if loaded is not current:
                    # A local worker's failure reloaded the scenario while the job ran: its result belongs to the
                    # instance that reserved the batch. The rebuilt trainer reserves the same rows and the
                    # backend replays the job under its marker, so the commit lands with the job's identity.
                    logger.warning("scenario %r was reloaded under its training job; the job replays", current.name)
                    return True
                if execution.outcome == "drop":
                    logger.warning("dropping stale training batch %r for scenario %r", batch.batch_id, current.name)
                    current.reject_pending(execution.metrics, component=component)
                    return True
                result = execution.result
                if result is None:
                    raise RuntimeContractError("commit execution must carry a training result")
                # Another component may have moved the head while the job ran. The
                # committer merges a dispatched result rather than refusing it: the
                # backend has published the weights already and its job can only be
                # finished, never taken back.
                self._commit_result(current.name, result, component)
                if result.training_job_id is not None:
                    backend.acknowledge_commit(current.scenario_step, result.training_job_id)
        return True

    def _record_training_error(
        self, scenario: str, value: str | None, *, source: str | None = TRAINING_THREAD_SOURCE
    ) -> None:
        """Record ``value`` for ``scenario`` under ``source``; ``None`` clears the error that source recorded.

        A local cycle that runs after the training thread's failure must not wipe that failure from the
        status; ``source=None`` clears whatever is recorded.
        """
        with self._training.lock:
            if value is None:
                for key in [key for key in self._training.errors if key[0] == scenario]:
                    if source is None or key[1] == source:
                        self._training.errors.pop(key, None)
            else:
                self.note_training_error(scenario, value, source or TRAINING_THREAD_SOURCE)

    def note_training_error(self, scenario: str, value: str, source: str) -> None:
        """Store an error and count the failure; the caller holds the training lock."""
        self._training.errors[(scenario, source)] = value
        self._training.failure_counts[scenario] = self._training.failure_counts.get(scenario, 0) + 1

    def _record_status_build_error(self, value: str | None) -> bool:
        """Record a failure to build training status; return whether it changed."""
        with self._training.lock:
            changed = value != self._training.status_build_error
            self._training.status_build_error = value
            return changed

    @staticmethod
    def _error_text(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"

    def _record_training_drain(self) -> None:
        with self._training.lock:
            self._training.last_drain = time.time()
            self._training.undrained_warned = False

    def _warn_if_undrained(self, scenario: str, last_drain: float | None) -> None:
        if last_drain is None or time.time() - last_drain < self.undrained_warning_seconds:
            return
        with self._training.lock:
            first = not self._training.undrained_warned
            self._training.undrained_warned = True
        if first:
            logger.warning(
                "scenario %r has a ready training batch undrained for %.0f seconds",
                scenario,
                time.time() - last_drain,
            )

    def _set_training_storage_status(self, value: Mapping[str, Any] | None) -> None:
        with self._training.lock:
            was_blocked = self._training.storage_status is not None
            self._training.storage_status = value
        if value is not None and not was_blocked:
            logger.warning(
                "training is blocked on checkpoint storage: %s; retrying every %.0f seconds until it clears",
                "; ".join(str(reason) for reason in value.get("reasons", ())) or "no reason reported",
                self.storage_retry_seconds,
            )
        elif value is None and was_blocked:
            logger.info("checkpoint storage block cleared; training resumes")

    def build_training_status(self) -> Mapping[str, Any]:
        """Assemble the training status block, checking training health as it goes.

        This is a method rather than a property because reading it is also the
        only health check Reef runs: it is where an undrained batch is warned
        about and where a training thread that died without recording an error
        is noticed. Both record state so a repeated read does not repeat the
        log line, so callers must expect a status read to have effects.
        """
        preload_errors = self._registry.preload_errors
        with self._training.lock:
            storage_status = self._training.storage_status
            last_drain = self._training.last_drain
        # One scenario's status failing stops the sweep: the recorded error goes
        # out with whatever was collected before it.
        scenarios: dict[str, dict[str, Any]] = {}
        failed = False
        for scenario_name in self._registry.training_status_scenario_names:
            try:
                block = self._scenario_status(scenario_name, storage_status, last_drain)
            except Exception as exc:
                failed = True
                if self._record_status_build_error(f"{scenario_name}: {self._error_text(exc)}"):
                    logger.exception("failed to build training status for scenario %r", scenario_name)
                break
            if block is not None:
                scenarios[scenario_name] = block
        if not failed:
            self._record_status_build_error(None)
        errors = self._training_errors()
        return {
            "error": "\n".join(errors) or None,
            "last_drain_at": last_drain,
            "loaded_scenarios": len(self._registry.loaded_scenarios()),
            "preload_errors": preload_errors,
            "scenarios": scenarios,
            "serving": self._serving_status(),
            "training_job": self.training_job_view(),
        }

    def training_job_view(self) -> dict[str, Any] | None:
        """The weight job the training runtime holds out, for an operator; ``None`` when none is out.

        ``owner`` names the scenario whose job it is, the one a delete refuses.
        """
        try:
            marker = self.in_flight_job_marker()
        except Exception as exc:
            return {"error": self._error_text(exc)}
        if marker is None:
            return None
        return {
            "status": marker.get("status"),
            "training_job_id": marker.get("training_job_id", marker.get("job_id")),
            "owner": marker.get("scenario"),
        }

    def _scenario_status(
        self,
        scenario_name: str,
        storage_status: Mapping[str, Any] | None,
        last_drain: float | None,
    ) -> dict[str, Any] | None:
        current = self._registry.get_optional(scenario_name)
        if current is None:
            return None
        runtime = current.runtime
        stepping_trainer = current.trainer
        batch_ready = stepping_trainer.batch_ready()
        if batch_ready:
            self._warn_if_undrained(scenario_name, last_drain)
        # The stepping trainer's status is read once and reported again in its component block.
        stepping_pipeline_status = stepping_trainer.pipeline_status()
        processor = dict(stepping_pipeline_status)
        if "buffered_requests" in processor:
            # Include instructions still unread in storage alongside the buffered ones.
            processor["pending_instructions"] = stepping_trainer.pending_instructions()
        block: dict[str, Any] = {
            **current.commit_status,
            # A version is current only after Reef commits its head
            # and reopens admission. The backend may report it
            # earlier while the update is still being published.
            "current_runtime_load_id": (runtime.current_runtime_load_id() if runtime is not None else None),
            "checkpoint_storage": storage_status,
            "batch_ready": batch_ready,
            "training_mode": stepping_trainer.pipeline.training_mode,
            "processor": processor,
            "inference_admission": runtime.inference_admission_status if runtime is not None else None,
        }
        if (
            runtime is not None
            and current.training_runtime is not None
            and current.training_runtime.concurrent_training_scenarios
        ):
            block["adapter_runtime_load_id"] = runtime.serving_adapter_runtime_load_id(scenario_name)
        if len(current.trainers) > 1:
            # Each component's trainer commits on its own; report each one beside the scenario-wide step.
            components: dict[str, Any] = {}
            for bound in current.trainers:
                last = current.last_commit_for(bound.component)
                components[str(bound.component)] = {
                    "batch_ready": bound.trainer.batch_ready(),
                    "training_mode": bound.trainer.pipeline.training_mode,
                    "processor": dict(
                        stepping_pipeline_status
                        if bound.trainer is stepping_trainer
                        else bound.trainer.pipeline_status()
                    ),
                    "stale_refusals_total": self.stale_refusals_total(scenario_name, bound.component),
                    "last_committed_step": (
                        None
                        if last is None
                        else {
                            "step": last.step,
                            "recorded_at": last.recorded_at,
                            "base_release_id": last.base_release_id,
                            "metrics": None if last.metrics is None else dict(last.metrics),
                        }
                    ),
                }
            block["components"] = components
        return block

    def _training_errors(self) -> list[str]:
        """Every recorded training error, noticing a training thread that died silently."""
        with self._training.lock:
            training_errors = dict(self._training.errors)
            status_build_error = self._training.status_build_error
            training_thread = self._training.thread
        training_scenario = self._registry.training_scenario_name or "<unbound>"
        if (
            (training_scenario, TRAINING_THREAD_SOURCE) not in training_errors
            and training_thread is not None
            and not training_thread.is_alive()
            and not self.workers.closed.is_set()
        ):
            error = "RuntimeError: training thread stopped unexpectedly"
            logger.error("%s: %s", training_scenario, error)
            self._record_training_error(training_scenario, error)
            training_errors[(training_scenario, TRAINING_THREAD_SOURCE)] = error
        errors = [f"{scenario}: {training_errors[(scenario, source)]}" for scenario, source in sorted(training_errors)]
        if status_build_error is not None:
            errors.append(status_build_error)
        return errors

    def _serving_status(self) -> dict[str, Any]:
        """Runtime-wide serving state for the deployment's recipe."""
        try:
            status = self._recipe.serving_status()
        except Exception as exc:
            status = {"error": self._error_text(exc)}
        return {} if status is None else {self._recipe.name: status}

    # -- Lifecycle -------------------------------------------------------

    def open_local_cycles(self) -> None:
        """Let local cycles run, and wake every loaded scenario's workers for the rows they hold."""
        self.workers.open()
        for current in self._registry.loaded_scenarios():
            self._wake_training(current)

    def stop_local_cycles(self) -> None:
        """Local cycles start no more, and one that ends from now on commits nothing; its rows wait for the next start.

        A stopping service takes its routes away before it closes this
        dispatcher, and a harness cycle's model calls go through them: a
        call that fails comes back as a skip, which would consume the batch.
        """
        self.workers.stop()

    def close(self) -> None:
        """Stop all scenario workers, then close this dispatcher's runtime."""
        if self.workers.closed.is_set():
            return
        local_workers = self.workers.begin_close()
        self._training.ready.set()
        if self._lifecycle.preload_thread is not None:
            self._lifecycle.preload_thread.join()
        if self._lifecycle.idle_unload_thread is not None:
            self._lifecycle.idle_unload_thread.join()
        if self._training.thread is not None:
            self._training.thread.join()
        for worker in local_workers:
            worker.thread.join()
        if self._lifecycle.metrics_thread is not None:
            self._lifecycle.metrics_thread.join()
            self.record_operational_metrics()
        errors: list[BaseException] = []
        replaced = self.workers.take_replaced()
        for scenario in (*replaced, *self._registry.loaded_scenarios()):
            # scenario.close(), not records.close(): processor teardown has to
            # precede the store closing, or a processor worker still in flight
            # observes a closed store.
            try:
                scenario.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self._storage.close()
        except BaseException as exc:
            errors.append(exc)
        if self._recipe.runtime is not None:
            self._recipe.runtime.pause_admission()
        for component in (self._recipe.training_runtime, self._recipe.runtime):
            if component is not None:
                try:
                    component.shutdown()
                except BaseException as exc:
                    errors.append(exc)
        try:
            self._experiment_tracker.close()
        except Exception:
            logger.exception("experiment tracker failed to close")
        if errors:
            raise errors[0]


def build_default_dispatcher(
    *,
    backend_factory: RepositoryBackendFactory | None = None,
    checkpoint_strategy: CheckpointStrategy | None = None,
    local_artifact_dir: Path | None = None,
    scenario_storage: ScenarioStorage,
) -> Dispatcher:
    """Build a Dispatcher serving the core record-only ``recipe``.

    Convenience for tests and service entrypoints that want a working
    dispatcher. The recipe is the same base ``Recipe`` a deployment gets
    from ``reef.recipe: recipe``.
    Uses an in-memory artifact backend when ``backend_factory`` is not
    provided. Record and commit storage must be supplied explicitly through
    ``scenario_storage``; this helper does not select a record backend.
    """
    if backend_factory is None:
        root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        initial = root / "initial"
        initial.mkdir()
        backend_factory = InMemoryRepositoryBackend.factory(initial, root=root / "repository")
    recipe = Recipe(
        checkpoint_strategy=checkpoint_strategy if checkpoint_strategy is not None else EveryNVersions(1),
    )
    return Dispatcher(
        recipe,
        backend_factory,
        local_artifact_dir=local_artifact_dir,
        scenario_storage=scenario_storage,
    )
