"""Process-wide coordinator that owns every scenario and its serving state.

The dispatcher delegates scenario table management (creation, resolution,
per-scenario locking) to :class:`ScenarioRegistry`, then layers record
acceptance, background training drains, and publication on top. Request
handling lives in ``reef.service``; the dispatcher is transport-free.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackendFactory
from reef.core.errors import UnknownScenario
from reef.core.records_types import AgentRecord, RequestType
from reef.observability import (
    ExperimentTracker,
    NullExperimentTracker,
    RollbackExperimentEvent,
    TrainingExperimentContext,
    TrainingExperimentEvent,
)
from reef.recipe.base import Recipe
from reef.runtime.base import RuntimeContractError, TrainingRuntime
from reef.scenario.checkpoint_strategy import CheckpointStrategy, EveryNVersions
from reef.scenario.registry import ScenarioRegistry
from reef.scenario.scenario import Scenario
from reef.train.types import TrainStepResult

logger = logging.getLogger(__name__)

_STORAGE_RETRY_SECONDS = 5.0
# Poll cadence while a processor reports asynchronous derivation in flight
# (see DataProcessor.derivation_pending): its judgments land without a new
# record ever setting the ready event, so readiness is re-checked on a
# bounded interval instead of sleeping until the next accept.
_DERIVATION_POLL_SECONDS = 1.0
# One drain, plus one more after reloading the scenario from durable state.
_DRAIN_ATTEMPTS = 2
# A ready batch should be reserved by the next drain; one that sits longer
# means the training thread is not waking. Status reads perform the check,
# so the alarm rides the health polling that already watches the service.
_UNDRAINED_WARNING_SECONDS = 60.0


@dataclass(frozen=True)
class _LocalBackendWorkerState:
    ready: Event
    thread: Thread


@dataclass
class _PublicationState:
    lock: Lock = field(default_factory=Lock)
    values: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Mapping[str, Any]:
        with self.lock:
            return dict(self.values)

    def record(self, scenario: str, value: Any) -> None:
        with self.lock:
            self.values[scenario] = value


class _ScenarioTrainingError(Exception):
    """One scenario's training turn failed; the others keep their state."""

    def __init__(self, scenario: str, cause: Exception) -> None:
        super().__init__(f"{scenario}: {cause}")
        self.scenario = scenario
        self.cause = cause


@dataclass
class _TrainingState:
    lock: Lock = field(default_factory=Lock)
    ready: Event = field(default_factory=Event)
    errors: dict[str, str] = field(default_factory=dict)
    status_build_error: str | None = None
    storage_status: Mapping[str, Any] | None = None
    last_drain: float | None = None
    undrained_warned: bool = False
    thread: Thread | None = None
    local_workers: dict[str, _LocalBackendWorkerState] = field(default_factory=dict)


@dataclass
class _LifecycleState:
    closed: Event = field(default_factory=Event)
    preload_thread: Thread | None = None


class Dispatcher:
    """Coordinate scenario creation, inference state, training, and model commits.

    Invariant: at most one weight-training scenario per process. Its serial
    thread is bound to the first scenario using the deployment's
    ``TrainingRuntime``; resolving a second raises (enforced in
    :class:`ScenarioRegistry`).
    Local training backends are unlimited and drain on per-scenario threads.
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
        self._recipe = recipe
        self._experiment_tracker = experiment_tracker if experiment_tracker is not None else NullExperimentTracker()
        self._registry = ScenarioRegistry(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            agent_record_dir=agent_record_dir,
            allow_implicit_creation=allow_implicit_creation,
            experiment_tracker=self._experiment_tracker,
        )
        self._registry.set_training_scenario_callback(self._start_training)
        self._publication = _PublicationState()
        self._training = _TrainingState()
        self._lifecycle = _LifecycleState()
        if isinstance(backend_factory, EnumerableRepositoryBackendFactory):
            self._lifecycle.preload_thread = Thread(
                target=self._preload_scenarios,
                args=(backend_factory.list_registrations(),),
                name="reef-preload",
                daemon=True,
            )
            self._lifecycle.preload_thread.start()

    @property
    def published(self) -> Mapping[str, Any]:
        return self._publication.snapshot()

    # -- Scenario API (delegates to registry) ----------------------------

    def has_scenario(self, scenario: str) -> bool:
        return self._registry.has(scenario)

    def has_loaded(self, scenario: str) -> bool:
        return self._registry.has_loaded(scenario)

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

    def list_scenarios(self) -> tuple[dict[str, Any], ...]:
        return self._registry.list()

    def recipe_has_files(self) -> bool:
        return self._registry.recipe_has_files()

    def list_releases(self, scenario: str) -> tuple[dict[str, Any], ...]:
        with self._registry.lock_for(scenario):
            return self._registry.require(scenario).releases()

    def scenario_contract(self, scenario: str) -> dict[str, Any]:
        with self._registry.lock_for(scenario):
            current = self._registry.require(scenario)
            processor = current.trainer.processor
            return {
                "scenario": scenario,
                "processor": type(processor).__name__,
                "required_request_types": sorted(rt.value for rt in processor.required_request_types),
            }

    def promote(self, scenario: str, release_id: str) -> ArtifactRef:
        """Serve a pending release: the rollback path under its own record operation."""
        return self.rollback(scenario, release_id, operation="promote")

    def rollback(self, scenario: str, release_id: str, *, operation: str = "rollback") -> ArtifactRef:
        with self._registry.lock_for(scenario):
            current = self._registry.require(scenario)
            source = current.current_artifact_ref()
            context = self._experiment_context(current)
            published = current.rollback(release_id, operation=operation)
            if published == source:
                return published
            event = RollbackExperimentEvent(
                scenario=scenario,
                recipe=self._recipe.name,
                step=current.scenario_step,
                run_segment=context.run_segment,
                source_artifact_ref=source,
                produced_artifact_ref=published,
                target_release_id=release_id,
            )
        try:
            self._experiment_tracker.record_rollback(event)
        except Exception:
            logger.exception("experiment tracker failed to record rollback")
        return published

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
            current = self.get_or_create_scenario(item.scenario, release_id=release_id)
            if current is None:
                raise UnknownScenario(f"unknown scenario {item.scenario!r}")
            return self._accept_record(current, item)

    def _accept_record(self, current: Scenario, item: AgentRecord) -> AgentRecord:
        # Schema enforcement: reject a malformed report before it is durably
        # appended, so the producer's POST fails with the violation naming
        # the broken field instead of the record dying silently at training
        # time. An undeclared schema keeps open ingress.
        if item.request_type is RequestType.REPORT and (report_type := current.report_type) is not None:
            report_type.from_dict(item.payload)
        appended = current.records.append_result(item)
        stored = appended.item
        if not appended.inserted:
            return stored
        if isinstance(current.runtime, TrainingRuntime):
            self._training.ready.set()
            return stored
        if current.trainer.training_backend is not None:
            self._start_local_backend_worker(current.name)
            return stored
        result = current.prepare_training_step()
        if result is not None:
            # scenario.commit is the single commit point: it commits the
            # trainer, appends the durable commit record, applies record
            # compaction, and moves the serving head — in that order, so a
            # crash in any gap is recovered by replaying the scenario's
            # commit log instead of silently losing the batch.
            self._commit_result(current.name, result)
        return stored

    # -- Commit & publication --------------------------------------------

    def _commit_result(self, scenario: str, result: TrainStepResult) -> None:
        current = self._registry.get(scenario)
        context = self._experiment_context(current)
        tracked_result = result
        try:
            correlation = dict(self._experiment_tracker.correlation_metrics(context))
            if correlation:
                tracked_result = current.trainer.add_commit_metrics(result, correlation)
        except Exception:
            logger.exception("experiment tracker failed to prepare correlation metadata")

        value = current.commit(tracked_result)
        self._publication.record(scenario, value)
        try:
            self._experiment_tracker.record(
                TrainingExperimentEvent(
                    context=context,
                    produced_artifact_ref=current.current_artifact_ref(),
                    metrics=dict(tracked_result.metrics),
                    outcome="rejected" if tracked_result.metrics.get("selected") is False else "committed",
                    training_job_id=tracked_result.training_job_id,
                    source_runtime_load_id=tracked_result.source_runtime_load_id,
                    produced_runtime_load_id=tracked_result.runtime_load_id,
                    checkpoint_path=tracked_result.checkpoint_path,
                )
            )
        except Exception:
            logger.exception("experiment tracker failed to record committed training step")

    def _experiment_context(self, current: Scenario) -> TrainingExperimentContext:
        backend = current.trainer.training_backend
        try:
            backend_config = None if backend is None else dict(backend.experiment_config())
        except Exception:
            logger.exception("training backend failed to describe experiment configuration")
            backend_config = None
        run_segment, run_step = (0, 0) if current.commit_log is None else current.commit_log.training_run_position()
        return TrainingExperimentContext(
            scenario=current.name,
            recipe=self._recipe.name,
            step=current.scenario_step + 1,
            source_artifact_ref=current.current_artifact_ref(),
            run_segment=run_segment,
            run_step=run_step,
            backend=None if backend is None else type(backend).__name__,
            backend_config=backend_config,
        )

    # -- Preload ---------------------------------------------------------

    def _preload_scenarios(self, scenarios: tuple[str, ...]) -> None:
        # Preload failures land in their own training_status field: a scenario
        # that cannot be re-loaded at boot is a different failure domain from
        # a training step that cannot commit.
        for scenario in scenarios:
            try:
                self.get_or_create_scenario(scenario)
            except Exception as exc:  # noqa: PERF203
                logger.exception("failed to preload scenario %r", scenario)
                self._registry.record_preload_error(scenario, f"{type(exc).__name__}: {exc}")

    # -- Background workers ---------------------------------------------

    def _start_training(self, scenario: Scenario) -> None:
        """Start the training drain thread if it hasn't been started yet.

        Called by the registry whenever it resolves a training scenario — the
        single notification point, so every resolution path (get_or_create,
        require, preload) triggers it without the dispatcher remembering to
        call from each entry point.
        """
        if self._lifecycle.closed.is_set():
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

    def _start_local_backend_worker(self, scenario: str) -> None:
        if self._lifecycle.closed.is_set():
            return
        with self._training.lock:
            worker = self._training.local_workers.get(scenario)
            if worker is None:
                ready = Event()
                thread = Thread(
                    target=self._run_local_backend_worker,
                    args=(scenario, ready),
                    name=f"reef-local-backend-{scenario}",
                    daemon=True,
                )
                worker = _LocalBackendWorkerState(ready=ready, thread=thread)
                self._training.local_workers[scenario] = worker
                thread.start()
        worker.ready.set()

    def _run_local_backend_worker(self, scenario: str, ready: Event) -> None:
        try:
            while True:
                ready.wait()
                ready.clear()
                if self._lifecycle.closed.is_set():
                    return
                self._drain_local_backend(scenario)
        except Exception as exc:
            logger.exception("local backend worker stopped unexpectedly for scenario %r", scenario)
            self._record_training_error(scenario, self._error_text(exc))

    def _drain_local_backend(self, scenario: str) -> None:
        try:
            while self._process_local_backend_step(scenario):
                pass
        except Exception as exc:
            logger.exception("local backend failed to commit for scenario %r", scenario)
            self._record_training_error(scenario, self._error_text(exc))

    def _reload_durable_local_scenario(self, scenario: str, current: Scenario) -> None:
        if current.commit_log is None:
            return
        with self._registry.lock_for(scenario):
            if self._registry.get_optional(scenario) is current:
                self._registry.reload(scenario)

    def _process_local_backend_step(self, scenario: str) -> bool:
        current = self._registry.get_optional(scenario)
        if current is None:
            raise RuntimeContractError(f"local backend scenario {scenario!r} is not loaded")
        if current.trainer.training_backend is None:
            raise RuntimeContractError(f"scenario {scenario!r} has no local backend")
        self._record_training_error(scenario, None)
        try:
            result = current.prepare_training_step()
        except Exception:
            # Durable records let recovery reconstruct the reserved batch. A
            # logless deployment must retain its in-memory pending batch for a
            # later wake instead.
            self._reload_durable_local_scenario(scenario, current)
            raise
        if result is None:
            return False
        # Keep only the short commit and recovery window under the scenario
        # registry lock. Candidate generation above can take minutes and must
        # not block record acceptance for this scenario.
        with self._registry.lock_for(scenario):
            if self._registry.get_optional(scenario) is not current:
                raise RuntimeContractError(f"local backend scenario {scenario!r} changed before commit")
            try:
                self._commit_result(scenario, result)
            except Exception:
                # A record may already have crossed the fsync commit point.
                # Reload before rollback or acceptance can observe the stale
                # in-memory step and append the same step number again.
                self._reload_durable_local_scenario(scenario, current)
                raise
        return True

    def _run_training(self) -> None:
        # Keep prepare, remote execution, and the trainer/version-chain commit
        # in one serial thread. Accepts may signal readiness while a commit is
        # slow; serial draining prevents two wake-ups from observing and
        # committing the same pending result as separate steps.
        try:
            while True:
                self._training.ready.wait(self._training_wait_timeout())
                self._training.ready.clear()
                if self._lifecycle.closed.is_set():
                    return
                self._drain_training()
                self._record_training_drain()
        except Exception as exc:
            name = self._registry.training_scenario_name or "<unbound>"
            logger.exception("training thread stopped unexpectedly for scenario %r", name)
            self._record_training_error(name, self._error_text(exc))

    def _training_wait_timeout(self) -> float | None:
        """How long the training thread may sleep before re-checking.

        Blocked checkpoint storage retries on its own cadence. A processor
        with asynchronous derivation in flight can become ready without a
        new record ever setting the event, so it is polled on a bounded
        interval. Otherwise sleep until the next accept.
        """
        timeout: float | None = _STORAGE_RETRY_SECONDS if self._training.storage_status is not None else None
        for name in self._training_scenario_names():
            current = self._registry.get_optional(name)
            if current is not None and current.trainer.processor.derivation_pending():
                timeout = _DERIVATION_POLL_SECONDS if timeout is None else min(timeout, _DERIVATION_POLL_SECONDS)
        return timeout

    def _drain_training(self) -> None:
        # One recovery attempt, not a retry loop: if draining fails we reload
        # the scenario from durable state and drain once more, so a crash
        # between a commit record and its compaction is healed on the spot. A
        # second failure still reloads (leaving a clean scenario for the next
        # wake-up) but is not spun on; the cause is reported through
        # training_status.
        for _ in range(_DRAIN_ATTEMPTS):
            try:
                while self._process_training():
                    pass
                return
            except _ScenarioTrainingError as failure:  # noqa: PERF203
                name = failure.scenario
                logger.exception("training thread failed to commit for scenario %r", name)
                self._record_training_error(name, self._error_text(failure.cause))
                self._registry.reload(name)
            except Exception as exc:
                name = self._registry.training_scenario_name or "<unbound>"
                logger.exception("training thread failed to commit for scenario %r", name)
                self._record_training_error(name, self._error_text(exc))
                self._registry.reload(name)

    def _training_scenario_names(self) -> tuple[str, ...]:
        names = getattr(self._registry, "training_scenario_names", None)
        if names is not None:
            return tuple(names)
        name = self._registry.training_scenario_name
        return () if name is None else (name,)

    def _process_training(self) -> bool:
        """Give every training scenario one turn; True when any of them progressed.

        Several scenarios share one training runtime only when it time-slices
        its adapter slot between them; the round-robin keeps the schedule
        deterministic and fair, and a failure in one scenario's turn reloads
        that scenario alone.
        """
        names = self._training_scenario_names()
        if not names:
            raise RuntimeContractError("training thread is not bound to a scenario")
        progressed = False
        for name in names:
            try:
                progressed = self._process_training_scenario(name) or progressed
            except Exception as exc:  # noqa: PERF203
                raise _ScenarioTrainingError(name, exc) from exc
        return progressed

    def _process_training_scenario(self, name: str) -> bool:
        current = self._registry.get_optional(name)
        if current is None:
            raise RuntimeContractError(f"training thread is not bound to scenario {name!r}")
        runtime = current.runtime
        if not isinstance(runtime, TrainingRuntime):
            raise RuntimeContractError(
                f"training thread requires a TrainingRuntime for scenario {current.name!r}, got {type(runtime).__name__}"
            )
        self._record_training_error(current.name, None)
        # A crash may leave remote serving updated but paused after Reef's
        # commit, or checkpointed before the weight update. Recover that
        # pending step before deciding whether another batch is available.
        backend = current.trainer.training_backend
        if backend is None or not backend.dispatched:
            raise RuntimeContractError(f"training scenario {current.name!r} has no dispatched training backend")
        backend.recover_pending_step(
            current.scenario_step,
            committed_training_job_id=current.committed_training_job_id,
            committed_training_without_job_id=current.committed_training_without_job_id,
        )
        if (batch := current.reserve_training_batch()) is None:
            return False
        execution = current.execute_reserved_training_step()
        if execution.outcome == "retry":
            if execution.storage is None:
                raise RuntimeContractError("retry execution must carry storage status")
            self._set_training_storage_status(dict(execution.storage))
            return False
        self._set_training_storage_status(None)
        if execution.outcome == "drop":
            logger.warning(
                "dropping stale training batch %r for scenario %r",
                batch.batch_id,
                current.name,
            )
            current.reject_pending(execution.metrics)
            return True
        result = execution.result
        if execution.outcome != "commit" or result is None:
            raise RuntimeContractError(f"training backend returned unsupported outcome: {execution.outcome!r}")
        self._commit_result(current.name, result)
        if result.training_job_id is not None:
            backend.acknowledge_commit(current.scenario_step, result.training_job_id)
        return True

    def _record_training_error(self, scenario: str, value: str | None) -> None:
        with self._training.lock:
            if value is None:
                self._training.errors.pop(scenario, None)
            else:
                self._training.errors[scenario] = value

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
        if last_drain is None or time.time() - last_drain < _UNDRAINED_WARNING_SECONDS:
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
            self._training.storage_status = value

    @property
    def storage_status(self) -> Mapping[str, Any] | None:
        """The training thread's last checkpoint-storage block, if any."""
        with self._training.lock:
            return self._training.storage_status

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
            "preload_errors": preload_errors,
            "scenarios": scenarios,
            "serving": self._serving_status(),
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
        batch_ready = current.trainer.batch_ready()
        if batch_ready:
            self._warn_if_undrained(scenario_name, last_drain)
        block: dict[str, Any] = {
            **current.commit_status,
            # A version is current only after Reef commits its head
            # and reopens admission. The backend may report it
            # earlier while the update is still being published.
            "current_runtime_load_id": (
                runtime.current_runtime_load_id() if isinstance(runtime, TrainingRuntime) else None
            ),
            "checkpoint_storage": storage_status,
            "batch_ready": batch_ready,
            "processor": current.trainer.processor_status(),
            "inference_admission": runtime.inference_admission_status if runtime is not None else None,
        }
        if isinstance(runtime, TrainingRuntime) and runtime.concurrent_training_scenarios:
            block["adapter_runtime_load_id"] = runtime.serving_adapter_runtime_load_id(scenario_name)
        return block

    def _training_errors(self) -> list[str]:
        """Every recorded training error, noticing a training thread that died silently."""
        with self._training.lock:
            training_errors = dict(self._training.errors)
            status_build_error = self._training.status_build_error
            training_thread = self._training.thread
        training_scenario = self._registry.training_scenario_name or "<unbound>"
        if (
            training_scenario not in training_errors
            and training_thread is not None
            and not training_thread.is_alive()
            and not self._lifecycle.closed.is_set()
        ):
            error = "RuntimeError: training thread stopped unexpectedly"
            logger.error("%s: %s", training_scenario, error)
            self._record_training_error(training_scenario, error)
            training_errors[training_scenario] = error
        errors = [f"{key}: {training_errors[key]}" for key in sorted(training_errors)]
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

    def close(self) -> None:
        if self._lifecycle.closed.is_set():
            return
        self._lifecycle.closed.set()
        self._training.ready.set()
        with self._training.lock:
            local_workers = tuple(self._training.local_workers.values())
        for worker in local_workers:
            worker.ready.set()
        if self._lifecycle.preload_thread is not None:
            self._lifecycle.preload_thread.join()
        if self._training.thread is not None:
            self._training.thread.join()
        for worker in local_workers:
            worker.thread.join()
        for scenario in self._registry.close_all():
            # scenario.close(), not records.close(): processor teardown has to
            # precede the store closing, or a processor worker still in flight
            # observes a closed store.
            scenario.close()
        try:
            self._experiment_tracker.close()
        except Exception:
            logger.exception("experiment tracker failed to close")


def build_default_dispatcher(
    *,
    backend_factory: RepositoryBackendFactory | None = None,
    checkpoint_strategy: CheckpointStrategy | None = None,
    local_artifact_dir: Path | None = None,
    agent_record_dir: Path | None = None,
) -> Dispatcher:
    """Build a Dispatcher serving the core record-only ``recipe``.

    Convenience for tests and service entrypoints that want a working
    dispatcher. The recipe is the same base ``Recipe`` a deployment gets
    from ``reef.recipe: recipe``.
    Uses an in-memory artifact backend when ``backend_factory`` is not
    provided.
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
        agent_record_dir=agent_record_dir,
    )
