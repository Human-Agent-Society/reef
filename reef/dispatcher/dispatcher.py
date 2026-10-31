"""The dispatcher: the process-wide owner of every scenario and its training workers."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from reef.artifact.artifact import Artifact, ArtifactRef
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.artifact.repository import EnumerableRepositoryBackendFactory, RepositoryBackendFactory
from reef.artifact.sources import SourceFiles
from reef.core.errors import ReefError, ScenarioCapacityFull, UnknownScenario
from reef.core.job_data import TrajectorySample
from reef.core.records_types import AgentRecord, RequestType
from reef.core.served_models import CACHE_SECONDS
from reef.dispatcher import ingest
from reef.dispatcher.data_sync_worker import DataSyncWorker, ItemSync
from reef.dispatcher.errors import TrainingErrors
from reef.dispatcher.local_workers import LocalTrainingWorkers
from reef.dispatcher.samples import LocalSampleProducers
from reef.dispatcher.served import ControlInstance, ServedModelsPublisher, take_control_lease
from reef.dispatcher.status import TrainingStatus
from reef.dispatcher.store_changes import StoreChanges, StoreChangesWatcher
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.evaluation.artifact_endpoint import CandidateArtifacts
from reef.execution.workers import SERVICE_COMPONENTS, ServiceSet, ServiceWorkers
from reef.observability import ExperimentTracker, NullExperimentTracker
from reef.recipe.base import Recipe
from reef.record2sample.service import SamplesWorker, request_selection
from reef.runtime.deployment import close_runtimes
from reef.runtime.interfaces import InferenceRuntime
from reef.scenario.admission import ScenarioAdmission
from reef.scenario.committer import ComponentArtifact
from reef.scenario.registry import ScenarioLifecycle, ScenarioRegistry
from reef.scenario.releases import ServedBesideArtifact
from reef.scenario.scenario import Scenario
from reef.storage.artifact_items import request_artifact_copy
from reef.storage.control_lease import ControlLeaseState
from reef.storage.coordination import Coordination, instance_id
from reef.storage.data import DataStore, check_owner
from reef.storage.data_sync import ScenarioUses, sync_scenario_samples, uploads
from reef.storage.record_items import request_record_copy, sync_scenario_records
from reef.storage.records import RecordRetention
from reef.storage.samples import StoredSample
from reef.storage.scenario import ScenarioSettings, ScenarioStorage
from reef.train.job_queue import JobExperimentRecorder, JobQueueComponent, TrainingJobWorker
from reef.train.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclass
class BackgroundThreads:
    """The dispatcher's own threads besides the local workers."""

    preload_thread: Thread | None = None
    metrics_thread: Thread | None = None
    idle_unload_thread: Thread | None = None


class Dispatcher(ScenarioLifecycle, StoreChanges, ControlInstance, ItemSync, CandidateArtifacts, ScenarioUses):
    """Coordinate scenario creation, inference state, training, and model commits.

    A process trains the weights of one scenario on the deployment's
    training component. It trains more than one only when the serving
    runtime serves each scenario its own weights
    (``shared_by_training_scenarios``). :class:`ScenarioRegistry` refuses any
    other weight-training scenario. Each trainer that trains runs on its own
    worker thread (:class:`LocalTrainingWorkers`).
    The dispatcher owns the runtimes of its recipe and closes them after all scenarios.
    It unloads an idle scenario (see :meth:`scenario_is_idle`) after
    ``scenario_idle_unload_seconds``. It also unloads one when scenario
    admission needs a load slot.
    """

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
        record_storage: ScenarioStorage | None = None,
        data_store: DataStore | None = None,
        hold_local_cycles: bool = False,
        scenario_idle_unload_seconds: float | None = None,
        scenario_admission: ScenarioAdmission | None = None,
        coordination: Coordination,
        control_lease: ControlLeaseState | None = None,
        workers: ServiceWorkers | None = None,
        components: frozenset[str] = frozenset(SERVICE_COMPONENTS),
        work_retention_seconds: float = 7 * 24 * 3600.0,
    ) -> None:
        """``hold_local_cycles``: run no local cycle until :meth:`open_local_cycles`.
        Use it when the recipe calls this same service. The service does not
        answer while the dispatcher starts.

        ``scenario_idle_unload_seconds``: unload a scenario that is idle for
        this time. With ``None``, every scenario stays loaded.

        ``scenario_admission``: decides whether a scenario is created or
        loaded. The default admits all scenarios.

        ``record_storage``: the scenario storage that the work outside a loaded scenario uses, such as the inference
        path that appends records and the data service that copies them. It is ``scenario_storage`` without what a
        scenario's session does when it opens (the data sync); None means ``scenario_storage``.

        ``control_lease``: the control lease of this instance, when it runs control. A standby waits for it before
        it builds the dispatcher (``wait_for_control_lease``); None takes the lease now, and raises ``ReefError``
        when another instance holds it. When another instance takes the lease, the local cycles stop and
        ``lease_lost`` is set: the process that owns the dispatcher ends, so that its supervisor starts it again as a
        standby.

        ``coordination`` and ``workers``: the coordination store and the background workers of the services
        that this process runs. A process that runs every service and keeps no state across restarts, such as an
        embedded dispatcher, gives it ``reef.storage.sql_coordination.memory_coordination()``.
        The constructor starts the workers; ``close`` stops them first and closes the store last. Control deletes
        the work items that ended more than ``work_retention_seconds`` ago (``EndedWorkSweeper``).

        The recipe's training component gets its jobs through the jobs queue: the dispatcher gives the trainers a
        ``JobQueueComponent`` over it and, with ``training``, runs a ``TrainingJobWorker`` on it.

        ``components``: the services that this process runs (``SERVICE_COMPONENTS``). With ``control``, the
        constructor first waits for the control lease, while another control instance holds it; then the
        dispatcher writes ``served_models``, watches the shared store, and runs the trainers' local workers, which
        add the training jobs. With ``samples``, it runs the samples worker; without ``control``, that worker loads
        each scenario on demand and leaves the jobs to control.

        ``data_store``: the deployment's data store (RFC #1028). Control asks the data service to copy each
        scenario's new records into it (``request_record_copy``), and runs the ``data_sync`` items that the data
        service adds after its writes (``DataSyncWorker``). The dispatcher closes it before the scenario storage,
        which owns the task file store that the data store shares.
        """
        if scenario_idle_unload_seconds is not None and scenario_idle_unload_seconds <= 0:
            raise ValueError("scenario_idle_unload_seconds must be positive")
        #: The deployment's coordination store: ``served_models``, the work queues and the control lease.
        self.coordination = coordination
        #: The background workers of the services that this process runs; the dispatcher stops them first.
        self.workers = workers if workers is not None else ServiceWorkers()
        #: The services that this process runs. Control runs the trainers' local workers, ``served_models`` and
        #: the store watcher.
        self.services = ServiceSet.of(components)
        #: The work queue of the samples, jobs and evaluations queues.
        self.queue = coordination.work_queue
        # Every job component's jobs go through the jobs queue: the weight trainer's and a harness's alike. The
        # training service runs one worker for each of them.
        job_components = recipe.job_components()
        if job_components:
            if self.services.runs_training:
                for component in job_components:
                    self.workers.add(
                        TrainingJobWorker(component, self.queue, worker_id=instance_id(f"training-{component.name}"))
                    )
            recipe = recipe.with_job_components(
                {
                    component.name: JobQueueComponent(
                        component, self.queue, owns_component=not self.services.runs_training
                    )
                    for component in job_components
                }
            )
        #: The deployment's recipe, before any scenario's own model settings; its trainers submit to the jobs queue.
        self.recipe = recipe
        self.storage = scenario_storage
        self.record_storage = scenario_storage if record_storage is None else record_storage
        self.data_store = data_store
        self.record_retention_lock = Lock()
        tracker = experiment_tracker if experiment_tracker is not None else NullExperimentTracker()
        if recipe.job_components() and not isinstance(tracker, NullExperimentTracker):
            # Control's tracker also writes each training job's run into the job's item, for the job listing.
            tracker = JobExperimentRecorder(tracker, self.queue)
        self.experiment_tracker = tracker
        self.registry = ScenarioRegistry(
            recipe,
            backend_factory,
            local_artifact_dir=local_artifact_dir,
            scenario_storage=scenario_storage,
            allow_implicit_creation=allow_implicit_creation,
            experiment_tracker=self.experiment_tracker,
            admission=scenario_admission,
            lifecycle=self,
            # Only control activates a scenario's artifact on the engines when it loads the scenario.
            activates_runtime=self.services.runs_control,
        )
        self.errors = TrainingErrors()
        self.background_threads = BackgroundThreads()
        self.local_workers = LocalTrainingWorkers(self.registry, self.errors, held=hold_local_cycles)
        #: The writer of ``served_models``, when this process runs control.
        self.publisher: ServedModelsPublisher | None = None
        #: Set when another control instance took the lease (``control_lease_lost``).
        self.lease_lost = Event()
        if self.services.runs_control:
            lease = (
                take_control_lease(coordination.control_lease, instance_id("control"))
                if control_lease is None
                else control_lease
            )
            runtime = recipe.runtime
            request_timeout_seconds = 300.0 if runtime is None else runtime.inference_timeout_s
            publisher = ServedModelsPublisher(
                self.registry,
                coordination,
                self,
                lease,
                runtime=runtime,
                instance_id=lease.instance_id,
                stop_delay_seconds=CACHE_SECONDS + request_timeout_seconds,
            )
            self.workers.add(publisher)
            self.publisher = publisher
            if runtime is not None:
                # Only control changes the engines. It takes them over, and does not activate again at start-up what
                # they still serve: the runtime adopts each value that served_models names for its engine, before
                # any scenario loads.
                runtime.open()
                for served in coordination.served_models.all_served():
                    if served.url == runtime.base_url:
                        runtime.adopt(served)
            # The publisher extends the lease; it starts now, so a slow start-up does not let the lease expire.
            publisher.start()
            self.workers.add(StoreChangesWatcher(self.registry, coordination.work_queue, self))
            self.workers.add(EndedWorkSweeper(coordination.work_queue, retention_seconds=work_retention_seconds))
            if data_store is not None:
                self.workers.add(DataSyncWorker(coordination.work_queue, self, worker_id=instance_id("data-sync")))
        if self.services.runs_samples:
            self.workers.add(
                SamplesWorker(
                    self.queue,
                    LocalSampleProducers(self.registry, self.local_workers if self.services.runs_control else None),
                    worker_id=instance_id("samples"),
                )
            )
        self.training_status = TrainingStatus(recipe, self.registry, self.errors)
        if self.services.runs_control and isinstance(backend_factory, EnumerableRepositoryBackendFactory):
            self.background_threads.preload_thread = Thread(
                target=self.preload_scenarios,
                args=(backend_factory.list_registrations(),),
                name="reef-preload",
                daemon=True,
            )
            self.background_threads.preload_thread.start()
        if self.experiment_tracker.operational_metrics_enabled:
            self.background_threads.metrics_thread = Thread(
                target=self.run_operational_metrics,
                name="reef-operational-metrics",
                daemon=True,
            )
            self.background_threads.metrics_thread.start()
        if scenario_idle_unload_seconds is not None:
            self.background_threads.idle_unload_thread = Thread(
                target=self.run_idle_unload,
                args=(scenario_idle_unload_seconds,),
                name="reef-idle-unload",
                daemon=True,
            )
            self.background_threads.idle_unload_thread.start()
        self.workers.start()

    # -- Scenario API (delegates to registry) ----------------------------

    def has_scenario(self, scenario: str) -> bool:
        return self.registry.has(scenario)

    def get_or_create_scenario(
        self,
        scenario: str,
        *,
        release_id: str | None = None,
        allow_implicit_creation: bool | None = None,
    ) -> Scenario | None:
        return self.registry.get_or_create(
            scenario,
            release_id,
            allow_implicit_creation=allow_implicit_creation,
        )

    def loaded_scenario(self, scenario: str, *, release_id: str | None = None) -> Scenario | None:
        """The loaded instance of ``scenario``, or None; see :meth:`ScenarioRegistry.get_loaded`."""
        return self.registry.get_loaded(scenario, release_id)

    def configure_scenario_model(
        self, scenario: str, model: object, *, create: bool = False, release_id: str | None = None
    ) -> Scenario:
        return self.registry.configure_model(scenario, model, create=create, release_id=release_id)

    def set_training_mode(self, scenario: str, training_mode: str) -> dict[str, Any]:
        with self.registry.lock_for(scenario):
            current = self.registry.set_training_mode(scenario, training_mode)
            self.local_workers.wake(current)
            return {"scenario": scenario, "training_mode": current.training_mode}

    def list_scenarios(self) -> tuple[dict[str, Any], ...]:
        return self.registry.list()

    def prune_record_archives(self, retention: RecordRetention) -> int:
        """Apply deployment-wide retention without racing scenario file moves."""
        with self.record_retention_lock:
            return self.storage.prune(max_bytes=retention.max_bytes)

    def delete_scenario(self, scenario: str) -> dict[str, Any]:
        """Remove a scenario from this deployment and move its own state aside.

        The method holds the scenario's lock, so no accept or commit runs
        meanwhile. It does these steps:

        - Its ``served_models`` rows go, so the inference service answers it 404.
        - The workers drop the scenario, and a training job it runs is cancelled.
        - The loaded instance closes.
        - The storage archives the records, the commits and the settings.
        - The recipe's own directories move to an ``archived`` sibling.
        - The repository registration is archived, so the name is free again.

        Artifacts that other scenarios share stay.
        """
        if "/" in scenario or scenario in ("", ".", ".."):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        with self.record_retention_lock, self.registry.lock_for(scenario):
            if not self.registry.has(scenario):
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            # The inference service stops answering the scenario first, so no record reaches the archived store.
            if self.publisher is not None:
                self.publisher.withdraw(scenario)
            dropped = self.registry.remove(scenario)
            self.local_workers.remove(scenario)
            self.errors.forget(scenario)
            if dropped is not None:
                for component_trainer in dropped.trainers:
                    # Nothing would read the job's checkpoints once its scenario is gone.
                    component_trainer.trainer.cancel_job()
                dropped.close()
            self.local_workers.close_replaced_instances(scenario)
            archived = [*self.archive_scenario_state(scenario), *self.registry.archive_registration(scenario)]
        self.registry.forget_lock(scenario)
        return {"scenario": scenario, "archived": archived}

    def scenario_loaded(self, instance: Scenario) -> None:
        """The registry calls this when it loads a scenario: its weight trainers resume their jobs and records.

        The data sync can add uploads while the scenario opens, and no record wakes a worker for them, so the
        workers wake when a trainer holds untrained samples.
        """
        if not self.services.runs_control:
            # Another process runs control: this one loaded the scenario only to make its samples.
            return
        self.scenario_committed(instance)
        self.request_samples(instance)
        self.local_workers.resume(instance)
        self.request_record_copy(instance.name)
        if any(component_trainer.trainer.holds_untrained_records() for component_trainer in instance.trainers):
            self.local_workers.wake(instance)

    def scenario_committed(self, instance: Scenario) -> None:
        """Write what ``instance`` serves now to ``served_models``, in the step of the commit that changed it, and
        ask the data service to copy the scenario's artifacts to artifact items.

        The commit is durable already, so a failed write does not undo it: the publisher's next round writes the rows
        again, and the next commit asks for the copy again.
        """
        try:
            if self.publisher is not None:
                self.publisher.publish(instance)
            self.request_artifact_copy(instance.name)
        except Exception:
            logger.exception(
                "scenario %r: the commit is durable, but its served models were not written", instance.name
            )

    def scenario_reloaded(self, instance: Scenario) -> None:
        """A failure rebuilt ``instance``: ask the samples service again for the records after its progress. Its
        workers keep the failure until the samples come and the next cycle runs."""
        if self.services.runs_control:
            self.request_samples(instance)

    def close_replaced(self, instance: Scenario) -> None:
        """Close a reloaded scenario's old instance once no local cycle runs on it."""
        self.local_workers.close_replaced(instance)

    def scenario_is_idle(self, current: Scenario) -> bool:
        """Whether Reef can unload ``current`` with no loss of data and no stop of work.

        A scenario is idle when all of these are true:

        - Its storage is durable.
        - No request runs on it.
        - No trainer has records that it would still train on.
        - No trainer has a training job that runs.
        - No reload or close of an instance waits.

        A weight-training scenario is never idle, because only a delete
        releases its training component binding. The caller holds the lock and the
        cycle lock of the scenario, so no record or cycle starts meanwhile.
        """
        if current.training_component is not None or not current.store.durable:
            return False
        if current.operations.active_count() > 0:
            return False
        if self.local_workers.instance_change_pending(current.name):
            return False
        return not any(
            component_trainer.trainer.job_running or component_trainer.trainer.holds_untrained_records()
            for component_trainer in current.trainers
        )

    def unload_if_idle(self, scenario: str) -> bool:
        """Unload ``scenario`` if it is idle, and return whether it was unloaded.

        Unload closes the loaded instance and stops its local workers. It is
        a delete without the archive: the storage, the repository
        registration and the settings stay, and the next request loads the
        scenario again. Counters in memory start again from zero. If another
        thread holds the lock or the cycle lock of the scenario, the scenario
        is busy, so this method never waits for them.
        """
        with self.registry.lock_for(scenario, blocking=False) as acquired:
            if not acquired:
                return False
            cycle_lock = self.local_workers.cycle_lock(scenario)
            if not cycle_lock.acquire(blocking=False):
                return False
            try:
                current = self.registry.get_optional(scenario)
                if current is None or not self.scenario_is_idle(current):
                    return False
                self.registry.unload(scenario)
                self.local_workers.remove(scenario)
                current.close()
            finally:
                cycle_lock.release()
            self.local_workers.close_replaced_instances(scenario)
        logger.info("unloaded idle scenario %r; its next request loads it again", scenario)
        return True

    def run_idle_unload(self, idle_seconds: float) -> None:
        """Until the dispatcher closes, unload each scenario that has no use for ``idle_seconds`` and is idle."""
        while not self.local_workers.closed.wait(min(idle_seconds, self.idle_unload_check_seconds)):
            for name in self.registry.unused_since(time.monotonic() - idle_seconds):
                self.unload_if_idle(name)

    def archive_scenario_state(self, scenario: str) -> list[str]:
        """Move the scenario's own files and directories under an ``archived`` sibling, stamped so a name can be deleted twice."""
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        moved = list(self.registry.archive_store(scenario))
        for directory in self.recipe.scenario_state_dirs(scenario):
            if directory.exists():
                destination = directory.parent / "archived" / f"{directory.name}-{stamp}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(directory), str(destination))
                moved.append(str(destination))
        return moved

    def list_artifacts(self, scenario: str) -> tuple[dict[str, Any], ...]:
        """Each component's artifacts of ``scenario``, newest first, with each component's active one marked."""
        with self.registry.lock_for(scenario):
            return self.registry.require(scenario).artifacts()

    def read_records(
        self, scenario: str, *, after_sequence: int = 0, limit: int = 50, request_type: RequestType | None = None
    ) -> dict[str, Any]:
        """Read retained summaries, optionally of one request type, without changing the training queue."""
        from reef.scenario.history import read_records

        with self.registry.lock_for(scenario):
            return read_records(
                self.registry.require(scenario),
                after_sequence=after_sequence,
                limit=limit,
                request_type=request_type,
            )

    def read_record(self, scenario: str, record_id: str) -> dict[str, Any] | None:
        """Read a retained trace within its scenario, including consumed records."""
        from reef.scenario.history import read_record

        with self.registry.lock_for(scenario):
            return read_record(self.registry.require(scenario), record_id)

    def read_commits(
        self, scenario: str, *, after_step: int = 0, limit: int = 50, record_ids: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        from reef.scenario.history import read_commits

        with self.registry.lock_for(scenario):
            return read_commits(
                self.registry.require(scenario), after_step=after_step, limit=limit, record_ids=record_ids
            )

    def scenario_contract(self, scenario: str) -> dict[str, Any]:
        with self.registry.lock_for(scenario):
            current = self.registry.require(scenario)
            pipeline = current.trainer.pipeline
            return {
                "scenario": scenario,
                # The key keeps its name for callers; the value names the recipe's sample-to-job stage.
                "processor": type(pipeline.sample2job).__name__,
                "training_mode": current.trainer.pipeline.training_mode,
                "status": dict(current.trainer.pipeline_status()),
                "required_request_types": sorted(rt.value for rt in pipeline.required_request_types),
                "capabilities": current.capabilities().to_json(),
            }

    def activate(self, scenario: str, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Make ``artifact_id`` the active artifact of ``component`` in ``scenario``; no training start moves."""
        with self.registry.lock_for(scenario):
            current = self.registry.require(scenario)
            changed = current.activate(artifact_id, component)
            self.scenario_committed(current)
            return changed

    def serve(self, scenario: str, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Serve ``artifact_id`` of ``component`` beside the active release of ``scenario``; that release stays."""
        with self.registry.lock_for(scenario):
            current = self.registry.require(scenario)
            changed = current.serve(artifact_id, component)
            self.scenario_committed(current)
            return changed

    def stop_serving(self, scenario: str, artifact_id: str) -> ServedBesideArtifact:
        """Stop serving ``artifact_id`` beside the active release of ``scenario``; the active release stays."""
        with self.registry.lock_for(scenario):
            current = self.registry.require(scenario)
            changed = current.stop_serving(artifact_id)
            self.scenario_committed(current)
            return changed

    def train_from(self, scenario: str, artifact_id: str, component: str | None = None) -> ComponentArtifact:
        """Start ``component``'s trainer of ``scenario`` from ``artifact_id``; what serves stays."""
        with self.registry.lock_for(scenario):
            return self.registry.require(scenario).train_from(artifact_id, component)

    def import_target(self, scenario: str, component: str | None) -> str:
        """The component of ``scenario`` an import into ``component`` lands in; raise as ``import_artifact`` does.

        Raise ``UnknownScenario`` for an unknown scenario, and ``ValueError`` for an unknown component or one that
        takes no import. An unknown scenario is refused before its lock is made, so a request for any name adds none.
        """
        if not self.registry.has(scenario):
            raise UnknownScenario(f"unknown scenario {scenario!r}")
        with self.registry.lock_for(scenario):
            return self.registry.require(scenario).import_target(component)

    def import_artifact(
        self,
        scenario: str,
        files: SourceFiles,
        component: str | None,
        *,
        source: Mapping[str, str],
        directory: Path,
    ) -> ComponentArtifact:
        """Publish ``files`` as a new artifact of ``component`` in ``scenario``, stored in the empty ``directory``.

        Nothing serves or trains it yet; see :meth:`ScenarioCommitter.import_artifact`.
        """
        with self.registry.lock_for(scenario):
            imported = self.registry.require(scenario).import_artifact(
                files, component, source=source, directory=directory
            )
        self.request_artifact_copy(scenario)
        return imported

    def sync_release(
        self,
        scenario: str,
        artifact: Artifact,
        *,
        component: str,
        expected_release_id: str,
        metrics: Mapping[str, Any],
    ) -> ArtifactRef:
        """Make a person's merged ``component`` tree ``scenario``'s active release and training start, under the
        scenario's lock, and write what the scenario serves now; see :meth:`ScenarioCommitter.sync_release`."""
        with self.registry.lock_for(scenario):
            current = self.registry.require(scenario)
            published = current.sync_release(
                artifact, component=component, expected_release_id=expected_release_id, metrics=metrics
            )
            self.scenario_committed(current)
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
        with self.registry.lock_for(item.scenario):
            # An instruction names a scenario that exists; inference and reports keep implicit creation.
            current = self.get_or_create_scenario(
                item.scenario,
                release_id=release_id,
                allow_implicit_creation=False if item.request_type is RequestType.TRAIN else None,
            )
            if current is None:
                raise UnknownScenario(f"unknown scenario {item.scenario!r}")
            stored, inserted = ingest.append_record(current, item)
            if inserted:
                self.process_accepted_records(current)
            return stored

    def accept_records(
        self, items: Sequence[AgentRecord], *, release_id: str | None = None
    ) -> tuple[AgentRecord, ...]:
        """Validate an ordered import batch, commit it, then wake its consumer."""
        if not items:
            raise ValueError("a record batch must not be empty")
        scenario = items[0].scenario
        if any(item.scenario != scenario or item.request_type is RequestType.TRAIN for item in items):
            raise ValueError("a record batch must contain inferences or reports from one scenario")
        with self.registry.lock_for(scenario):
            current = self.get_or_create_scenario(scenario, release_id=release_id)
            if current is None:
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            stored, inserted = ingest.append_records(current, items)
            if inserted:
                self.process_accepted_records(current)
            return stored

    def scenario_owner(self, scenario: str) -> str | None:
        """The owner of ``scenario`` in the data store, or None when the scenario does not exist.

        A scenario that the data sync has not opened yet has the owner with its own name.
        """
        if not self.has_scenario(scenario):
            return None
        return self.storage.read_settings(scenario).owner or scenario

    def set_scenario_owner(self, scenario: str, owner: str) -> None:
        """Give ``scenario`` the owner ``owner``, and sync its uploads with that owner's items.

        A scenario gets its owner when it is created, so only a scenario that still has the default owner (its
        own name) and no uploads can take another. Raise ``ValueError`` otherwise, and ``UnknownScenario`` for a
        scenario that does not exist.
        """
        check_owner(owner)
        with self.locked_scenario(scenario) as current:
            settings = self.storage.read_settings(scenario)
            if settings.owner == owner:
                return
            if settings.owner != scenario or next(uploads(current.store.samples), None) is not None:
                raise ValueError(f"scenario {scenario!r} has the owner {settings.owner!r}; it is set at creation")
            self.storage.write_settings(scenario, replace(settings, owner=owner))
            self.apply_sync(current, owner)

    def check_use(
        self, scenario: str, trajectories: Sequence[tuple[str, TrajectorySample]], train_tasks: bool
    ) -> None:
        """Raise ``ValueError`` unless ``scenario`` can train on the items that it is to use (``ingest.check_use``)."""
        with self.locked_scenario(scenario) as current:
            ingest.check_use(current, trajectories, train_tasks)

    def sync_items(self, owner: str, scenarios: Iterable[str]) -> None:
        """Sync the uploads of each loaded scenario of ``owner`` among ``scenarios`` with that owner's items.

        A scenario that is not loaded syncs when it opens. A scenario of another owner does not use the items.
        """
        if self.data_store is None:
            return
        for scenario in sorted(set(scenarios)):
            with self.registry.lock_for(scenario):
                current = self.registry.get_loaded(scenario)
                if current is not None and self.storage.read_settings(scenario).owner == owner:
                    self.apply_sync(current, owner)

    def apply_sync(self, current: Scenario, owner: str) -> None:
        """Sync the uploads and record copies of the loaded scenario ``current`` with the items of ``owner``.

        Its trainers forget the uploads that left it, and its workers wake for new uploads and copies. The caller
        holds the scenario's lock.
        """
        if self.data_store is None:
            raise ReefError("this dispatcher has no data store")
        result = sync_scenario_samples(current.store.samples, self.data_store, owner, current.name)
        for component_trainer in current.trainers:
            component_trainer.trainer.forget_samples(result.removed)
        if sync_scenario_records(current.store.records, self.data_store, owner, current.name):
            self.process_accepted_records(current)
        elif result.added:
            self.local_workers.wake(current)

    def sample_buffer(self, scenario: str, component: str | None = None) -> Mapping[str, Any]:
        """The buffer of the trainer of ``component`` (the only one when None): its mode and its samples."""
        with self.locked_scenario(scenario) as current:
            return self.trainer_of(current, component).buffer_listing()

    def change_buffer(
        self, scenario: str, sample_ids: tuple[str, ...], *, insert: bool, component: str | None = None
    ) -> Mapping[str, Any]:
        """Insert ``sample_ids`` into the manual buffer of the trainer of ``component``, or remove them.

        Raise ``ManualBufferRefused`` in automatic mode and ``ValueError`` for an ID that the trainer cannot take.
        """
        with self.locked_scenario(scenario) as current:
            listing = self.trainer_of(current, component).change_buffer(sample_ids, insert=insert)
            # A queued instruction may now have samples to train.
            self.local_workers.wake(current)
            return listing

    def stored_samples(self, scenario: str, *, after_sequence: int, limit: int) -> tuple[StoredSample, ...]:
        """The scenario's stored samples after ``after_sequence``, at most ``limit``, in arrival order."""
        with self.locked_scenario(scenario) as current:
            return current.store.samples.page(after_sequence=after_sequence, limit=limit)

    @staticmethod
    def trainer_of(current: Scenario, component: str | None) -> Trainer:
        """The trainer of ``component`` in ``current``; ``ValueError`` for None in a scenario of several."""
        if component is None and len(current.trainers) > 1:
            raise ValueError(f"scenario {current.name!r} has several components: name one with component")
        return current.trainer_for(component)

    @contextmanager
    def locked_scenario(self, scenario: str) -> Iterator[Scenario]:
        """The loaded scenario ``scenario``, loaded again when it was unloaded, held under its lock.

        Raise ``UnknownScenario`` when the scenario does not exist. The caller must not keep the scenario, or its
        storage session, after the block: an unload closes the session.
        """
        with self.registry.lock_for(scenario):
            current = self.get_or_create_scenario(scenario, allow_implicit_creation=False)
            if current is None:
                raise UnknownScenario(f"unknown scenario {scenario!r}")
            yield current

    def set_evaluation_and_activation(
        self, scenario: str, *, artifact_evaluation: Mapping[str, object] | None, activation: object
    ) -> Mapping[str, object] | None:
        """Store how Reef evaluates and activates each new artifact of ``scenario``; see
        :meth:`ScenarioRegistry.set_evaluation_and_activation`."""
        return self.registry.set_evaluation_and_activation(
            scenario, artifact_evaluation=artifact_evaluation, activation=activation
        )

    def stored_settings(self, scenario: str) -> ScenarioSettings:
        """The settings stored for ``scenario``; defaults when none are stored."""
        return self.registry.stored_settings(scenario)

    def scenario_settings(self, scenario: str) -> dict[str, Any]:
        """The settings of an existing scenario as JSON, for ``GET /reef/scenarios/{scenario}/settings``.

        ``training_mode`` and ``model`` are what the loaded scenario runs. ``artifact_evaluation`` and
        ``activation`` are the stored settings, or None. ``effective_activation`` is the rule that applies: the
        scenario's ``activation``, else the deployment's. Raise ``UnknownScenario`` for an unknown scenario.
        """
        with self.locked_scenario(scenario) as current:
            settings = self.registry.stored_settings(scenario)
            return {
                "scenario": scenario,
                "training_mode": current.training_mode,
                "model": current.model_config.view(),
                "artifact_evaluation": settings.artifact_evaluation,
                "activation": settings.activation,
                "effective_activation": self.registry.effective_activation(settings),
            }

    def imported_tasks(self, scenario: str) -> tuple[StoredSample, ...]:
        """The scenario's imported tasks in the order they arrived, each with its train counts."""
        with self.locked_scenario(scenario) as current:
            return current.store.samples.imported_tasks()

    def process_accepted_records(self, current: Scenario) -> None:
        """Start consumption only after the accepted records are durable, and copy them to record items."""
        self.request_record_copy(current.name)
        # Every component's trainer sees the record. The samples service makes the samples of a trainer that
        # trains, and its worker builds jobs from them. A trainer that trains nothing only advances its record
        # cursor, here.
        self.request_samples(current)
        self.local_workers.wake(current)
        for component_trainer in current.trainers:
            if not component_trainer.trainer.trains:
                current.prepare_training_step(component_trainer.component)

    def control_lease_lost(self) -> None:
        """Another control instance took the lease: no local cycle of this instance runs or commits any more, and
        ``lease_lost`` is set for the process that owns the dispatcher."""
        self.local_workers.stop()
        self.lease_lost.set()

    def candidate(self, scenario: str, artifact_id: str, component: str | None) -> tuple[InferenceRuntime, Artifact]:
        """The runtime of ``scenario`` and the materialized artifact ``artifact_id`` (``CandidateArtifacts``)."""
        with self.locked_scenario(scenario) as current:
            runtime = current.runtime
            if runtime is None:
                raise ReefError(f"scenario {scenario!r} has no runtime")
            _, release = current.resolve_artifact(artifact_id, component)
            return runtime, release.materialize()

    def request_artifact_copy(self, scenario: str) -> None:
        """Ask the data service to copy the artifacts of ``scenario`` to artifact items, when there is a data store."""
        if self.data_store is not None:
            request_artifact_copy(self.queue, scenario)

    def request_record_copy(self, scenario: str) -> None:
        """Ask the data service to copy the new records of ``scenario`` to record items, when there is a data store."""
        if self.data_store is not None:
            request_record_copy(self.queue, scenario)

    def records_added(self, current: Scenario) -> None:
        """Another process appended records to ``current``: ask for their samples and copy them to record items."""
        self.request_record_copy(current.name)
        self.request_samples(current)

    def samples_added(self, scenario: str) -> None:
        """The samples service stored samples of ``scenario``: its local workers build the next job from them."""
        current = self.registry.get_loaded(scenario)
        if current is not None:
            self.local_workers.wake(current)

    def job_changed(self, scenario: str) -> None:
        """A training job of ``scenario`` saved a checkpoint or ended: its local workers commit it now."""
        current = self.registry.get_loaded(scenario)
        if current is not None:
            self.local_workers.wake(current)

    def request_samples(self, current: Scenario) -> None:
        """Ask the samples service for a selection of the new records of each trainer of ``current`` that trains."""
        for component_trainer in current.trainers:
            if component_trainer.trainer.trains:
                request_selection(self.queue, current.name, component_trainer.component)

    # -- Preload ---------------------------------------------------------

    def preload_scenarios(self, scenarios: tuple[str, ...]) -> None:
        # Preload failures go to their own training-status field. A scenario
        # that does not load at start is a different failure from a training
        # step that does not commit.
        for scenario in scenarios:
            try:
                current = self.get_or_create_scenario(scenario)
                if current is not None:
                    # Rows that the last run did not read train now, before the next record arrives.
                    self.local_workers.wake(current)
            except ScenarioCapacityFull:
                # This is not a preload failure. The scenario loads on its first request, when a load slot is free.
                logger.info("scenario %r is not preloaded: the service holds its limit of loaded scenarios", scenario)
            except Exception as exc:
                logger.exception("failed to preload scenario %r", scenario)
                self.registry.record_preload_error(scenario, f"{type(exc).__name__}: {exc}")

    # -- Background workers ---------------------------------------------

    def run_operational_metrics(self) -> None:
        """Sample while training is idle, blocked, or failing, without status polling."""
        while not self.local_workers.closed.wait(self.operational_metrics_interval_seconds):
            self.record_operational_metrics()

    def record_operational_metrics(self) -> None:
        """Publish numeric state only; one unavailable scenario must not hide others."""
        for current in self.registry.loaded_scenarios():
            self.record_scenario_operational_metrics(current)

    def record_scenario_operational_metrics(self, current: Scenario) -> None:
        """Keep recipe, storage, and provider sampling failures local to one scenario."""
        try:
            metrics = current.trainer.operational_metrics()
            metrics.update(current.operations.snapshot())
            metrics["training/failed_attempts_total"] = self.errors.failure_count(current.name)
            metrics["training/error"] = int(self.errors.has_error(current.name))
            current.trainer.pipeline.experiment_logger.log(metrics, namespace="operations")
        except Exception as exc:
            logger.warning("operational metrics unavailable for scenario %r (%s)", current.name, type(exc).__name__)

    def build_training_status(self) -> Mapping[str, Any]:
        """The training status block; reading it also checks training health (see :class:`TrainingStatus`)."""
        return self.training_status.build()

    # -- Lifecycle -------------------------------------------------------

    def open_local_cycles(self) -> None:
        """Let local cycles run, and wake every loaded scenario's workers for the rows they hold."""
        self.local_workers.open()
        for current in self.registry.loaded_scenarios():
            self.local_workers.wake(current)

    def stop_local_cycles(self) -> None:
        """Start no more local cycles, and commit nothing from a cycle that ends from now on.

        The rows of such a cycle wait for the next start. A stopping service
        removes its routes before it closes this dispatcher. A harness
        cycle's model calls use those routes. A call that fails comes back as
        a skip, and a skip would consume the job data.
        """
        self.local_workers.stop()

    def close(self) -> None:
        """Stop all scenario workers, then close this dispatcher's runtime."""
        if self.local_workers.closed.is_set():
            return
        local_workers = self.local_workers.begin_close()
        if self.background_threads.preload_thread is not None:
            self.background_threads.preload_thread.join()
        if self.background_threads.idle_unload_thread is not None:
            self.background_threads.idle_unload_thread.join()
        for worker in local_workers:
            worker.thread.join()
        # The service workers stop after the local cycles, so that the publisher releases the control lease only
        # when no cycle of this instance can commit any more.
        self.workers.stop()
        if self.background_threads.metrics_thread is not None:
            self.background_threads.metrics_thread.join()
            self.record_operational_metrics()
        errors: list[BaseException] = []
        replaced = self.local_workers.take_replaced()
        for scenario in (*replaced, *self.registry.loaded_scenarios()):
            # Use scenario.close(), not records.close(). The processors must
            # stop before the store closes, or a processor worker in flight
            # finds a closed store.
            try:
                scenario.close()
            except BaseException as exc:
                errors.append(exc)
        if self.data_store is not None:
            try:
                self.data_store.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self.storage.close()
        except BaseException as exc:
            errors.append(exc)
        errors.extend(close_runtimes(self.recipe.job_components(), self.recipe.runtime))
        # Send the experiment events that are still queued before the tracker closes.
        self.registry.experiment_reports.close()
        try:
            self.experiment_tracker.close()
        except Exception:
            logger.exception("experiment tracker failed to close")
        try:
            self.coordination.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]


def build_default_dispatcher(
    *,
    backend_factory: RepositoryBackendFactory | None = None,
    local_artifact_dir: Path | None = None,
    scenario_storage: ScenarioStorage,
    coordination: Coordination,
) -> Dispatcher:
    """Build a Dispatcher serving the core record-only ``recipe``.

    Convenience for tests and service entrypoints that want a working
    dispatcher. The recipe is the same base ``Recipe`` a deployment gets
    from ``reef.recipe: recipe``.
    Uses an in-memory artifact backend when ``backend_factory`` is not
    provided. Record and commit storage must be supplied explicitly through
    ``scenario_storage``, and the coordination store through ``coordination``; this helper selects neither.
    """
    if backend_factory is None:
        root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
        initial = root / "initial"
        initial.mkdir()
        backend_factory = InMemoryRepositoryBackend.factory(initial, root=root / "repository")
    return Dispatcher(
        Recipe(),
        backend_factory,
        local_artifact_dir=local_artifact_dir,
        scenario_storage=scenario_storage,
        coordination=coordination,
    )
