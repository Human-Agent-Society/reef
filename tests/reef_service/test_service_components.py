"""``reef serve --component``: each process runs the workers of its own services on the shared store."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support._trajectories import policy_trajectory
from tests.support.artifact_versions import ImportingComponent
from tests.support.harness_steps import IdleMethod, harness_recipe
from tests.support.runtime_stubs import StubTrainingComponent, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.sources import ArtifactSources, DirectorySource
from reef.core.job_data import JobData
from reef.dispatcher import Dispatcher
from reef.dispatcher.data_sync_worker import DataSyncWorker
from reef.dispatcher.samples import LocalSampleProducers
from reef.dispatcher.served import ServedModelsPublisher
from reef.dispatcher.store_changes import StoreChanges, StoreChangesWatcher
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.evaluation.rollout_worker import EvaluationRolloutWorker
from reef.execution.workers import SERVICE_COMPONENTS, ServiceWorker, service_components
from reef.recipe import CompositeRecipe, Recipe
from reef.record2sample.service import (
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    SamplesWorker,
    Selection,
    request_selection,
)
from reef.service import assembly
from reef.service import components as component_apps
from reef.service.app import create_data_app
from reef.service.components import build_component_app
from reef.service.data_service import DataJob, DataWorker
from reef.service.data_transfer import ArchiveWriters
from reef.service.deploy.cli import build_serve_parser
from reef.service.deploy.service_config import ServiceConfig
from reef.service.scenario_files import artifact_directory
from reef.storage.sql_coordination import memory_coordination, open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import NewWork
from reef.train.component import TrainingJob
from reef.train.harness_backend.component import HarnessTrainingComponent
from reef.train.job_queue import JOBS_QUEUE, JobQueueComponent, TrainingJobWorker
from reef.train.training_requests import TrainingRequestQueue

MEMORY_BACKEND = "tests.support.artifact_backend_plugin:MemoryArtifactBackend"
TOKEN = "service-token"
import signal

from tests.support.job_data import job_data_ref

from reef.core.errors import ReefError
from reef.dispatcher.dispatcher import build_default_dispatcher


def settings(tmp_path: Path) -> ServiceConfig:
    return ServiceConfig(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        artifact_backend={"module": MEMORY_BACKEND, "config": {"label": "memory"}},
        tokens=(TOKEN,),
    )


def worker_types(tmp_path: Path, components: frozenset[str]) -> set[type]:
    dispatcher = assembly.build_dispatcher(settings(tmp_path), environ={}, components=components)
    try:
        return {type(worker) for worker in dispatcher.workers.workers}
    finally:
        dispatcher.close()


@pytest.mark.unit
def test_a_process_runs_the_workers_of_its_own_services_only(tmp_path: Path) -> None:
    control = {ServedModelsPublisher, StoreChangesWatcher, EndedWorkSweeper, DataSyncWorker}
    assert worker_types(tmp_path / "all", frozenset(SERVICE_COMPONENTS)) == control | {
        SamplesWorker,
        DataWorker,
        EvaluationRolloutWorker,
    }
    assert worker_types(tmp_path / "control", frozenset({"control"})) == control
    assert worker_types(tmp_path / "samples", frozenset({"samples"})) == {SamplesWorker}
    assert worker_types(tmp_path / "data", frozenset({"data"})) == {DataWorker}
    assert worker_types(tmp_path / "evaluation", frozenset({"evaluation"})) == {EvaluationRolloutWorker}
    with pytest.raises(ValueError, match="unknown services"):
        service_components({"control", "router"})
    with pytest.raises(ValueError, match="one or more"):
        service_components(frozenset())


def test_the_training_service_runs_queued_jobs_without_opening_scenario_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(settings(tmp_path), task_file_dir=str(tmp_path / "task-files"))
    component = StubTrainingComponent(tmp_path / "checkpoints")
    recipe = Recipe(training_component=component)
    monkeypatch.setattr(assembly, "_serving_recipe", lambda selected, settings, environ: recipe)
    trajectory = policy_trajectory("sample", (1, 2), (1,), (-0.1,), 1.0, artifact_id="v1")
    coordination = assembly.open_service_coordination(config)
    view = JobQueueComponent(component, coordination.work_queue, owns_component=False)
    view.submit(
        TrainingJob(
            "one",
            component.name,
            component.base_start,
            job_data_ref(JobData("data", (trajectory,))),
            {},
            scenario="math",
        )
    )

    async def run() -> None:
        app = build_component_app("training", config, environ={})
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/healthz")).status == 200
            deadline = asyncio.get_running_loop().time() + 5
            while view.status("one").state != "finished":
                assert asyncio.get_running_loop().time() < deadline, "the training service did not finish its job"
                await asyncio.sleep(0.01)
            assert len(view.status("one").checkpoints) == 1
            assert coordination.control_lease.current() is None
            assert not Path(config.task_file_dir).exists()
            assert not (tmp_path / "work").exists()
            assert not (tmp_path / "cache").exists()
        assert component.closed

    try:
        asyncio.run(run())
    finally:
        coordination.close()


@pytest.mark.parametrize("failure_stage", ["thread", "worker"])
def test_a_training_worker_start_failure_closes_its_component_and_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    component = StubTrainingComponent()
    recipe = Recipe(training_component=component)
    coordination = memory_coordination()
    monkeypatch.setattr(assembly, "_serving_recipe", lambda selected, settings, environ: recipe)
    monkeypatch.setattr(component_apps, "open_service_coordination", lambda settings: coordination)
    threads: list[Thread | None] = []
    start = TrainingJobWorker.start
    failure = RuntimeError("worker startup failed")

    def fail_start(worker: TrainingJobWorker) -> None:
        start(worker)
        threads.append(worker.thread)
        raise failure

    def fail_thread_start(thread: Thread) -> None:
        threads.append(thread)
        raise failure

    if failure_stage == "thread":
        monkeypatch.setattr(Thread, "start", fail_thread_start)
    else:
        monkeypatch.setattr(TrainingJobWorker, "start", fail_start)
    with pytest.raises(RuntimeError) as raised:
        build_component_app("training", settings(tmp_path), environ={})
    assert raised.value is failure
    assert component.closed
    assert all(thread is not None and not thread.is_alive() for thread in threads)
    with pytest.raises(ReefError, match="closed"):
        coordination.work_queue.items(JOBS_QUEUE)


def test_the_training_service_keeps_composite_harness_evaluation_on_the_served_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(settings(tmp_path), served_url="http://reef-control:8000")
    recipe = CompositeRecipe(components={"harness": harness_recipe(tmp_path, IdleMethod()), "records": Recipe()})
    monkeypatch.setattr(assembly, "_serving_recipe", lambda selected, settings, environ: recipe)
    bindings = []
    start = TrainingJobWorker.start

    def observe_start(worker: TrainingJobWorker) -> None:
        component = worker.component
        assert isinstance(component, HarnessTrainingComponent)
        bindings.append(component.steps.recipe.model_binding("math"))
        start(worker)

    monkeypatch.setattr(TrainingJobWorker, "start", observe_start)

    async def run() -> None:
        async with TestClient(TestServer(build_component_app("training", config, environ={}))) as client:
            assert (await client.get("/healthz")).status == 200

    asyncio.run(run())
    assert len(bindings) == 1
    assert bindings[0].base_url == "http://reef-control:8000/reef/scenarios/math/components/harness/evaluation"
    assert bindings[0].api_key is None


@pytest.mark.parametrize("failure_stage", ["jobs", "start"])
@pytest.mark.parametrize("stop_fails", [False, True])
def test_failed_assembly_stops_workers_before_closing_their_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str, stop_fails: bool
) -> None:
    config = settings(tmp_path)
    startup_error = RuntimeError("data worker assembly failed")
    dispatchers: list[Dispatcher] = []
    workers: list[ServiceWorker] = []
    threads: list[Thread] = []
    closed_with_running_workers: list[bool] = []
    data_jobs = assembly.data_jobs
    start_data_worker = DataWorker.start
    stop_watcher = StoreChangesWatcher.stop

    def observe_jobs(dispatcher: Dispatcher, sources: ArtifactSources, writers: ArchiveWriters) -> tuple[DataJob, ...]:
        dispatchers.append(dispatcher)
        workers.extend(dispatcher.workers.workers)
        publisher = dispatcher.publisher
        assert publisher is not None and publisher.thread is not None
        threads.append(publisher.thread)
        close_storage = dispatcher.storage.close

        def close() -> None:
            closed_with_running_workers.append(any(thread.is_alive() for thread in threads))
            close_storage()

        monkeypatch.setattr(dispatcher.storage, "close", close)
        if failure_stage == "jobs":
            raise startup_error
        return data_jobs(dispatcher, sources, writers)

    def fail_start(worker: DataWorker) -> None:
        workers.append(worker)
        start_data_worker(worker)
        assert worker.thread is not None
        threads.append(worker.thread)
        raise startup_error

    def fail_stop(watcher: StoreChangesWatcher) -> None:
        stop_watcher(watcher)
        raise RuntimeError("store watcher cleanup failed")

    monkeypatch.setattr(assembly, "data_jobs", observe_jobs)
    monkeypatch.setattr(DataWorker, "start", fail_start)
    if stop_fails:
        monkeypatch.setattr(StoreChangesWatcher, "stop", fail_stop)
    try:
        with pytest.raises(RuntimeError) as raised:
            assembly.build_dispatcher(config, environ={}, components=frozenset({"control", "data"}))
        assert raised.value is startup_error
        assert closed_with_running_workers == [False]
        assert all(not thread.is_alive() for thread in threads)
        coordination = assembly.open_service_coordination(config)
        try:
            assert coordination.control_lease.acquire("replacement-control") is not None
        finally:
            coordination.close()
    finally:
        # Also reap leaked workers when the regression runs against the old implementation.
        monkeypatch.setattr(StoreChangesWatcher, "stop", stop_watcher)
        for worker in reversed(workers):
            with suppress(Exception):
                worker.stop()
        for dispatcher in dispatchers:
            with suppress(Exception):
                dispatcher.close()


class Quiet(Selection):
    """A selection that drops its progress."""

    def progress(self, records_read: int, samples_created: int) -> None:
        return


@pytest.mark.unit
def test_the_samples_service_of_another_process_loads_the_scenario_and_leaves_the_jobs_to_control() -> None:
    made: list[str] = []
    trainer = SimpleNamespace(trains=True)
    current = SimpleNamespace(
        trainers=(SimpleNamespace(component="weights", trainer=trainer),),
        produce_samples=lambda component, selection: made.append(component) or False,
    )
    loads: list[tuple[str, object]] = []

    class Registry:
        def get_or_create(self, scenario: str, *, allow_implicit_creation: bool) -> SimpleNamespace:
            loads.append((scenario, allow_implicit_creation))
            return current

    assert LocalSampleProducers(Registry(), None).produce("math", "weights", Quiet("t")) is False
    assert made == ["weights"] and loads == [("math", False)]


class Changes(StoreChanges):
    def __init__(self) -> None:
        self.records: list[str] = []
        self.samples: list[str] = []
        self.jobs: list[str] = []
        self.training_requests: list[str] = []

    def records_added(self, current) -> None:
        self.records.append(current.name)

    def samples_added(self, scenario: str) -> None:
        self.samples.append(scenario)

    def job_changed(self, scenario: str) -> None:
        self.jobs.append(scenario)

    def requests_added(self, scenario: str) -> None:
        self.training_requests.append(scenario)


@pytest.mark.unit
@pytest.mark.parametrize("initial_sequence", [0, 3])
def test_control_learns_of_records_samples_and_jobs_that_other_processes_add(initial_sequence) -> None:
    sequence = {"math": initial_sequence}
    scenario = SimpleNamespace(name="math", records=SimpleNamespace(last_sequence=lambda name: sequence[name]))
    registry = SimpleNamespace(loaded_scenarios=lambda: (scenario,))
    queue = memory_coordination().work_queue
    changes = Changes()
    watcher = StoreChangesWatcher(registry, queue, changes)

    watcher.run_once()
    first_records = [] if initial_sequence == 0 else ["math"]
    assert changes.records == first_records and changes.samples == []
    sequence["math"] = 5
    request_selection(queue, "math", "weights", trigger="records")
    item = queue.take(SAMPLES_QUEUE, "samples-1", [RECORD_SELECTION_KIND])
    queue.finish(SAMPLES_QUEUE, item.id, "samples-1", {"changed": True, "background": False})
    watcher.run_once()
    assert changes.records == [*first_records, "math"] and changes.samples == ["math"]
    watcher.run_once()
    assert changes.records == [*first_records, "math"] and changes.samples == ["math"]
    # A training request that another process queued wakes control once.
    TrainingRequestQueue(queue).of("math").file("one", {"text": "train"})
    watcher.run_once()
    watcher.run_once()
    assert changes.training_requests == ["math"]

    # A job that a worker takes changes nothing to commit; its checkpoint and its end each wake control.
    queue.add(NewWork(JOBS_QUEUE, "job-1", "math", "job:stub"))
    job = queue.take(JOBS_QUEUE, "training-1", ["job:stub"])
    watcher.run_once()
    assert changes.jobs == []
    queue.add_progress(JOBS_QUEUE, job.id, "training-1", {"step": 0}, number=0)
    watcher.run_once()
    assert changes.jobs == ["math"]
    queue.finish(JOBS_QUEUE, job.id, "training-1", {"state": "finished", "metrics": {}})
    watcher.run_once()
    assert changes.jobs == ["math", "math"]


@pytest.mark.unit
def test_the_inference_component_answers_from_served_models_alone(tmp_path: Path) -> None:
    async def run() -> None:
        app = build_component_app("inference", settings(tmp_path))
        async with TestClient(TestServer(app)) as client:
            assert (await client.get("/healthz")).status == 200
            auth = {"Authorization": f"Bearer {TOKEN}"}
            assert (await client.post("/v1/chat/completions", json={})).status == 401
            assert (await client.post("/v1/chat/completions", headers=auth, json={})).status == 400
            missing = await client.post(
                "/v1/chat/completions", headers={**auth, "x-reef-scenario": "math"}, json={"model": "m"}
            )
            assert missing.status == 404

    asyncio.run(run())


@pytest.mark.unit
def test_the_evaluation_service_runs_on_its_own_and_an_unknown_service_is_refused(tmp_path: Path) -> None:
    # It serves its candidates itself, so it starts with the config alone: no other service's URL.
    async def run() -> None:
        async with TestClient(TestServer(build_component_app("evaluation", settings(tmp_path)))) as client:
            assert (await client.get("/healthz")).status == 200

    asyncio.run(run())
    with pytest.raises(ValueError, match="unknown service"):
        build_component_app("router", settings(tmp_path))


@pytest.mark.unit
def test_the_harness_service_needs_a_recipe_that_evolves_a_harness(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="evolves no harness, so it has no harness service"):
        build_component_app("harness", settings(tmp_path))


@pytest.mark.unit
def test_reef_serve_names_one_service_with_component() -> None:
    parser = build_serve_parser()
    assert parser.parse_args(["--component", "evaluation"]).component == "evaluation"
    assert parser.parse_args([]).component is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--component", "router"])


@pytest.mark.unit
def test_control_and_data_load_a_scenario_without_activating_its_engine(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("w0")
    component = ImportingComponent(tmp_path / "checkpoints")
    # Both processes reach the same artifact repository.
    repository = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")

    def dispatcher(components: frozenset[str]) -> Dispatcher:
        return Dispatcher(
            TestPolicyRecipe(**runtime_bindings(component, component.inference)),
            repository,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            coordination=open_coordination("sqlite", agent_record_dir=str(tmp_path), record_database_url=None),
            components=components,
        )

    control = dispatcher(frozenset({"control"}))
    try:
        control.get_or_create_scenario("math")
    finally:
        control.close()
    assert component.inference.activated == []
    data = dispatcher(frozenset({"data"}))
    try:
        with data.locked_scenario("math") as current:
            assert current.name == "math"
    finally:
        data.close()
    assert component.inference.activated == []


@pytest.mark.unit
def test_the_data_service_reads_the_artifact_that_control_commits_after_the_data_service_loaded(
    tmp_path: Path,
) -> None:
    component = ImportingComponent(tmp_path / "checkpoints")
    repository = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")

    def dispatcher(components: frozenset[str]) -> Dispatcher:
        return Dispatcher(
            TestPolicyRecipe(**runtime_bindings(component, component.inference)),
            repository,
            scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
            coordination=open_coordination("sqlite", agent_record_dir=str(tmp_path), record_database_url=None),
            components=components,
        )

    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}")
    (weights / "model.safetensors").write_bytes(b"w9")
    directory = tmp_path / "store"
    directory.mkdir()
    control = dispatcher(frozenset({"control"}))
    data = dispatcher(frozenset({"data"}))
    try:
        control.get_or_create_scenario("math")
        before = {row["artifact_id"] for row in _listed(data)}
        imported = control.import_artifact(
            "math",
            DirectorySource(weights, source_version="", link=False),
            None,
            source={"uri": weights.as_uri(), "version": ""},
            directory=directory,
        )
        assert imported.artifact_id not in before
        assert imported.artifact_id in {row["artifact_id"] for row in _listed(data)}
        assert artifact_directory(data, "math", imported.artifact_id, None).is_dir()
    finally:
        data.close()
        control.close()


def _listed(dispatcher: Dispatcher) -> tuple[dict[str, object], ...]:
    """The data service's artifact list. It loads the scenario on its first read."""
    dispatcher.get_or_create_scenario("math")
    return dispatcher.list_artifacts("math")


@pytest.mark.unit
def test_the_data_service_serves_the_data_routes_and_the_reads_and_control_alone_serves_neither_data_nor_writes(
    tmp_path: Path,
) -> None:
    auth = {"Authorization": f"Bearer {TOKEN}"}

    async def run() -> None:
        data_app = create_data_app(
            assembly.build_dispatcher(settings(tmp_path / "data"), environ={}, components=frozenset({"data"})),
            tokens=(TOKEN,),
        )
        async with TestClient(TestServer(data_app)) as client:
            assert (await client.get("/reef/scenarios", headers=auth)).status == 200
            assert (await client.get("/reef/namespaces/o/data", headers=auth)).status == 200
            # Writes to a scenario and the trainer's buffer are control's.
            assert (await client.post("/reef/scenarios", headers=auth, json={"name": "s"})).status == 405
            assert (await client.get("/reef/scenarios/s/buffer", headers=auth)).status == 404
        control_app = assembly.build_app(settings(tmp_path / "control"), environ={}, components=frozenset({"control"}))
        async with TestClient(TestServer(control_app)) as client:
            assert (await client.get("/reef/namespaces/o/data", headers=auth)).status == 404

    asyncio.run(run())


@pytest.mark.unit
def test_a_control_dispatcher_refuses_a_lease_that_another_instance_holds(tmp_path: Path) -> None:
    coordination = memory_coordination()
    assert coordination.control_lease.acquire("control-other") is not None
    with pytest.raises(ReefError, match="holds the control lease"):
        build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=coordination)


@pytest.mark.unit
def test_a_lost_lease_stops_the_local_cycles_and_the_app_owner_ends_the_process(monkeypatch) -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    signals: list[int] = []
    monkeypatch.setattr(assembly.signal, "raise_signal", signals.append)
    try:
        dispatcher.control_lease_lost()
        # An embedded dispatcher ends no process: it only says that the lease is lost.
        assert dispatcher.lease_lost.is_set() and not dispatcher.local_workers.cycles_run() and signals == []
        asyncio.run(assembly.EndOnLeaseLoss(dispatcher, poll_seconds=0.01).watch())
        assert signals == [signal.SIGTERM]
    finally:
        dispatcher.close()
