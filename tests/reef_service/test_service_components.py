"""``reef serve --component``: each process runs the workers of its own services on the shared store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.artifact_versions import ImportingComponent
from tests.support.runtime_stubs import runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.dispatcher.data_sync_worker import DataSyncWorker
from reef.dispatcher.samples import LocalSampleProducers
from reef.dispatcher.served import ServedModelsPublisher
from reef.dispatcher.store_changes import StoreChanges, StoreChangesWatcher
from reef.dispatcher.work_retention import EndedWorkSweeper
from reef.evaluation.rollout_worker import EvaluationRolloutWorker
from reef.execution.workers import SERVICE_COMPONENTS, service_components
from reef.record2sample.service import (
    RECORD_SELECTION_KIND,
    SAMPLES_QUEUE,
    SamplesWorker,
    TaskGenerationWorker,
    request_selection,
)
from reef.service import assembly
from reef.service.app import create_data_app
from reef.service.components import build_component_app
from reef.service.data_service import DataWorker
from reef.service.deploy.cli import build_serve_parser
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.sql_coordination import memory_coordination, open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import NewWork
from reef.train.instructions import InstructionQueue
from reef.train.job_queue import JOBS_QUEUE

MEMORY_BACKEND = "tests.support.artifact_backend_plugin:MemoryArtifactBackend"
TOKEN = "service-token"
import signal

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
        TaskGenerationWorker,
        DataWorker,
        EvaluationRolloutWorker,
    }
    assert worker_types(tmp_path / "control", frozenset({"control"})) == control
    assert worker_types(tmp_path / "samples", frozenset({"samples"})) == {SamplesWorker, TaskGenerationWorker}
    assert worker_types(tmp_path / "data", frozenset({"data"})) == {DataWorker}
    assert worker_types(tmp_path / "evaluation", frozenset({"evaluation"})) == {EvaluationRolloutWorker}
    with pytest.raises(ValueError, match="unknown services"):
        service_components({"control", "router"})
    with pytest.raises(ValueError, match="one or more"):
        service_components(frozenset())


@pytest.mark.unit
def test_the_samples_service_of_another_process_loads_the_scenario_follows_its_stored_mode_and_leaves_the_jobs_to_control() -> (
    None
):
    made: list[str] = []
    trainer = SimpleNamespace(trains=True, produce_samples=lambda: made.append("weights") or False)
    current = SimpleNamespace(trainers=(SimpleNamespace(component="weights", trainer=trainer),))
    loads: list[tuple[str, object]] = []

    class Registry:
        def get_or_create(self, scenario: str, *, allow_implicit_creation: bool) -> SimpleNamespace:
            loads.append((scenario, allow_implicit_creation))
            return current

        def apply_stored_training_mode(self, scenario: str) -> None:
            loads.append((scenario, "stored mode"))

    assert LocalSampleProducers(Registry(), None).produce("math", "weights") is False
    assert made == ["weights"] and loads == [("math", False), ("math", "stored mode")]


class Changes(StoreChanges):
    def __init__(self) -> None:
        self.records: list[str] = []
        self.samples: list[str] = []
        self.jobs: list[str] = []
        self.instructions: list[str] = []

    def records_added(self, current) -> None:
        self.records.append(current.name)

    def samples_added(self, scenario: str) -> None:
        self.samples.append(scenario)

    def job_changed(self, scenario: str) -> None:
        self.jobs.append(scenario)

    def instructions_added(self, scenario: str) -> None:
        self.instructions.append(scenario)


@pytest.mark.unit
def test_control_learns_of_records_samples_and_jobs_that_other_processes_add() -> None:
    sequence = {"math": 3}
    scenario = SimpleNamespace(name="math", records=SimpleNamespace(last_sequence=lambda name: sequence[name]))
    registry = SimpleNamespace(loaded_scenarios=lambda: (scenario,))
    queue = memory_coordination().work_queue
    changes = Changes()
    watcher = StoreChangesWatcher(registry, queue, changes)

    watcher.run_once()
    assert changes.records == [] and changes.samples == []  # a scenario seen first asks for samples when it loads
    sequence["math"] = 5
    request_selection(queue, "math", "weights")
    item = queue.take(SAMPLES_QUEUE, "samples-1", [RECORD_SELECTION_KIND])
    queue.finish(SAMPLES_QUEUE, item.id, "samples-1", {"changed": True, "background": False})
    watcher.run_once()
    assert changes.records == ["math"] and changes.samples == ["math"]
    watcher.run_once()
    assert changes.records == ["math"] and changes.samples == ["math"]
    # A training instruction that another process queued wakes control once.
    InstructionQueue(queue).of("math").file("one", {"text": "train"})
    watcher.run_once()
    watcher.run_once()
    assert changes.instructions == ["math"]

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
def test_a_process_without_control_loads_a_scenario_without_activating_it(tmp_path: Path) -> None:
    # The engines are control's: a process that loads a scenario for its reads or its samples loads no weights.
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("w0")
    component = ImportingComponent(tmp_path / "checkpoints")
    # Both processes reach the same artifact repository.
    repository = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")

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
    activated = list(component.inference.activated)
    assert activated, "control activates the scenario it loads"
    data = dispatcher(frozenset({"data"}))
    try:
        with data.locked_scenario("math") as current:
            assert current.name == "math"
    finally:
        data.close()
    assert component.inference.activated == activated


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
