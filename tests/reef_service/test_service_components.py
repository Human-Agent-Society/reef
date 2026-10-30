"""``reef serve --component``: each process runs the workers of its own services on the shared store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.dispatcher.samples import LocalSampleProducers
from reef.dispatcher.served import ServedModelsPublisher
from reef.dispatcher.store_changes import StoreChanges, StoreChangesWatcher
from reef.execution.workers import SERVICE_COMPONENTS, service_components
from reef.record2sample.service import RECORD_SELECTION_KIND, SAMPLES_QUEUE, SamplesWorker, request_selection
from reef.service import assembly
from reef.service.components import build_component_app
from reef.service.deploy.cli import build_serve_parser
from reef.service.deploy.service_config import ServiceConfig
from reef.storage.work_queue import MemoryWorkQueue

MEMORY_BACKEND = "tests.support.artifact_backend_plugin:MemoryArtifactBackend"
TOKEN = "service-token"


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
    control = {ServedModelsPublisher, StoreChangesWatcher}
    assert worker_types(tmp_path / "all", frozenset(SERVICE_COMPONENTS)) == control | {SamplesWorker}
    assert worker_types(tmp_path / "control", frozenset({"control"})) == control
    assert worker_types(tmp_path / "samples", frozenset({"samples"})) == {SamplesWorker}
    with pytest.raises(ValueError, match="unknown services"):
        service_components({"control", "router"})
    with pytest.raises(ValueError, match="one or more"):
        service_components(frozenset())


@pytest.mark.unit
def test_the_samples_service_of_another_process_loads_the_scenario_and_leaves_the_jobs_to_control() -> None:
    made: list[str] = []
    trainer = SimpleNamespace(trains=True, produce_samples=lambda: made.append("weights") or False)
    current = SimpleNamespace(trainers=(SimpleNamespace(component="weights", trainer=trainer),))
    loads: list[tuple[str, object]] = []

    class Registry:
        def get_or_create(self, scenario: str, *, allow_implicit_creation: bool) -> SimpleNamespace:
            loads.append((scenario, allow_implicit_creation))
            return current

    assert LocalSampleProducers(Registry(), None).produce("math", "weights") is False
    assert made == ["weights"] and loads == [("math", False)]


class Changes(StoreChanges):
    def __init__(self) -> None:
        self.records: list[str] = []
        self.samples: list[str] = []

    def records_added(self, current) -> None:
        self.records.append(current.name)

    def samples_added(self, scenario: str) -> None:
        self.samples.append(scenario)


@pytest.mark.unit
def test_control_learns_of_records_and_samples_that_other_processes_add() -> None:
    sequence = {"math": 3}
    scenario = SimpleNamespace(name="math", records=SimpleNamespace(last_sequence=lambda name: sequence[name]))
    registry = SimpleNamespace(loaded_scenarios=lambda: (scenario,))
    queue = MemoryWorkQueue()
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
def test_the_evaluation_component_needs_the_inference_service_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--inference-service-url"):
        build_component_app("evaluation", settings(tmp_path))
    with pytest.raises(ValueError, match="unknown service"):
        build_component_app("router", settings(tmp_path))


@pytest.mark.unit
def test_reef_serve_names_one_service_with_component() -> None:
    # With the service arguments, so that the flag cannot collide with a configuration option.
    parser = build_serve_parser()
    args = parser.parse_args(["--component", "evaluation", "--inference-service-url", "http://inference:8000"])
    assert (args.component, args.inference_service_url) == ("evaluation", "http://inference:8000")
    assert parser.parse_args([]).component is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--component", "router"])
