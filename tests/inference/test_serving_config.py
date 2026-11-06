"""A scenario's serving config (RFC #1097): control checks and stores it, inference applies it without an activation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.core.values import JsonValue
from reef.dispatcher import Dispatcher
from reef.dispatcher.serving import ServingWorker
from reef.inference.interfaces import FixedEngineClients, InferenceHandler
from reef.service.app import create_app
from reef.service.inference_service import build_inference_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

CHAT = {"model": "reef", "messages": [{"role": "user", "content": "hello"}]}
HEADERS = {"x-reef-scenario": "math"}


class ScalingRuntime(StubInferenceRuntime):
    """An engine with one engine field, ``min_replicas`` (deployment value 0), and the handler's ``temperature``
    (deployment value 1.0). ``applied`` lists each override that inference applied."""

    def __init__(self) -> None:
        super().__init__()
        self.applied: list[tuple[str, dict[str, JsonValue]]] = []

    def serving_config(self, override: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        unknown = sorted(set(override) - {"min_replicas", "temperature"})
        if unknown:
            raise ValueError(f"unknown serving fields {unknown}")
        replicas = override.get("min_replicas", 0)
        if not isinstance(replicas, int) or isinstance(replicas, bool) or replicas < 0:
            raise ValueError("min_replicas must be a non-negative integer")
        return {"min_replicas": replicas, "temperature": override.get("temperature", 1.0)}

    def apply_serving_config(self, scenario: str, override: Mapping[str, JsonValue]) -> None:
        self.applied.append((scenario, dict(override)))

    def served_model_name(self, scenario: str) -> str:
        return "test-model"


class TemperatureHandler(InferenceHandler):
    """An engine that keeps each body it receives; the scenario's temperature applies when a request has none."""

    def __init__(self) -> None:
        self.received: list[dict] = []

    def request_payload(self, payload, *, serving_config):
        if "temperature" in serving_config and "temperature" not in payload:
            return {**payload, "temperature": serving_config["temperature"]}
        return dict(payload)

    async def inference(self, artifact, path, payload):
        self.received.append(payload)
        return {"choices": [{"message": {"content": "ready"}}]}


@pytest.fixture
def serving(tmp_path):
    runtime = ScalingRuntime()
    handler = TemperatureHandler()
    recipe = TestPolicyRecipe(**runtime_bindings(StubTrainingComponent(tmp_path / "checkpoints"), runtime))
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "records")

    def control(coordination=coordination, storage=storage) -> Dispatcher:
        return Dispatcher(
            recipe,
            backends,
            scenario_storage=storage,
            coordination=coordination,
            components=frozenset({"control"}),
            local_artifact_dir=tmp_path / "control",
        )

    worker = ServingWorker(
        recipe, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    inference = build_inference_app(coordination.served_models, storage, FixedEngineClients(handler), tokens=None)
    # Each test stops the worker before it closes control, which closes the coordination store.
    return control, worker, runtime, handler, inference


@pytest.mark.unit
def test_a_changed_serving_config_reaches_the_engine_and_each_request_without_an_activation(serving, tmp_path):
    control_of, worker, runtime, handler, inference_app = serving
    control = control_of()

    async def run() -> None:
        async with (
            TestClient(TestServer(create_app(control))) as control_client,
            TestClient(TestServer(inference_app)) as inference_client,
        ):
            assert (await control_client.post("/reef/scenarios", json={"name": "math"})).status == 201
            assert await asyncio.to_thread(worker.run_once)
            assert runtime.applied == [("math", {})] and len(runtime.activated) == 1

            update = {"serving_config": {"weights": {"temperature": 0.3, "min_replicas": 1}}}
            response = await control_client.post("/reef/scenarios/math/update", json=update)
            assert response.status == 200, await response.text()
            changed = await response.json()
            assert changed["serving_config"] == {"weights": {"temperature": 0.3, "min_replicas": 1}}
            assert changed["effective_serving_config"] == {"weights": {"temperature": 0.3, "min_replicas": 1}}

            assert await asyncio.to_thread(worker.run_once)
            assert runtime.applied[-1] == ("math", {"temperature": 0.3, "min_replicas": 1})
            assert len(runtime.activated) == 1
            assert (await inference_client.post("/v1/chat/completions", headers=HEADERS, json=CHAT)).status == 200
            own = {**CHAT, "temperature": 0.9}
            assert (await inference_client.post("/v1/chat/completions", headers=HEADERS, json=own)).status == 200
            assert [body["temperature"] for body in handler.received] == [0.3, 0.9]

            # Null removes a field; the other field stays.
            removal = {"serving_config": {"weights": {"temperature": None}}}
            assert (await control_client.post("/reef/scenarios/math/update", json=removal)).status == 200
            settings = await (await control_client.get("/reef/scenarios/math/settings")).json()
            assert settings["serving_config"] == {"weights": {"min_replicas": 1}}
            assert settings["effective_serving_config"] == {"weights": {"temperature": 1.0, "min_replicas": 1}}
            assert await asyncio.to_thread(worker.run_once)
            assert runtime.applied[-1] == ("math", {"min_replicas": 1})
            assert (await inference_client.post("/v1/chat/completions", headers=HEADERS, json=CHAT)).status == 200
            assert "temperature" not in handler.received[-1]

    try:
        asyncio.run(run())
    finally:
        worker.stop()
        control.close()

    # A restart, which keeps only the scenario files, serves with the stored config again.
    restarted = control_of(memory_coordination(), SQLiteScenarioStorage(tmp_path / "records"))
    try:
        restarted.scenario_settings("math")
        payload = restarted.coordination.served_models.current_request("math").payload
        assert payload["serving_config"] == {"min_replicas": 1}
    finally:
        restarted.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "update",
    [
        {"weights": {"gpu": "B200"}},
        {"weights": {"min_replicas": -1}},
        {"harness": {"temperature": 0.3}},
        {"weights": "fast"},
    ],
)
def test_a_refused_serving_config_answers_400_and_stores_nothing(serving, update):
    control_of, worker = serving[0], serving[1]
    control = control_of()

    async def run() -> None:
        async with TestClient(TestServer(create_app(control))) as client:
            assert (await client.post("/reef/scenarios", json={"name": "math"})).status == 201
            response = await client.post("/reef/scenarios/math/update", json={"serving_config": update})
            assert response.status == 400, await response.text()
            settings = await (await client.get("/reef/scenarios/math/settings")).json()
            assert settings["serving_config"] == {}
            assert settings["effective_serving_config"] == {"weights": {"min_replicas": 0, "temperature": 1.0}}

    try:
        asyncio.run(run())
    finally:
        worker.stop()
        control.close()


@pytest.mark.unit
def test_another_service_reads_the_stored_serving_config_that_control_changed(serving, tmp_path):
    control_of, worker = serving[0], serving[1]
    control = control_of()
    # The data service reads settings with its own dispatcher, which loaded the scenario before the change.
    data = control_of(memory_coordination(), SQLiteScenarioStorage(tmp_path / "records"))
    try:
        control.create_scenario("math")
        assert data.scenario_settings("math")["effective_serving_config"] == {
            "weights": {"min_replicas": 0, "temperature": 1.0}
        }
        control.set_serving_config("math", {"weights": {"min_replicas": 2}})
        settings = data.scenario_settings("math")
        assert settings["serving_config"] == {"weights": {"min_replicas": 2}}
        assert settings["effective_serving_config"] == {"weights": {"min_replicas": 2, "temperature": 1.0}}
    finally:
        worker.stop()
        data.close()
        control.close()
