"""Inference startup runs independently of control and reports the selected artifact's result."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import (
    RunningEngines,
    StubFacts,
    StubInferenceRuntime,
    StubTrainingComponent,
    live_objects,
)
from tests.support.serving import Engines

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.artifact import Artifact, SnapshotRef
from reef.core.served_models import ScheduledStop, ServedArtifact
from reef.dispatcher import Dispatcher, serving
from reef.dispatcher.serving import ServingWorker
from reef.inference.checkpoint_slots import CheckpointSlotsRuntime
from reef.inference.http import InferenceProxyRuntime
from reef.inference.interfaces import (
    FixedEngineClients,
    InferenceHandler,
    InferenceOwnerLostError,
    RuntimeContractError,
)
from reef.inference.null import NullInferenceRuntime
from reef.inference.serving import FullWeightServingRuntime, OpenedEngines
from reef.inference.startup import StartupFailedError, StartupStoppedError, redeploying, report_startup, stopping
from reef.recipe.base import Recipe
from reef.recipe.composite import CompositeRecipe
from reef.scenario.live_objects import LiveObjects
from reef.service.app import create_app
from reef.service.inference_service import ARTIFACT_HEADER, RECORD_HEADER, build_inference_app
from reef.storage.served_models import REDEPLOY, SERVING_KIND, SERVING_QUEUE
from reef.storage.sql_coordination import memory_coordination, open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import WorkLeaseLostError

STATUS = "/reef/scenarios/math/inference/status"
CHAT = {"model": "reef", "messages": [{"role": "user", "content": "hello"}]}


class PausedRuntime(StubInferenceRuntime):
    """A named engine whose activation waits for the test to release it."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_start = False
        self.opens = 0
        self.shutdowns = 0
        self.stops = []
        self.binding_known = True

    def open(self) -> None:
        self.opens += 1

    def shutdown(self) -> None:
        self.shutdowns += 1

    def stop_serving(self, served) -> None:
        assert self.opens > 0
        self.stops.append(served)

    def activate_checkpoint(self, artifact, *, scenario=None):
        report_startup("prepare", "Deploying the engine")
        report_startup("start", "Waiting for a GPU", estimate_seconds=300.0)
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("the test did not release startup")
        if self.fail_start:
            raise ValueError("the selected model cannot load")
        result = super().activate_checkpoint(artifact, scenario=scenario)
        self.binding_known = True
        return result

    def served_artifact(self, artifact_id, scenario):
        if not self.binding_known:
            return None
        return super().served_artifact(artifact_id, scenario)

    def served_model_name(self, scenario: str) -> str:
        return "test-model"


class Answer(InferenceHandler):
    async def inference(self, artifact, path, payload):
        return {"choices": [{"message": {"content": "ready"}}]}


@pytest.fixture
def startup(tmp_path):
    runtime = PausedRuntime()
    recipe = TestPolicyRecipe()
    live = live_objects(StubTrainingComponent(tmp_path / "checkpoints"), runtime)
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "records")
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=storage,
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "control",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    inference = build_inference_app(coordination.served_models, storage, FixedEngineClients(Answer()), tokens=None)
    try:
        yield control, worker, runtime, inference
    finally:
        runtime.release.set()
        worker.stop()
        control.close()


@pytest.mark.unit
@pytest.mark.parametrize("lost_binding", [False, True])
def test_cached_activation_is_reused_only_while_the_runtime_still_has_its_route(startup, lost_binding):
    control, worker, runtime, _ = startup
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    item = control.coordination.served_models.current_request("math")
    assert item.state == "finished"
    runtime.entered.clear()
    runtime.binding_known = not lost_binding

    # A serving-config or alias update can select the same artifact again.
    result = worker.run_item(item, threading.Event())
    assert runtime.entered.is_set() is lost_binding
    assert result["active"]["artifact_id"] == item.payload["artifact_id"]


@pytest.mark.unit
def test_an_activation_that_leaves_no_weight_binding_fails_despite_an_endpoint_model(startup, monkeypatch):
    control, worker, runtime, _ = startup
    # The activation returns, but the runtime holds no weight binding for the artifact afterwards.
    monkeypatch.setattr(runtime, "served_artifact", lambda artifact_id, scenario: None)
    control.create_scenario("math")
    runtime.release.set()
    while worker.run_once():
        pass
    routes = control.coordination.served_models
    failed = routes.current_request("math")
    assert runtime.activated and runtime.endpoint_model_name("math") == "test-model"
    assert failed.state == "failed" and "provides no route for the selected artifact" in failed.error
    assert routes.read_scenario("math") == {}


@pytest.mark.unit
@pytest.mark.parametrize("composite", [False, True])
def test_full_weight_worker_restores_its_component_after_a_failed_load_and_takeover(tmp_path, composite):
    class AttachedEngines(RunningEngines):
        """Attach to the same backend and signal ``checked_twice`` after two healthy ticks."""

        def __init__(self, backend):
            super().__init__(backend)
            self.checks = 0
            self.checked_twice = threading.Event()

        def open(self):
            return OpenedEngines(self.backend, attached=True)

        def check_health(self):
            self.checks += 1
            if self.checks >= 2:
                self.checked_twice.set()

    class WaitingRuntime(FullWeightServingRuntime):
        """A full-weight runtime whose activation waits for ``release`` after signaling ``activating``."""

        def __init__(self, engines):
            super().__init__(engines, Answer(), model_path="test-model", health_interval_s=0.01)
            self.activating = threading.Event()
            self.release = threading.Event()

        def activate_checkpoint(self, artifact, *, scenario=None):
            self.activating.set()
            if not self.release.wait(5):
                raise TimeoutError("the test did not release activation")
            return super().activate_checkpoint(artifact, scenario=scenario)

    engines = Engines()
    runtime = FullWeightServingRuntime(RunningEngines(engines), Answer(), model_path="test-model")
    weights = TestPolicyRecipe()
    recipe = CompositeRecipe(components={"weights": weights, "records": Recipe()}) if composite else weights
    # Control and inference hold the facts of the training backend, and no training component.
    live = LiveObjects(runtime=runtime, training_facts=StubFacts(StubTrainingComponent()))
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "control",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    try:
        control.create_scenario("math")
        assert worker.run_once()
        item = coordination.served_models.current_request("math")
        assert item.state == "finished"
        selected = item.payload["selection"]["weights"]
        assert item.payload["artifact_id"] == runtime.served_artifact_id == selected
        original_path = engines.loads()[-1][-1]

        candidate = tmp_path / "failed-candidate"
        candidate.mkdir()
        engines.misreport = "unexpected-load"
        with pytest.raises(RuntimeContractError):
            runtime.activate_checkpoint(
                Artifact(SnapshotRef("candidate", None), None, local_path=candidate, artifact_id="artifact:candidate")
            )
        assert runtime.served_artifact(selected, "math") is None

        # The worker cached this selection before the failed load. It must still restore the weights.
        engines.misreport = None
        result = worker.run_item(item, threading.Event())
        assert engines.loads()[-1][-1] == original_path
        assert engines.events[-1] == "resume"
        assert result["active"]["artifact_id"] == selected
        assert runtime.served_artifact(selected, "math") is not None

        worker.stop()
        engines.events.clear()
        attached = AttachedEngines(engines)
        restarted = WaitingRuntime(attached)
        worker = ServingWorker(
            recipe,
            replace(live, runtime=restarted),
            backends,
            coordination,
            worker_id="restarted-inference",
            local_artifact_dir=tmp_path / "restarted-inference",
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            started = executor.submit(worker.run_once)
            try:
                assert restarted.activating.wait(2)
                assert attached.checked_twice.wait(2)
                assert restarted.recovery_pending
                assert "pause" in engines.events and "resume" not in engines.events
                assert coordination.served_models.read_scenario("math") == {}
            finally:
                restarted.release.set()
            assert started.result(timeout=5)
        assert engines.events[-1] == "resume"
        assert coordination.served_models.read_scenario("math")["active"].artifact_id == selected
    finally:
        worker.stop()
        control.close()


@pytest.mark.unit
@pytest.mark.parametrize("fails", [False, True])
def test_slow_start_reports_its_result_without_blocking_control(startup, fails, caplog):
    control, worker, runtime, inference_app = startup
    runtime.fail_start = fails
    caplog.set_level(logging.WARNING, logger="reef.execution.workers")

    async def run() -> None:
        async with (
            TestClient(TestServer(create_app(control))) as control_client,
            TestClient(TestServer(inference_app)) as inference_client,
        ):
            created = await asyncio.wait_for(control_client.post("/reef/scenarios", json={"name": "math"}), 2)
            assert created.status == 201
            assert runtime.opens == 0 and not runtime.entered.is_set()
            pending = await (await inference_client.get(STATUS)).json()
            assert pending["state"] == "pending" and pending["error"] is None

            work = asyncio.create_task(asyncio.to_thread(worker.run_once))
            try:
                assert await asyncio.to_thread(runtime.entered.wait, 2)
                status = await (await inference_client.get(STATUS)).json()
                startup = status.pop("startup")
                assert status == {
                    **{key: value for key, value in pending.items() if key != "startup"},
                    "state": "starting",
                }
                # RFC #1283: the runtime's reports are the phases of this artifact's startup.
                prepare, start = startup["steps"]
                assert (prepare["phase"], prepare["detail"], start["phase"], start["detail"]) == (
                    "prepare",
                    "Deploying the engine",
                    "start",
                    "Waiting for a GPU",
                )
                assert prepare["ended_at"] == start["started_at"] and start["ended_at"] is None
                assert startup["estimate_seconds"] == 300.0
                settings = await asyncio.wait_for(control_client.get("/reef/scenarios/math/settings"), 2)
                assert settings.status == 200
                waiting = await inference_client.post(
                    "/v1/chat/completions", headers={"x-reef-scenario": "math"}, json=CHAT
                )
                assert waiting.status == 503 and waiting.headers["Retry-After"] == "30"
                assert (await waiting.json())["error"]["type"] == "inference_starting"
            finally:
                runtime.release.set()
                assert await asyncio.wait_for(work, 5)

            result = await (await inference_client.get(STATUS)).json()
            assert result["artifact_id"] == pending["artifact_id"]
            assert result["state"] == ("failed" if fails else "ready")
            assert (result["error"] is not None) == fails
            response = await inference_client.post(
                "/v1/chat/completions", headers={"x-reef-scenario": "math"}, json=CHAT
            )
            assert response.status == (503 if fails else 200)
            if fails:
                assert (await response.json())["error"]["type"] == "inference_start_failed"
                failed = control.coordination.served_models.current_request("math")
                warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
                assert warnings == [
                    f"work item serving/{failed.id} of scenario math failed: ValueError: the selected model cannot load"
                ]
            assert (await control_client.get("/reef/scenarios/math/settings")).status == 200
            assert runtime.opens == 1

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("recreate", [False, True])
def test_deleted_or_recreated_scenario_rejects_the_old_startup_result(startup, recreate):
    control, worker, runtime, inference_app = startup

    async def run() -> None:
        async with TestClient(TestServer(inference_app)) as client:
            control.create_scenario("math")
            first = control.coordination.served_models.current_request("math")
            work = asyncio.create_task(asyncio.to_thread(worker.run_once))
            try:
                assert await asyncio.to_thread(runtime.entered.wait, 2)
                await asyncio.wait_for(asyncio.to_thread(control.delete_scenario, "math"), 2)
                assert (await client.get(STATUS)).status == 404
                if recreate:
                    control.create_scenario("math")
                    current = control.coordination.served_models.current_request("math")
                    assert current.id != first.id
                    assert (await (await client.get(STATUS)).json())["state"] == "pending"
            finally:
                runtime.release.set()
                assert await asyncio.wait_for(work, 5)

            assert control.coordination.served_models.read_scenario("math") == {}
            if recreate:
                assert (await (await client.get(STATUS)).json())["state"] == "pending"
                assert await asyncio.to_thread(worker.run_once)
                assert (await (await client.get(STATUS)).json())["state"] == "ready"
            else:
                assert (await client.get(STATUS)).status == 404

    asyncio.run(run())


@pytest.mark.unit
def test_an_inference_worker_resumes_expired_startup_and_keeps_completed_work_ready(startup):
    control, worker, runtime, inference_app = startup
    control.create_scenario("math")
    queue = control.coordination.work_queue
    abandoned = queue.take(SERVING_QUEUE, "previous-process", [SERVING_KIND])
    assert abandoned is not None
    queue.start(SERVING_QUEUE, abandoned.id, "previous-process")
    assert queue.requeue_expired(now=abandoned.lease_expires_at + 1)[0].state == "queued"
    runtime.release.set()

    async def run() -> None:
        async with TestClient(TestServer(inference_app)) as client:
            assert (await (await client.get(STATUS)).json())["state"] == "starting"
            assert await asyncio.to_thread(worker.run_once)
            ready = await (await client.get(STATUS)).json()
            assert ready["state"] == "ready"
            completed = control.coordination.served_models.current_request("math")
            assert completed.id == abandoned.id and completed.attempt_count == 2
            assert completed.result and control.coordination.served_models.read_scenario("math")
            with pytest.raises(WorkLeaseLostError):
                control.coordination.served_models.complete_request(
                    abandoned, "previous-process", {}, serving_config={}, stop_delay_seconds=0
                )
            assert not await asyncio.to_thread(worker.run_once)
            assert await (await client.get(STATUS)).json() == ready
            assert runtime.opens == 1

    asyncio.run(run())


@pytest.mark.unit
def test_a_standby_recovers_completed_selections_after_the_owner_stops(startup, tmp_path):
    control, worker, runtime, inference_app = startup
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    replacement = PausedRuntime()
    recipe = TestPolicyRecipe()
    live = live_objects(StubTrainingComponent(tmp_path / "replacement"), replacement)
    standby = ServingWorker(recipe, live, worker.backends, control.coordination, worker_id="standby")

    async def run() -> None:
        async with TestClient(TestServer(inference_app)) as client:
            assert (await (await client.get(STATUS)).json())["state"] == "ready"
            assert not await asyncio.to_thread(standby.run_once)
            assert replacement.opens == 0
            worker.stop()
            assert runtime.shutdowns == 1
            work = asyncio.create_task(asyncio.to_thread(standby.run_once))
            try:
                assert await asyncio.to_thread(replacement.entered.wait, 2)
                assert (await (await client.get(STATUS)).json())["state"] == "starting"
                assert control.coordination.served_models.read_scenario("math") == {}
            finally:
                replacement.release.set()
                assert await asyncio.wait_for(work, 5)
            assert (await (await client.get(STATUS)).json())["state"] == "ready"
            assert replacement.opens == 1

    try:
        asyncio.run(run())
    finally:
        replacement.release.set()
        standby.stop()


@pytest.mark.unit
def test_a_fresh_owner_opens_the_runtime_before_cleanup(startup, tmp_path):
    control, worker, runtime, _ = startup
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    [served] = set(control.coordination.served_models.read_scenario("math").values())
    control.delete_scenario("math")
    control.coordination.served_models.schedule_stop(
        ScheduledStop(served, 0), control_lease_number=control.publisher.lease_number
    )
    worker.stop()
    replacement = PausedRuntime()
    recipe = TestPolicyRecipe()
    live = live_objects(StubTrainingComponent(tmp_path / "replacement"), replacement)
    standby = ServingWorker(recipe, live, worker.backends, control.coordination, worker_id="standby")
    try:
        assert standby.run_once()
        assert replacement.opens == 1 and replacement.stops == [served]
    finally:
        standby.stop()


@pytest.mark.unit
def test_owner_lease_loss_stops_the_runtime_and_prevents_reacquisition(startup):
    control, worker, runtime, _ = startup
    control.coordination.work_queue.lease_seconds = 0.3
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    owner = worker.owner
    assert owner is not None
    control.coordination.work_queue.cancel(owner.item.queue, owner.item.id, "test takeover")
    assert owner.lease_lost.wait(2)
    owner.stop()
    assert runtime.shutdowns == 1 and worker.stopped.is_set()
    assert not worker.run_once()


@pytest.mark.unit
def test_receiver_ownership_loss_preserves_the_selection_even_when_cleanup_is_fenced(startup, monkeypatch):
    control, worker, runtime, _ = startup
    control.create_scenario("math")

    def fenced(*args, **kwargs):
        raise InferenceOwnerLostError("the receiver accepted another owner")

    monkeypatch.setattr(runtime, "activate_checkpoint", fenced)
    monkeypatch.setattr(runtime, "shutdown", fenced)
    assert worker.run_once()
    item = control.coordination.served_models.current_request("math")
    assert item.state == "running" and item.error is None
    assert item.attempt_count == 1
    assert worker.owner.lease_lost.is_set() and worker.stopped.is_set()
    assert not worker.run_once()


@pytest.mark.unit
def test_an_owner_retries_recovery_before_opening_its_runtime(startup, monkeypatch):
    control, worker, runtime, _ = startup
    control.create_scenario("math")
    runtime.release.set()
    recover = worker.routes.recover_completed
    attempts = []

    def interrupted(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("coordination temporarily unavailable")
        return recover(*args, **kwargs)

    monkeypatch.setattr(worker.routes, "recover_completed", interrupted)
    with pytest.raises(OSError):
        worker.run_once()
    assert runtime.opens == 0
    assert worker.run_once()
    assert len(attempts) == 2 and runtime.opens == 1


@pytest.mark.unit
def test_a_stop_shuts_down_a_runtime_whose_open_failed(startup, monkeypatch):
    control, worker, runtime, _ = startup
    control.create_scenario("math")

    def refuse_open():
        raise RuntimeError("one engine group did not open")

    monkeypatch.setattr(runtime, "open", refuse_open)
    assert worker.run_once()
    # The failed open can have started engines, so the stop shuts down the runtime that never opened.
    assert not worker.opened and runtime.shutdowns == 0
    worker.stop()
    assert runtime.shutdowns == 1


@pytest.mark.unit
def test_an_idle_owner_releases_what_no_route_and_no_scheduled_stop_uses(startup, monkeypatch):
    control, worker, runtime, _ = startup
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    [published] = set(control.coordination.served_models.read_scenario("math").values())
    retiring = ServedArtifact("retiring", "http://inference-runtime", "sampler:retiring")
    control.coordination.served_models.schedule_stop(
        ScheduledStop(retiring, time.time() + 3600), control_lease_number=control.publisher.lease_number
    )
    calls = []

    def release(retained):
        calls.append(set(retained))
        if len(calls) == 1:
            raise RuntimeError("the engines did not pause")

    monkeypatch.setattr(runtime, "release_unreferenced_artifacts", release)
    # A serving work ran. The next round keeps the published route and the stop that is due in an hour, and a
    # release that fails runs again in the round after it.
    with pytest.raises(RuntimeError, match="did not pause"):
        worker.run_once()
    worker.run_once()
    assert calls == [{published, retiring}, {published, retiring}]
    # No serving work ran since the release, so an idle round releases nothing again.
    worker.run_once()
    assert len(calls) == 2 and runtime.stops == []


@pytest.mark.unit
def test_a_release_that_the_receiver_fences_stops_the_owner(startup, monkeypatch):
    control, worker, runtime, _ = startup
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()

    def fenced(retained):
        raise InferenceOwnerLostError("the receiver accepted another owner")

    monkeypatch.setattr(runtime, "release_unreferenced_artifacts", fenced)
    with pytest.raises(WorkLeaseLostError, match="newer owner"):
        worker.run_once()
    assert worker.owner.lease_lost.is_set() and worker.stopped.is_set() and runtime.shutdowns == 1
    assert not worker.run_once()


@pytest.mark.unit
def test_an_offline_training_runtime_has_no_startup_request(tmp_path):
    runtime = NullInferenceRuntime()
    recipe = TestPolicyRecipe()
    live = live_objects(StubTrainingComponent(tmp_path / "checkpoints"), runtime)
    coordination = memory_coordination()
    control = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        live=live,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=coordination,
        components=frozenset({"control"}),
    )
    try:
        control.create_scenario("math")
        request = coordination.served_models.current_request("math")
        assert request is not None and not request.payload
        assert coordination.served_models.read_scenario("math") == {}
    finally:
        control.close()


@contextmanager
def served_scenarios(tmp_path, recipe, live, *scenarios):
    """Create ``scenarios`` on a control dispatcher over ``recipe`` and ``live`` and run an inference worker until no
    serving work is left; yield the routes, the record storage and an inference app that answers each route with
    ``Answer``."""
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "records")
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=storage,
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "control",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    try:
        for scenario in scenarios:
            control.create_scenario(scenario)
        while worker.run_once():
            pass
        inference = build_inference_app(coordination.served_models, storage, FixedEngineClients(Answer()), tokens=None)
        yield coordination.served_models, storage, inference
    finally:
        worker.stop()
        control.close()


def chat(app, scenario):
    """The status, headers and JSON body of one chat request to ``scenario`` through ``app``."""

    async def send():
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/chat/completions", headers={"x-reef-scenario": scenario}, json=CHAT)
            return response.status, response.headers, await response.json()

    return asyncio.run(send())


@pytest.mark.unit
def test_a_proxy_routes_each_snapshot_without_weights_to_its_configured_model(tmp_path):
    """A record-only snapshot loads no weights, so it gets no activation: its route names the scenario's own selected
    artifact at the proxy's configured model. The proxy still binds no weights to any artifact."""
    runtime = InferenceProxyRuntime(base_url="http://provider.invalid", model_path="configured-model")
    live = LiveObjects(runtime=runtime)
    with served_scenarios(tmp_path, Recipe(), live, "math", "code") as (routes, storage, inference):
        selected = {scenario: routes.current_request(scenario).payload["artifact_id"] for scenario in ("math", "code")}
        assert selected["math"] != selected["code"]
        for scenario, artifact_id in selected.items():
            route = ServedArtifact(artifact_id, "http://provider.invalid", "configured-model")
            published = routes.read_scenario(scenario)
            assert published["active"] == route and set(published.values()) == {route}
            assert runtime.served_artifact(artifact_id, scenario) is None
        assert runtime.served_artifact("an-arbitrary-artifact", "math") is None
        status, headers, _ = chat(inference, "math")
        assert (status, headers[ARTIFACT_HEADER]) == (200, selected["math"])
        store = storage.open("math")
        try:
            [record] = store.records.replay("math")
        finally:
            store.close()
        assert (record.agent_record_id, record.artifact_id) == (headers[RECORD_HEADER], selected["math"])


@pytest.mark.unit
def test_a_proxy_refuses_a_recipe_that_loads_weights_even_with_a_configured_model(tmp_path):
    runtime = InferenceProxyRuntime(base_url="http://provider.invalid", model_path="configured-model")
    live = LiveObjects(runtime=runtime, training_facts=StubFacts(StubTrainingComponent()))
    with served_scenarios(tmp_path, TestPolicyRecipe(), live, "math") as (routes, _, inference):
        failed = routes.current_request("math")
        # The weights cannot load, and the configured model is no route for them.
        assert failed.state == "failed" and "an inference proxy loads no weights" in failed.error
        assert routes.read_scenario("math") == {}
        status, _, body = chat(inference, "math")
        assert (status, body["error"]["type"]) == (503, "inference_start_failed")


@pytest.mark.unit
def test_a_weight_runtime_gives_no_route_to_a_snapshot_without_weights(tmp_path):
    runtime = FullWeightServingRuntime(RunningEngines(Engines()), Answer(), model_path="test-model")
    with served_scenarios(tmp_path, Recipe(), LiveObjects(runtime=runtime), "math") as (routes, _, _):
        failed = routes.current_request("math")
        assert failed.state == "failed" and "provides no route for the selected artifact" in failed.error
        assert routes.read_scenario("math") == {}
    assert runtime.endpoint_model_name("math") is None


@pytest.mark.unit
def test_the_worker_logs_its_ownership_its_opening_and_each_activation(startup, caplog):
    control, worker, runtime = startup[0], startup[1], startup[2]
    runtime.release.set()
    control.create_scenario("math")
    with caplog.at_level("INFO", logger="reef.dispatcher.serving"):
        assert worker.run_once()
        worker.stop()
    messages = [record.getMessage() for record in caplog.records if record.name == "reef.dispatcher.serving"]
    assert any("holds runtime ownership" in message for message in messages)
    assert any("opened the inference runtime" in message for message in messages)
    assert any("kept 0 of 0 routes from before" in message for message in messages)
    assert any("scenario math: activated artifact" in message for message in messages)
    assert any("released runtime ownership" in message for message in messages)


class StoppableRuntime(PausedRuntime):
    """An engine whose start waits until the test releases it or the serving work must stop (RFC #1299)."""

    def __init__(self, slots: int = 1) -> None:
        super().__init__()
        self.slots = slots
        self.redeploys: list[bool] = []
        self.running = 0
        self.most_running = 0
        self.lock = threading.Lock()

    @property
    def serving_slots(self) -> int:
        return self.slots

    @property
    def shared_by_training_scenarios(self) -> bool:
        return True

    def activate_checkpoint(self, artifact, *, scenario=None):
        self.redeploys.append(redeploying())
        with self.lock:
            self.running += 1
            self.most_running = max(self.most_running, self.running)
        self.entered.set()
        try:
            while not self.release.wait(0.02):
                if stopping():
                    raise StartupStoppedError
        finally:
            with self.lock:
                self.running -= 1
        if self.fail_start:
            raise StartupFailedError("--language-model-only does not support this model")
        return StubInferenceRuntime.activate_checkpoint(self, artifact, scenario=scenario)


@pytest.fixture
def stoppable(tmp_path, monkeypatch, request):
    monkeypatch.setattr(serving, "REPLACED_CHECK_SECONDS", 0.05)
    runtime = StoppableRuntime(getattr(request, "param", 1))
    training = StubTrainingComponent(tmp_path / "checkpoints", shares_scenarios=True)
    recipe = TestPolicyRecipe()
    live = live_objects(training, runtime)
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "control",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    try:
        yield control, worker, runtime
    finally:
        runtime.release.set()
        worker.stop()
        control.close()


@pytest.mark.unit
def test_a_redeploy_stops_the_held_start_and_the_next_control_round_keeps_it(stoppable):
    control, worker, runtime = stoppable
    control.create_scenario("math")
    routes = control.coordination.served_models
    first = routes.current_request("math")
    held = threading.Thread(target=worker.run_once)
    held.start()
    assert runtime.entered.wait(2)
    control.redeploy("math")
    redeploy = routes.current_request("math")
    assert redeploy.id != first.id and REDEPLOY in redeploy.payload
    # The held start stops within seconds, not when the engine would have started.
    held.join(5)
    assert not held.is_alive()
    stopped = control.coordination.work_queue.read(SERVING_QUEUE, first.id)
    assert (stopped.state, stopped.error) == ("failed", serving.REPLACED)
    # A control round compares the payloads without the redeploy number, so it keeps the redeploy request.
    control.publisher.run_once()
    assert routes.current_request("math").id == redeploy.id
    runtime.release.set()
    assert worker.run_once()
    assert runtime.redeploys == [False, True]
    assert routes.current_request("math").state == "finished" and routes.read_scenario("math")


@pytest.mark.unit
@pytest.mark.parametrize("stoppable", [1, 2], indirect=True)
def test_the_runtime_s_slots_start_that_many_scenarios_at_once(stoppable):
    control, worker, runtime = stoppable
    control.create_scenario("math")
    control.create_scenario("code")
    worker.start()
    deadline = time.monotonic() + 5
    while runtime.most_running < runtime.slots and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.2)
    assert runtime.most_running == runtime.slots
    runtime.release.set()
    routes = control.coordination.served_models
    while time.monotonic() < deadline and not all(routes.read_scenario(name) for name in ("math", "code")):
        time.sleep(0.02)
    assert all(routes.read_scenario(name) for name in ("math", "code"))


@pytest.mark.unit
def test_the_status_gives_the_message_of_a_startup_failure(stoppable, tmp_path):
    control, worker, runtime = stoppable
    runtime.fail_start = True
    runtime.release.set()
    control.create_scenario("math")
    assert worker.run_once()
    inference = build_inference_app(
        control.coordination.served_models,
        SQLiteScenarioStorage(tmp_path / "records"),
        FixedEngineClients(Answer()),
        tokens=None,
    )

    async def run() -> None:
        async with TestClient(TestServer(inference)) as client:
            status = await (await client.get(STATUS)).json()
            assert (status["state"], status["error"]) == (
                "failed",
                "--language-model-only does not support this model",
            )

    asyncio.run(run())


def stop_all_serving(control, worker, runtime):
    """Serve scenario ``math``, then stop all its serving; return the artifact and the endpoint that served it."""
    control.create_scenario("math")
    runtime.release.set()
    assert worker.run_once()
    artifact_id = control.get_or_create_scenario("math").serving_artifact_id()
    [served] = set(control.coordination.served_models.read_scenario("math").values())
    assert control.set_served("math", {}) == {"served": {}}
    return artifact_id, served


@pytest.mark.unit
def test_a_redeploy_of_a_scenario_that_serves_nothing_starts_no_engine(startup):
    control, worker, runtime, _ = startup
    stop_all_serving(control, worker, runtime)
    routes = control.coordination.served_models
    withdrawn = routes.current_request("math")
    assert withdrawn.state == "finished" and not withdrawn.payload

    control.redeploy("math")
    control.publisher.run_once()
    assert routes.current_request("math").id == withdrawn.id
    # The worker has no serving work, and the runtime activated only the first start.
    assert not worker.run_once()
    assert len(runtime.activated) == 1 and runtime.opens == 1
    assert routes.read_scenario("math") == {}
    assert control.scenario_settings("math")["served"] == {"weights": {}}


def deployment(tmp_path, runtime):
    """Control and an inference worker on the record and coordination files of ``tmp_path``, as one start of the
    service has them."""
    recipe = TestPolicyRecipe()
    live = live_objects(StubTrainingComponent(tmp_path / "checkpoints"), runtime)
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = open_coordination("sqlite", agent_record_dir=str(tmp_path / "records"), record_database_url=None)
    control = Dispatcher(
        recipe,
        backends,
        live=live,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=coordination,
        components=frozenset({"control"}),
        local_artifact_dir=tmp_path / "control",
    )
    worker = ServingWorker(
        recipe, live, backends, coordination, worker_id="inference", local_artifact_dir=tmp_path / "inference"
    )
    return control, worker


@pytest.mark.unit
def test_a_restart_of_a_scenario_that_serves_nothing_starts_no_engine(tmp_path):
    runtime = PausedRuntime()
    control, worker = deployment(tmp_path, runtime)
    try:
        _, served = stop_all_serving(control, worker, runtime)
        withdrawn = control.coordination.served_models.current_request("math")
    finally:
        worker.stop()
        control.close()

    restarted_runtime = PausedRuntime()
    restarted_runtime.release.set()
    restarted, restarted_worker = deployment(tmp_path, restarted_runtime)
    try:
        routes = restarted.coordination.served_models
        assert not restarted.get_or_create_scenario("math").serves
        restarted.publisher.run_once()
        assert not restarted_worker.run_once()
        restarted.redeploy("math")
        assert not restarted_worker.run_once()
        assert routes.current_request("math").id == withdrawn.id
        assert restarted_runtime.opens == 0 and restarted_runtime.activated == []

        # The stop of the endpoint from before the restart comes due: the new runtime stops it and starts nothing.
        routes.schedule_stop(ScheduledStop(served, 0), control_lease_number=restarted.publisher.lease_number)
        assert restarted_worker.run_once()
        assert restarted_runtime.stops == [served] and restarted_runtime.activated == []
        assert routes.due_stops(time.time()) == ()
        restarted.publisher.run_once()
        assert not restarted_worker.run_once()
        assert routes.current_request("math").id == withdrawn.id
        assert routes.read_scenario("math") == {}
    finally:
        restarted_worker.stop()
        restarted.close()


@pytest.mark.unit
def test_a_scenario_that_stopped_all_serving_activates_the_same_artifact_again(startup):
    control, worker, runtime, _ = startup
    artifact_id, served = stop_all_serving(control, worker, runtime)
    routes = control.coordination.served_models

    # Before the stop of its endpoint is due, the engine still serves the artifact: the route comes back, and the
    # runtime loads nothing.
    assert control.set_served("math", {artifact_id: 100}) == {"served": {artifact_id: 100}}
    assert worker.run_once()
    assert routes.read_scenario("math")["active"] == served
    assert len(runtime.activated) == 1 and runtime.stops == []

    # After the runtime stopped the endpoint, the same artifact needs a new activation.
    control.set_served("math", {})
    routes.schedule_stop(ScheduledStop(served, 0), control_lease_number=control.publisher.lease_number)
    assert worker.run_once()
    assert runtime.stops == [served]
    control.set_served("math", {artifact_id: 100})
    assert worker.run_once()
    assert len(runtime.activated) == 2
    assert routes.read_scenario("math")["active"] == served


@pytest.mark.unit
def test_checkpoint_slots_keep_a_withdrawn_checkpoint_until_its_stop_and_then_load_it_again(tmp_path):
    class SlotEngines(RunningEngines):
        """Running engines at ``url``: each checkpoint slot has its own base URL."""

        def __init__(self, url):
            super().__init__(Engines())
            self.url = url

        @property
        def base_url(self):
            return self.url

    first, second = SlotEngines("http://slot-a.example"), SlotEngines("http://slot-b.example")
    runtime = CheckpointSlotsRuntime(
        [FullWeightServingRuntime(engines, Answer(), model_path="test-model") for engines in (first, second)]
    )
    control, worker = deployment(tmp_path, runtime)
    routes = control.coordination.served_models
    try:
        control.create_scenario("math")
        assert worker.run_once()
        artifact_id = control.get_or_create_scenario("math").serving_artifact_id()
        [served] = set(routes.read_scenario("math").values())
        assert runtime.bindings == {artifact_id: served} and served.url == "http://slot-a.example"

        # The scenario serves nothing. Its withdrawn route waits for its stop, so the release of what no route uses
        # runs and keeps the checkpoint in its slot.
        assert control.set_served("math", {}) == {"served": {}}
        assert not worker.run_once()
        assert not worker.release_due and runtime.bindings == {artifact_id: served}

        # The stop releases the slot, and the worker forgets its activation. The same artifact then loads again
        # into the free slot, under the route of the new load.
        routes.schedule_stop(ScheduledStop(served, 0), control_lease_number=control.publisher.lease_number)
        assert worker.run_once()
        assert runtime.bindings == {} and worker.activated == {} and first.backend.events[-1] == "pause"
        control.set_served("math", {artifact_id: 100})
        assert worker.run_once()
        restored = routes.read_scenario("math")["active"]
        assert restored == runtime.bindings[artifact_id] and restored != served and restored.url == served.url
        assert len(first.backend.loads()) == 2 and first.backend.events[-1] == "resume"
        assert second.backend.loads() == []
    finally:
        worker.stop()
        control.close()
