"""Inference startup runs independently of control and reports the selected artifact's result."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.core.served_models import ScheduledStop
from reef.dispatcher import Dispatcher
from reef.dispatcher.serving import ServingWorker
from reef.inference.interfaces import FixedEngineClients, InferenceHandler
from reef.inference.null import NullInferenceRuntime
from reef.service.app import create_app
from reef.service.inference_service import build_inference_app
from reef.storage.served_models import SERVING_KIND, SERVING_QUEUE
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.work_queue import WorkLeaseLost

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

    def open(self) -> None:
        self.opens += 1

    def shutdown(self) -> None:
        self.shutdowns += 1

    def stop_serving(self, served) -> None:
        assert self.opens > 0
        self.stops.append(served)

    def activate_checkpoint(self, artifact, *, scenario=None):
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("the test did not release startup")
        if self.fail_start:
            raise ValueError("the selected model cannot load")
        return super().activate_checkpoint(artifact, scenario=scenario)

    def served_model_name(self, scenario: str) -> str:
        return "test-model"


class Answer(InferenceHandler):
    async def inference(self, artifact, path, payload):
        return {"choices": [{"message": {"content": "ready"}}]}


@pytest.fixture
def startup(tmp_path):
    runtime = PausedRuntime()
    recipe = TestPolicyRecipe(**runtime_bindings(StubTrainingComponent(tmp_path / "checkpoints"), runtime))
    backends = InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")
    coordination = memory_coordination()
    storage = SQLiteScenarioStorage(tmp_path / "records")
    control = Dispatcher(
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
    inference = build_inference_app(coordination.served_models, storage, FixedEngineClients(Answer()), tokens=None)
    try:
        yield control, worker, runtime, inference
    finally:
        runtime.release.set()
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
                assert status == {**pending, "state": "starting"}
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
            with pytest.raises(WorkLeaseLost):
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
    recipe = TestPolicyRecipe(**runtime_bindings(StubTrainingComponent(tmp_path / "replacement"), replacement))
    standby = ServingWorker(recipe, worker.backends, control.coordination, worker_id="standby")

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
    recipe = TestPolicyRecipe(**runtime_bindings(StubTrainingComponent(tmp_path / "replacement"), replacement))
    standby = ServingWorker(recipe, worker.backends, control.coordination, worker_id="standby")
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
def test_an_offline_training_runtime_has_no_startup_request(tmp_path):
    runtime = NullInferenceRuntime()
    recipe = TestPolicyRecipe(**runtime_bindings(StubTrainingComponent(tmp_path / "checkpoints"), runtime))
    coordination = memory_coordination()
    control = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
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
