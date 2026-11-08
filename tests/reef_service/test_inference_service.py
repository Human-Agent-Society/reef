"""The inference service: requests answered from ``served_models`` and recorded in the shared record store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.core.records_types import RequestType
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import EngineClients, InferenceHandler
from reef.service.inference_service import ARTIFACT_HEADER, RECORD_HEADER, build_inference_app
from reef.storage.commits import CommitRecord
from reef.storage.served_models import SERVING_KIND, SERVING_QUEUE
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

TOKEN = "service-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "x-reef-scenario": "math", "x-reef-tag-session": "s1"}


def engine_app() -> web.Application:
    """An engine that answers as the model name of the request."""
    app = web.Application()

    async def complete(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response(
            {"model": body["model"], "choices": [{"message": {"role": "assistant", "content": "4"}}]}
        )

    app.router.add_post("/v1/chat/completions", complete)
    return app


class Engines(EngineClients):
    """An HTTP handler for each engine URL."""

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        return HttpInferenceHandler(served.url)


def commit(release_id: str) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=1,
        snapshot=SnapshotRef(release_id, "base"),
        checkpoint=True,
        algorithm_state={},
        consumed_ids=frozenset(),
        high_water_sequence=0,
        high_water_offset=0,
        recorded_at=1000.0,
    )


def inference_records(records: Any) -> tuple[Any, ...]:
    return tuple(item for item in records.replay("math") if item.request_type is RequestType.INFERENCE)


async def ask(tmp_path: Path, *, write: bool = True) -> tuple[int, dict[str, Any], Any]:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    store = storage.open("math")
    store.commit_step(expected_step=0, commit=commit("release-1"))
    coordination = memory_coordination()
    table = coordination.served_models
    number = coordination.control_lease.acquire("control-1").number
    async with TestServer(engine_app()) as engine:
        if write:
            served = ServedArtifact("release-1", str(engine.make_url("/")), "math-v1")
            table.write(ScenarioModel("math", ACTIVE), served, control_lease_number=number)
        app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions", headers=HEADERS, json={"model": "any", "messages": [{"role": "user"}]}
            )
            return response.status, dict(response.headers), store.records


@pytest.mark.unit
def test_a_request_goes_to_the_served_artifact_and_its_record_joins_the_record_store(tmp_path: Path) -> None:
    status, headers, records = asyncio.run(ask(tmp_path))
    assert status == 200
    assert headers[ARTIFACT_HEADER] == "release-1"
    (item,) = inference_records(records)
    assert item.agent_record_id == headers[RECORD_HEADER]
    assert item.artifact_id == "release-1"
    assert item.payload["metadata"]["tags"] == {"session": "s1"}
    assert item.payload["response"]["model"] == "math-v1"


@pytest.mark.unit
def test_a_scenario_without_a_served_model_is_not_found(tmp_path: Path) -> None:
    status, _, records = asyncio.run(ask(tmp_path, write=False))
    assert status == 404
    assert inference_records(records) == ()


class BrokenStorage(SQLiteScenarioStorage):
    """A record store that refuses every record."""

    def open(self, scenario: str):
        store = super().open(scenario)
        if self.refuse:

            def refuse(item):
                raise OSError("the disk is full")

            store.records.append = refuse
        return store


async def ask_failing(tmp_path: Path, *, engine_url: str | None, refuse_records: bool) -> int:
    storage = BrokenStorage(tmp_path / "records")
    storage.refuse = False
    storage.open("math").commit_step(expected_step=0, commit=commit("release-1"))
    storage.refuse = refuse_records
    coordination = memory_coordination()
    table = coordination.served_models
    number = coordination.control_lease.acquire("control-1").number
    async with TestServer(engine_app()) as engine:
        url = engine_url or str(engine.make_url("/"))
        table.write(
            ScenarioModel("math", ACTIVE), ServedArtifact("release-1", url, "math-v1"), control_lease_number=number
        )
        async with TestClient(TestServer(build_inference_app(table, storage, Engines(), tokens=TOKEN))) as client:
            response = await client.post(
                "/v1/chat/completions", headers=HEADERS, json={"model": "any", "messages": [{"role": "user"}]}
            )
            return response.status


@pytest.mark.unit
def test_an_answer_without_its_record_is_never_returned_and_a_dead_engine_is_a_bad_gateway(tmp_path: Path) -> None:
    assert asyncio.run(ask_failing(tmp_path / "records", engine_url=None, refuse_records=True)) == 503
    assert asyncio.run(ask_failing(tmp_path / "engine", engine_url="http://127.0.0.1:9", refuse_records=False)) == 502


@pytest.mark.unit
def test_startup_status_follows_work_and_a_retry_remains_starting(tmp_path: Path) -> None:
    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")
        table, queue = coordination.served_models, coordination.work_queue
        lease = coordination.control_lease.acquire("control")
        item = table.request(
            "math",
            {
                "artifact_id": "release-1",
                "aliases": ["release-1:weights"],
                "beside": {"release-0": {"component": "weights", "artifact_id": "release-0"}},
            },
            control_lease_number=lease.number,
        )
        app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                path = "/reef/scenarios/math/inference/status"
                assert (await client.get(path)).status == 401
                before = table.current_request("math")
                response = await client.get(path, headers=HEADERS)
                assert await response.json() == {
                    "artifact_id": "release-1",
                    "state": "pending",
                    "error": None,
                    "engine": None,
                    "engine_detail": None,
                }
                assert response.headers["Cache-Control"] == "no-store"
                assert table.current_request("math") == before
                alias = await client.get(path + "?model=release-1:weights", headers=HEADERS)
                assert (await alias.json())["artifact_id"] == "release-1"
                beside = await client.get(path + "?model=release-0", headers=HEADERS)
                assert (await beside.json())["artifact_id"] == "release-0"
                assert (await client.get(path + "?model=unknown", headers=HEADERS)).status == 404

                response = await client.post("/v1/chat/completions", headers=HEADERS, json={"model": "any"})
                assert response.status == 503
                assert response.headers["Retry-After"] == "30"
                assert (await response.json())["error"]["type"] == "inference_starting"

                taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
                assert taken.id == item.id
                assert (await (await client.get(path, headers=HEADERS)).json())["state"] == "starting"
                queue.start(SERVING_QUEUE, taken.id, "worker")
                queue.fail(SERVING_QUEUE, taken.id, "worker", "temporary failure", retryable=True)
                assert table.current_request("math").state == "queued"
                assert (await (await client.get(path, headers=HEADERS)).json())["state"] == "starting"

                taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
                queue.start(SERVING_QUEUE, taken.id, "worker")
                table.complete_request(
                    taken,
                    "worker",
                    {ACTIVE: ServedArtifact("release-1", "http://engine", "math")},
                    serving_config={},
                    stop_delay_seconds=60,
                )
                response = await client.get(path, headers=HEADERS)
                # The plain HTTP handler cannot tell its engine's state.
                assert await response.json() == {
                    "artifact_id": "release-1",
                    "state": "ready",
                    "error": None,
                    "engine": None,
                    "engine_detail": None,
                }
        finally:
            storage.close()
            coordination.close()

    asyncio.run(check())


@pytest.mark.unit
def test_startup_failure_is_safe_and_withdrawal_is_not_starting(tmp_path: Path) -> None:
    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")
        table, queue = coordination.served_models, coordination.work_queue
        lease = coordination.control_lease.acquire("control")
        table.request("math", {"artifact_id": "release-1"}, control_lease_number=lease.number)
        taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
        queue.fail(SERVING_QUEUE, taken.id, "worker", "private startup diagnostic", retryable=False)
        app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                status = await response.json()
                assert status == {
                    "artifact_id": "release-1",
                    "state": "failed",
                    "error": "The inference engine did not start. Check the service logs.",
                    "engine": None,
                    "engine_detail": None,
                }
                response = await client.post("/v1/chat/completions", headers=HEADERS, json={"model": "any"})
                assert response.status == 503
                assert "Retry-After" not in response.headers
                assert (await response.json())["error"]["type"] == "inference_start_failed"
                table.request("math", {}, control_lease_number=lease.number)
                assert (await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)).status == 404
                assert (await client.get("/reef/scenarios/unknown/inference/status", headers=HEADERS)).status == 404
                response = await client.post("/v1/chat/completions", headers=HEADERS, json={"model": "any"})
                assert response.status == 404
        finally:
            storage.close()
            coordination.close()

    asyncio.run(check())


@pytest.mark.unit
def test_startup_status_does_not_guess_readiness_when_the_store_fails(tmp_path: Path, monkeypatch) -> None:
    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")

        def unavailable(scenario: str):
            raise OSError("private database diagnostic")

        monkeypatch.setattr(coordination.served_models, "current_request", unavailable)
        app = build_inference_app(coordination.served_models, storage, Engines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                assert response.status == 503
                assert await response.text() == "Inference startup status is unavailable."
        finally:
            storage.close()
            coordination.close()

    asyncio.run(check())


@pytest.mark.unit
def test_a_ready_old_route_does_not_make_its_replacement_ready(tmp_path: Path) -> None:
    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")
        table = coordination.served_models
        lease = coordination.control_lease.acquire("control")
        try:
            async with TestServer(engine_app()) as engine:
                table.write(
                    ScenarioModel("math", ACTIVE),
                    ServedArtifact("release-1", str(engine.make_url("/")), "math-v1"),
                    control_lease_number=lease.number,
                )
                table.request("math", {"artifact_id": "release-2"}, control_lease_number=lease.number)
                app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
                async with TestClient(TestServer(app)) as client:
                    response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                    assert await response.json() == {
                        "artifact_id": "release-2",
                        "state": "pending",
                        "error": None,
                        "engine": None,
                        "engine_detail": None,
                    }
                    response = await client.post("/v1/chat/completions", headers=HEADERS, json={"model": "any"})
                    assert response.status == 200
                    assert response.headers[ARTIFACT_HEADER] == "release-1"
        finally:
            storage.close()
            coordination.close()

    asyncio.run(check())


def test_inference_records_do_not_need_snapshot_history(tmp_path):
    # The serving row alone identifies the responding artifact, including before
    # the first training commit and after older operations have been pruned.
    storage = SQLiteScenarioStorage(tmp_path / "records")
    coordination = memory_coordination()
    try:
        from reef.service.inference_service import InferenceReplica, ServedModelsCache
        from reef.service.wire import RequestHeaders

        replica = InferenceReplica(ServedModelsCache(coordination.served_models), storage, Engines())
        receipt = replica.record(
            RequestHeaders(scenario="math", request_type=RequestType.INFERENCE),
            {"model": "reef", "artifact_id": "model-artifact"},
            {"choices": []},
            "model-artifact",
        )
        store = storage.open("math")
        try:
            assert store.history() == ()
            assert store.records.get("math", receipt).artifact_id == "model-artifact"
        finally:
            store.close()
    finally:
        storage.close()
        coordination.close()


class ColdHandler(HttpInferenceHandler):
    """A handler whose engine scales to zero: it reports a state and never calls the engine."""

    state: Any = "idle"

    def engine_state(self, served: ServedArtifact) -> Any:
        return self.state

    def engine_detail(self, served: ServedArtifact) -> str:
        return "Loading weights · 0:42"


@pytest.mark.unit
def test_a_ready_status_names_the_engine_state_that_its_handler_reports(tmp_path: Path) -> None:
    """RFC #1269: a ready artifact's status names its engine's state, and the read asks the handler only."""

    class ColdEngines(EngineClients):
        def handler(self, served: ServedArtifact) -> InferenceHandler:
            return ColdHandler(served.url)

    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")
        table, queue = coordination.served_models, coordination.work_queue
        lease = coordination.control_lease.acquire("control")
        table.request("math", {"artifact_id": "release-1"}, control_lease_number=lease.number)
        taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
        queue.start(SERVING_QUEUE, taken.id, "worker")
        served = ServedArtifact("release-1", "http://127.0.0.1:9", "math")
        table.complete_request(taken, "worker", {ACTIVE: served}, serving_config={}, stop_delay_seconds=60)
        app = build_inference_app(table, storage, ColdEngines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                for state in ("idle", "starting", "warm"):
                    ColdHandler.state = state
                    response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                    status = await response.json()
                    assert status["engine"] == state
                    # Only a starting engine says what it does now.
                    assert status["engine_detail"] == ("Loading weights · 0:42" if state == "starting" else None)
        finally:
            ColdHandler.state = "idle"
            storage.close()
            coordination.close()

    asyncio.run(check())
