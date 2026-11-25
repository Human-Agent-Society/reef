"""The inference service: requests answered from ``served_models`` and recorded in the shared record store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.api.inference_service import (
    ARTIFACT_HEADER,
    ATTEMPT_HEADER,
    BINDING_HEADER,
    RECORD_HEADER,
    build_inference_app,
)
from reef.core.records_types import RequestType
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.inference.http import HttpInferenceHandler
from reef.inference.interfaces import EngineClients, EngineReplicas, InferenceHandler, StartupStep, StartupSteps
from reef.storage.coordination.served_models import SERVING_KIND, SERVING_QUEUE
from reef.storage.scenario.commits import CommitRecord
from reef.storage.sql.coordination import memory_coordination
from reef.storage.sql.sqlite import SQLiteScenarioStorage

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


def commit(revision: str) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        commit_number=1,
        snapshot=SnapshotRef(revision, "base"),
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
    store.append_commit(expected_commit_number=0, commit=commit("artifact-1"))
    coordination = memory_coordination()
    table = coordination.served_models
    number = coordination.control_lease.acquire("control-1").number
    async with TestServer(engine_app()) as engine:
        if write:
            served = ServedArtifact("artifact-1", str(engine.make_url("/")), "math-v1")
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
    assert headers[ARTIFACT_HEADER] == "artifact-1"
    (item,) = inference_records(records)
    assert item.agent_record_id == headers[RECORD_HEADER]
    assert item.artifact_id == "artifact-1"
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
    storage.open("math").append_commit(expected_commit_number=0, commit=commit("artifact-1"))
    storage.refuse = refuse_records
    coordination = memory_coordination()
    table = coordination.served_models
    number = coordination.control_lease.acquire("control-1").number
    async with TestServer(engine_app()) as engine:
        url = engine_url or str(engine.make_url("/"))
        table.write(
            ScenarioModel("math", ACTIVE), ServedArtifact("artifact-1", url, "math-v1"), control_lease_number=number
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


#: An engine's refusal of a prompt that fills the model's context window, in Tinker's words.
CONTEXT_ERROR = (
    "context_length_exceeded: the prompt exceeds this model's maximum context length. "
    "66527 prompt tokens fill the context window of 65536"
)


def refusing_engine_app(status: int, message: str) -> web.Application:
    """An engine that answers every request with ``status`` and the error ``message``."""
    app = web.Application()

    async def refuse(request: web.Request) -> web.Response:
        return web.json_response({"error": {"message": message}}, status=status)

    app.router.add_post("/v1/chat/completions", refuse)
    return app


async def ask_refused(tmp_path: Path, status: int, message: str, *, stream: bool) -> tuple[int, str]:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    storage.open("math").append_commit(expected_commit_number=0, commit=commit("artifact-1"))
    coordination = memory_coordination()
    table = coordination.served_models
    number = coordination.control_lease.acquire("control-1").number
    async with TestServer(refusing_engine_app(status, message)) as engine:
        served = ServedArtifact("artifact-1", str(engine.make_url("/")), "math-v1")
        table.write(ScenarioModel("math", ACTIVE), served, control_lease_number=number)
        async with TestClient(TestServer(build_inference_app(table, storage, Engines(), tokens=TOKEN))) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers=HEADERS,
                json={"model": "any", "messages": [{"role": "user"}], "stream": stream},
            )
            return response.status, await response.text()


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_an_engine_that_refuses_a_request_answers_with_its_own_status_and_one_that_fails_is_a_bad_gateway(
    tmp_path: Path, stream: bool
) -> None:
    # An agent reads the refusal to repair its run: Harbor's terminus-2 ends or compacts on a context error.
    status, body = asyncio.run(ask_refused(tmp_path / "refused", 400, CONTEXT_ERROR, stream=stream))
    assert status == 400 and "context_length_exceeded" in body
    status, body = asyncio.run(ask_refused(tmp_path / "failed", 500, "the engine ran out of memory", stream=stream))
    assert status == 502 and "the engine ran out of memory" in body


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
                "artifact_id": "artifact-1",
                "aliases": ["artifact-1:weights"],
                "beside": {"artifact-0": {"component": "weights", "artifact_id": "artifact-0"}},
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
                    "artifact_id": "artifact-1",
                    "state": "pending",
                    "error": None,
                    "engine": None,
                    "engine_detail": None,
                    "engine_replicas": None,
                    # RFC #1283: a startup that waits is in its prepare phase from the request.
                    "startup": {
                        "steps": [
                            {"phase": "prepare", "started_at": before.created_at, "ended_at": None, "detail": None}
                        ],
                        "estimate_seconds": None,
                    },
                }
                assert response.headers["Cache-Control"] == "no-store"
                assert table.current_request("math") == before
                alias = await client.get(path + "?model=artifact-1:weights", headers=HEADERS)
                assert (await alias.json())["artifact_id"] == "artifact-1"
                beside = await client.get(path + "?model=artifact-0", headers=HEADERS)
                assert (await beside.json())["artifact_id"] == "artifact-0"
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
                    {ACTIVE: ServedArtifact("artifact-1", "http://engine", "math")},
                    serving_config={},
                    stop_delay_seconds=60,
                )
                response = await client.get(path, headers=HEADERS)
                # The plain HTTP handler cannot tell its engine's state.
                assert await response.json() == {
                    "artifact_id": "artifact-1",
                    "state": "ready",
                    "error": None,
                    "engine": None,
                    "engine_detail": None,
                    "engine_replicas": None,
                    "startup": None,
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
        table.request("math", {"artifact_id": "artifact-1"}, control_lease_number=lease.number)
        taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
        queue.fail(SERVING_QUEUE, taken.id, "worker", "private startup diagnostic", retryable=False)
        app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                status = await response.json()
                assert status == {
                    "artifact_id": "artifact-1",
                    "state": "failed",
                    "error": "The inference engine did not start. Check the service logs.",
                    "engine": None,
                    "engine_detail": None,
                    "engine_replicas": None,
                    "startup": None,
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
def test_an_artifact_that_serves_already_stays_ready_while_another_one_starts_or_fails(tmp_path: Path) -> None:
    """One request serves the active artifact and each beside one: the start of a beside artifact, and its failure,
    do not change the state of the active artifact that a route serves already."""

    async def check() -> None:
        coordination = memory_coordination()
        storage = SQLiteScenarioStorage(tmp_path / "records")
        table, queue = coordination.served_models, coordination.work_queue
        lease = coordination.control_lease.acquire("control")
        table.write(
            ScenarioModel("math", ACTIVE),
            ServedArtifact("artifact-1", "http://127.0.0.1:9", "math-v1"),
            control_lease_number=lease.number,
        )
        payload = {
            "artifact_id": "artifact-1",
            "aliases": [],
            "beside": {"artifact-0": {"component": "weights", "artifact_id": "artifact-0"}},
        }
        table.request("math", payload, control_lease_number=lease.number)
        app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
        try:
            async with TestClient(TestServer(app)) as client:
                path = "/reef/scenarios/math/inference/status"

                async def states() -> tuple[str, str]:
                    active = await (await client.get(path, headers=HEADERS)).json()
                    beside = await (await client.get(path + "?model=artifact-0", headers=HEADERS)).json()
                    return active["state"], beside["state"]

                assert await states() == ("ready", "pending")
                taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
                queue.start(SERVING_QUEUE, taken.id, "worker")
                assert await states() == ("ready", "starting")
                queue.fail(SERVING_QUEUE, taken.id, "worker", "the beside engine did not start", retryable=False)
                assert await states() == ("ready", "failed")
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
                    ServedArtifact("artifact-1", str(engine.make_url("/")), "math-v1"),
                    control_lease_number=lease.number,
                )
                table.request("math", {"artifact_id": "artifact-2"}, control_lease_number=lease.number)
                app = build_inference_app(table, storage, Engines(), tokens=TOKEN)
                async with TestClient(TestServer(app)) as client:
                    response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                    created_at = table.current_request("math").created_at
                    prepare = {"phase": "prepare", "started_at": created_at, "ended_at": None, "detail": None}
                    assert await response.json() == {
                        "artifact_id": "artifact-2",
                        "state": "pending",
                        "error": None,
                        "engine": None,
                        "engine_detail": None,
                        "engine_replicas": None,
                        "startup": {"steps": [prepare], "estimate_seconds": None},
                    }
                    response = await client.post("/v1/chat/completions", headers=HEADERS, json={"model": "any"})
                    assert response.status == 200
                    assert response.headers[ARTIFACT_HEADER] == "artifact-1"
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
        from reef.api.inference_service import InferenceReplica, ServedModelsCache
        from reef.api.wire import RequestHeaders

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
            assert store.records.read("math", receipt).artifact_id == "model-artifact"
        finally:
            store.close()
    finally:
        storage.close()
        coordination.close()


class ColdHandler(HttpInferenceHandler):
    """A handler whose engine scales to zero: it reports a state and never calls the engine."""

    state: Any = "idle"
    adding = False

    def engine_state(self, served: ServedArtifact) -> Any:
        return self.state

    def engine_detail(self, served: ServedArtifact) -> str:
        return "Loading weights · 0:42"

    def engine_replicas(self, served: ServedArtifact) -> EngineReplicas:
        return EngineReplicas(serving=int(self.state == "warm"), starting=int(self.state == "starting" or self.adding))

    def engine_steps(self, served: ServedArtifact) -> StartupSteps:
        return StartupSteps((StartupStep("start", 100.0, 108.0), StartupStep("load", 108.0, detail="5 of 6")), 60.0)


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
        table.request("math", {"artifact_id": "artifact-1"}, control_lease_number=lease.number)
        taken = queue.take(SERVING_QUEUE, "worker", [SERVING_KIND])
        queue.start(SERVING_QUEUE, taken.id, "worker")
        served = ServedArtifact("artifact-1", "http://127.0.0.1:9", "math")
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
                    # The replicas come in every engine state: a warm engine can add replicas.
                    assert status["engine_replicas"] == {
                        "serving": int(state == "warm"),
                        "starting": int(state == "starting"),
                    }
                    # RFC #1283: and its phases, with their times.
                    load = {"phase": "load", "started_at": 108.0, "ended_at": None, "detail": "5 of 6"}
                    start = {"phase": "start", "started_at": 100.0, "ended_at": 108.0, "detail": None}
                    expected = {"steps": [start, load], "estimate_seconds": 60.0}
                    assert status["startup"] == (expected if state == "starting" else None)
                # A warm engine that adds a replica gives the replica's step and phases.
                ColdHandler.adding = True
                response = await client.get("/reef/scenarios/math/inference/status", headers=HEADERS)
                status = await response.json()
                assert (status["engine"], status["engine_detail"], status["engine_replicas"]) == (
                    "warm",
                    "Loading weights · 0:42",
                    {"serving": 1, "starting": 1},
                )
                assert status["startup"] == expected
        finally:
            ColdHandler.state, ColdHandler.adding = "idle", False
            storage.close()
            coordination.close()

    asyncio.run(check())


@pytest.mark.unit
@pytest.mark.parametrize(
    "change",
    [
        "none",
        "replaced",
        "removed",
        "scenario",
        "missing-header",
        "stream",
        "route",
        "config",
        "database",
        "changing",
        "invalid-attempt",
        "protocol",
    ],
)
def test_bound_chat_checks_target_before_execution(
    tmp_path: Path, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def check() -> None:
        calls = []

        async def complete(request: web.Request) -> web.Response:
            calls.append(await request.json())
            return web.json_response(
                {"choices": [{"message": {"content": "4"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
            )

        engine = web.Application()
        engine.router.add_post("/v1/chat/completions", complete)
        coordination = memory_coordination()
        table = coordination.served_models
        lease = coordination.control_lease.acquire("control").number
        storage = SQLiteScenarioStorage(tmp_path / "records")
        storage.open("math").append_commit(expected_commit_number=0, commit=commit("artifact-1"))
        async with TestServer(engine) as upstream:
            served = ServedArtifact("artifact-1", str(upstream.make_url("/")), "load-1")
            for scenario in ("math", "other"):
                for name in (ACTIVE, "artifact-1"):
                    table.write(ScenarioModel(scenario, name), served, control_lease_number=lease)
            async with TestClient(TestServer(build_inference_app(table, storage, Engines(), tokens=TOKEN))) as client:
                target_path = "/reef/scenarios/math/inference/target"
                assert (await client.get(target_path)).status == 401
                assert (await client.get(target_path + "?model=missing", headers=HEADERS)).status == 404
                selected = await client.get(target_path + "?model=artifact-1", headers=HEADERS)
                target = await selected.json()
                assert selected.headers["Cache-Control"] == "no-store"
                assert target == {
                    "contract_version": 1,
                    "model": "artifact-1",
                    "artifact_id": "artifact-1",
                    "binding_id": target["binding_id"],
                    "protocol": "openai_chat_completions",
                    "streaming": False,
                }
                assert len(target["binding_id"]) == 64
                assert not calls
                headers = {**HEADERS, BINDING_HEADER: target["binding_id"], ATTEMPT_HEADER: "attempt-1"}
                path = "/reef/scenarios/math/artifacts/artifact-1/v1/chat/completions/attempt"
                body = {"model": "ignored", "messages": [{"role": "user", "content": "2+2"}]}
                expected = 200
                if change == "replaced":
                    table.write(
                        ScenarioModel("math", "artifact-1"),
                        ServedArtifact("artifact-1", served.url, "load-2"),
                        control_lease_number=lease,
                    )
                    expected = 409
                elif change == "removed":
                    table.delete(ScenarioModel("math", "artifact-1"), control_lease_number=lease)
                    expected = 404
                elif change == "scenario":
                    path = path.replace("/math/", "/other/")
                    expected = 409
                elif change == "missing-header":
                    headers.pop(ATTEMPT_HEADER)
                    expected = 400
                elif change == "stream":
                    body["stream"] = True
                    expected = 400
                elif change == "route":
                    path = "/v1/chat/completions"
                    expected = 400
                elif change == "invalid-attempt":
                    headers[ATTEMPT_HEADER] = "two attempts"
                    expected = 400
                elif change == "protocol":
                    path = path.replace("chat/completions", "responses")
                    expected = 404
                elif change == "config":
                    monkeypatch.setattr(table, "read_serving_config", lambda scenario: {"temperature": 0.5})
                    expected = 409
                elif change == "database":
                    # Populate the normal cache first; strict calls must not reuse it on read failure.
                    assert (await client.post(path.removesuffix("/attempt"), headers=HEADERS, json=body)).status == 200
                    calls.clear()

                    def unavailable(scenario: str):
                        raise OSError("database unavailable")

                    monkeypatch.setattr(table, "read_scenario", unavailable)
                    expected = 503
                elif change == "changing":

                    def changing(scenario: str):
                        table.write(
                            ScenarioModel("math", "artifact-1"),
                            ServedArtifact("artifact-1", served.url, "load-2"),
                            control_lease_number=lease,
                        )
                        return {}

                    monkeypatch.setattr(table, "read_serving_config", changing)
                    expected = 503
                else:
                    # A new active model does not change the retained artifact's binding.
                    table.write(
                        ScenarioModel("math", ACTIVE),
                        ServedArtifact("artifact-2", served.url, "load-2"),
                        control_lease_number=lease,
                    )
                assert (
                    await client.post(
                        "/reef/scenarios/math/artifacts/artifact-1/v1/chat/completions/attempt",
                        headers=HEADERS,
                        json=body,
                    )
                ).status == 400
                response = await client.post(path, headers=headers, json=body)
                assert response.status == expected, await response.text()
                if expected == 200:
                    assert response.headers[ARTIFACT_HEADER] == "artifact-1"
                    assert response.headers[BINDING_HEADER] == target["binding_id"]
                    assert response.headers[ATTEMPT_HEADER] == "attempt-1"
                    assert (await response.json())["usage"] == {"prompt_tokens": 3, "completion_tokens": 1}
                    assert len(calls) == 1 and calls[0]["model"] == "load-1"
                else:
                    assert calls == []
        coordination.close()

    asyncio.run(check())


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["mismatch", "abort"])
def test_bound_chat_does_not_retry_engine_execution(tmp_path: Path, failure: str) -> None:
    async def check() -> None:
        calls = []

        async def complete(request: web.Request) -> web.Response:
            calls.append(await request.json())
            if failure == "mismatch":
                return web.json_response({"error": {"message": "binding changed"}}, status=421)
            return web.json_response({"choices": [{"finish_reason": "abort"}]})

        engine = web.Application()
        engine.router.add_post("/v1/chat/completions", complete)
        coordination = memory_coordination()
        table = coordination.served_models
        lease = coordination.control_lease.acquire("control").number
        async with TestServer(engine) as upstream:
            table.write(
                ScenarioModel("math", "artifact-1"),
                ServedArtifact("artifact-1", str(upstream.make_url("/")), "load-1"),
                control_lease_number=lease,
            )
            app = build_inference_app(table, SQLiteScenarioStorage(tmp_path / "records"), Engines(), tokens=TOKEN)
            async with TestClient(TestServer(app)) as client:
                target = await (
                    await client.get("/reef/scenarios/math/inference/target?model=artifact-1", headers=HEADERS)
                ).json()
                response = await client.post(
                    "/reef/scenarios/math/artifacts/artifact-1/evaluation/v1/chat/completions/attempt",
                    headers={**HEADERS, BINDING_HEADER: target["binding_id"], ATTEMPT_HEADER: "attempt-1"},
                    json={"messages": [{"role": "user", "content": "2+2"}]},
                )
                assert response.status == (503 if failure == "mismatch" else 409), await response.text()
                assert len(calls) == 1
        coordination.close()

    asyncio.run(check())


@pytest.mark.unit
def test_bound_request_keeps_its_target_during_activation(tmp_path: Path) -> None:
    async def check() -> None:
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def complete(request: web.Request) -> web.Response:
            calls.append(await request.json())
            started.set()
            await finish.wait()
            return web.json_response({"choices": [{"message": {"content": "old answer"}}]})

        engine = web.Application()
        engine.router.add_post("/v1/chat/completions", complete)
        coordination = memory_coordination()
        table = coordination.served_models
        lease = coordination.control_lease.acquire("control").number
        async with TestServer(engine) as upstream:
            served = ServedArtifact("artifact-1", str(upstream.make_url("/")), "load-1")
            table.write(ScenarioModel("math", ACTIVE), served, control_lease_number=lease)
            app = build_inference_app(table, SQLiteScenarioStorage(tmp_path / "records"), Engines(), tokens=TOKEN)
            async with TestClient(TestServer(app)) as client:
                target = await (await client.get("/reef/scenarios/math/inference/target", headers=HEADERS)).json()
                headers = {**HEADERS, BINDING_HEADER: target["binding_id"], ATTEMPT_HEADER: "attempt-1"}
                path = "/reef/scenarios/math/artifacts/active/evaluation/v1/chat/completions/attempt"
                request = asyncio.create_task(client.post(path, headers=headers, json={"messages": []}))
                await asyncio.wait_for(started.wait(), 5)
                table.write(
                    ScenarioModel("math", ACTIVE),
                    ServedArtifact("artifact-2", served.url, "load-2"),
                    control_lease_number=lease,
                )
                finish.set()
                response = await asyncio.wait_for(request, 5)
                assert response.status == 200
                assert response.headers[ARTIFACT_HEADER] == "artifact-1"
                assert len(calls) == 1 and calls[0]["model"] == "load-1"
                assert (await client.post(path, headers=headers, json={"messages": []})).status == 409
                assert len(calls) == 1
        coordination.close()

    asyncio.run(check())
