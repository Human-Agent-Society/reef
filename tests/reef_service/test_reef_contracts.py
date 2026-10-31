from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from tests.support.runtime_stubs import StubInferenceRuntime, runtime_bindings

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactError,
    ArtifactPublicationError,
    InMemoryRepositoryBackend,
    Repository,
)
from reef.core import ReefError, RequestType
from reef.core.records_types import ROLLOUT_HEADER
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.sync import create_harness_surface
from reef.inference.http import HttpInferenceHandler
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler, InferenceStream
from reef.service.app import RequestService, create_app
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import Surface, create_weight_surface
from reef.train import TrainStepResult
from reef.train.types import JobData


class ContractInferenceHandler(InferenceHandler):
    def __init__(self, handler) -> None:
        self._handler = handler

    async def inference(self, artifact, path, payload):
        return await self._handler(artifact, path, payload)


class WeightRecipe(Recipe):
    """A recipe whose scenarios serve weights (the core ``recipe`` implementation serves a plain evolution surface)."""

    def build_surface(self, scenario: str) -> Surface:
        return create_weight_surface(kind="model")


class ForkFailingBackend(InMemoryRepositoryBackend):
    """A repository whose scenario fork always fails."""

    def fork(self, release_id: str | None = None, *, metadata=None):
        raise ArtifactError("cannot fork math")


def base_directory(tmp_path: Path, **files: str) -> Path:
    """The base artifact directory ``tmp_path/initial`` holding ``files`` (name to text)."""
    initial = tmp_path / "initial"
    initial.mkdir()
    for name, text in files.items():
        (initial / name).write_text(text)
    return initial


#: The model name under which the stub engine serves every scenario.
SERVED_MODEL = "stub-model"


def serving_runtime() -> StubInferenceRuntime:
    """A stub runtime whose engine has a URL and a model name, so control writes a ``served_models`` row for it."""
    return StubInferenceRuntime(model_path=SERVED_MODEL)


def weight_dispatcher(tmp_path, initial, *, training: bool = False):
    """A dispatcher of weight scenarios that a stub runtime serves; ``training`` gives them a training component.
    The commit history is durable, so a commit's activation moves the served row."""
    return Dispatcher(
        (
            WeightRecipe(**runtime_bindings(runtime=serving_runtime()))
            if training
            else WeightRecipe(runtime=serving_runtime())
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )


def serving_client(dispatcher, backend=None, **options) -> TestClient:
    """A test client of ``dispatcher``'s app; ``backend`` is an async inference function, when given."""
    if backend is not None:
        options["inference_handler"] = ContractInferenceHandler(backend)
    return TestClient(TestServer(create_app(dispatcher, **options)))


def serving_dispatcher(tmp_path) -> Dispatcher:
    """A dispatcher of the record-only recipe whose runtime serves the created scenario "chat"; a /v1 call answers
    only for a scenario that exists. The commit history is durable, so a commit's activation moves the served row."""
    dispatcher = Dispatcher(
        Recipe(runtime=serving_runtime()),
        InMemoryRepositoryBackend.factory(
            base_directory(tmp_path, **{"model.txt": "base"}), root=tmp_path / "repository"
        ),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    dispatcher.get_or_create_scenario("chat")
    return dispatcher


@asynccontextmanager
async def proxied_client(tmp_path, path: str, upstream) -> AsyncIterator[tuple[TestClient, Dispatcher]]:
    """A client of a Reef app whose HTTP inference handler forwards to the ``upstream`` route at ``path``; the
    scenario "chat" serves the committed version ``v1``, so its records name that release."""
    upstream_app = web.Application()
    upstream_app.router.add_post(path, upstream)
    async with TestServer(upstream_app) as upstream_server:
        dispatcher = serving_dispatcher(tmp_path)
        commit_saved_version(dispatcher, "chat", "v1", tmp_path)
        backend = HttpInferenceHandler(str(upstream_server.make_url("")).rstrip("/"))
        async with serving_client(dispatcher, inference_handler=backend) as client:
            yield client, dispatcher


@pytest.mark.unit
def test_service_rejects_missing_scenario_and_inference_accept() -> None:
    service = RequestService(
        build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    )

    with pytest.raises(ReefError, match="x-reef-scenario"):
        service.accept({}, {"score": 1.0}, request_type=RequestType.REPORT)
    with pytest.raises(ValueError, match=r"inference requests must use infer\(\)"):
        service.accept(
            {"x-reef-scenario": "math"},
            {"model": "reef"},
            request_type=RequestType.INFERENCE,
        )


@pytest.mark.unit
def test_service_has_no_control_route_and_provider_native_generation_stays_behind_inference_backend() -> None:
    with closing(
        build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    ) as dispatcher:
        routes = {route.resource.canonical for route in create_app(dispatcher).router.routes()}

    assert {"/v1/chat/completions", "/v1/responses", "/v1/messages"} <= routes
    assert "/generate" not in routes
    assert "/reef/control" not in routes


@pytest.mark.unit
def test_dispatcher_isolates_agent_record_and_trainer_state_by_scenario() -> None:
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    service = RequestService(dispatcher)

    math = service.accept(
        {"x-reef-scenario": "math"},
        {"score": 1.0},
        request_type=RequestType.REPORT,
        agent_record_id="math-report",
    )
    code = service.accept(
        {"x-reef-scenario": "code"},
        {"score": 0.5},
        request_type=RequestType.REPORT,
        agent_record_id="code-report",
    )

    assert math.scenario == "math" and math.request_type is RequestType.REPORT
    assert code.scenario == "code" and code.request_type is RequestType.REPORT
    assert [r.agent_record_id for r in dispatcher.get_or_create_scenario("math").records.replay("math")] == [
        "math-report"
    ]
    assert [r.agent_record_id for r in dispatcher.get_or_create_scenario("code").records.replay("code")] == [
        "code-report"
    ]
    assert dispatcher.get_or_create_scenario("math").records.count("code") == 0
    assert dispatcher.get_or_create_scenario("code").records.count("math") == 0
    assert (
        dispatcher.get_or_create_scenario("math").repository.current_artifact.release_id
        != dispatcher.get_or_create_scenario("code").repository.current_artifact.release_id
    )


@pytest.mark.unit
def test_scenario_release_id_is_bound_on_first_request(tmp_path) -> None:
    backend = InMemoryRepositoryBackend.factory(base_directory(tmp_path, **{"model.txt": "base"}))
    selected = backend("math").resolve_release().release_id
    dispatcher = build_default_dispatcher(
        backend_factory=backend, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )
    service = RequestService(dispatcher)
    headers = {
        "x-reef-scenario": "math",
        "x-reef-release-id": selected,
    }

    service.accept(headers, {"score": 1.0}, request_type=RequestType.REPORT)

    assert dispatcher.get_or_create_scenario("math").repository.base_artifact.release_id == selected
    with pytest.raises(ArtifactConflict, match="already bound"):
        service.accept({**headers, "x-reef-release-id": "other"}, {"score": 2.0}, request_type=RequestType.REPORT)


@pytest.mark.unit
def test_dispatcher_uses_the_served_recipe_factory(tmp_path) -> None:
    selected: list[str] = []

    @dataclass(frozen=True)
    class RecordingRecipe(Recipe):
        def build(
            self,
            scenario,
            records,
            *,
            algorithm_state=None,
            experiment_logger=None,
            task_storage=None,
            releases=None,
            samples=None,
        ):
            selected.append(self.name)
            return super().build(
                scenario,
                records,
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
            )

    dispatcher = Dispatcher(
        RecordingRecipe(name="tttd"),
        InMemoryRepositoryBackend.factory(base_directory(tmp_path)),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )

    dispatcher.get_or_create_scenario("discovery")

    assert selected == ["tttd"]


@pytest.mark.unit
def test_report_first_request_forks_scenario_artifact(tmp_path) -> None:
    backend = InMemoryRepositoryBackend.factory(base_directory(tmp_path))
    dispatcher = build_default_dispatcher(
        backend_factory=backend, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )

    RequestService(dispatcher).accept(
        {"x-reef-scenario": "math"},
        {"score": 1.0},
        request_type=RequestType.REPORT,
    )

    assert backend("math").current() == dispatcher.get_or_create_scenario("math").repository.current_artifact


@pytest.mark.unit
def test_new_dispatcher_recovers_scenario_metadata_from_repository(tmp_path) -> None:
    backend = InMemoryRepositoryBackend.factory(base_directory(tmp_path))

    first = build_default_dispatcher(
        backend_factory=backend, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )
    created = first.get_or_create_scenario("math")
    second = build_default_dispatcher(
        backend_factory=backend, scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    )

    recovered = second.get_or_create_scenario("math")

    assert recovered.repository.base_artifact == created.repository.base_artifact
    assert recovered.repository.current_artifact == created.repository.checkpoint_artifact
    assert recovered.scenario_step == 0

    with pytest.raises(ArtifactConflict, match="already bound"):
        second.get_or_create_scenario("math", release_id="another-version")


@pytest.mark.unit
def test_failed_artifact_fork_returns_service_unavailable_and_does_not_leave_runtime(tmp_path) -> None:
    dispatcher = build_default_dispatcher(
        backend_factory=ForkFailingBackend.factory(base_directory(tmp_path), root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )

    async def run() -> None:
        async with serving_client(dispatcher) as client:
            response = await client.post("/reef/report", headers={"x-reef-scenario": "math"}, json={"score": 1.0})
            assert response.status == 503
            assert await response.text() == "cannot fork math"

    asyncio.run(run())
    with pytest.raises(ArtifactError, match="cannot fork math"):
        dispatcher.get_or_create_scenario("math")
    assert dispatcher.registry.get_optional("math") is None


@pytest.mark.unit
def test_recipe_pipeline_has_no_update_algorithm() -> None:
    runtime = build_default_dispatcher(
        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
    ).get_or_create_scenario("math")

    assert runtime.trainer.pipeline.output_schema is JobData
    assert runtime.trainer.candidate_backend is None


@pytest.mark.unit
def test_inference_payload_is_recorded_without_reef_body_fields(tmp_path) -> None:
    dispatcher = serving_dispatcher(tmp_path)
    commit_saved_version(dispatcher, "chat", "v1", tmp_path)
    payload = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}

    async def backend(artifact: Artifact, path: str, request_payload: dict) -> dict:
        # The engine gets the payload under the served model name.
        assert request_payload == {**payload, "model": SERVED_MODEL}
        return {"provider": "ok"}

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            response = await client.post("/v1/chat/completions", headers={"x-reef-scenario": "chat"}, json=payload)
            assert response.status == 200

    asyncio.run(run())

    scenario = dispatcher.get_or_create_scenario("chat")
    record = scenario.records.replay("chat")[0]
    # The record names the artifact that answered.
    assert record.payload == {**payload, "artifact_id": scenario.serving_artifact_id(), "response": {"provider": "ok"}}
    assert record.artifact_ref is not None
    assert "scenario" not in record.payload
    assert "type" not in record.payload


@pytest.mark.unit
def test_http_app_records_inference_and_returns_backend_response(tmp_path) -> None:
    used_artifact = None

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        nonlocal used_artifact
        used_artifact = artifact
        assert not hasattr(artifact.ref, "scenario")
        assert path == "/v1/chat/completions"
        return {"choices": [{"message": {"content": payload["messages"][0]["content"]}}]}

    dispatcher = serving_dispatcher(tmp_path)
    commit_saved_version(dispatcher, "chat", "v1", tmp_path)

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"model": "reef", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status == 200
            assert response.headers["x-reef-agent-record-id"]
            assert (await response.json())["choices"][0]["message"]["content"] == "hi"

    asyncio.run(run())
    records = dispatcher.get_or_create_scenario("chat").records.replay("chat")
    assert [record.request_type for record in records] == [RequestType.INFERENCE]
    assert records[0].payload["response"]["choices"][0]["message"]["content"] == "hi"
    assert records[0].artifact_ref == used_artifact.ref
    assert "artifact" not in records[0].payload


@pytest.mark.unit
def test_a_rollout_call_keeps_no_record_and_returns_its_training_block(tmp_path) -> None:
    training = {"tokens": [1, 2, 3], "loss_mask": [1], "rollout_log_probs": [-0.5]}

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        # The engine's own name for its loaded weights stays inside the engine.
        engine_training = {**training, "runtime_load_id": "engine:3", "runtime_load_spans": []}
        return {"choices": [{"message": {"content": "hi"}}], "training": engine_training}

    dispatcher = serving_dispatcher(tmp_path)
    served_artifact_id = dispatcher.get_or_create_scenario("chat").serving_artifact_id()

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            rollout = {"x-reef-scenario": "chat", ROLLOUT_HEADER: "1"}
            body = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}
            response = await client.post("/v1/chat/completions", headers=rollout, json=body)
            assert response.status == 200 and "x-reef-agent-record-id" not in response.headers
            # The training block names the artifact that answered.
            assert (await response.json())["training"] == {**training, "artifact_id": served_artifact_id}
            streamed = await client.post("/v1/chat/completions", headers=rollout, json={**body, "stream": True})
            assert streamed.status == 400
            served = await client.post("/v1/chat/completions", headers={"x-reef-scenario": "chat"}, json=body)
            assert "training" not in await served.json(), "a served call never shows its training block"

    asyncio.run(run())
    records = dispatcher.get_or_create_scenario("chat").records.replay("chat")
    assert len(records) == 1, "only the served call is a record"
    assert records[0].payload["artifact_id"] == served_artifact_id
    assert records[0].payload["response"]["training"] == training


@pytest.mark.unit
def test_artifact_proxy_backend_forwards_native_payload_unchanged(tmp_path) -> None:
    received = {}

    async def upstream(request):
        received["path"] = request.path
        received["payload"] = await request.json()
        received["release_id"] = request.headers["x-reef-release-id"]
        received["artifact_path"] = request.headers["x-reef-artifact-path"]
        return web.json_response({"provider": "ok"})

    repository_backend = InMemoryRepositoryBackend("chat", base_directory(tmp_path), root=tmp_path / "repository")
    artifact = repository_backend.materialize(repository_backend.fork())
    payload = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}

    async def run() -> None:
        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        async with TestServer(upstream_app) as server:
            inference_handler = HttpInferenceHandler(str(server.make_url("")).rstrip("/"))
            assert await inference_handler.inference(artifact, "/v1/chat/completions", payload) == {"provider": "ok"}

    asyncio.run(run())
    assert received == {
        "path": "/v1/chat/completions",
        "payload": payload,
        "release_id": artifact.ref.release_id,
        "artifact_path": str(artifact.local_path),
    }


@pytest.mark.unit
def test_http_app_forwards_inference_stream_before_upstream_finishes(tmp_path) -> None:
    first_chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\r\n\n'
    final_chunk = b"data: [DONE]\n\r"
    received = {}

    async def run() -> None:
        finish_upstream = asyncio.Event()

        async def upstream(request):
            received["path"] = request.path
            received["payload"] = await request.json()
            received["release_id"] = request.headers["x-reef-release-id"]
            response = web.StreamResponse(
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Upstream-Response": "preserved",
                }
            )
            await response.prepare(request)
            await response.write(first_chunk)
            await finish_upstream.wait()
            await response.write(final_chunk)
            return response

        async with proxied_client(tmp_path, "/v1/chat/completions", upstream) as (client, dispatcher):
            try:
                payload = {
                    "model": "reef",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"x-reef-scenario": "chat"},
                    json=payload,
                )

                assert response.status == 200
                assert response.headers["Content-Type"].startswith("text/event-stream")
                assert response.headers["Cache-Control"] == "no-cache"
                assert response.headers["X-Upstream-Response"] == "preserved"
                assert "x-reef-agent-record-id" not in response.headers

                first = await asyncio.wait_for(response.content.readexactly(len(first_chunk)), timeout=1)
                assert first == first_chunk
                assert dispatcher.get_or_create_scenario("chat").records.replay("chat") == ()

                finish_upstream.set()
                metadata_frame = await asyncio.wait_for(response.content.readuntil(b"\n\n"), timeout=1)
                metadata = json.loads(metadata_frame.removeprefix(b"data: "))
                agent_record_id = metadata["reef"]["agent_record_id"]

                records = dispatcher.get_or_create_scenario("chat").records.replay("chat")
                assert len(records) == 1
                assert records[0].agent_record_id == agent_record_id
                assert await asyncio.wait_for(response.read(), timeout=1) == final_chunk
                assert records[0].payload["response"]["stream"] is True
                assert records[0].payload["response"]["complete"] is True
                assert records[0].payload["response"]["status"] == 200
                assert records[0].payload["response"]["body"] == (first_chunk + final_chunk).decode()
                assert records[0].artifact_ref is not None
                assert received == {
                    "path": "/v1/chat/completions",
                    "payload": {**payload, "model": SERVED_MODEL},
                    "release_id": records[0].artifact_ref.release_id,
                }
            finally:
                finish_upstream.set()

    asyncio.run(run())


@pytest.mark.unit
def test_anthropic_stream_attaches_receipt_only_after_record_is_stored(tmp_path) -> None:
    content = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
    stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def run() -> None:
        release_stop = asyncio.Event()

        async def upstream(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(content)
            await release_stop.wait()
            await response.write(stop)
            return response

        async with proxied_client(tmp_path, "/v1/messages", upstream) as (client, dispatcher):
            try:
                response = await client.post(
                    "/v1/messages",
                    headers={"x-reef-scenario": "chat"},
                    json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
                )
                assert "x-reef-agent-record-id" not in response.headers
                assert await response.content.readexactly(len(content)) == content
                assert dispatcher.get_or_create_scenario("chat").records.replay("chat") == ()

                release_stop.set()
                terminal = await response.read()
                metadata = json.loads(terminal.decode().split("data: ", 1)[1])
                [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
                assert record.agent_record_id == metadata["reef"]["agent_record_id"]
                assert record.payload["response"]["body"] == (content + stop).decode()
                assert record.payload["response"]["complete"] is True
            finally:
                release_stop.set()

    asyncio.run(run())


@pytest.mark.unit
def test_responses_stream_is_forwarded_and_attaches_its_receipt_to_response_completed(tmp_path) -> None:
    delta = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    completed = b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_1"}}\n\n'
    received = {}

    async def upstream(request):
        received["payload"] = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(delta)
        await response.write(completed)
        return response

    async def run() -> None:
        async with proxied_client(tmp_path, "/v1/responses", upstream) as (client, dispatcher):
            payload = {"model": "gpt-4o", "input": [{"role": "user", "content": "hi"}], "stream": True}
            response = await client.post("/v1/responses", headers={"x-reef-scenario": "chat"}, json=payload)
            assert response.status == 200
            body = await response.read()
            assert body.startswith(delta)
            terminal = body[len(delta) :].decode()
            assert terminal.startswith("event: response.completed\n")
            metadata = json.loads(terminal.split("data: ", 1)[1])
            assert metadata["response"] == {"id": "resp_1"}
            [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
            assert metadata["reef"]["agent_record_id"] == record.agent_record_id
            assert record.payload["response"]["complete"] is True
            assert record.payload["response"]["message"] == {"role": "assistant", "content": "hi"}
            assert received["payload"] == {**payload, "model": SERVED_MODEL}

    asyncio.run(run())


@pytest.mark.unit
def test_sse_without_terminal_event_has_no_receipt_and_is_recorded_incomplete(tmp_path) -> None:
    content = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

    async def upstream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(content)
        return response

    async def run() -> None:
        async with proxied_client(tmp_path, "/v1/chat/completions", upstream) as (client, dispatcher):
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert "x-reef-agent-record-id" not in response.headers
            assert await response.read() == content
            [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
            assert record.payload["response"]["complete"] is False
            assert record.payload["response"]["error"] == "the engine's SSE stream ended without a terminal event"

    asyncio.run(run())


@pytest.mark.unit
def test_published_candidate_is_used_by_next_inference(tmp_path) -> None:
    used_versions = []

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        used_versions.append(artifact.ref.release_id)
        return {"version": artifact.ref.release_id}

    repository_backend = InMemoryRepositoryBackend.factory(
        base_directory(tmp_path, **{"model.txt": "base"}), root=tmp_path / "repository"
    )
    dispatcher = Dispatcher(
        Recipe(runtime=serving_runtime()),
        repository_backend,
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    old_ref = dispatcher.get_or_create_scenario("chat").repository.current_artifact

    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    (candidate_path / "model.txt").write_text("trained")
    committed = dispatcher.registry.get("chat")
    committed.commit(TrainStepResult(state=None, artifact=Artifact.local(candidate_path)))
    dispatcher.scenario_committed(committed)
    new_ref = dispatcher.get_or_create_scenario("chat").repository.current_artifact

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"model": "reef", "messages": []},
            )
            assert response.status == 200

    asyncio.run(run())
    assert old_ref.release_id != new_ref.release_id
    assert used_versions == [new_ref.release_id]


def commit_saved_version(dispatcher, scenario: str, version: str, root: Path) -> None:
    """Commit a training step whose weights the trainer saved, so the scenario serves them as ``version``; control
    writes the scenario's ``served_models`` rows in the step of the commit, as a local worker's commit does."""
    checkpoint = root / f"checkpoint-{version}"
    checkpoint.mkdir()
    (checkpoint / "model.txt").write_text(version)
    current = dispatcher.registry.get(scenario)
    current.commit(TrainStepResult(state=None, artifact=Artifact.local(checkpoint)))
    dispatcher.scenario_committed(current)


def serving_weight_dispatcher(tmp_path, *, training: bool = False) -> Dispatcher:
    """A weight dispatcher whose scenario "chat" serves the saved version ``sglang-v1``."""
    dispatcher = weight_dispatcher(tmp_path, base_directory(tmp_path, **{"model.txt": "base"}), training=training)
    dispatcher.get_or_create_scenario("chat")
    commit_saved_version(dispatcher, "chat", "sglang-v1", tmp_path)
    return dispatcher


@pytest.mark.unit
def test_aborted_inference_restarts_before_recording(tmp_path) -> None:
    """A backend ``abort`` finish reason retries; only the completed response is recorded."""
    calls = []

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        calls.append(artifact.ref)
        if len(calls) == 1:
            return {"choices": [{"finish_reason": "abort"}]}
        return {"choices": [{"message": {"content": "hi"}, "meta_info": {"runtime_load_id": "sglang-v1"}}]}

    dispatcher = serving_weight_dispatcher(tmp_path)
    head = dispatcher.get_or_create_scenario("chat").repository.current_artifact

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            response = await client.post(
                "/v1/chat/completions", headers={"x-reef-scenario": "chat"}, json={"messages": []}
            )
            assert response.status == 200
            assert await response.json() == {
                "choices": [{"message": {"content": "hi"}, "meta_info": {"runtime_load_id": "sglang-v1"}}]
            }

    asyncio.run(run())
    assert calls == [head, head]
    recorded = dispatcher.get_or_create_scenario("chat").records.replay("chat")
    assert len(recorded) == 1
    assert recorded[0].artifact_ref == calls[-1]


class StreamRecordHandler(InferenceHandler):
    """Streams fixed frames and the record response of one turn.

    ``frames`` is a nonempty tuple of byte frames, sent in order; the constructor rejects an
    empty tuple. With ``record_before_terminal`` the record response is set after every frame
    but the last is sent, as SGLang's token-by-token stream does. Otherwise it is set when the
    stream is created, as a stream served from a buffered result (vLLM) is.
    """

    def __init__(
        self,
        record_response: dict,
        frames: tuple[bytes, ...],
        *,
        record_before_terminal: bool,
        content_type: str = "text/event-stream",
    ) -> None:
        if not frames:
            raise ValueError("StreamRecordHandler needs at least one frame")
        self.record_response = record_response
        self.frames = frames
        self.record_before_terminal = record_before_terminal
        self.content_type = content_type
        self.calls = 0

    async def inference(self, artifact, path, payload):
        raise AssertionError("a streamed request must not call the buffered route")

    async def inference_stream(self, artifact, path, payload):
        self.calls += 1

        async def chunks():
            for frame in self.frames[:-1]:
                yield frame
            if self.record_before_terminal:
                stream.record_response = self.record_response
            yield self.frames[-1]

        stream = InferenceStream(
            status=200,
            headers={"Content-Type": self.content_type},
            chunks=chunks(),
            record_response=None if self.record_before_terminal else self.record_response,
            record_response_pending=self.record_before_terminal,
        )
        return stream


STREAM_TRAINING = {"tokens": [10, 11, 20, 21], "loss_mask": [1, 1], "rollout_log_probs": [-0.5, -0.5]}
CHAT_STREAM_FRAMES = (b'data: {"choices":[{"delta":{"content":"a b"}}]}\n\n', b"data: [DONE]\n\n")
MESSAGES_STREAM_FRAMES = (
    b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"a b"}}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)


def chat_stream_record(finish_reason: str) -> dict:
    return {
        "choices": [{"index": 0, "message": {"content": "a b"}, "finish_reason": finish_reason}],
        "training": STREAM_TRAINING,
    }


#: An Anthropic record shows an abort only in its private training block: stop_reason maps abort to end_turn.
ABORTED_MESSAGES_RECORD = {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "a b"}],
    "stop_reason": "end_turn",
    "training": {**STREAM_TRAINING, "finish_reason": "abort"},
}


def post_stream(dispatcher, handler, path: str):
    """Post one streamed turn to ``path``; the response's status, headers and body."""

    async def run():
        async with serving_client(dispatcher, inference_handler=handler) as client:
            response = await client.post(
                path,
                headers={"x-reef-scenario": "chat"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            return response.status, dict(response.headers), await response.read()

    return asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("record_before_terminal", [True, False], ids=["record-before-terminal", "record-with-stream"])
@pytest.mark.parametrize(
    ("path", "record_response", "frames"),
    [
        ("/v1/chat/completions", chat_stream_record("abort"), CHAT_STREAM_FRAMES),
        ("/v1/messages", ABORTED_MESSAGES_RECORD, MESSAGES_STREAM_FRAMES),
    ],
    ids=["chat", "messages"],
)
@pytest.mark.parametrize("serves_weights", [False, True], ids=["default", "weights"])
def test_aborted_stream_has_no_receipt_and_is_recorded_incomplete(
    tmp_path, path, record_response, frames, record_before_terminal, serves_weights
) -> None:
    """A stream cannot restart after it sent frames, so an aborted stream ends like an incomplete one. On a
    scenario that serves weights, an engine abort before the first sampled token reports no runtime-load ID; it
    is not a 409."""
    handler = StreamRecordHandler(record_response, frames, record_before_terminal=record_before_terminal)
    with closing(
        serving_weight_dispatcher(tmp_path) if serves_weights else serving_dispatcher(tmp_path)
    ) as dispatcher:
        status, _, body = post_stream(dispatcher, handler, path)
        assert status == 200
        assert body == b"".join(frames)
        assert handler.calls == 1
        [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
        recorded = record.payload["response"]
        assert recorded["training"] == record_response["training"]
        assert recorded["stream_delivery"]["complete"] is False
        assert recorded["stream_delivery"]["error"] == "upstream engine aborted the generation"


@pytest.mark.unit
def test_stream_record_handler_needs_a_frame() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        StreamRecordHandler(chat_stream_record("stop"), (), record_before_terminal=False)


@pytest.mark.unit
def test_aborted_non_sse_stream_has_no_receipt_header(tmp_path) -> None:
    body = json.dumps(chat_stream_record("abort")).encode()
    handler = StreamRecordHandler(
        chat_stream_record("abort"), (body,), record_before_terminal=False, content_type="application/json"
    )
    with closing(serving_dispatcher(tmp_path)) as dispatcher:
        status, headers, received = post_stream(dispatcher, handler, "/v1/chat/completions")
        assert status == 200
        assert received == body
        assert "x-reef-agent-record-id" not in headers
        [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
        recorded = record.payload["response"]
        assert recorded["training"] == STREAM_TRAINING
        assert recorded["stream_delivery"]["complete"] is False
        assert recorded["stream_delivery"]["error"] == "upstream engine aborted the generation"


@pytest.mark.unit
@pytest.mark.parametrize("record_before_terminal", [True, False], ids=["record-before-terminal", "record-with-stream"])
def test_finished_stream_keeps_its_receipt_and_training_block(tmp_path, record_before_terminal) -> None:
    handler = StreamRecordHandler(
        chat_stream_record("stop"), CHAT_STREAM_FRAMES, record_before_terminal=record_before_terminal
    )
    with closing(serving_dispatcher(tmp_path)) as dispatcher:
        status, _, body = post_stream(dispatcher, handler, "/v1/chat/completions")
        assert status == 200
        [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
        assert record.agent_record_id.encode() in body
        recorded = record.payload["response"]
        assert recorded["training"] == STREAM_TRAINING
        assert recorded["stream_delivery"]["complete"] is True
        assert "error" not in recorded["stream_delivery"]


@pytest.mark.unit
def test_an_engine_that_aborts_twice_is_a_conflict_and_nothing_is_recorded(tmp_path) -> None:
    calls = []

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        calls.append(path)
        return {"choices": [{"finish_reason": "abort"}]}

    async def run() -> None:
        async with serving_client(dispatcher, backend) as client:
            response = await client.post(
                "/v1/chat/completions", headers={"x-reef-scenario": "chat"}, json={"messages": []}
            )
            assert response.status == 409
            assert "upstream engine aborted the generation" in await response.text()

    with closing(serving_dispatcher(tmp_path)) as dispatcher:
        asyncio.run(run())
        assert len(calls) == 2
        assert not dispatcher.get_or_create_scenario("chat").records.replay("chat")


@pytest.mark.unit
def test_default_local_artifact_root_is_cleaned_up_with_repository(tmp_path) -> None:
    import gc

    backend = InMemoryRepositoryBackend("chat", base_directory(tmp_path), root=tmp_path / "repository")
    repository = Repository(
        backend,
        backend.resolve_release(),
    )
    local_root = repository.local_root
    assert local_root.is_dir()

    del repository
    gc.collect()

    assert not local_root.exists()


@pytest.mark.unit
def test_release_ids_are_independent_and_ignore_results_without_candidates(tmp_path) -> None:
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(base_directory(tmp_path), root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )

    dispatcher.get_or_create_scenario("math")
    dispatcher.get_or_create_scenario("code")
    math_head = dispatcher.get_or_create_scenario("math").repository.require_current_artifact()
    dispatcher.registry.get("math").commit(TrainStepResult(state={"label": "no artifact"}))
    # A no-artifact commit advances the step (the trainer consumed records and
    # state) but does not move the serving head: there is no candidate to publish.
    assert dispatcher.get_or_create_scenario("math").scenario_step == 1
    assert dispatcher.get_or_create_scenario("math").repository.require_current_artifact() == math_head

    for scenario in ("math", "code", "math", "code"):
        version = dispatcher.get_or_create_scenario(scenario).scenario_step + 1
        candidate = tmp_path / f"{scenario}-{version}"
        candidate.mkdir()
        dispatcher.registry.get(scenario).commit(
            TrainStepResult(state={"version": version}, artifact=Artifact.local(candidate))
        )

    assert dispatcher.get_or_create_scenario("math").scenario_step == 3
    assert dispatcher.get_or_create_scenario("code").scenario_step == 2


@pytest.mark.unit
def test_artifact_publication_failure_leaves_runtime_unchanged(tmp_path) -> None:
    class FailingBackend(InMemoryRepositoryBackend):
        def publish(self, candidate, *, expected_parent, advance_head=True):
            raise ArtifactPublicationError("repository failed")

    dispatcher = Dispatcher(
        Recipe(),
        FailingBackend.factory(base_directory(tmp_path), root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    runtime = dispatcher.get_or_create_scenario("chat")
    previous = (runtime.repository.current_artifact, runtime.repository.checkpoint_artifact, runtime.scenario_step)
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    with pytest.raises(ArtifactPublicationError, match="repository failed"):
        dispatcher.registry.get("chat").commit(
            TrainStepResult(state={"label": "new"}, artifact=Artifact.local(candidate))
        )

    assert (
        runtime.repository.current_artifact,
        runtime.repository.checkpoint_artifact,
        runtime.scenario_step,
    ) == previous
    assert list((tmp_path / "local").rglob("*")) == []


def test_healthz_answers_without_token_while_other_routes_require_it(tmp_path) -> None:
    dispatcher = build_default_dispatcher(
        backend_factory=InMemoryRepositoryBackend.factory(base_directory(tmp_path), root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )

    async def run() -> None:
        async with serving_client(dispatcher, tokens="secret") as client:
            health = await client.get("/healthz")
            assert health.status == 200

            unauthenticated = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "math"},
                json={"score": 1.0},
            )
            assert unauthenticated.status == 401

            authenticated = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "math", "Authorization": "Bearer secret"},
                json={"score": 1.0},
            )
            assert authenticated.status != 401

    asyncio.run(run())


def test_any_accepted_token_authenticates_and_rotation_keeps_scenarios_reachable(tmp_path) -> None:
    """The token is the service boundary, not a per-user identity: every
    accepted token is equivalent, so a caller can rotate from one to another
    without losing access to the scenarios it created."""
    dispatcher = build_default_dispatcher(
        backend_factory=InMemoryRepositoryBackend.factory(base_directory(tmp_path), root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    current = {"Authorization": "Bearer current"}
    rotated = {"Authorization": "Bearer next"}
    retired = {"Authorization": "Bearer retired"}

    async def run() -> None:
        async with serving_client(dispatcher, tokens=["current", "next"]) as client:
            created = await client.post(
                "/reef/scenarios",
                headers=current,
                json={"name": "shared"},
            )
            assert created.status == 201
            created_body = await created.json()
            assert created_body["scenario"] == "shared"
            assert "recipe" not in created_body

            after_rotation = await client.post(
                "/reef/report",
                headers={**rotated, "x-reef-scenario": "shared"},
                json={"score": 1.0},
            )
            assert after_rotation.status == 200

            listed = await client.get("/reef/scenarios", headers=rotated)
            listed_rows = (await listed.json())["scenarios"]
            assert [row["scenario"] for row in listed_rows] == ["shared"]
            assert all("recipe" not in row for row in listed_rows)

            outsider = await client.post(
                "/reef/report",
                headers={**retired, "x-reef-scenario": "shared"},
                json={"score": 1.0},
            )
            assert outsider.status == 401

            anonymous = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "shared"},
                json={"score": 1.0},
            )
            assert anonymous.status == 401

    asyncio.run(run())


@pytest.mark.unit
def test_tag_headers_ride_the_inference_record_and_nothing_else() -> None:
    """``x-reef-tag-*`` reaches a processor; the service reads no value but ``release`` (see the record test).

    The streaming path builds its own record and carries the tags as well: a streaming agent-loop call is exactly
    the one a correlating processor needs.
    """
    from reef.service.wire import parse_request_headers, with_request_tags

    headers = {
        "x-reef-scenario": "math",
        "X-Reef-Tag-Session": "hw-3",
        "x-reef-tag-blank": "   ",
    }
    parsed = parse_request_headers(headers, RequestType.INFERENCE)
    assert dict(parsed.tags) == {"session": "hw-3"}  # blank values are not tags
    assert with_request_tags({"messages": []}, parsed) == {"messages": [], "metadata": {"tags": {"session": "hw-3"}}}
    assert with_request_tags({"messages": [], "stream": True}, parsed)["metadata"]["tags"] == {"session": "hw-3"}

    # A report is a judgment about turns, not a served exchange: no tags.
    reported = parse_request_headers(headers, RequestType.REPORT)
    assert with_request_tags({"score": 1.0}, reported) == {"score": 1.0}


class HarnessRecipe(Recipe):
    """A recipe whose clients pull a file tree: the served exchange is the same on every release. A stub runtime
    answers its model calls."""

    def build_surface(self, scenario: str) -> Surface:
        return create_harness_surface(get_adapter("pi"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "recipe",
    [HarnessRecipe(runtime=serving_runtime()), WeightRecipe(runtime=serving_runtime())],
    ids=["harness", "weights"],
)
def test_a_record_names_the_served_release_and_keeps_the_release_the_client_runs_as_a_tag(tmp_path, recipe) -> None:
    """The inference service records every call against the release that served it: it reads served_models only,
    never the catalog. The release that a session says it runs stays in the record's tags."""

    async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
        return {"ok": True}

    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(
            base_directory(tmp_path, **{"rules.md": "first"}), root=tmp_path / "repository"
        ),
        local_artifact_dir=tmp_path / "local",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
    )
    releases = [dispatcher.get_or_create_scenario("chat").current_artifact_ref().release_id]
    for version in (1, 2):
        if isinstance(recipe, WeightRecipe):
            commit_saved_version(dispatcher, "chat", f"sglang-v{version}", tmp_path)
        else:
            candidate = tmp_path / f"candidate-{version}"
            candidate.mkdir()
            (candidate / "rules.md").write_text(f"v{version}")
            current = dispatcher.registry.get("chat")
            current.commit(TrainStepResult(state=None, artifact=Artifact.local(candidate)))
            dispatcher.scenario_committed(current)
        releases.append(dispatcher.get_or_create_scenario("chat").current_artifact_ref().release_id)
    assert len(set(releases)) == 3

    async def run() -> list[str]:
        async with serving_client(dispatcher, backend) as client:
            for tag in (releases[1], releases[0], "no-such-release", None):
                headers = {"x-reef-scenario": "chat", **({"x-reef-tag-release": tag} if tag else {})}
                response = await client.post("/v1/chat/completions", headers=headers, json={"messages": []})
                assert response.status == 200
        store = dispatcher.storage.open("chat")
        try:
            rows = [item for item in store.records.replay("chat") if item.request_type is RequestType.INFERENCE]
        finally:
            store.close()
        return [
            (item.artifact_ref.release_id, item.payload.get("metadata", {}).get("tags", {}).get("release"))
            for item in rows
        ]

    recorded = asyncio.run(run())
    first, middle, served = releases
    assert recorded == [(served, middle), (served, first), (served, "no-such-release"), (served, None)]


@pytest.mark.parametrize("close_dispatcher", [False, True])
def test_http_app_closes_dispatcher_only_when_requested(monkeypatch, close_dispatcher) -> None:
    with closing(
        build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    ) as dispatcher:
        close = dispatcher.close
        calls = []

        def close_dispatcher_once():
            calls.append("close")
            close()

        monkeypatch.setattr(dispatcher, "close", close_dispatcher_once)

        async def run():
            runner = web.AppRunner(create_app(dispatcher, close_dispatcher=close_dispatcher))
            await runner.setup()
            await runner.cleanup()

        asyncio.run(run())
        assert calls == (["close"] if close_dispatcher else [])


def test_ingest_metrics_distinguish_new_records_retries_rejections_and_write_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from reef.core import AgentRecord
    from reef.core.reports import ReportValidationError
    from reef.storage.records import RecordConflict

    with closing(
        build_default_dispatcher(scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination())
    ) as dispatcher:
        item = AgentRecord.create(scenario="math", request_type=RequestType.INFERENCE, payload={})
        dispatcher.accept_record(item)
        dispatcher.accept_record(item)
        with pytest.raises(RecordConflict):
            dispatcher.accept_record(replace(item, payload={"different": True}))
        with pytest.raises(ReportValidationError):
            dispatcher.accept_record(
                AgentRecord.create(
                    scenario="math",
                    request_type=RequestType.REPORT,
                    payload={"score": 1},
                    references=("missing",),
                )
            )
        with pytest.raises(ValueError, match="training_mode"):
            dispatcher.accept_record(
                AgentRecord.create(
                    scenario="math",
                    request_type=RequestType.TRAIN,
                    payload={"text": "train"},
                )
            )
        scenario = dispatcher.get_or_create_scenario("math")

        def fail_write(item: AgentRecord) -> None:
            raise OSError("storage unavailable")

        monkeypatch.setattr(scenario.records, "append_result", fail_write)
        with pytest.raises(OSError):
            dispatcher.accept_record(replace(item, agent_record_id="another"))
        metrics = scenario.operations.snapshot()
        assert metrics["ingest/accepted_total"] == 1
        assert metrics["ingest/duplicates_total"] == 1
        assert metrics["ingest/rejected_conflict_total"] == 1
        assert metrics["ingest/rejected_report_total"] == 1
        assert metrics["ingest/rejected_request_total"] == 1
        assert metrics["ingest/write/failed_total"] == 2
        assert metrics["ingest/write/completed_total"] == 2
        assert metrics["ingest/write/active"] == 0
        assert metrics["ingest/write/duration_seconds_total"] > 0
        assert dispatcher.get_or_create_scenario("code").operations.snapshot()["ingest/accepted_total"] == 0
