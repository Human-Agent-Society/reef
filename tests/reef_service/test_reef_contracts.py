from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactError,
    ArtifactPublicationError,
    InMemoryRepositoryBackend,
    LiveWeightArtifactRef,
    Repository,
)
from reef.core import ReefError, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import Recipe
from reef.runtime.inference import HttpInferenceBackend, InferenceBackend, default_artifact_request_headers
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.service.app import InferenceRetryPolicy, RequestService, create_app
from reef.surface import Surface, create_weight_surface
from reef.train import TrainStepResult
from reef.train.types import PolicyBatch


class ContractInferenceBackend(InferenceBackend):
    def __init__(self, handler) -> None:
        self._handler = handler

    async def inference(self, artifact, path, payload):
        return await self._handler(artifact, path, payload)


def dispatcher_with_checkpoint_strategy(strategy, *, backend_factory, local_artifact_dir):
    recipe = Recipe(checkpoint_strategy=strategy)
    return Dispatcher(
        recipe,
        backend_factory,
        local_artifact_dir=local_artifact_dir,
    )


class WeightRecipe(Recipe):
    """A recipe whose scenarios commit live weights, so serving-version
    verification applies (the core ``recipe`` implementation serves a plain
    evolution surface and never verifies a runtime load ID)."""

    def build_surface(self, scenario: str) -> Surface:
        return create_weight_surface()


def weight_dispatcher(tmp_path, initial):
    return Dispatcher(
        WeightRecipe(checkpoint_strategy=EveryNVersions(3)),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
    )


@pytest.mark.unit
def test_service_rejects_missing_scenario_and_inference_accept() -> None:
    service = RequestService(build_default_dispatcher())

    with pytest.raises(ReefError, match="x-reef-scenario"):
        service.accept({}, {"score": 1.0}, request_type=RequestType.REPORT)
    with pytest.raises(ValueError, match=r"inference requests must use infer\(\)"):
        service.accept(
            {"x-reef-scenario": "math"},
            {"model": "reef"},
            request_type=RequestType.INFERENCE,
        )


@pytest.mark.unit
def test_service_has_no_control_route() -> None:
    assert all(route.resource.canonical != "/reef/control" for route in create_app().router.routes())


@pytest.mark.unit
def test_provider_native_generation_stays_behind_inference_backend() -> None:
    routes = {route.resource.canonical for route in create_app().router.routes()}

    assert "/v1/chat/completions" in routes
    assert "/generate" not in routes


@pytest.mark.unit
def test_live_artifact_headers_use_version_without_a_checkpoint_path() -> None:
    ref = LiveWeightArtifactRef("artifact-1", "live:weight-v1", "checkpoint", "weight-v1")
    artifact = Artifact(ref, None)

    assert default_artifact_request_headers(artifact) == {
        "x-reef-release-id": "live:weight-v1",
    }


@pytest.mark.unit
def test_dispatcher_isolates_agent_record_and_trainer_state_by_scenario() -> None:
    dispatcher = build_default_dispatcher()
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
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")
    backend = InMemoryRepositoryBackend.factory(initial)
    selected = backend("math").resolve_release().release_id
    dispatcher = build_default_dispatcher(backend_factory=backend)
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
    initial = tmp_path / "initial"
    initial.mkdir()
    selected: list[str] = []

    @dataclass(frozen=True)
    class RecordingRecipe(Recipe):
        def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
            selected.append(self.name)
            return super().build(
                scenario,
                records,
                algorithm_state=algorithm_state,
                experiment_logger=experiment_logger,
            )

    dispatcher = Dispatcher(
        RecordingRecipe(name="tttd"),
        InMemoryRepositoryBackend.factory(initial),
    )

    dispatcher.get_or_create_scenario("discovery")

    assert selected == ["tttd"]


@pytest.mark.unit
def test_report_first_request_forks_scenario_artifact(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial)
    dispatcher = build_default_dispatcher(backend_factory=backend)

    RequestService(dispatcher).accept(
        {"x-reef-scenario": "math"},
        {"score": 1.0},
        request_type=RequestType.REPORT,
    )

    assert backend("math").current() == dispatcher.get_or_create_scenario("math").repository.current_artifact


@pytest.mark.unit
def test_new_dispatcher_recovers_scenario_snapshot_from_repository(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend.factory(initial)

    first = build_default_dispatcher(backend_factory=backend)
    created = first.get_or_create_scenario("math")
    second = build_default_dispatcher(backend_factory=backend)

    recovered = second.get_or_create_scenario("math")

    assert recovered.repository.base_artifact == created.repository.base_artifact
    assert recovered.repository.current_artifact == created.repository.checkpoint_artifact
    assert recovered.scenario_step == 0

    with pytest.raises(ArtifactConflict, match="already bound"):
        second.get_or_create_scenario("math", release_id="another-version")


@pytest.mark.unit
def test_failed_artifact_fork_does_not_leave_runtime(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()

    class FailingBackend(InMemoryRepositoryBackend):
        def fork(self, release_id: str | None = None, *, metadata=None):
            del release_id
            del metadata
            raise ArtifactError("cannot fork math")

    dispatcher = build_default_dispatcher(
        backend_factory=FailingBackend.factory(initial, root=tmp_path / "repository")
    )

    with pytest.raises(ArtifactError, match="cannot fork math"):
        dispatcher.get_or_create_scenario("math")

    assert not dispatcher.has_loaded("math")


@pytest.mark.unit
def test_http_artifact_failure_returns_service_unavailable(tmp_path) -> None:
    async def run() -> None:
        initial = tmp_path / "initial"
        initial.mkdir()

        class FailingBackend(InMemoryRepositoryBackend):
            def fork(self, release_id: str | None = None, *, metadata=None):
                del release_id
                del metadata
                raise ArtifactError("cannot fork math")

        dispatcher = build_default_dispatcher(
            backend_factory=FailingBackend.factory(initial, root=tmp_path / "repository")
        )
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            response = await client.post(
                "/reef/report",
                headers={"x-reef-scenario": "math"},
                json={"score": 1.0},
            )
            assert response.status == 503
            assert await response.text() == "cannot fork math"
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_recipe_pipeline_has_no_update_algorithm() -> None:
    runtime = build_default_dispatcher().get_or_create_scenario("math")

    assert runtime.trainer.processor.output_schema is PolicyBatch
    assert runtime.trainer.training_backend is None


@pytest.mark.unit
def test_inference_payload_is_recorded_without_reef_body_fields() -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher()
        service = RequestService(dispatcher)
        payload = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}

        async def backend(artifact: Artifact, path: str, request_payload: dict) -> dict:
            del artifact, path
            assert request_payload == payload
            return {"provider": "ok"}

        await service.infer(
            {"x-reef-scenario": "chat"},
            payload,
            "/v1/chat/completions",
            ContractInferenceBackend(backend),
        )

        record = dispatcher.get_or_create_scenario("chat").records.replay("chat")[0]
        assert record.payload == {**payload, "response": {"provider": "ok"}}
        assert record.artifact_ref is not None
        assert "scenario" not in record.payload
        assert "type" not in record.payload

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_http_app_records_inference_and_returns_backend_response() -> None:
    async def run() -> None:
        used_artifact = None

        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            nonlocal used_artifact
            used_artifact = artifact
            assert not hasattr(artifact.ref, "scenario")
            assert path == "/v1/chat/completions"
            return {"choices": [{"message": {"content": payload["messages"][0]["content"]}}]}

        dispatcher = build_default_dispatcher()
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=ContractInferenceBackend(backend))))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"model": "reef", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status == 200
            assert response.headers["x-reef-agent-record-id"]
            assert (await response.json())["choices"][0]["message"]["content"] == "hi"
            records = dispatcher.get_or_create_scenario("chat").records.replay("chat")
            assert [record.request_type for record in records] == [RequestType.INFERENCE]
            assert records[0].payload["response"]["choices"][0]["message"]["content"] == "hi"
            assert records[0].artifact_ref == used_artifact.ref
            assert "artifact" not in records[0].payload
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_artifact_proxy_backend_forwards_native_payload_unchanged() -> None:
    async def run() -> None:
        received = {}

        async def upstream(request):
            received["path"] = request.path
            received["payload"] = await request.json()
            received["release_id"] = request.headers["x-reef-release-id"]
            received["artifact_path"] = request.headers["x-reef-artifact-path"]
            return __import__("aiohttp").web.json_response({"provider": "ok"})

        from aiohttp import web

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        server = TestServer(upstream_app)
        await server.start_server()
        try:
            inference_backend = HttpInferenceBackend(str(server.make_url("")).rstrip("/"))
            payload = {"model": "reef", "messages": [{"role": "user", "content": "hi"}]}
            initial = tmp_path / "initial"
            initial.mkdir()
            repository_backend = InMemoryRepositoryBackend("chat", initial, root=tmp_path / "repository")
            materialized = repository_backend.materialize(repository_backend.fork())
            artifact = materialized
            assert await inference_backend.inference(artifact, "/v1/chat/completions", payload) == {"provider": "ok"}
            assert received == {
                "path": "/v1/chat/completions",
                "payload": payload,
                "release_id": artifact.ref.release_id,
                "artifact_path": str(artifact.local_path),
            }
        finally:
            await server.close()

    import asyncio
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        asyncio.run(run())


@pytest.mark.unit
def test_http_app_forwards_inference_stream_before_upstream_finishes(tmp_path) -> None:
    async def run() -> None:
        import asyncio

        from aiohttp import web

        first_chunk = b'data: {"choices":[{"delta":{"content":"hello"}}]}\r\n\n'
        final_chunk = b"data: [DONE]\n\r"
        finish_upstream = asyncio.Event()
        received = {}

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

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()

        dispatcher = build_default_dispatcher(local_artifact_dir=tmp_path / "local")
        backend = HttpInferenceBackend(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=backend)))
        await client.start_server()
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
                "payload": payload,
                "release_id": records[0].artifact_ref.release_id,
            }
        finally:
            finish_upstream.set()
            await client.close()
            await upstream_server.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_anthropic_stream_attaches_receipt_only_after_record_is_stored(tmp_path) -> None:
    async def run() -> None:
        import asyncio

        from aiohttp import web

        content = b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n'
        stop = b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        release_stop = asyncio.Event()

        async def upstream(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(content)
            await release_stop.wait()
            await response.write(stop)
            return response

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/messages", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        dispatcher = build_default_dispatcher(local_artifact_dir=tmp_path / "local")
        backend = HttpInferenceBackend(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=backend)))
        await client.start_server()
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
            data = terminal.decode().split("data: ", 1)[1]
            metadata = json.loads(data)
            receipt = metadata["reef"]["agent_record_id"]
            [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
            assert record.agent_record_id == receipt
            assert record.payload["response"]["body"] == (content + stop).decode()
            assert record.payload["response"]["complete"] is True
        finally:
            release_stop.set()
            await client.close()
            await upstream_server.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_sse_without_terminal_event_has_no_receipt_and_is_recorded_incomplete(tmp_path) -> None:
    async def run() -> None:
        from aiohttp import web

        content = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

        async def upstream(request):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(content)
            return response

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/chat/completions", upstream)
        upstream_server = TestServer(upstream_app)
        await upstream_server.start_server()
        dispatcher = build_default_dispatcher(local_artifact_dir=tmp_path / "local")
        backend = HttpInferenceBackend(str(upstream_server.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_backend=backend)))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
            )
            assert "x-reef-agent-record-id" not in response.headers
            assert await response.read() == content
            [record] = dispatcher.get_or_create_scenario("chat").records.replay("chat")
            assert record.payload["response"]["complete"] is False
            assert record.payload["response"]["error"] == "upstream SSE ended without a terminal event"
        finally:
            await client.close()
            await upstream_server.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_published_candidate_is_used_by_next_inference(tmp_path) -> None:
    async def run() -> None:
        used_versions = []

        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del path, payload
            used_versions.append(artifact.ref.release_id)
            return {"version": artifact.ref.release_id}

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        repository_backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
        dispatcher = build_default_dispatcher(backend_factory=repository_backend)
        old_ref = dispatcher.get_or_create_scenario("chat").repository.current_artifact

        candidate_path = tmp_path / "candidate"
        candidate_path.mkdir()
        (candidate_path / "model.txt").write_text("trained")
        dispatcher._commit_result("chat", TrainStepResult(state=None, artifact=Artifact.local(candidate_path)))
        new_ref = dispatcher.get_or_create_scenario("chat").repository.current_artifact

        client = TestClient(TestServer(create_app(dispatcher, inference_backend=ContractInferenceBackend(backend))))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"model": "reef", "messages": []},
            )
            assert response.status == 200
        finally:
            await client.close()

        assert old_ref.release_id != new_ref.release_id
        assert used_versions == [new_ref.release_id]

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_every_version_serves_locally_and_only_selected_versions_checkpoint(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")

    class CountingBackend(InMemoryRepositoryBackend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.publish_count = 0
            self.expected_parents = []

        def publish(self, candidate, *, expected_parent):
            self.publish_count += 1
            self.expected_parents.append(expected_parent)
            return super().publish(candidate, expected_parent=expected_parent)

    backend = CountingBackend.factory(initial, root=tmp_path / "repository")
    dispatcher = build_default_dispatcher(
        backend_factory=backend,
        checkpoint_strategy=EveryNVersions(3),
        local_artifact_dir=tmp_path / "local",
    )
    original_checkpoint = dispatcher.get_or_create_scenario("chat").repository.current_artifact

    for version in (1, 2, 3):
        candidate = tmp_path / f"candidate-{version}"
        candidate.mkdir()
        (candidate / "model.txt").write_text(f"v{version}")
        dispatcher._commit_result("chat", TrainStepResult(state=None, artifact=Artifact.local(candidate)))

        runtime = dispatcher.get_or_create_scenario("chat")
        selected = dispatcher.current_artifact("chat")
        assert selected.local_path is None
        materialized = selected.materialize()
        assert materialized.local_path.joinpath("model.txt").read_text() == f"v{version}"
        assert runtime.scenario_step == version
        if version < 3:
            assert runtime.repository.current_artifact.release_id.startswith("local:")
            assert runtime.repository.checkpoint_artifact == original_checkpoint
            assert backend("chat").publish_count == 0

    runtime = dispatcher.get_or_create_scenario("chat")
    assert not runtime.repository.current_artifact.release_id.startswith("local:")
    assert runtime.repository.checkpoint_artifact == runtime.repository.current_artifact
    assert backend("chat").publish_count == 1
    assert backend("chat").expected_parents == [original_checkpoint]
    assert backend("chat").current() == runtime.repository.current_artifact
    recovered = build_default_dispatcher(backend_factory=backend).get_or_create_scenario("chat")
    assert recovered.scenario_step == 3


@pytest.mark.unit
def test_non_checkpointed_version_is_used_by_next_inference(tmp_path) -> None:
    async def run() -> None:
        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del path, payload
            assert artifact.local_path is None
            assert isinstance(artifact.ref, LiveWeightArtifactRef)
            # Real engines report the serving version in meta_info.
            return {"meta_info": {"runtime_load_id": artifact.ref.runtime_load_id}}

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        dispatcher = weight_dispatcher(tmp_path, initial)
        dispatcher.get_or_create_scenario("chat")
        checkpoint = dispatcher.get_or_create_scenario("chat").repository.checkpoint_artifact
        dispatcher._commit_result("chat", TrainStepResult(state=None, runtime_load_id="sglang-v1"))
        runtime = dispatcher.get_or_create_scenario("chat")
        assert runtime.repository.current_artifact.release_id.startswith("live:")
        assert runtime.repository.checkpoint_artifact == checkpoint
        assert runtime.scenario_step == 1

        client = TestClient(TestServer(create_app(dispatcher, inference_backend=ContractInferenceBackend(backend))))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": []},
            )
            assert response.status == 200
            assert "x-reef-runtime-load-ID" not in response.headers
            assert await response.json() == {"meta_info": {"runtime_load_id": "sglang-v1"}}
            listed = await client.get("/reef/scenarios/chat/releases")
            current = next(row for row in (await listed.json())["releases"] if row["current"])
            assert current["runtime_load_id"] == "sglang-v1"
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_aborted_inference_restarts_before_recording(tmp_path) -> None:
    """A backend ``abort`` finish reason retries; only the completed response is recorded."""

    async def run() -> None:
        calls = []

        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del path, payload
            calls.append(artifact.ref)
            if len(calls) == 1:
                return {"choices": [{"finish_reason": "abort"}]}
            return {"choices": [{"message": {"content": "hi"}, "meta_info": {"runtime_load_id": "sglang-v1"}}]}

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        dispatcher = weight_dispatcher(tmp_path, initial)
        dispatcher.get_or_create_scenario("chat")
        dispatcher._commit_result("chat", TrainStepResult(state=None, runtime_load_id="sglang-v1"))
        head = dispatcher.get_or_create_scenario("chat").repository.current_artifact
        assert isinstance(head, LiveWeightArtifactRef)
        assert head.runtime_load_id == "sglang-v1"

        client = TestClient(TestServer(create_app(dispatcher, inference_backend=ContractInferenceBackend(backend))))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": []},
            )
            assert response.status == 200
            assert await response.json() == {
                "choices": [{"message": {"content": "hi"}, "meta_info": {"runtime_load_id": "sglang-v1"}}]
            }
        finally:
            await client.close()

        assert [ref.runtime_load_id for ref in calls] == ["sglang-v1", "sglang-v1"]
        recorded = dispatcher.get_or_create_scenario("chat").records.replay("chat")
        assert len(recorded) == 1
        assert recorded[0].artifact_ref == calls[-1]

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_completed_inference_with_mismatched_runtime_load_id_fails_loudly(tmp_path) -> None:
    """A completed response with unverifiable runtime-load-ID information returns 409 and is never recorded."""

    async def run() -> None:
        calls = []

        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del path, payload
            calls.append(artifact.ref)
            return {"choices": [{"message": {"content": "hi"}, "meta_info": {"runtime_load_id": "sglang-v2"}}]}

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        dispatcher = weight_dispatcher(tmp_path, initial)
        dispatcher.get_or_create_scenario("chat")
        dispatcher._commit_result("chat", TrainStepResult(state=None, runtime_load_id="sglang-v1"))

        client = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    inference_backend=ContractInferenceBackend(backend),
                    inference_retry_policy=InferenceRetryPolicy(initial_s=0.001, max_s=0.002, timeout_s=0.01),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": []},
            )
            assert response.status == 409
            assert "without token-level runtime-load-ID spans" in await response.text()
        finally:
            await client.close()

        assert [ref.runtime_load_id for ref in calls] == ["sglang-v1"]
        assert dispatcher.get_or_create_scenario("chat").records.replay("chat") == ()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_interrupted_inference_stops_at_the_configured_retry_deadline() -> None:
    async def run() -> None:
        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del artifact, path, payload
            return {"choices": [{"finish_reason": "abort"}]}

        dispatcher = build_default_dispatcher()
        client = TestClient(
            TestServer(
                create_app(
                    dispatcher,
                    inference_backend=ContractInferenceBackend(backend),
                    inference_retry_policy=InferenceRetryPolicy(initial_s=0.001, max_s=0.002, timeout_s=0.01),
                )
            )
        )
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": []},
            )
            assert response.status == 503
            assert "retry deadline exceeded" in await response.text()
            assert not dispatcher.get_or_create_scenario("chat").records.replay("chat")
        finally:
            await client.close()
            dispatcher.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_inference_fails_loudly_when_the_engine_reports_no_version(tmp_path) -> None:
    """Live artifacts only exist behind engines that report runtime_load_id;

    a response without one means the serving version is unverifiable — 409, no record.
    """

    async def run() -> None:
        async def backend(artifact: Artifact, path: str, payload: dict) -> dict:
            del artifact, path, payload
            return {"choices": [{"message": {"content": "hi"}}]}

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "model.txt").write_text("base")
        dispatcher = weight_dispatcher(tmp_path, initial)
        dispatcher.get_or_create_scenario("chat")
        dispatcher._commit_result("chat", TrainStepResult(state=None, runtime_load_id="sglang-v1"))

        client = TestClient(TestServer(create_app(dispatcher, inference_backend=ContractInferenceBackend(backend))))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"x-reef-scenario": "chat"},
                json={"messages": []},
            )
            assert response.status == 409
            assert "reports no runtime_load_id" in await response.text()
        finally:
            await client.close()

        assert dispatcher.get_or_create_scenario("chat").records.replay("chat") == ()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_new_dispatcher_falls_back_to_latest_checkpoint(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("base")
    backend = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    first_dispatcher = build_default_dispatcher(
        backend_factory=backend,
        checkpoint_strategy=EveryNVersions(3),
        local_artifact_dir=tmp_path / "first-local",
    )
    checkpoint = first_dispatcher.get_or_create_scenario("chat").repository.checkpoint_artifact
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.txt").write_text("v1")
    first_dispatcher._commit_result("chat", TrainStepResult(state=None, artifact=Artifact.local(candidate)))
    assert first_dispatcher.get_or_create_scenario("chat").repository.current_artifact != checkpoint

    second_dispatcher = build_default_dispatcher(
        backend_factory=backend,
        checkpoint_strategy=EveryNVersions(3),
        local_artifact_dir=tmp_path / "second-local",
    )

    assert second_dispatcher.get_or_create_scenario("chat").repository.current_artifact == checkpoint
    restored = second_dispatcher.current_artifact("chat").materialize()
    assert restored.local_path.joinpath("model.txt").read_text() == "base"


@pytest.mark.unit
def test_default_local_artifact_root_is_cleaned_up_with_repository(tmp_path) -> None:
    import gc

    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend("chat", initial, root=tmp_path / "repository")
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
    initial = tmp_path / "initial"
    initial.mkdir()
    decisions = []

    class RecordingStrategy:
        def should_checkpoint(self, scenario, version, result):
            decisions.append((scenario, version, result.state))
            return False

    dispatcher = dispatcher_with_checkpoint_strategy(
        RecordingStrategy(),
        backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
    )

    dispatcher.get_or_create_scenario("math")
    dispatcher.get_or_create_scenario("code")
    math_head = dispatcher.get_or_create_scenario("math").repository.require_current_artifact()
    dispatcher._commit_result("math", TrainStepResult(state="no artifact"))
    # A no-artifact commit advances the step (the trainer consumed records and
    # state) but does not move the serving head or consult the checkpoint
    # strategy: there is no candidate to checkpoint.
    assert dispatcher.get_or_create_scenario("math").scenario_step == 1
    assert dispatcher.get_or_create_scenario("math").repository.require_current_artifact() == math_head
    assert dispatcher.published["math"] == "no artifact"
    assert decisions == []

    for scenario in ("math", "code", "math", "code"):
        version = dispatcher.get_or_create_scenario(scenario).scenario_step + 1
        candidate = tmp_path / f"{scenario}-{version}"
        candidate.mkdir()
        dispatcher._commit_result(scenario, TrainStepResult(state=version, artifact=Artifact.local(candidate)))

    assert dispatcher.get_or_create_scenario("math").scenario_step == 3
    assert dispatcher.get_or_create_scenario("code").scenario_step == 2
    assert decisions == [
        ("math", 2, 2),
        ("code", 1, 1),
        ("math", 3, 3),
        ("code", 2, 2),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("failure_source", ["strategy", "repository"])
def test_artifact_publication_failure_leaves_runtime_unchanged(tmp_path, failure_source) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()

    class FailingBackend(InMemoryRepositoryBackend):
        def publish(self, candidate, *, expected_parent):
            if failure_source == "repository":
                raise ArtifactPublicationError("repository failed")
            return super().publish(candidate, expected_parent=expected_parent)

    class FailingStrategy:
        def should_checkpoint(self, scenario, version, result):
            if failure_source == "strategy":
                raise RuntimeError("strategy failed")
            return True

    dispatcher = dispatcher_with_checkpoint_strategy(
        FailingStrategy(),
        backend_factory=FailingBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "local",
    )
    runtime = dispatcher.get_or_create_scenario("chat")
    previous = (runtime.repository.current_artifact, runtime.repository.checkpoint_artifact, runtime.scenario_step)
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    error = RuntimeError if failure_source == "strategy" else ArtifactPublicationError
    with pytest.raises(error, match=f"{failure_source} failed"):
        dispatcher._commit_result("chat", TrainStepResult(state="new", artifact=Artifact.local(candidate)))

    assert (
        runtime.repository.current_artifact,
        runtime.repository.checkpoint_artifact,
        runtime.scenario_step,
    ) == previous
    assert "chat" not in dispatcher.published
    assert list((tmp_path / "local").rglob("*")) == []


def test_healthz_answers_without_token_while_other_routes_require_it(tmp_path) -> None:
    async def run() -> None:
        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = build_default_dispatcher(
            backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
        )
        client = TestClient(TestServer(create_app(dispatcher, tokens="secret")))
        await client.start_server()
        try:
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
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


def test_any_accepted_token_authenticates_and_rotation_keeps_scenarios_reachable(tmp_path) -> None:
    """The token is the service boundary, not a per-user identity: every
    accepted token is equivalent, so a caller can rotate from one to another
    without losing access to the scenarios it created."""

    async def run() -> None:
        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = build_default_dispatcher(
            backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
        )
        client = TestClient(TestServer(create_app(dispatcher, tokens=["current", "next"])))
        await client.start_server()
        current = {"Authorization": "Bearer current"}
        rotated = {"Authorization": "Bearer next"}
        retired = {"Authorization": "Bearer retired"}
        try:
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
        finally:
            await client.close()

    import asyncio

    asyncio.run(run())


@pytest.mark.unit
def test_tag_headers_ride_the_inference_record_and_nothing_else() -> None:
    """``x-reef-tag-*`` reaches a processor; the service never reads a value."""
    from reef.service.request_service import _with_tags
    from reef.service.wire import parse_request_headers

    headers = {
        "x-reef-scenario": "math",
        "X-Reef-Tag-Session": "hw-3",
        "x-reef-tag-blank": "   ",
    }
    parsed = parse_request_headers(headers, RequestType.INFERENCE)
    assert dict(parsed.tags) == {"session": "hw-3"}  # blank values are not tags
    assert _with_tags({"messages": []}, parsed) == {"messages": [], "metadata": {"tags": {"session": "hw-3"}}}

    # A report is a judgment about turns, not a served exchange: no tags.
    reported = parse_request_headers(headers, RequestType.REPORT)
    assert _with_tags({"score": 1.0}, reported) == {"score": 1.0}


@pytest.mark.unit
def test_tags_ride_the_streaming_record_too() -> None:
    """The streaming path builds its own record; it must carry tags as well.

    A streaming agent-loop call is exactly the one a correlating processor
    needs, so a tag that only survives the buffered path tags nothing that
    matters.
    """
    from reef.service.request_service import _with_tags
    from reef.service.wire import parse_request_headers

    parsed = parse_request_headers({"x-reef-scenario": "math", "x-reef-tag-session": "hw-3"}, RequestType.INFERENCE)
    streamed = _with_tags({"messages": [], "stream": True}, parsed)
    assert streamed["metadata"]["tags"] == {"session": "hw-3"}
