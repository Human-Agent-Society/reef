"""The request that an engine's handler gives for a client's body (``InferenceHandler.request_payload``).

A handler that sets values itself, such as a model's fixed sampling values, returns the body with them. The inference
service sends that body to the engine and records it, so a record holds the values that the engine used.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact.artifact import Artifact
from reef.core import RequestType
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.inference.interfaces import FixedEngineClients, InferenceHandler, InferenceStream
from reef.service.inference_service import InferenceReplica, ServedModelsCache, add_inference_routes
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

SCENARIO = "reef-sampling"
COMPLETION = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
CHUNKS = (
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
)


class EchoHandler(InferenceHandler):
    """An engine that answers every request and keeps each body that it receives."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.received.append(payload)
        return COMPLETION

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        self.received.append(payload)

        async def chunks() -> AsyncIterator[bytes]:
            for chunk in CHUNKS:
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return InferenceStream(status=200, headers={"content-type": "text/event-stream"}, chunks=chunks())


class FixedSamplingHandler(EchoHandler):
    """An engine served at a fixed temperature, which refuses a reasoning effort it does not know."""

    def request_payload(self, payload: Mapping[str, Any], *, serving_config: Mapping[str, Any]) -> dict[str, Any]:
        if payload.get("reasoning_effort") not in (None, "low"):
            raise ValueError(f"reasoning_effort {payload['reasoning_effort']!r} is not served")
        return {**payload, "temperature": 0.6}


def replica_client(
    tmp_path: Path, handler: InferenceHandler, *, refuses_images: bool = False
) -> tuple[TestClient, SQLiteScenarioStorage]:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    coordination = memory_coordination()
    number = coordination.control_lease.acquire("control-1").number
    coordination.served_models.write(
        ScenarioModel(SCENARIO, ACTIVE),
        ServedArtifact("release-1", "http://engine.invalid", "engine-model"),
        control_lease_number=number,
    )
    replica = InferenceReplica(
        ServedModelsCache(coordination.served_models),
        storage,
        FixedEngineClients(handler),
        refuses_images=refuses_images,
    )
    app = web.Application()
    add_inference_routes(app, replica)
    return TestClient(TestServer(app)), storage


def recorded_requests(storage: SQLiteScenarioStorage) -> list[dict[str, Any]]:
    store = storage.open(SCENARIO)
    try:
        return [
            {key: value for key, value in item.payload.items() if key != "response"}
            for item in store.records.replay(SCENARIO)
            if item.request_type is RequestType.INFERENCE
        ]
    finally:
        store.close()


def post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/v1/chat/completions", json=body, headers={"x-reef-scenario": SCENARIO})


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_the_engine_receives_and_the_record_holds_the_body_that_the_handler_gives(
    tmp_path: Path, stream: bool
) -> None:
    handler = FixedSamplingHandler()
    body = {"model": "client-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0}
    body = {**body, "stream": True} if stream else body

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler)
        async with client:
            response = await post(client, body)
            assert response.status == 200
            await response.read()
        # The engine samples at the handler's temperature, under its own model name.
        (received,) = handler.received
        assert (received["temperature"], received["model"]) == (0.6, "engine-model")
        # The record keeps the client's model and the temperature that the engine used.
        assert recorded_requests(storage) == [{**body, "temperature": 0.6, "artifact_id": "release-1"}]

    asyncio.run(run())


@pytest.mark.unit
def test_a_body_that_the_handler_refuses_answers_400_without_a_record(tmp_path: Path) -> None:
    handler = FixedSamplingHandler()

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler)
        async with client:
            response = await post(client, {"model": "m", "messages": [], "reasoning_effort": "high"})
            assert response.status == 400
            assert "reasoning_effort 'high' is not served" in await response.text()
        assert handler.received == [] and recorded_requests(storage) == []

    asyncio.run(run())


IMAGE_BODY = {
    "model": "m",
    "messages": [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        }
    ],
}


@pytest.mark.unit
@pytest.mark.parametrize("stream", [False, True])
def test_a_recipe_that_refuses_images_answers_400_before_the_engine_and_the_record(
    tmp_path: Path, stream: bool
) -> None:
    handler = EchoHandler()

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler, refuses_images=True)
        async with client:
            response = await post(client, {**IMAGE_BODY, "stream": stream})
            assert response.status == 400
            assert f"recipe of scenario {SCENARIO!r} does not support images" in await response.text()
            text = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            assert (await post(client, text)).status == 200
        assert len(handler.received) == 1 and len(recorded_requests(storage)) == 1

    asyncio.run(run())


@pytest.mark.unit
def test_an_unrecorded_call_passes_an_image(tmp_path: Path) -> None:
    handler = EchoHandler()

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler, refuses_images=True)
        async with client:
            response = await client.post(f"/reef/scenarios/{SCENARIO}/evaluation/v1/chat/completions", json=IMAGE_BODY)
            assert response.status == 200, await response.text()
        assert len(handler.received) == 1 and recorded_requests(storage) == []

    asyncio.run(run())


@pytest.mark.unit
def test_a_recipe_that_keeps_images_passes_and_records_an_image(tmp_path: Path) -> None:
    handler = EchoHandler()

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler)
        async with client:
            assert (await post(client, IMAGE_BODY)).status == 200
        assert len(handler.received) == 1 and len(recorded_requests(storage)) == 1

    asyncio.run(run())


@pytest.mark.unit
def test_by_default_the_engine_receives_and_the_record_holds_the_clients_body(tmp_path: Path) -> None:
    handler = EchoHandler()
    body = {"model": "client-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 1.0}

    async def run() -> None:
        client, storage = replica_client(tmp_path, handler)
        async with client:
            response = await post(client, body)
            assert response.status == 200
        assert handler.received == [{**body, "model": "engine-model"}]
        assert recorded_requests(storage) == [{**body, "artifact_id": "release-1"}]

    asyncio.run(run())
