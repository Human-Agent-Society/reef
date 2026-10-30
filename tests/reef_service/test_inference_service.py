"""The inference service: requests answered from ``served_models`` and recorded in the shared record store."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.core.artifact_ref import ArtifactRef
from reef.core.records_types import RequestType
from reef.core.served_models import ACTIVE, ScenarioModel, ServedArtifact
from reef.service.inference_service import RECORD_HEADER, RELEASE_HEADER, build_inference_app
from reef.storage.commits import CommitRecord
from reef.storage.control_lease import MemoryControlLease
from reef.storage.served_models import MemoryServedModels
from reef.storage.sqlite import SQLiteScenarioStorage

TOKEN = "service-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}", "x-reef-scenario": "math", "x-reef-tag-session": "s1"}


def engine_app(answers: list[str]) -> web.Application:
    """An engine that answers as the model name of the request, unless ``answers`` names another."""
    app = web.Application()

    async def complete(request: web.Request) -> web.Response:
        body = await request.json()
        model = answers.pop(0) if answers else body["model"]
        return web.json_response({"model": model, "choices": [{"message": {"role": "assistant", "content": "4"}}]})

    app.router.add_post("/v1/chat/completions", complete)
    return app


def commit(release_id: str) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=1,
        artifact_ref=ArtifactRef("content:1", release_id, "base"),
        checkpoint=True,
        algorithm_state={},
        consumed_ids=frozenset(),
        high_water_sequence=0,
        high_water_offset=0,
        recorded_at=1000.0,
    )


def inference_records(records: Any) -> tuple[Any, ...]:
    return tuple(item for item in records.replay("math") if item.request_type is RequestType.INFERENCE)


async def ask(tmp_path: Path, answers: list[str], *, write: bool = True) -> tuple[int, dict[str, Any], Any]:
    storage = SQLiteScenarioStorage(tmp_path / "records")
    store = storage.open("math")
    store.commit_step(expected_step=0, commit=commit("release-1"))
    table = MemoryServedModels(MemoryControlLease())
    number = table.control_lease.acquire("control-1").number
    async with TestServer(engine_app(answers)) as engine:
        if write:
            served = ServedArtifact("release-1", str(engine.make_url("/")), "math-v1")
            table.write(ScenarioModel("math", ACTIVE), served, control_lease_number=number)
        app = build_inference_app(table, storage, tokens=TOKEN)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(
                "/v1/chat/completions", headers=HEADERS, json={"model": "any", "messages": [{"role": "user"}]}
            )
            return response.status, dict(response.headers), store.records


@pytest.mark.unit
def test_a_request_goes_to_the_served_artifact_and_its_record_joins_the_record_store(tmp_path: Path) -> None:
    status, headers, records = asyncio.run(ask(tmp_path, []))
    assert status == 200
    assert headers[RELEASE_HEADER] == "release-1"
    (item,) = inference_records(records)
    assert item.agent_record_id == headers[RECORD_HEADER]
    assert item.artifact_ref is not None and item.artifact_ref.release_id == "release-1"
    assert item.payload["metadata"]["tags"] == {"session": "s1"}
    assert item.payload["response"]["model"] == "math-v1"


@pytest.mark.unit
def test_a_scenario_without_a_served_model_is_not_found(tmp_path: Path) -> None:
    status, _, records = asyncio.run(ask(tmp_path, [], write=False))
    assert status == 404
    assert inference_records(records) == ()


@pytest.mark.unit
def test_an_engine_that_answers_as_another_model_twice_is_a_conflict_and_nothing_is_recorded(tmp_path: Path) -> None:
    status, _, records = asyncio.run(ask(tmp_path, ["math-v0", "math-v0"]))
    assert status == 409
    assert inference_records(records) == ()
    # One wrong answer is read again and retried.
    assert asyncio.run(ask(tmp_path / "again", ["math-v0"]))[0] == 200
