"""Malformed client input is a 400 at the wire, never a traceback (#139, #140)."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.dispatcher import build_default_dispatcher
from reef.service.app import create_app
from reef.service.wire import ReportPayload
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage

HEADERS = {"x-reef-scenario": "body-validation"}
NON_OBJECTS = ([], "text", 1, True, None)


@pytest.mark.unit
def test_object_routes_reject_non_object_bodies_with_400() -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            json_headers = {**HEADERS, "Content-Type": "application/json"}
            for path in ("/reef/scenarios", "/reef/report", "/v1/chat/completions"):
                for body in NON_OBJECTS:
                    # data= so JSON null really travels as null; json=None would send no body at all.
                    response = await client.post(path, data=json.dumps(body), headers=json_headers)
                    assert response.status == 400, (path, body, response.status)
                    # The inference routes read their body in the inference service, with its own message.
                    expected = (
                        "request body must be a JSON object"
                        if path.startswith("/v1/")
                        else "request body must be an object"
                    )
                    assert expected in await response.text()
                empty = await client.post(path, data="", headers=json_headers)
                assert empty.status == 400, (path, empty.status)
            created = await client.post("/reef/scenarios", json={"name": "still-works"}, headers=HEADERS)
            assert created.status in (200, 201)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_report_payload_rejects_non_finite_scores(score: float) -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        ReportPayload.from_dict({"score": score, "feedback": "text"})


@pytest.mark.unit
def test_report_payload_keeps_finite_scores() -> None:
    assert ReportPayload.from_dict({"score": 3}).score == 3.0
    assert ReportPayload.from_dict({"score": -0.5}).score == -0.5


@pytest.mark.unit
def test_report_route_answers_400_for_a_non_finite_score() -> None:
    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            # 1e999 decodes to infinity, so the value reaches the wire contract itself.
            response = await client.post(
                "/reef/report",
                data='{"score": 1e999, "feedback": "x", "references": []}',
                headers={**HEADERS, "Content-Type": "application/json"},
            )
            assert response.status == 400
            assert "score must be finite" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_artifact_id_routes_reject_empty_and_whitespace_bodies_with_400() -> None:
    """#228 — an empty artifact_id is a malformed request, not a missing artifact."""

    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            for path, key in (
                ("/reef/scenarios", "artifact_id"),
                ("/reef/scenarios/x/activate", "artifact_id"),
                ("/reef/scenarios/x/train-from", "artifact_id"),
            ):
                for value in ("", "   "):
                    response = await client.post(path, json={"name": "s", key: value})
                    assert response.status == 400, (path, value, response.status)
                    assert f"{key} must be a non-empty string" in await response.text()

                if path == "/reef/scenarios":
                    # A scenario without artifact_id starts from the base.
                    missing = await client.post(path, json={"name": "s"})
                    assert missing.status != 400, (path, missing.status)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_create_and_update_refuse_the_retired_release_id_and_model_keys() -> None:
    """RFC #1062: a scenario starts from an artifact item, and it has no model setting of its own."""

    async def run() -> None:
        dispatcher = build_default_dispatcher(
            scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
        )
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            for key, value in (("release_id", "base"), ("model", {"url": "http://model.test", "model": "m"})):
                response = await client.post("/reef/scenarios", json={"name": "s", key: value})
                assert response.status == 400, (key, response.status)
                text = await response.text()
                assert key in text and "artifact_id" in text
            component_alone = await client.post("/reef/scenarios", json={"name": "s", "component": "policy"})
            assert component_alone.status == 400 and "artifact_id" in await component_alone.text()
            assert not dispatcher.has_scenario("s")
            assert (await client.post("/reef/scenarios", json={"name": "s"})).status == 201
            update = await client.post("/reef/scenarios/s/update", json={"model": None})
            assert update.status == 400 and "training_mode, artifact_evaluation or activation" in await update.text()

    asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/reef/scenarios/x/activate", "/reef/scenarios/x/train-from"])
def test_activation_routes_reject_non_object_bodies_with_400(path: str) -> None:
    """#228 — activate and train-from share the read_object path with the other JSON routes."""

    async def run() -> None:
        client = TestClient(
            TestServer(
                create_app(
                    build_default_dispatcher(
                        scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
                    )
                )
            )
        )
        await client.start_server()
        try:
            for body in NON_OBJECTS:
                response = await client.post(path, data=json.dumps(body), headers={"Content-Type": "application/json"})
                assert response.status == 400, (body, response.status)
                assert "request body must be an object" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())


def test_report_route_rejects_missing_references_and_eligibility_flags() -> None:
    async def run() -> None:
        dispatcher = build_default_dispatcher(
            scenario_storage=SQLiteScenarioStorage(), coordination=memory_coordination()
        )
        client = TestClient(TestServer(create_app(dispatcher)))
        await client.start_server()
        try:
            for body, message in (
                ({"score": 1.0, "references": ["missing"]}, "existing inference"),
                ({"score": 1.0, "metadata": {"training": {"eligible": False}}}, "eligible"),
            ):
                response = await client.post("/reef/report", json=body, headers=HEADERS)
                assert response.status == 400
                assert message in await response.text()
            scenario = dispatcher.get_or_create_scenario(HEADERS["x-reef-scenario"])
            assert scenario.records.count(scenario.name) == 0
        finally:
            await client.close()

    asyncio.run(run())
