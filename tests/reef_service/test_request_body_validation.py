"""Malformed client input is a 400 at the wire, never a traceback (#139, #140)."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.dispatcher import build_default_dispatcher
from reef.service.app import create_app
from reef.service.wire import ReportPayload

HEADERS = {"x-reef-scenario": "body-validation"}
NON_OBJECTS = ([], "text", 1, True, None)


@pytest.mark.unit
def test_object_routes_reject_non_object_bodies_with_400() -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()
        try:
            json_headers = {**HEADERS, "Content-Type": "application/json"}
            for path in ("/reef/scenarios", "/reef/report", "/v1/chat/completions"):
                for body in NON_OBJECTS:
                    # data= so JSON null really travels as null; json=None would send no body at all.
                    response = await client.post(path, data=json.dumps(body), headers=json_headers)
                    assert response.status == 400, (path, body, response.status)
                    assert "request body must be an object" in await response.text()
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
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
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
def test_release_id_routes_reject_empty_and_whitespace_bodies_with_400() -> None:
    """#228 — an empty release_id is a malformed request, not a missing release."""

    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()
        try:
            for path in ("/reef/scenarios", "/reef/scenarios/x/rollback", "/reef/scenarios/x/promote"):
                for release_id in ("", "   "):
                    response = await client.post(path, json={"name": "s", "release_id": release_id})
                    assert response.status == 400, (path, release_id, response.status)
                    assert "release_id must be a non-empty string" in await response.text()

                if path == "/reef/scenarios":
                    # create keeps its current behavior: no release_id is allowed there.
                    missing = await client.post(path, json={"name": "s"})
                    assert missing.status != 400, (path, missing.status)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_create_scenario_strips_surrounding_whitespace_from_release_id() -> None:
    """#228 — a supplied release_id is stripped before use, matching rollback/promote."""

    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()
        try:
            # A whitespace-padded release id is stripped, so the lookup names "no-such-release",
            # not " no-such-release " — the 404 blames the real (unknown) release id.
            response = await client.post(
                "/reef/scenarios", json={"name": "padded", "release_id": "  no-such-release  "}
            )
            assert response.status == 404, response.status
            body = await response.text()
            assert "no-such-release" in body
            assert '"  no-such-release  "' not in body
            assert "no link for scenario" not in body
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_promote_route_rejects_non_object_body_with_400() -> None:
    """#228 — promote now shares the read_object path with the other JSON routes."""

    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()
        try:
            for body in NON_OBJECTS:
                response = await client.post(
                    "/reef/scenarios/x/promote", data=json.dumps(body), headers={"Content-Type": "application/json"}
                )
                assert response.status == 400, (body, response.status)
                assert "request body must be an object" in await response.text()
        finally:
            await client.close()

    asyncio.run(run())
