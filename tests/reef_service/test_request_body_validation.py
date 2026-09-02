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
