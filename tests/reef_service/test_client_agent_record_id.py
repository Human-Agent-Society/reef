"""Client-supplied agent_record_id on /reef/report.

reef-protocols.md has always promised "clients choose any unique string";
these tests pin the HTTP behavior: echo, retry-dedup, content conflict, and
validation.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.core import AgentRecord, RequestType
from reef.dispatcher import build_default_dispatcher
from reef.records import RecordConflict, RecordStore
from reef.service.app import create_app

HEADERS = {"x-reef-scenario": "idempotency"}


def _report(**extra):
    return {"score": 1.0, "feedback": "ok", "references": [], **extra}


@pytest.mark.unit
def test_report_accepts_and_echoes_client_agent_record_id() -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()

        first = await client.post("/reef/report", json=_report(agent_record_id="grader:turn-1"), headers=HEADERS)
        assert first.status == 200
        assert (await first.json())["agent_record_id"] == "grader:turn-1"

        # Identical retry (fresh timestamp) dedups onto the stored record.
        retry = await client.post("/reef/report", json=_report(agent_record_id="grader:turn-1"), headers=HEADERS)
        assert retry.status == 200
        assert (await retry.json())["agent_record_id"] == "grader:turn-1"

        # Same id, different content: conflict, nothing overwritten.
        conflict = await client.post(
            "/reef/report", json=_report(score=-1.0, agent_record_id="grader:turn-1"), headers=HEADERS
        )
        assert conflict.status == 409

        await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_report_rejects_non_string_agent_record_id_without_storing() -> None:
    async def run() -> None:
        client = TestClient(TestServer(create_app(build_default_dispatcher())))
        await client.start_server()
        for bad in (123, "", "   ", None):
            response = await client.post("/reef/report", json=_report(agent_record_id=bad), headers=HEADERS)
            assert response.status == 400, bad
        await client.close()

    asyncio.run(run())


@pytest.mark.unit
def test_store_dedups_identical_content_across_timestamps() -> None:
    store = RecordStore()
    payload = {"score": 1.0}
    first = AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload=payload, agent_record_id="fixed", created_at=1.0
    )
    second = AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload=payload, agent_record_id="fixed", created_at=2.0
    )
    assert store.append_result(first).inserted is True
    result = store.append_result(second)
    assert result.inserted is False
    assert result.item.created_at == 1.0

    different = AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload={"score": 0.0}, agent_record_id="fixed"
    )
    with pytest.raises(RecordConflict):
        store.append_result(different)
