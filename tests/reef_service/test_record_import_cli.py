"""File imports bound memory and recover from partial uploads and lost replies."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from reef_client.record_import import import_records_file

from reef.recipe import Recipe
from reef.service.request_service import RequestService

from .test_record_import import dispatcher_for, imported


def write_records(path, count):
    path.write_bytes(b"".join(json.dumps(imported(str(index))).encode() + b"\n" for index in range(count)))


def test_file_import_resumes_after_a_lost_reply_and_partial_upload(tmp_path):
    dispatcher = dispatcher_for(tmp_path, Recipe())
    service = RequestService(dispatcher)
    source = tmp_path / "records.jsonl"
    progress = tmp_path / "progress.json"
    write_records(source, 7)
    received = []

    async def accept(request):
        assert request.headers["Authorization"] == "Bearer fixture-token"
        body = await request.json()
        received.append([item["agent_record_id"] for item in body["records"]])
        if len(received) == 3:
            return web.Response(status=503)
        items = await asyncio.to_thread(service.import_records, request.headers, body)
        if len(received) == 1:
            # The server committed, but the client did not receive its receipts.
            return web.Response(status=503)
        return web.json_response(
            {
                "records": [
                    {
                        "agent_record_id": item.agent_record_id,
                        "scenario": item.scenario,
                        "request_type": item.request_type.value,
                    }
                    for item in items
                ]
            }
        )

    async def run():
        app = web.Application()
        app.router.add_post("/reef/records/batch", accept)
        async with TestServer(app) as server:
            options = {
                "url": str(server.make_url("")),
                "scenario": "s",
                "progress_path": progress,
                "batch_size": 2,
                "token": "fixture-token",
                "max_retries": 0,
            }
            with pytest.raises(ValueError, match="503"):
                await asyncio.to_thread(import_records_file, source, **options)
            assert json.loads(progress.read_text())["offset"] == 0
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 2
            with pytest.raises(ValueError, match="503"):
                await asyncio.to_thread(import_records_file, source, **options)
            assert json.loads(progress.read_text())["count"] == 2
            assert await asyncio.to_thread(import_records_file, source, **options) == 7
            assert received == [["0", "1"], ["0", "1"], ["2", "3"], ["2", "3"], ["4", "5"], ["6"]]
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 7
            before = len(received)
            assert await asyncio.to_thread(import_records_file, source, **options) == 7
            assert len(received) == before
            with pytest.raises(ValueError, match="different file, destination or scenario"):
                await asyncio.to_thread(import_records_file, source, **{**options, "scenario": "other"})
            write_records(source, 8)
            with pytest.raises(ValueError, match="different file, destination or scenario"):
                await asyncio.to_thread(import_records_file, source, **options)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_transient_response_retries_the_same_batch_without_advancing_progress(tmp_path):
    source = tmp_path / "records.jsonl"
    progress = tmp_path / "progress.json"
    write_records(source, 3)
    received = []

    async def accept(request):
        body = await request.json()
        received.append(body)
        if len(received) <= 2:
            assert json.loads(progress.read_text())["count"] == 0
        if len(received) == 1:
            return web.Response(status=429, headers={"Retry-After": "0"})
        return web.json_response(
            {
                "records": [
                    {"agent_record_id": item["agent_record_id"], "request_type": item["request_type"], "scenario": "s"}
                    for item in body["records"]
                ]
            }
        )

    async def run():
        app = web.Application()
        app.router.add_post("/reef/records/batch", accept)
        async with TestServer(app) as server:
            assert (
                await asyncio.to_thread(
                    import_records_file,
                    source,
                    url=str(server.make_url("")),
                    scenario="s",
                    progress_path=progress,
                    batch_size=2,
                    max_retries=1,
                )
                == 3
            )
        assert len(received) == 3
        assert received[0] == received[1]
        assert json.loads(progress.read_text())["count"] == 3

    asyncio.run(run())
