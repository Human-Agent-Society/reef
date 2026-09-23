"""File imports bound memory and recover from partial uploads and lost replies."""

import asyncio
import fcntl
import io
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.recipe import Recipe
from reef.service.record_import import import_records_file, record_batches
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
                await import_records_file(source, **options)
            assert json.loads(progress.read_text())["offset"] == 0
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 2
            with pytest.raises(ValueError, match="503"):
                await import_records_file(source, **options)
            assert json.loads(progress.read_text())["count"] == 2
            assert await import_records_file(source, **options) == 7
            assert received == [["0", "1"], ["0", "1"], ["2", "3"], ["2", "3"], ["4", "5"], ["6"]]
            assert dispatcher.get_or_create_scenario("s").records.count("s") == 7
            before = len(received)
            assert await import_records_file(source, **options) == 7
            assert len(received) == before
            with pytest.raises(ValueError, match="different file, destination or scenario"):
                await import_records_file(source, **{**options, "scenario": "other"})
            write_records(source, 8)
            with pytest.raises(ValueError, match="different file, destination or scenario"):
                await import_records_file(source, **options)

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_batches_bound_bytes_count_and_preserve_resume_offsets():
    rows = [
        json.dumps(imported(str(index), {"text": "汉字" * index}), ensure_ascii=False).encode() for index in range(8)
    ]
    data = b"\n\n".join(rows) + b"\n"
    source = io.BytesIO(data)
    batches = list(record_batches(source, batch_size=3, max_bytes=400))
    assert sum(count for _, _, count in batches) == 8
    assert all(len(body) <= 400 and count <= 3 for body, _, count in batches)
    _, first_offset, _ = batches[0]
    source.seek(first_offset)
    resumed = list(record_batches(source, batch_size=3, max_bytes=400))
    assert resumed == batches[1:]
    ids = [item["agent_record_id"] for body, _, _ in batches for item in json.loads(body)["records"]]
    assert ids == [str(index) for index in range(8)]
    with pytest.raises(ValueError, match="exceeds"):
        list(record_batches(io.BytesIO(b"x" * 401), batch_size=3, max_bytes=400))
    with pytest.raises(ValueError, match="invalid JSONL"):
        list(record_batches(io.BytesIO(b"not json\n"), batch_size=3, max_bytes=400))


def test_import_rejects_concurrent_checkpoint_use(tmp_path):
    source = tmp_path / "records.jsonl"
    write_records(source, 1)
    progress = tmp_path / "progress.json"
    with progress.with_name(progress.name + ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another importer"):
            asyncio.run(import_records_file(source, url="http://localhost:8900", scenario="s", progress_path=progress))


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
                await import_records_file(
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
