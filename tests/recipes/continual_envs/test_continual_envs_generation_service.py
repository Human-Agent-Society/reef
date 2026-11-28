"""The generation service on Reef Platform's generation API: its calls, its retries, and the task downloads."""

from __future__ import annotations

import asyncio
import io
import tarfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from recipes.beta.continual_envs import generation_service
from recipes.beta.continual_envs.generation_service import GenerationServiceError, PlatformGenerationService

RUN = {"id": "run-1", "state": "queued", "error": None, "created_at": 1700000000.5}


def task_archive(name: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"Solve it.\n"
        member = tarfile.TarInfo(f"{name}/instruction.md")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


class FakePlatform:
    """The generation routes of the Platform. ``failures`` 503 answers come before the first good one of a GET."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[str, str, dict[str, str], object]] = []

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/api/v1/generation/{path:.*}", self.generation)
        app.router.add_get("/downloads/{name}", self.signed)
        return app

    async def generation(self, request: web.Request) -> web.Response:
        body = await request.json() if request.can_read_body else None
        self.calls.append((request.method, request.path, dict(request.query), body))
        if request.headers.get("Authorization") != "Bearer rft_1":
            return web.Response(status=401, text="no token")
        if self.failures:
            self.failures -= 1
            return web.Response(status=503, text="the Cell restarts")
        path = request.match_info["path"]
        if path == "runs" and request.method == "POST":
            return web.json_response(RUN, status=202)
        if path == "runs":
            return web.json_response({"runs": [RUN | {"state": "finished"}]})
        if path == "runs/run-1/items" and "after" not in request.query:
            items = [
                {"id": "seed-2", "run_id": "run-1", "fields": {"task": "fail-000000000002__b"}},
                {"id": "seed-old", "run_id": "run-0", "fields": {"task": "fail-000000000000__old"}},
            ]
            return web.json_response({"items": items, "next": "tasks/seed-2"})
        if path == "runs/run-1/items":
            return web.json_response(
                {
                    "items": [{"id": "seed-1", "run_id": "run-1", "fields": {"task": "fail-000000000001__a"}}],
                    "next": None,
                }
            )
        if path.startswith("tasks/") and path.endswith("/archive"):
            name = path.split("/")[1]
            raise web.HTTPFound(str(request.url.with_path(f"/downloads/{name}").with_query({})))
        if path == "runs/run-1/cancel":
            return web.json_response(RUN | {"state": "cancelled"})
        return web.json_response(RUN)

    async def signed(self, request: web.Request) -> web.Response:
        # A signed address takes no token: the service must not send it there.
        if "Authorization" in request.headers or request.match_info["name"] == "missing":
            return web.Response(status=403)
        return web.Response(body=task_archive(request.match_info["name"]))


def served(platform: FakePlatform, work: Callable[[PlatformGenerationService], Awaitable[object]]) -> object:
    async def run() -> object:
        server = TestServer(platform.app())
        await server.start_server()
        try:
            return await work(PlatformGenerationService(str(server.make_url("")), "rft_1"))
        finally:
            await server.close()

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def no_wait(monkeypatch) -> None:
    monkeypatch.setattr(generation_service, "RETRY_SECONDS", 0.0)


def test_a_run_starts_with_its_sources_and_is_read_back() -> None:
    platform = FakePlatform()

    async def work(service: PlatformGenerationService) -> object:
        sources = [{"name": "traffic", "kind": "records", "id": "generation-00000", "files": {"r/session.json": "{}"}}]
        started = await service.start(spec="terminal-failures", generator="tw", sources=sources, params={}, targets={})
        return started, await service.run("run-1"), await service.runs("tw")

    started, read, listed = served(platform, work)
    assert (started.run_id, started.state, started.created_at, started.ended) == (
        "run-1",
        "queued",
        1700000000.5,
        False,
    )
    assert read.state == "queued" and [run.state for run in listed] == ["finished"]
    method, path, _, body = platform.calls[0]
    assert (method, path) == ("POST", "/api/v1/generation/runs")
    assert body == {
        "spec": "terminal-failures",
        "generator": "tw",
        "sources": [
            {"name": "traffic", "kind": "records", "id": "generation-00000", "files": {"r/session.json": "{}"}}
        ],
        "params": {},
    }
    assert platform.calls[2][2] == {"generator": "tw", "limit": "20"}


def test_the_published_tasks_of_a_run_are_read_over_every_page() -> None:
    platform = FakePlatform()
    names = served(platform, lambda service: service.published_tasks("run-1", "tasks"))
    # An item that another run of the generator wrote is not a task of this run.
    assert names == ("fail-000000000001__a", "fail-000000000002__b")
    assert [call[2] for call in platform.calls] == [
        {"stage": "tasks", "state": "verified", "limit": "500"},
        {"stage": "tasks", "state": "verified", "limit": "500", "after": "tasks/seed-2"},
    ]


def test_a_read_is_tried_again_after_a_failure_that_can_pass_and_a_start_is_not() -> None:
    platform = FakePlatform(failures=2)
    assert served(platform, lambda service: service.run("run-1")).run_id == "run-1"
    assert len(platform.calls) == 3
    platform = FakePlatform(failures=generation_service.ATTEMPTS)
    with pytest.raises(GenerationServiceError, match="answered 503: the Cell restarts"):
        served(platform, lambda service: service.run("run-1"))
    platform = FakePlatform(failures=1)
    with pytest.raises(GenerationServiceError, match="answered 503"):
        served(platform, lambda service: service.start(spec="s", generator="g", sources=[], params={}, targets={}))
    assert len(platform.calls) == 1


def test_a_refusal_is_not_tried_again() -> None:
    async def work(service: PlatformGenerationService) -> object:
        service.headers = {"Authorization": "Bearer wrong"}
        return await service.run("run-1")

    platform = FakePlatform()
    with pytest.raises(GenerationServiceError, match=r"was refused \(401\): no token"):
        served(platform, work)
    assert len(platform.calls) == 1


def test_a_task_downloads_from_its_signed_address_without_the_token(tmp_path: Path) -> None:
    path = served(FakePlatform(), lambda service: service.download_task("fail-000000000001__a", tmp_path))
    assert path == tmp_path / "fail-000000000001__a" and (path / "instruction.md").read_text() == "Solve it.\n"
    with pytest.raises(GenerationServiceError, match="the download of the task missing answered 403"):
        served(FakePlatform(), lambda service: service.download_task("missing", tmp_path))


def test_an_archive_with_another_directory_is_refused(tmp_path: Path, monkeypatch) -> None:
    async def other(self: FakePlatform, request: web.Request) -> web.Response:
        return web.Response(body=task_archive("other-task"))

    monkeypatch.setattr(FakePlatform, "signed", other)
    with pytest.raises(GenerationServiceError, match="holds another directory"):
        served(FakePlatform(), lambda service: service.download_task("fail-000000000001__a", tmp_path))
