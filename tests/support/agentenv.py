"""A local stand-in for the AgentENV API that a trial of a prepared image calls: the status of a template, the cold
start of a sandbox from an image, and the deletion of a sandbox."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence

from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.execution.sandbox.agentenv.build import AgentEnvTemplates

#: The answer of the stand-in to one cold start: ``(status, body)``, with a JSON body for a mapping and an HTML page
#: for a string; ``"drop"``, which closes the connection before any answer; ``"slow"``, which answers 201 after one
#: second; or ``"redirect"``, which answers 307 to the stand-in's ``/elsewhere``.
ColdAnswer = tuple[int, Mapping[str, object] | str] | str
#: The API key that the stand-in accepts.
API_KEY = "key"


class FakeAgentEnvApi:
    """The AgentENV API of one test.

    Values: ``cold_answers`` holds the answers to the cold starts, in order; the last one repeats. ``statuses`` maps
    a template ID to the status of its build; to an HTTP status that its read answers; to ``"slow"``, which answers
    ``ready`` after one second; or to ``"redirect"``, which answers 307 to ``/elsewhere``. A template that it lacks
    answers 404. ``cold_bodies`` keeps the JSON body of each cold start, ``deleted`` the ID of each deleted sandbox,
    and ``followed`` the method of each request to ``/elsewhere``, in order. Each request to the API must carry
    ``API_KEY``.
    """

    def __init__(
        self,
        cold_answers: Sequence[ColdAnswer] = ((201, {"sandboxID": "sandbox-1"}),),
        statuses: Mapping[str, str | int] | None = None,
    ) -> None:
        self.cold_answers = list(cold_answers)
        self.statuses = dict(statuses or {})
        self.cold_bodies: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self.followed: list[str] = []

    def app(self) -> web.Application:
        async def cold(request: web.Request) -> web.StreamResponse:
            assert request.headers["X-API-Key"] == API_KEY
            self.cold_bodies.append(await request.json())
            answer = self.cold_answers.pop(0) if len(self.cold_answers) > 1 else self.cold_answers[0]
            if answer == "drop":
                assert request.transport is not None
                request.transport.close()
                return web.Response(status=201)
            if answer == "slow":
                await asyncio.sleep(1)
                return web.json_response({"sandboxID": "late"}, status=201)
            if answer == "redirect":
                return web.Response(status=307, headers={"Location": "/elsewhere"})
            status, body = answer
            if isinstance(body, str):
                return web.Response(status=status, text=body, content_type="text/html")
            return web.json_response(dict(body), status=status)

        async def status(request: web.Request) -> web.Response:
            assert request.headers["X-API-Key"] == API_KEY
            template = request.match_info["template"]
            assert request.match_info["build"] == template
            answer = self.statuses.get(template)
            if answer is None:
                return web.json_response({"message": "template not found"}, status=404)
            if answer == "invalid-json":
                return web.Response(body=b"{", content_type="application/json")
            if answer == "missing-status":
                return web.json_response({})
            if answer == "redirect":
                return web.Response(status=307, headers={"Location": "/elsewhere"})
            if answer == "slow":
                await asyncio.sleep(1)
                answer = "ready"
            if isinstance(answer, int):
                return web.json_response({"message": "refused"}, status=answer)
            return web.json_response({"templateID": template, "buildID": template, "status": answer})

        async def delete(request: web.Request) -> web.Response:
            assert request.headers["X-API-Key"] == API_KEY
            self.deleted.append(request.match_info["sandbox"])
            return web.Response(status=204)

        async def elsewhere(request: web.Request) -> web.Response:
            self.followed.append(request.method)
            return web.json_response({"sandboxID": "elsewhere", "status": "ready"}, status=201)

        app = web.Application()
        app.router.add_post("/sandboxes-cold", cold)
        app.router.add_get("/templates/{template}/builds/{build}/status", status)
        app.router.add_delete("/sandboxes/{sandbox}", delete)
        app.router.add_route("*", "/elsewhere", elsewhere)
        return app


@contextlib.asynccontextmanager
async def serving(api: FakeAgentEnvApi) -> AsyncIterator[AgentEnvTemplates]:
    """Serve ``api`` on a free local port for the time of the block, and yield the deployment that reaches it."""
    server = TestServer(api.app())
    await server.start_server()
    try:
        yield AgentEnvTemplates(str(server.make_url("")).rstrip("/"), API_KEY)
    finally:
        await server.close()


__all__ = ["API_KEY", "ColdAnswer", "FakeAgentEnvApi", "serving"]
