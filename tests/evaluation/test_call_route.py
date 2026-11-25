"""The public route of rollout calls: signed tickets, the hop to the process that runs the rollout, and auth."""

from __future__ import annotations

import asyncio
import socket

import pytest
from aiohttp import ClientSession, web

from reef.api.web import service_application
from reef.evaluation.call_route import (
    FORWARDED_HEADER,
    ROLLOUT_CALL_PATH,
    RolloutCallRoute,
    register_rollout_call_route,
)

TOKENS = ["service-token"]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def started(app: web.Application, port: int) -> web.AppRunner:
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def capture_proxy() -> web.Application:
    """A stand-in capture proxy: it streams the path, the query and the body of each call back in two pieces."""

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "X-Model": "m"})
        await response.prepare(request)
        await response.write(f"data: {request.path}?{request.query_string}\n\n".encode())
        await response.write(b"data: " + body + b"\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", answer)
    return app


def route(call_url: str, port: int, tokens: list[str] = TOKENS) -> RolloutCallRoute:
    return RolloutCallRoute(call_url, tokens, address="127.0.0.1", server_port=port)


@pytest.mark.unit
def test_a_ticket_names_its_rollout_and_no_other_key_or_changed_byte_opens_it() -> None:
    owner = route("https://calls.example/rollout-calls/cell-f/", 8900)
    url = owner.open(41000)
    assert owner.host == "calls.example" and url.startswith("https://calls.example/rollout-calls/cell-f/")
    ticket = url.rsplit("/", 1)[1]
    target = owner.read(ticket)
    assert target is not None and (target.address, target.server_port, target.proxy_port) == ("127.0.0.1", 8900, 41000)
    assert owner.is_local(target) and owner.is_open(target)
    # Another deployment's tokens, or a changed payload, open nothing.
    assert route("https://calls.example", 8900, ["other-token"]).read(ticket) is None
    assert owner.read(ticket.replace(ticket[2], "A" if ticket[2] != "A" else "B", 1)) is None
    assert owner.read("no-signature") is None
    # A rotation keeps the old tickets good: a process that accepts both tokens reads them.
    assert route("https://calls.example", 8900, ["a-new-token", *TOKENS]).read(ticket) is not None
    owner.close(41000)
    assert not owner.is_open(target)
    # An unset ${VAR} in the config leaves the setting empty: no route.
    assert RolloutCallRoute.from_settings("", TOKENS, address=None, server_port=8900) is None
    with pytest.raises(ValueError):
        RolloutCallRoute("https://calls.example", [], address="127.0.0.1", server_port=8900)
    with pytest.raises(ValueError):
        RolloutCallRoute("ftp://calls.example", TOKENS, address="127.0.0.1", server_port=8900)


@pytest.mark.unit
def test_a_call_at_any_evaluation_process_reaches_its_rollouts_capture_proxy_and_streams_back() -> None:
    async def run() -> None:
        proxy_port, owner_port, other_port = free_port(), free_port(), free_port()
        # Both processes sit behind one ingress, which sends <call_url>/ to ROLLOUT_CALL_PATH/ of either.
        owner = route(f"http://127.0.0.1:{owner_port}{ROLLOUT_CALL_PATH}", owner_port)
        other = route(f"http://127.0.0.1:{other_port}{ROLLOUT_CALL_PATH}", other_port)
        servers = [await started(capture_proxy(), proxy_port)]
        for process, port in ((owner, owner_port), (other, other_port)):
            app = service_application(TOKENS)
            register_rollout_call_route(app, process)
            servers.append(await started(app, port))
        try:
            ticket = owner.open(proxy_port).rsplit("/", 1)[1]
            async with ClientSession() as client:
                for port in (owner_port, other_port):
                    url = f"http://127.0.0.1:{port}{ROLLOUT_CALL_PATH}/{ticket}/v1/chat/completions?beta=true"
                    async with client.post(url, data=b'{"messages": []}') as response:
                        assert response.status == 200 and response.headers["X-Model"] == "m"
                        assert await response.read() == (
                            b"data: /v1/chat/completions?beta=true\n\ndata: " + b'{"messages": []}\n\n'
                        )
                # A call that one process sent to another for a third's rollout goes no further.
                misdirected = f"http://127.0.0.1:{other_port}{ROLLOUT_CALL_PATH}/{ticket}/v1/models"
                async with client.get(misdirected, headers={FORWARDED_HEADER: "1"}) as response:
                    assert response.status == 421
                forged = route("http://x", owner_port, ["other-token"]).open(proxy_port).rsplit("/", 1)[1]
                async with client.get(f"http://127.0.0.1:{owner_port}{ROLLOUT_CALL_PATH}/{forged}/v1/models") as r:
                    assert r.status == 404
                # The route asks no service token, and every other route still does.
                async with client.get(f"http://127.0.0.1:{owner_port}/reef/scenarios") as response:
                    assert response.status == 401
                owner.close(proxy_port)
                async with client.post(f"http://127.0.0.1:{other_port}{ROLLOUT_CALL_PATH}/{ticket}/v1/x") as r:
                    assert r.status == 410
        finally:
            for server in servers:
                await server.cleanup()

    asyncio.run(run())
