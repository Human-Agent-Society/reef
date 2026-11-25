"""The public route of rollout calls: signed tickets, and the hop to the process that runs the rollout."""

from __future__ import annotations

import asyncio
import socket
import threading
import urllib.error
import urllib.request

import pytest
from aiohttp import web

from reef.harness.harbor.call_route import FORWARDED_HEADER, ROLLOUT_CALL_PATH, RolloutCallRoute

TOKENS = ["service-token"]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def route(port: int, tokens: list[str] = TOKENS) -> RolloutCallRoute:
    """The route of a process at 127.0.0.1 whose ingress is the process itself."""
    return RolloutCallRoute(f"http://127.0.0.1:{port}{ROLLOUT_CALL_PATH}", tokens, address="127.0.0.1", port=port)


def serve_capture_proxy(port: int) -> asyncio.AbstractEventLoop:
    """A stand-in capture proxy on ``port``: it streams the path, the query and the body of each call back in two
    pieces, on a loop of its own."""

    async def answer(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "X-Model": "m"})
        await response.prepare(request)
        await response.write(f"data: {request.path}?{request.query_string}\n\n".encode())
        await response.write(b"data: " + body + b"\n\n")
        await response.write_eof()
        return response

    async def start() -> None:
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", answer)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(start())
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


def status_and_body(url: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, headers=headers or {}, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.headers.get("X-Model") == "m"
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, b""


@pytest.mark.unit
def test_a_ticket_names_its_rollout_and_no_other_key_or_changed_byte_opens_it() -> None:
    owner = RolloutCallRoute("https://calls.example/rollout-calls/cell-f/", TOKENS, address="127.0.0.1", port=8901)
    url = owner.open(41000)
    assert owner.host == "calls.example" and url.startswith("https://calls.example/rollout-calls/cell-f/")
    ticket = url.rsplit("/", 1)[1]
    target = owner.read(ticket)
    assert target is not None and (target.address, target.route_port, target.proxy_port) == ("127.0.0.1", 8901, 41000)
    assert owner.is_local(target) and owner.is_open(target)
    # Another deployment's tokens, or a changed payload, open nothing.
    assert route(8901, ["other-token"]).read(ticket) is None
    assert owner.read(ticket.replace(ticket[2], "A" if ticket[2] != "A" else "B", 1)) is None
    assert owner.read("no-signature") is None
    # A rotation keeps the old tickets good: a process that accepts both tokens reads them.
    assert route(8901, ["a-new-token", *TOKENS]).read(ticket) is not None
    owner.close(41000)
    assert not owner.is_open(target)
    # An unset ${VAR} in the config leaves the setting empty: no route.
    assert RolloutCallRoute.from_settings("", TOKENS, address=None, port=8901) is None
    with pytest.raises(ValueError):
        RolloutCallRoute("https://calls.example", [], address="127.0.0.1", port=8901)
    with pytest.raises(ValueError):
        RolloutCallRoute("ftp://calls.example", TOKENS, address="127.0.0.1", port=8901)


@pytest.mark.unit
def test_a_call_at_any_evaluation_process_reaches_its_rollouts_capture_proxy_and_streams_back() -> None:
    proxy_port = free_port()
    proxy_loop = serve_capture_proxy(proxy_port)
    # Both processes sit behind one ingress, which sends <call_url>/ to ROLLOUT_CALL_PATH/ of either.
    owner, other = route(free_port()), route(free_port())
    owner.start()
    other.start()
    try:
        ticket = owner.open(proxy_port).rsplit("/", 1)[1]
        for process in (owner, other):
            url = f"http://127.0.0.1:{process.port}{ROLLOUT_CALL_PATH}/{ticket}/v1/chat/completions?beta=true"
            assert status_and_body(url, b'{"messages": []}') == (
                200,
                b"data: /v1/chat/completions?beta=true\n\ndata: " + b'{"messages": []}\n\n',
            )
        # A call that one process sent to another for a third's rollout goes no further.
        misdirected = f"http://127.0.0.1:{other.port}{ROLLOUT_CALL_PATH}/{ticket}/v1/models"
        assert status_and_body(misdirected, headers={FORWARDED_HEADER: "1"})[0] == 421
        forged = route(owner.port, ["other-token"]).open(proxy_port).rsplit("/", 1)[1]
        assert status_and_body(f"http://127.0.0.1:{owner.port}{ROLLOUT_CALL_PATH}/{forged}/v1/models")[0] == 404
        owner.close(proxy_port)
        assert status_and_body(f"http://127.0.0.1:{other.port}{ROLLOUT_CALL_PATH}/{ticket}/v1/x", b"{}")[0] == 410
    finally:
        owner.stop()
        other.stop()
        proxy_loop.call_soon_threadsafe(proxy_loop.stop)
    # A stopped route closes its port.
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f"http://127.0.0.1:{owner.port}/healthz", timeout=5)
