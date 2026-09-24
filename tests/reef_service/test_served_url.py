"""The address a recipe's own evaluation calls reach the service at."""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit

import pytest
from aiohttp import web

from reef.service.assembly import _served_url


@pytest.mark.unit
@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", "http://127.0.0.1:8900"),
        ("::", "http://[::1]:8900"),
        ("", "http://127.0.0.1:8900"),
        ("127.0.0.1", "http://127.0.0.1:8900"),
        ("10.0.0.5", "http://10.0.0.5:8900"),
        ("::1", "http://[::1]:8900"),
        ("fe80::1", "http://[fe80::1]:8900"),
        ("reef.internal", "http://reef.internal:8900"),
    ],
)
def test_the_served_url_names_an_address_the_service_reaches_itself_at(host: str, expected: str) -> None:
    assert _served_url(host, 8900) == expected


@pytest.mark.unit
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "127.0.0.1", "::1"])
def test_the_served_url_answers_for_a_real_bind(host: str) -> None:
    """The service binds as aiohttp binds it, and a connection to the served URL reaches it."""

    async def bind_and_connect() -> None:
        app = web.Application()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, 0)
        try:
            await site.start()
        except OSError as exc:
            await runner.cleanup()
            pytest.skip(f"this host cannot bind {host}: {exc}")
        try:
            server = site._server
            assert server is not None
            port = server.sockets[0].getsockname()[1]
            address = urlsplit(_served_url(host, port))
            _reader, writer = await asyncio.open_connection(address.hostname, address.port)
            writer.close()
        finally:
            await runner.cleanup()

    asyncio.run(bind_and_connect())
