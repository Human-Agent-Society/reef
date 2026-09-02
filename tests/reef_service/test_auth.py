"""Bearer authentication middleware tests (issue #145)."""

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _make_client(tokens) -> TestClient:
    from reef.service.auth import create_authentication_middleware

    app = web.Application(middlewares=[create_authentication_middleware(tokens)])

    async def _ok(request: web.Request) -> web.Response:
        del request
        return web.Response(text="ok")

    app.router.add_get("/protected", _ok)
    app.router.add_get("/healthz", _ok)
    return TestClient(TestServer(app))


def test_scheme_is_case_insensitive() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
                resp = await client.get("/protected", headers={"Authorization": f"{scheme} secret"})
                assert resp.status == 200, scheme

    asyncio.run(run())


def test_credential_stays_case_sensitive() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            resp = await client.get("/protected", headers={"Authorization": "Bearer SECRET"})
            assert resp.status == 401

    asyncio.run(run())


def test_malformed_or_wrong_scheme_is_rejected() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            for header in ("", "Bearer", "Bearer ", "Basic secret", "BearerSecret secret"):
                resp = await client.get("/protected", headers={"Authorization": header})
                assert resp.status == 401, header
            resp = await client.get("/protected")
            assert resp.status == 401

    asyncio.run(run())


def test_wrong_token_is_rejected() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            resp = await client.get("/protected", headers={"Authorization": "Bearer nope"})
            assert resp.status == 401

    asyncio.run(run())


def test_multiple_accepted_tokens_all_match_any_case() -> None:
    async def run() -> None:
        client = _make_client(["one", "two"])
        async with client:
            first = await client.get("/protected", headers={"Authorization": "bearer one"})
            second = await client.get("/protected", headers={"Authorization": "BEARER two"})
            assert first.status == 200
            assert second.status == 200

    asyncio.run(run())


def test_healthz_reachable_without_credentials() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            resp = await client.get("/healthz")
            assert resp.status == 200

    asyncio.run(run())
