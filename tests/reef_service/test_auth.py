"""Bearer authentication middleware tests (issue #145)."""

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _make_client(tokens, evaluation_tokens=None) -> TestClient:
    from reef.service.auth import create_authentication_middleware

    app = web.Application(middlewares=[create_authentication_middleware(tokens, evaluation_tokens=evaluation_tokens)])

    async def _ok(request: web.Request) -> web.Response:
        del request
        return web.Response(text="ok")

    app.router.add_get("/protected", _ok)
    app.router.add_get("/healthz", _ok)
    # The two harness pages a browser opens by a link, and their neighbours that are no page.
    app.router.add_get("/reef/harness/requests/{record_id}/page", _ok)
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/page", _ok)
    app.router.add_post("/reef/harness/requests/{record_id}/page", _ok)
    app.router.add_get("/reef/harness/releases", _ok)
    app.router.add_get(r"/reef/harness/releases/{step:\d{1,9}}/records", _ok)
    app.router.add_get("/reef/harness/requests/{record_id}/progress", _ok)
    app.router.add_post("/reef/scenarios/{scenario}/evaluation/v1/{route:.+}", _ok)
    app.router.add_post("/reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route:.+}", _ok)
    app.router.add_get("/reef/scenarios/{scenario}/evaluation/v1/{route:.+}", _ok)
    app.router.add_post("/v1/chat/completions", _ok)
    app.router.add_post("/reef/scenarios/{scenario}/rollback", _ok)
    return TestClient(TestServer(app))


def test_scheme_is_case_insensitive() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            for scheme in ("Bearer", "bearer", "BEARER", "BeArEr"):
                resp = await client.get("/protected", headers={"Authorization": f"{scheme} secret"})
                assert resp.status == 200, scheme

    asyncio.run(run())


def test_the_anthropic_dialect_presents_the_token_in_its_own_header() -> None:
    """An evaluation episode or proposer bound through the Anthropic dialect sends x-api-key, not Bearer."""

    async def run() -> None:
        client = _make_client("secret")
        async with client:
            resp = await client.get("/protected", headers={"x-api-key": "secret"})
            assert resp.status == 200
            resp = await client.get("/protected", headers={"x-api-key": "wrong"})
            assert resp.status == 401
            # The Bearer header is judged when both are present.
            resp = await client.get("/protected", headers={"Authorization": "Bearer wrong", "x-api-key": "secret"})
            assert resp.status == 401

    asyncio.run(run())


def test_an_evaluation_token_opens_the_evaluation_routes_alone() -> None:
    """The token a recipe's evaluation calls carry runs candidate code: it samples the served release through the
    evaluation routes, in either header, and every other route refuses it."""

    async def run() -> None:
        client = _make_client("secret", evaluation_tokens="episode")
        async with client:
            for path in (
                "/reef/scenarios/agent/evaluation/v1/chat/completions",
                "/reef/scenarios/team%2Fagent/evaluation/v1/messages",
                "/reef/scenarios/agent/components/harness/evaluation/v1/responses",
            ):
                resp = await client.post(path, headers={"Authorization": "Bearer episode"})
                assert resp.status == 200, path
                resp = await client.post(path, headers={"x-api-key": "episode"})
                assert resp.status == 200, path
                resp = await client.post(path, headers={"Authorization": "Bearer secret"})
                assert resp.status == 200, path
            for method, path in (
                ("GET", "/protected"),
                ("GET", "/reef/scenarios/agent/evaluation/v1/chat/completions"),
                ("POST", "/v1/chat/completions"),
                ("POST", "/reef/scenarios/agent/rollback"),
            ):
                resp = await client.request(method, path, headers={"Authorization": "Bearer episode"})
                assert resp.status == 401, (method, path)

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


def test_the_two_pages_accept_the_token_as_a_query_parameter() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            for path in ("/reef/harness/requests/3f1c2a9d0b7e/page", "/reef/harness/releases/3/page"):
                resp = await client.get(path, params={"token": "secret"})
                assert resp.status == 200, path
                resp = await client.get(path, params={"token": "nope"})
                assert resp.status == 401, path
                resp = await client.get(path, params={"token": ""})
                assert resp.status == 401, path
                resp = await client.get(path)
                assert resp.status == 401, path

    asyncio.run(run())


def test_the_query_token_is_refused_off_the_two_pages() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            for path in ("/protected", "/reef/harness/releases", "/reef/harness/releases/3/records"):
                resp = await client.get(path, params={"token": "secret"})
                assert resp.status == 401, path
            # The page path with another method is no page a browser opens.
            resp = await client.post("/reef/harness/requests/3f1c2a9d0b7e/page", params={"token": "secret"})
            assert resp.status == 401

    asyncio.run(run())


def test_the_authorization_header_wins_over_the_query_token() -> None:
    async def run() -> None:
        client = _make_client("secret")
        async with client:
            path = "/reef/harness/releases/3/page"
            resp = await client.get(path, params={"token": "secret"}, headers={"Authorization": "Bearer nope"})
            assert resp.status == 401
            resp = await client.get(path, params={"token": "secret"}, headers={"Authorization": "Basic secret"})
            assert resp.status == 401
            resp = await client.get(path, params={"token": "nope"}, headers={"Authorization": "Bearer secret"})
            assert resp.status == 200

    asyncio.run(run())


def test_a_page_key_opens_the_two_pages_of_its_scenario_and_nothing_else() -> None:
    """The links a session's model reads carry the scenario's page key, never the token: the key opens that
    scenario's request and step pages, and 401s on another scenario and on every other route."""
    from reef.harness.page_key import page_key

    async def run() -> None:
        client = _make_client(["old", "secret"])
        key = page_key("secret", "mine")
        async with client:
            for path in ("/reef/harness/requests/3f1c2a9d0b7e/page", "/reef/harness/releases/3/page"):
                resp = await client.get(path, params={"scenario": "mine", "key": key})
                assert resp.status == 200, path
                resp = await client.get(path, params={"scenario": "mine", "key": page_key("old", "mine")})
                assert resp.status == 200, path
                # Another scenario, by the query or by the header the page would render instead.
                resp = await client.get(path, params={"scenario": "theirs", "key": key})
                assert resp.status == 401, path
                resp = await client.get(
                    path, params={"scenario": "mine", "key": key}, headers={"x-reef-scenario": "b"}
                )
                assert resp.status == 401, path
                for params in ({"key": key}, {"scenario": "mine", "key": page_key("nope", "mine")}):
                    resp = await client.get(path, params=params)
                    assert resp.status == 401, path
            for path in (
                "/protected",
                "/reef/harness/releases",
                "/reef/harness/releases/3/records",
                "/reef/harness/requests/3f1c2a9d0b7e/progress",
            ):
                resp = await client.get(path, params={"scenario": "mine", "key": key})
                assert resp.status == 401, path
            resp = await client.post(
                "/reef/harness/requests/3f1c2a9d0b7e/page", params={"scenario": "mine", "key": key}
            )
            assert resp.status == 401

    asyncio.run(run())
