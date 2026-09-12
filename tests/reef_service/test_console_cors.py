"""Browser access must preserve authentication and protect state-changing routes."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.dispatcher import build_default_dispatcher
from reef.service import assembly
from reef.service.auth import create_authentication_middleware
from reef.service.cors import configure_browser_access
from reef.service.deploy.service_config import service_settings_from_config
from reef.storage.sqlite import SQLiteScenarioStorage

ORIGIN = "https://api.reefinfra.ai"


def make_client(origins=(ORIGIN,)):
    app = web.Application(middlewares=[create_authentication_middleware("local-secret")])
    calls = []

    async def mutation(request):
        calls.append(request.path)
        return web.json_response({"scenario": request.headers.get("x-reef-scenario")})

    async def stream(request):
        response = web.StreamResponse(headers={"content-type": "text/event-stream", "x-reef-release-id": "release-1"})
        await response.prepare(request)
        await response.write(b"data: hello\n\n")
        await response.write_eof()
        return response

    app.router.add_post("/reef/train", mutation)
    app.router.add_get("/stream", stream)
    configure_browser_access(app, origins)
    return TestClient(TestServer(app)), calls


def test_preflight_needs_no_token_but_actual_request_does():
    async def run():
        client, calls = make_client()
        async with client:
            headers = {
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type, X-Reef-Scenario",
                "Access-Control-Request-Private-Network": "true",
            }
            preflight = await client.options("/reef/train", headers=headers)
            assert preflight.status == 204
            assert preflight.headers["Access-Control-Allow-Origin"] == ORIGIN
            assert preflight.headers["Access-Control-Allow-Private-Network"] == "true"
            assert "Access-Control-Allow-Credentials" not in preflight.headers
            denied = await client.post("/reef/train", headers={"Origin": ORIGIN})
            assert denied.status == 401
            assert denied.headers["Access-Control-Allow-Origin"] == ORIGIN
            assert calls == []
            accepted = await client.post(
                "/reef/train",
                headers={"Origin": ORIGIN, "Authorization": "Bearer local-secret", "x-reef-scenario": "mine"},
            )
            assert accepted.status == 200
            assert await accepted.json() == {"scenario": "mine"}
            assert len(calls) == 1

    asyncio.run(run())


def test_untrusted_origins_cannot_execute_even_simple_requests():
    async def run():
        client, calls = make_client()
        async with client:
            for origin in ("https://evil.example", "https://api.reefinfra.ai.evil.example", "null"):
                response = await client.post(
                    "/reef/train", headers={"Origin": origin, "Authorization": "Bearer local-secret"}
                )
                assert response.status == 403
                assert "Access-Control-Allow-Origin" not in response.headers
            assert calls == []
            # Existing SDKs have no Origin and retain their prior behavior.
            response = await client.post("/reef/train", headers={"Authorization": "Bearer local-secret"})
            assert response.status == 200
            assert "Access-Control-Allow-Origin" not in response.headers

    asyncio.run(run())


def test_preflight_rejects_unlisted_headers_and_methods():
    async def run():
        client, calls = make_client()
        async with client:
            for method, header in (("PATCH", "authorization"), ("POST", "x-forwarded-host")):
                response = await client.options(
                    "/reef/train",
                    headers={
                        "Origin": ORIGIN,
                        "Access-Control-Request-Method": method,
                        "Access-Control-Request-Headers": header,
                    },
                )
                assert response.status == 403
                assert "Access-Control-Allow-Methods" not in response.headers
            assert calls == []

    asyncio.run(run())


def test_streaming_responses_and_errors_have_cors_headers():
    async def run():
        client, _ = make_client()
        async with client:
            response = await client.get("/stream", headers={"Origin": ORIGIN, "Authorization": "Bearer local-secret"})
            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == ORIGIN
            assert "x-reef-release-id" in response.headers["Access-Control-Expose-Headers"]
            assert "Origin" in response.headers.getall("Vary")
            assert await response.text() == "data: hello\n\n"
            response = await client.get("/missing", headers={"Origin": ORIGIN, "Authorization": "Bearer local-secret"})
            assert response.status == 404
            assert response.headers["Access-Control-Allow-Origin"] == ORIGIN

    asyncio.run(run())


def test_browser_access_is_disabled_by_default():
    async def run():
        client, _ = make_client(())
        async with client:
            response = await client.options(
                "/reef/train", headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST"}
            )
            assert response.status == 401
            assert "Access-Control-Allow-Origin" not in response.headers

    asyncio.run(run())


@pytest.mark.parametrize(
    "origins",
    [
        ["*"],
        ["null"],
        ["https://api.reefinfra.ai/"],
        ["https://name:secret@example.com"],
        ["https://example.com?a=1"],
        ["https://example.com#fragment"],
        ["https://example.com:bad"],
        [12],
        ORIGIN,
    ],
)
def test_config_rejects_non_origins(origins):
    with pytest.raises(ValueError):
        service_settings_from_config({"reef": {"recipe": "recipe", "console_origins": origins}})


def test_config_defaults_and_explicit_origins():
    assert service_settings_from_config({"reef": {"recipe": "recipe"}}).console_origins == ()
    assert service_settings_from_config(
        {"reef": {"recipe": "recipe", "console_origins": [ORIGIN, "http://localhost:3000", ORIGIN]}}
    ).console_origins == (ORIGIN, "http://localhost:3000")


def test_serve_settings_enable_browser_access_on_real_scenario_routes(monkeypatch):
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage())
    dispatcher.get_or_create_scenario("existing-local")
    monkeypatch.setattr(assembly, "build_dispatcher", lambda *args, **kwargs: dispatcher)
    settings = service_settings_from_config(
        {"reef": {"recipe": "recipe", "tokens": ["local-secret"], "console_origins": [ORIGIN]}}
    )

    async def run():
        async with TestClient(TestServer(assembly.build_app(settings))) as client:
            preflight = await client.options(
                "/reef/scenarios",
                headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
            )
            assert preflight.status == 204
            assert preflight.headers["Access-Control-Allow-Origin"] == ORIGIN
            denied = await client.get("/reef/scenarios", headers={"Origin": ORIGIN})
            assert denied.status == 401
            response = await client.get(
                "/reef/scenarios", headers={"Origin": ORIGIN, "Authorization": "Bearer local-secret"}
            )
            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == ORIGIN
            assert [row["scenario"] for row in (await response.json())["scenarios"]] == ["existing-local"]

    asyncio.run(run())
