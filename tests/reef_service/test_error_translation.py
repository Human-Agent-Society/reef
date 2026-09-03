"""The error-to-status table is the service's whole error contract.

Every route raises domain errors and lets ``translate_errors`` answer, so the
table in ``reef/service/errors.py`` decides what a client sees. Two properties
carry the contract and neither is visible from a single route test: the table's
most-specific-first ordering (a subclass must not inherit its base's status),
and the upstream relay, which passes a provider's own 4xx through so an agent
can repair the next attempt but refuses to blame the caller for a 5xx.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact.artifact import ArtifactConflict, ArtifactNotFound, ArtifactSourceError
from reef.artifact.release_chain import ReleaseNotRestorable
from reef.core.errors import ReefError, UnknownScenario
from reef.records import RecordConflict
from reef.runtime.inference import UpstreamStatusError
from reef.service.errors import translate_error, translate_errors
from reef.service.request_service import InferenceRetryTimeout
from reef.surface.weights import RuntimeLoadMismatch


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error", "status"),
    [
        # ArtifactNotFound and ArtifactConflict precede ArtifactError: a
        # missing or conflicting artifact is the caller's answer, not the
        # 503 that its base class reports for unreachable storage.
        (ArtifactNotFound("no such artifact"), 404),
        (ArtifactConflict("artifact moved under you"), 409),
        (ArtifactSourceError("source unreachable"), 503),
        (UnknownScenario("no such scenario"), 404),
        (ReleaseNotRestorable("release has no durable bytes"), 409),
        (RecordConflict("record id already used"), 409),
        (RuntimeLoadMismatch("engine served other weights"), 409),
        (InferenceRetryTimeout("retries exhausted"), 503),
        (ReefError("generic reef failure"), 400),
        (ValueError("bad field"), 400),
    ],
)
def test_domain_errors_translate_to_their_status(error: Exception, status: int) -> None:
    translated = translate_error(error)

    assert translated is not None, type(error).__name__
    assert translated.status == status
    # The message travels: agents read these bodies to repair the next attempt.
    assert str(error) in translated.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("upstream_status", "status"),
    [
        (400, 400),
        (401, 401),
        (403, 403),
        (404, 404),
        (408, 408),
        (409, 409),
        (422, 422),
        (429, 429),
        # 413's aiohttp class needs a max_size argument, so it is deliberately
        # absent from the relay table and falls back to 400 rather than crash.
        (413, 400),
        (451, 400),
        # The upstream's own fault, not the caller's request.
        (500, 502),
        (503, 502),
    ],
)
def test_upstream_statuses_relay_or_become_a_bad_gateway(upstream_status: int, status: int) -> None:
    error = UpstreamStatusError(f"upstream said {upstream_status}", status=upstream_status)

    translated = translate_error(error)

    assert translated is not None
    assert translated.status == status
    assert str(error) in translated.text


@pytest.mark.unit
@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyError("missing"), TypeError("wrong type")])
def test_unmapped_errors_stay_untranslated(error: Exception) -> None:
    assert translate_error(error) is None


@pytest.mark.unit
def test_middleware_answers_mapped_errors_and_lets_the_rest_become_a_500() -> None:
    async def run() -> None:
        app = web.Application(middlewares=[translate_errors])

        async def conflict(request: web.Request) -> web.Response:
            raise RecordConflict("record id already used")

        async def unmapped(request: web.Request) -> web.Response:
            raise RuntimeError("boom")

        app.router.add_get("/conflict", conflict)
        app.router.add_get("/unmapped", unmapped)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            answered = await client.get("/conflict")
            assert answered.status == 409
            assert "record id already used" in await answered.text()

            # An unmapped error is a bug, not a client-facing contract: it
            # stays a 500 and never leaks a translated status.
            crashed = await client.get("/unmapped")
            assert crashed.status == 500
        finally:
            await client.close()

    asyncio.run(run())
