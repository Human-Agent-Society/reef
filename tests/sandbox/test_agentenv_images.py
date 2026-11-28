"""``reef.execution.sandbox.agentenv.images``: AgentENV's answers about the snapshot of a template, and the native
start of a sandbox from a prepared image, on a local stand-in for the AgentENV API."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from tests.support.agentenv import ColdAnswer, FakeAgentEnvApi, serving

from reef.execution.sandbox.agentenv.images import ColdSandbox, ColdSandboxError, is_missing_snapshot
from reef.execution.sandbox.process import SandboxCapacityError, SandboxRequestError

#: A synthetic image of a task's environment, by digest.
IMAGE = "registry.example:5000/tasks/add@sha256:" + "a" * 64


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "message", "missing"),
    [
        (400, "400: template checked-0 not found", True),
        (400, "400: worker error: template checked-0 not found", True),
        (500, "500: worker error: snapshot 'checked-0' is not ready", True),
        # Another template, a longer ID, or the same text with another status is not this template's answer.
        (400, "400: template checked-1 not found", False),
        (400, "400: template checked-0-copy not found", False),
        (400, "400: subtemplate checked-0 not found", False),
        (500, "500: template checked-0 not found", False),
        (400, "400: snapshot 'checked-0' is not ready", False),
        (404, "404: template checked-0 not found", False),
        (503, "503: no nodes available", False),
        (400, "400: cpuCount must be at most 16", False),
    ],
)
def test_only_agentenvs_answers_of_a_missing_snapshot_of_the_template_count(
    status: int, message: str, missing: bool
) -> None:
    assert is_missing_snapshot(status, message, "checked-0") is missing


def cold_sandbox(**changes: object) -> ColdSandbox:
    """The cold start of ``IMAGE`` for a trial of an allowlisted task with 2 CPUs and 4096 MB, with ``changes``."""
    fields: dict[str, object] = {
        "image": IMAGE,
        "timeout_seconds": 4920,
        "allow_internet_access": True,
        "network": {"allow_out": ["pypi.org"], "deny_out": ["0.0.0.0/0"]},
        "metadata": {"environment_name": "add", "session_id": "t__env"},
        "env_vars": {"TASK_SETTING": "from-the-task"},
        "cpu_count": 2,
        "memory_mb": 4096,
    }
    return ColdSandbox(**{**fields, **changes})


def cold_start(api: FakeAgentEnvApi, wait_seconds: float = 30.0) -> str:
    """``cold_sandbox().create`` on ``api``, served on a free local port."""

    async def run() -> str:
        async with serving(api) as templates:
            return await cold_sandbox().create(templates, wait_seconds)

    return asyncio.run(run())


@pytest.mark.unit
def test_a_cold_start_sends_the_values_of_the_template_start_and_returns_the_sandbox() -> None:
    api = FakeAgentEnvApi([(201, {"sandboxID": "sandbox-7", "templateID": "cold"})])
    assert cold_start(api) == "sandbox-7"
    assert api.cold_bodies == [
        {
            "image": IMAGE,
            "timeout": 4920,
            # Deleted at the end of its lifetime, as a template's sandbox of a trial is.
            "autoPause": False,
            "secure": True,
            "allowInternetAccess": True,
            "network": {"allowOut": ["pypi.org"], "denyOut": ["0.0.0.0/0"]},
            "metadata": {"environment_name": "add", "session_id": "t__env"},
            "envVars": {"TASK_SETTING": "from-the-task"},
            "cpuCount": 2,
            "memoryMB": 4096,
        }
    ]
    # No internet at creation, and AgentENV's default resources, are sent as they are.
    plain = cold_sandbox(allow_internet_access=False, network=None, cpu_count=None, memory_mb=None).body()
    assert plain["allowInternetAccess"] is False
    assert not {"network", "cpuCount", "memoryMB"} & set(plain)


@pytest.mark.unit
def test_a_cold_start_refuses_a_network_option_that_it_cannot_send() -> None:
    with pytest.raises(ValueError, match="cannot apply the network options block_dns"):
        cold_sandbox(network={"deny_out": ["0.0.0.0/0"], "block_dns": True})


@pytest.mark.unit
def test_a_cold_start_retries_503_with_backoff_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.execution.sandbox.agentenv import images as agentenv_images

    real_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        if seconds == 0:
            await real_sleep(0)
            return
        sleeps.append(seconds)

    monkeypatch.setattr(agentenv_images.asyncio, "sleep", fake_sleep)
    busy = (503, {"message": "no nodes available"})
    api = FakeAgentEnvApi([busy, busy, (201, {"sandboxID": "sandbox-recovered"})])
    assert cold_start(api) == "sandbox-recovered"
    assert len(api.cold_bodies) == 3
    assert len(sleeps) == 2 and all(delay > 0 for delay in sleeps)


@pytest.mark.unit
def test_a_cold_start_surfaces_capacity_error_when_503_budget_exhausts(monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.execution.sandbox.agentenv import images as agentenv_images

    real_sleep = asyncio.sleep
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        if seconds == 0:
            await real_sleep(0)
            return
        sleeps.append(seconds)

    monkeypatch.setattr(agentenv_images.asyncio, "sleep", fake_sleep)
    api = FakeAgentEnvApi([(503, {"message": "no nodes available"})])
    with pytest.raises(SandboxCapacityError, match="HTTP 503") as raised:
        cold_start(api)
    assert len(api.cold_bodies) == agentenv_images.COLD_START_MAX_ATTEMPTS
    assert len(sleeps) == agentenv_images.COLD_START_MAX_ATTEMPTS - 1
    assert f"{agentenv_images.COLD_START_MAX_ATTEMPTS} attempts" in str(raised.value)


@pytest.mark.unit
def test_a_cold_start_stops_retrying_503_when_wait_budget_is_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.execution.sandbox.agentenv import images as agentenv_images

    monkeypatch.setattr(agentenv_images, "COLD_START_FIRST_WAIT_SECONDS", 10.0)
    api = FakeAgentEnvApi([(503, {"message": "no nodes available"})])
    with pytest.raises(SandboxCapacityError, match="HTTP 503"):
        cold_start(api, wait_seconds=0.5)
    assert len(api.cold_bodies) == 1


@pytest.mark.unit
def test_a_cold_start_sends_no_503_retry_when_its_wait_ends_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reef.execution.sandbox.agentenv import images as agentenv_images

    real_sleep = asyncio.sleep

    async def late_sleep(seconds: float) -> None:
        # A stalled event loop: the wait ends after the deadline of the start.
        await real_sleep(0.4 if seconds else 0)

    monkeypatch.setattr(agentenv_images, "COLD_START_FIRST_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(agentenv_images.asyncio, "sleep", late_sleep)
    api = FakeAgentEnvApi([(503, {"message": "no nodes available"}), (201, {"sandboxID": "sandbox-late"})])
    # The capacity error keeps the rollout on its deferral path; a late request could time out as ColdSandboxError.
    with pytest.raises(SandboxCapacityError, match="after 1 attempts"):
        cold_start(api, wait_seconds=0.3)
    assert len(api.cold_bodies) == 1


@pytest.mark.unit
def test_a_cold_start_does_not_resend_after_payload_error_on_201() -> None:
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    from reef.execution.sandbox.agentenv.build import AgentEnvTemplates

    requests = 0

    async def truncated_201(request: web.Request) -> web.StreamResponse:
        nonlocal requests
        requests += 1
        response = web.StreamResponse(status=201, headers={"Content-Length": "100"})
        await response.prepare(request)
        await response.write(b'{"sandboxID": "s')
        assert request.transport is not None
        request.transport.close()
        return response

    async def run() -> None:
        app = web.Application()
        app.router.add_post("/sandboxes-cold", truncated_201)
        server = TestServer(app)
        await server.start_server()
        try:
            templates = AgentEnvTemplates(str(server.make_url("")).rstrip("/"), "key")
            with pytest.raises(ColdSandboxError, match="lost its answer"):
                await cold_sandbox().create(templates, 30.0)
        finally:
            await server.close()

    asyncio.run(run())
    assert requests == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("answer", "raised", "match"),
    [
        ((429, {"message": "too many requests"}), SandboxCapacityError, "HTTP 429"),
        ((400, {"message": "cpuCount must be at most 16"}), SandboxRequestError, "at most 16"),
        # A gateway or proxy stopped waiting, or the connection dropped: AgentENV can still start the sandbox, so the
        # caller sends the request once and raises ColdSandboxError instead of leaking another sandbox or deferring
        # forever.
        ((502, "<html>502 Bad Gateway</html>"), ColdSandboxError, "HTTP 502"),
        ((504, "<html>Gateway Time-out</html>"), ColdSandboxError, "HTTP 504"),
        ((500, "<html>Internal Server Error</html>"), ColdSandboxError, "HTTP 500"),
        ("slow", ColdSandboxError, "in time"),
        ("drop", ColdSandboxError, "lost its answer"),
        ((201, {"templateID": "cold"}), ColdSandboxError, "named no sandbox ID"),
        # A redirect is not followed: the create is not sent again, and the API key goes to no other address.
        ("redirect", ColdSandboxError, "HTTP 307"),
        # A credential or service error keeps its meaning.
        ((401, {"message": "invalid API key"}), ColdSandboxError, "HTTP 401"),
        ((403, {"message": "image not allowed"}), ColdSandboxError, "HTTP 403"),
        ((500, {"message": "image pull failed"}), ColdSandboxError, "image pull failed"),
    ],
    ids=[
        "rate",
        "bad-request",
        "bad-gateway",
        "gateway",
        "proxy-500",
        "slow",
        "dropped",
        "no-id",
        "redirect",
        "unauthorized",
        "forbidden",
        "service",
    ],
)
def test_a_cold_start_that_does_not_start_raises_by_its_answer_and_is_sent_once(
    answer: ColdAnswer, raised: type[Exception], match: str
) -> None:
    api = FakeAgentEnvApi([answer])
    with pytest.raises(raised, match=match):
        cold_start(api, wait_seconds=0.2 if answer == "slow" else 30.0)
    assert (len(api.cold_bodies), api.followed) == (1, [])


@pytest.mark.unit
def test_only_a_ready_template_is_ready() -> None:
    statuses: dict[str, str | int] = {"ready-1": "ready", "building-1": "building", "failed-1": "error"}
    api = FakeAgentEnvApi(statuses={**statuses, "denied-1": 403, "moved-1": "redirect", "slow-1": "slow"})

    async def read(*templates: str, wait_seconds: float = 30.0) -> list[bool]:
        async with serving(api) as deployment:
            return [await deployment.is_ready(template, wait_seconds) for template in templates]

    assert asyncio.run(read(*statuses, "gone-1")) == [True, False, False, False]
    # A refused read, and a redirect, which the read does not follow with the API key, raise.
    with pytest.raises(aiohttp.ClientResponseError, match="403"):
        asyncio.run(read("denied-1"))
    with pytest.raises(aiohttp.ClientResponseError, match="redirected"):
        asyncio.run(read("moved-1"))
    assert api.followed == []
    # A read that gets no answer in its time ends.
    with pytest.raises(TimeoutError):
        asyncio.run(read("slow-1", wait_seconds=0.2))
