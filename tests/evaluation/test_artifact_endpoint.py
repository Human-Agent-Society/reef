from __future__ import annotations

import asyncio
import gc
import json
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import aiohttp
import pytest
from tests.support.polling import wait_until

from reef.artifact.artifact import Artifact
from reef.core.errors import UpstreamLimit, UpstreamStatusError
from reef.core.evaluation import PolicyModel
from reef.core.served_models import ServedArtifact
from reef.core.snapshot import SnapshotRef
from reef.evaluation import artifact_endpoint
from reef.evaluation.artifact_endpoint import CLOSED_ERROR, ArtifactEndpoint, ProviderCooldowns, refusal_limit
from reef.harness.capture import CaptureProxy
from reef.inference.interfaces import REQUEST_SESSION_KEY, InferenceArtifact, InferenceHandler, InferenceRuntime


class RefusingHandler(InferenceHandler):
    """Answers a call whose last message says ``ok``, refuses ``bad`` with 422, and fails on anything else."""

    async def inference(self, artifact, path, payload):
        content = payload["messages"][-1]["content"]
        if content == "ok":
            return {"artifact_id": artifact.artifact_id, "path": path}
        if content == "bad":
            raise UpstreamStatusError("the model refused the request", status=422)
        raise ValueError("the sampler returned nothing")


class Runtime(InferenceRuntime):
    """A runtime without engines that answers through ``RefusingHandler``.

    Values: ``any_artifact`` is ``serves_any_artifact``. With it, ``artifact_model`` names the base model at the
    artifact's revision; without it, the base refusal applies. ``activate_checkpoint`` loads nothing and returns
    "load-0". ``inference_timeout_s`` is the runtime's request timeout. Purpose: the tests build an
    ``ArtifactEndpoint`` on it.
    """

    def __init__(self, *, any_artifact: bool, inference_timeout_s: float = 300.0) -> None:
        super().__init__(base_url="http://runtime.invalid", inference_timeout_s=inference_timeout_s)
        self.any_artifact = any_artifact
        self.handler = RefusingHandler()

    @property
    def inference_handler(self):
        return self.handler

    @property
    def serves_any_artifact(self) -> bool:
        return self.any_artifact

    def artifact_model(self, artifact):
        if not self.any_artifact:
            return super().artifact_model(artifact)
        return PolicyModel("base", artifact.ref.revision)

    def activate_checkpoint(self, artifact, *, scenario=None):
        return "load-0"

    def served_artifact(self, artifact_id: str, scenario: str) -> ServedArtifact | None:
        """No route: these tests give each artifact endpoint its route themselves."""
        return None


def call(url: str, path: str, body: object) -> tuple[int, dict]:
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class EngineHandler(InferenceHandler):
    """An engine's handler, which sets its own sampling values and keeps each body."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.session_keys: list[str | None] = []

    def request_payload(self, payload: Mapping[str, Any], *, serving_config: Mapping[str, Any]) -> dict[str, Any]:
        return {**payload, "temperature": 0.2}

    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(payload)
        self.session_keys.append(REQUEST_SESSION_KEY.get())
        return {"artifact_id": artifact.artifact_id}


class StartedElsewhereRuntime(Runtime):
    """A runtime that another process started an engine of: it has no handler of its own, as a Modal runtime has none
    before an activation."""

    def __init__(self) -> None:
        super().__init__(any_artifact=True)
        self.engine = EngineHandler()

    @property
    def inference_handler(self) -> InferenceHandler:
        raise RuntimeError("no engine serves yet")

    def handler_for(self, url: str) -> InferenceHandler | None:
        return self.engine


@pytest.mark.unit
def test_a_call_reaches_an_engine_that_another_process_started_with_the_engines_values() -> None:
    artifact = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = StartedElsewhereRuntime()
    served = ServedArtifact("artifact:r7", "https://engine.invalid", "engine-model")
    with ArtifactEndpoint(runtime, artifact, served) as endpoint:
        chat = {"model": "from-the-agent", "messages": [{"role": "user", "content": "ok"}]}
        assert call(endpoint.url, "/v1/chat/completions", chat) == (200, {"artifact_id": "artifact:r7"})
    assert runtime.engine.bodies == [{**chat, "model": "engine-model", "temperature": 0.2}]


@pytest.mark.unit
def test_each_call_of_an_endpoint_has_its_session_key_and_no_other_call_sees_it() -> None:
    artifact = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = StartedElsewhereRuntime()
    served = ServedArtifact("artifact:r7", "https://engine.invalid", "engine-model")
    chat = {"model": "from-the-agent", "messages": [{"role": "user", "content": "ok"}]}
    # Two rollouts send the same first messages, as the tasks of one benchmark can.
    with (
        ArtifactEndpoint(runtime, artifact, served, session_key="rollout-0") as first,
        ArtifactEndpoint(runtime, artifact, served, session_key="rollout-1") as second,
    ):
        for url in (first.url, second.url, first.url):
            assert call(url, "/v1/chat/completions", chat)[0] == 200
    assert runtime.engine.session_keys == ["rollout-0", "rollout-1", "rollout-0"]
    assert REQUEST_SESSION_KEY.get() is None


@pytest.mark.unit
def test_an_endpoint_with_log_probs_asks_each_protocol_for_log_probabilities_and_token_ids() -> None:
    artifact = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = StartedElsewhereRuntime()
    served = ServedArtifact("artifact:r7", "https://engine.invalid", "engine-model")
    chat = {"messages": [{"role": "user", "content": "ok"}], "logprobs": False, "top_logprobs": 3}
    messages = {"messages": [{"role": "user", "content": "ok"}], "max_tokens": 8}
    responses = {"input": "ok", "include": ["reasoning.encrypted_content"]}
    with ArtifactEndpoint(runtime, artifact, served, log_probs=True) as endpoint:
        for path, body in (("/v1/chat/completions", chat), ("/v1/messages", messages), ("/v1/responses", responses)):
            assert call(endpoint.url, path, body)[0] == 200
    engine = {"model": "engine-model", "temperature": 0.2}
    assert runtime.engine.bodies == [
        {**chat, **engine, "logprobs": True, "return_token_ids": True, "return_prompt_token_ids": True},
        {**messages, **engine, "logprobs": True, "return_token_ids": True},
        {**engine, "input": "ok", "include": ["reasoning.encrypted_content", "message.output_text.logprobs"]},
    ]


@pytest.mark.unit
def test_the_endpoint_answers_every_call_from_its_artifact_and_maps_each_failure_to_a_status() -> None:
    artifact = Artifact(
        SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7"
    )
    runtime = Runtime(any_artifact=True)
    with ArtifactEndpoint(runtime, artifact) as endpoint:
        chat = {"messages": [{"role": "user", "content": "ok"}]}
        assert call(endpoint.url, "/v1/chat/completions", chat) == (
            200,
            {"artifact_id": "artifact:r7", "path": "/v1/chat/completions"},
        )
        assert call(endpoint.url, "/v1/embeddings", chat)[0] == 404
        assert call(endpoint.url, "/v1/chat/completions", [1, 2])[0] == 400
        assert call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "bad"}]})[0] == 422
        status, body = call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "?"}]})
        assert status == 502 and "the sampler returned nothing" in body["error"]["message"]
        assert runtime.inference_admission_status == {"open": True, "active": 0}, "every call releases admission"
        runtime.close_admission()
        assert call(endpoint.url, "/v1/chat/completions", chat)[0] == 503
    elsewhere = ServedArtifact("artifact:r7", "http://elsewhere.invalid", "m")
    with ArtifactEndpoint(Runtime(any_artifact=True), artifact, elsewhere) as endpoint:
        status, body = call(endpoint.url, "/v1/chat/completions", {"messages": [{"content": "ok"}]})
        assert status == 502 and "no handler" in body["error"]["message"]
    with pytest.raises(ValueError, match="serves only its current artifact"):
        ArtifactEndpoint(Runtime(any_artifact=False), artifact)


ARTIFACT = Artifact(SnapshotRef(revision="storage:checkpoint", parent_revision=None), None, artifact_id="artifact:r7")
CHAT = {"messages": [{"role": "user", "content": "add 2 and 3"}]}
REPLY = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "five"}, "finish_reason": "stop"}]}


def refused(status: int, **fields: Any) -> UpstreamStatusError:
    return UpstreamStatusError(f"inference provider returned {status}: busy", status=status, **fields)


def rate_limited(status: int = 429, wait_seconds: float | None = None) -> UpstreamStatusError:
    return refused(status, retry_after_seconds=wait_seconds, limit=UpstreamLimit.RATE)


class ScriptedHandler(InferenceHandler):
    """A provider's handler that meets each try with the next of ``outcomes``, and with the last one after them.

    An outcome is a reply that it returns or an error that it raises. Before it, the try waits the next of ``delays``
    in seconds, and the last one after them. ``payloads`` keeps the payload of each try, and ``sent_at`` the
    ``time.monotonic()`` time at which it came, in order. Purpose: the tests count the tries of a call, check that
    each try sends the same call, and check when each try came.
    """

    def __init__(self, *outcomes: Mapping[str, Any] | BaseException, delays: Sequence[float] = (0.0,)) -> None:
        self.outcomes = outcomes
        self.delays = delays
        self.payloads: list[dict[str, Any]] = []
        self.sent_at: list[float] = []

    async def inference(self, artifact: InferenceArtifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.payloads.append(payload)
        self.sent_at.append(time.monotonic())
        index = len(self.payloads) - 1
        await asyncio.sleep(self.delays[min(index, len(self.delays) - 1)])
        outcome = self.outcomes[min(index, len(self.outcomes) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return dict(outcome)


def scripted(
    *outcomes: Mapping[str, Any] | BaseException, delays: Sequence[float] = (0.0,), inference_timeout_s: float = 300.0
) -> tuple[Runtime, ScriptedHandler]:
    runtime = Runtime(any_artifact=True, inference_timeout_s=inference_timeout_s)
    runtime.handler = ScriptedHandler(*outcomes, delays=delays)
    return runtime, runtime.handler


def answered(url: str, body: object) -> tuple[int, dict, str | None]:
    """The status, the JSON body and the ``Retry-After`` header of one chat call to ``url``."""
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read()), response.headers.get("Retry-After")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read()), exc.headers.get("Retry-After")


@pytest.fixture
def short_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_FIRST_WAIT_SECONDS", 0.01)


@pytest.mark.unit
def test_a_call_that_a_rate_limit_refused_is_sent_again_unchanged_until_it_is_answered(short_waits) -> None:
    # A 429, then OpenRouter's temporary in-flight 402 with no wait to name, then the answer.
    runtime, handler = scripted(rate_limited(), rate_limited(402, wait_seconds=0.0), REPLY)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        assert answered(endpoint.url, CHAT) == (200, REPLY, None)
    assert handler.payloads == [CHAT, CHAT, CHAT]
    assert runtime.inference_admission_status == {"open": True, "active": 0}, "each try releases its admission"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "status", "code", "wait_seconds"),
    [
        (refused(402, limit=UpstreamLimit.PAYMENT), 402, "payment_required", None),
        # A payment refusal is not sent again, also when an intermediary added a wait; the wait reaches the agent.
        (refused(402, limit=UpstreamLimit.PAYMENT, retry_after_seconds=5.0), 402, "payment_required", 5),
        (refused(429, limit=UpstreamLimit.PAYMENT), 429, "payment_required", None),
        # An older handler's 402 that names no limit requires payment, and its 401 refuses access. Only the handler
        # that read a 403's body names its limit.
        (refused(402), 402, "payment_required", None),
        (refused(401), 401, "access_denied", None),
        (refused(403), 403, None, None),
        (refused(403, limit=UpstreamLimit.ACCESS), 403, "access_denied", None),
        # A 5xx keeps the provider's wait in its body, but no Retry-After header tells the agent to wait.
        (refused(503, retry_after_seconds=7.0), 503, None, 7),
        # A limit on a status that names none is no limit.
        (refused(503, limit=UpstreamLimit.RATE), 503, None, None),
        (TimeoutError("the provider took too long"), 502, None, None),
        (ConnectionResetError("the provider closed the connection"), 502, None, None),
    ],
)
def test_no_other_failure_is_sent_again_and_a_refusal_keeps_its_status_limit_and_wait(
    short_waits, failure: BaseException, status: int, code: str | None, wait_seconds: int | None
) -> None:
    runtime, handler = scripted(failure, REPLY)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        answer_status, body, retry_after = answered(endpoint.url, CHAT)
    header = None if wait_seconds is None or status not in artifact_endpoint.REFUSAL_STATUSES else str(wait_seconds)
    assert (answer_status, body["error"].get("code"), retry_after) == (status, code, header)
    assert body["error"].get("retry_after_seconds") == wait_seconds
    assert len(handler.payloads) == 1


@pytest.mark.unit
def test_a_rate_limited_call_has_a_bounded_number_of_tries_and_no_wait_past_its_time(short_waits) -> None:
    runtime, handler = scripted(rate_limited())
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        status, body, retry_after = answered(endpoint.url, CHAT)
    assert (status, body["error"]["code"], retry_after) == (429, "rate_limit", None)
    assert len(handler.payloads) == artifact_endpoint.RATE_LIMIT_TRIES
    # A wait that ends after RATE_LIMIT_SECONDS is not made: the agent gets the provider's wait at once.
    runtime, handler = scripted(rate_limited(wait_seconds=artifact_endpoint.RATE_LIMIT_SECONDS + 1))
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        started = time.monotonic()
        status, body, retry_after = answered(endpoint.url, CHAT)
    assert (status, retry_after, body["error"]["retry_after_seconds"], len(handler.payloads)) == (429, "61", 61, 1)
    assert time.monotonic() - started < 5


@pytest.mark.unit
def test_the_runtimes_request_timeout_bounds_every_try_after_the_first(short_waits) -> None:
    # A two-second wait passes a half-second timeout: no second try, and the agent gets the provider's wait.
    runtime, handler = scripted(rate_limited(wait_seconds=2.0), REPLY, inference_timeout_s=0.5)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        status, body, retry_after = answered(endpoint.url, CHAT)
    assert (status, body["error"]["code"], retry_after, len(handler.payloads)) == (429, "rate_limit", "2", 1)
    # A try after a wait gets only the time left of the timeout, and a try that times out is not sent again.
    runtime, handler = scripted(rate_limited(wait_seconds=0.0), REPLY, delays=(0.0, 30.0), inference_timeout_s=0.5)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        started = time.monotonic()
        status, body, _ = answered(endpoint.url, CHAT)
    assert (status, len(handler.payloads)) == (502, 2) and time.monotonic() - started < 5
    # A first try with no wait before it keeps the handler's own timeout, also past the runtime's.
    runtime, handler = scripted(REPLY, delays=(0.3,), inference_timeout_s=0.1)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        assert answered(endpoint.url, CHAT) == (200, REPLY, None)


@pytest.mark.unit
def test_a_call_that_waits_to_try_again_ends_when_its_endpoint_closes_or_its_task_is_cancelled() -> None:
    runtime, handler = scripted(rate_limited(wait_seconds=30.0), REPLY)
    with ThreadPoolExecutor(1) as pool:
        with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
            pending = pool.submit(answered, endpoint.url, CHAT)
            wait_until(lambda: len(handler.payloads) == 1)
            closed_at = time.monotonic()
        # The rollout ended: the call answers with the refusal at once, and the provider gets no second try.
        status, body, retry_after = pending.result(5)
    assert (status, body["error"]["code"]) == (429, "rate_limit")
    assert body["error"]["message"] == "inference provider returned 429: busy"
    assert 25 <= int(retry_after) <= 30 and body["error"]["retry_after_seconds"] == int(retry_after)
    assert time.monotonic() - closed_at < 5 and len(handler.payloads) == 1

    runtime, handler = scripted(rate_limited(wait_seconds=30.0), REPLY)
    endpoint = ArtifactEndpoint(runtime, ARTIFACT)

    async def cancel_during_the_wait() -> None:
        answer = asyncio.create_task(endpoint.answer("/v1/chat/completions", dict(CHAT)))
        while not handler.payloads:
            await asyncio.sleep(0.01)
        answer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await answer

    asyncio.run(cancel_during_the_wait())
    assert len(handler.payloads) == 1
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_a_streaming_agent_gets_one_stream_after_its_call_was_sent_again(short_waits) -> None:
    runtime, handler = scripted(rate_limited(), REPLY)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        # The evaluation's capture proxy sends each call without streaming, and makes the agent's stream from the
        # answer.
        proxy = CaptureProxy(endpoint.url, "s", is_unrecorded=True)
        proxy.start()
        try:
            request = urllib.request.Request(
                f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                data=json.dumps({**CHAT, "stream": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                stream = response.read().decode()
            turns = proxy.drain()
        finally:
            proxy.stop()
    assert handler.payloads == [CHAT, CHAT], "the tries ran before the agent got any output"
    assert stream.count('"content": "five"') == 1 and stream.count("[DONE]") == 1
    assert [(turn["status"], turn["response"]) for turn in turns] == [(200, REPLY)]


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [
        aiohttp.ServerDisconnectedError(),
        aiohttp.ClientPayloadError("the body ended early"),
        OSError("the connection was reset"),
        RuntimeError("the handler failed"),
    ],
)
def test_a_handler_that_fails_without_a_status_gives_a_502_that_the_capture_keeps(failure) -> None:
    runtime, _ = scripted(failure)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        proxy = CaptureProxy(endpoint.url, "s", is_unrecorded=True)
        proxy.start()
        try:
            status, _ = call(f"http://127.0.0.1:{proxy.port}", "/v1/chat/completions", CHAT)
            turns = proxy.drain()
        finally:
            proxy.stop()
    # The capture keeps the failure, so the rollout counts it as a failure of the inference service.
    assert (status, [turn["status"] for turn in turns]) == (502, [502])


@pytest.mark.unit
@pytest.mark.parametrize(("provider_wait", "held_seconds"), [(600.0, range(590, 601)), (None, range(10, 31))])
def test_a_rate_limit_holds_the_calls_to_its_model_route_and_no_other(monkeypatch, provider_wait, held_seconds):
    # The last try's refusal holds the route for the provider's wait, or for a backoff of 15 to 30 s without one.
    # Both pass the runtime's 5 s timeout, so a held call answers at once.
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_FIRST_WAIT_SECONDS", 30.0)
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_TRIES", 1)
    cooldowns = ProviderCooldowns()
    runtime, handler = scripted(rate_limited(wait_seconds=provider_wait), REPLY, inference_timeout_s=5.0)
    other_owner, other_handler = scripted(REPLY)
    other_artifact = Artifact(SnapshotRef(revision="storage:other", parent_revision=None), None, artifact_id="r8")
    model_a, model_b = {**CHAT, "model": "a"}, {**CHAT, "model": "b"}
    with (
        ArtifactEndpoint(runtime, ARTIFACT, cooldowns=cooldowns) as first,
        ArtifactEndpoint(runtime, other_artifact, cooldowns=cooldowns) as peer,
        ArtifactEndpoint(other_owner, ARTIFACT, cooldowns=cooldowns) as other,
    ):
        # The refusal names only the provider's own wait.
        status, body, retry_after = answered(first.url, model_a)
        assert (status, body["error"]["code"], retry_after) == (429, "rate_limit", "600" if provider_wait else None)
        # A peer call of the same model through the same handler, for another artifact, reaches no provider.
        status, body, retry_after = answered(peer.url, model_a)
        assert (status, body["error"]["code"]) == (429, "rate_limit") and int(retry_after) in held_seconds
        # Another model of the handler, and the model through another handler and credential, go at once.
        assert answered(peer.url, model_b) == (200, REPLY, None)
        assert answered(other.url, model_a) == (200, REPLY, None)
    assert (handler.payloads, other_handler.payloads) == ([model_a, model_b], [model_a])


@pytest.mark.unit
def test_a_peer_call_waits_until_the_route_wait_ends_and_then_runs(short_waits) -> None:
    cooldowns = ProviderCooldowns()
    runtime, handler = scripted(rate_limited(wait_seconds=0.3), REPLY)
    with (
        ArtifactEndpoint(runtime, ARTIFACT, cooldowns=cooldowns) as first,
        ArtifactEndpoint(runtime, ARTIFACT, cooldowns=cooldowns) as peer,
        ThreadPoolExecutor(2) as pool,
    ):
        refused_first = pool.submit(answered, first.url, CHAT)
        wait_until(lambda: cooldowns.resume_at(handler, "") > 0)
        held = pool.submit(answered, peer.url, CHAT)
        assert refused_first.result(5) == held.result(5) == (200, REPLY, None)
    # Neither the refused call nor its peer sent a try before the provider's wait ended.
    assert len(handler.sent_at) == 3 and min(handler.sent_at[1:]) - handler.sent_at[0] >= 0.3


@pytest.mark.unit
@pytest.mark.parametrize("ends_by", ["closing", "cancelling"])
def test_a_call_that_its_route_holds_ends_with_its_rollout_and_sends_nothing(ends_by: str) -> None:
    cooldowns = ProviderCooldowns()
    runtime, handler = scripted(REPLY)
    cooldowns.hold(handler, "", time.monotonic() + 30)
    endpoint = ArtifactEndpoint(runtime, ARTIFACT, cooldowns=cooldowns)

    async def end_the_held_call() -> tuple[int, Mapping[str, Any]] | None:
        answer = asyncio.create_task(endpoint.answer("/v1/chat/completions", dict(CHAT)))
        await asyncio.sleep(0.05)
        if ends_by == "cancelling":
            answer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await answer
            return None
        endpoint.closed.set()
        return await answer

    ended = asyncio.run(end_the_held_call())
    assert handler.payloads == []
    # The provider refused none of the call's tries, so the closed endpoint answers for itself.
    assert ended in (None, (503, {"error": {"message": CLOSED_ERROR}}))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "body", "limit"),
    [
        (429, {"error": {"code": "rate_limit", "message": "busy"}}, UpstreamLimit.RATE),
        (402, {"error": {"code": "rate_limit", "message": "too many in flight"}}, UpstreamLimit.RATE),
        (402, {"error": {"code": "payment_required", "message": "refused"}}, UpstreamLimit.PAYMENT),
        (429, {"error": {"code": "payment_required", "message": "no quota"}}, UpstreamLimit.PAYMENT),
        # A call that an older endpoint answered names no limit, and its status does: 429 and 402.
        (429, {"error": {"message": "inference provider returned 429: busy"}}, UpstreamLimit.RATE),
        (429, None, UpstreamLimit.RATE),
        (402, {"error": {"message": "inference provider returned 402: refused"}}, UpstreamLimit.PAYMENT),
        (402, {"error": {"code": 402, "message": "refused"}}, UpstreamLimit.PAYMENT),
        (502, {"error": {"message": "the runtime could not answer from the artifact"}}, None),
        (503, {"error": {"code": "rate_limit"}}, None),
        # A 403 names a limit only by its code. A 401 refuses access.
        (403, {"error": {"code": "payment_required"}}, UpstreamLimit.PAYMENT),
        (403, {"error": {"code": "access_denied"}}, UpstreamLimit.ACCESS),
        (403, {"error": {"message": "Forbidden"}}, None),
        (401, {"error": {"message": "invalid key"}}, UpstreamLimit.ACCESS),
        (401, None, UpstreamLimit.ACCESS),
        # A 200 answer is no refusal, whatever error code its body holds.
        (200, {"choices": [], "error": {"code": "payment_required"}}, None),
        (200, {"error": {"code": "rate_limit"}}, None),
    ],
)
def test_a_refusal_names_its_limit_and_an_older_one_its_status(
    status: int, body: dict | None, limit: UpstreamLimit | None
) -> None:
    assert refusal_limit(status, body) is limit


@pytest.mark.unit
@pytest.mark.parametrize(("status", "wait_seconds"), [(429, None), (402, 0.0)])
def test_an_older_handlers_status_only_refusal_is_retried_by_its_status(short_waits, status, wait_seconds) -> None:
    # A bare 429, and a 402 with a Retry-After, from a handler that names no limit, are rate limits.
    older_refusal = UpstreamStatusError("temporary", status=status, retry_after_seconds=wait_seconds)
    runtime, handler = scripted(older_refusal, REPLY)
    with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
        assert answered(endpoint.url, CHAT) == (200, REPLY, None)
    assert handler.payloads == [CHAT, CHAT]


class GatedRuntime(Runtime):
    """A runtime whose admission sets ``waiting``, and admits once ``admitted`` is set."""

    def __init__(self) -> None:
        super().__init__(any_artifact=True)
        self.waiting, self.admitted = asyncio.Event(), asyncio.Event()

    async def acquire_inference(self):
        self.waiting.set()
        await self.admitted.wait()
        return await super().acquire_inference()


@pytest.mark.unit
def test_a_call_after_its_rollout_ended_reaches_no_provider_and_releases_its_admission() -> None:
    closed_answer = (503, {"error": {"message": CLOSED_ERROR}})
    # Closed before the call: no try.
    runtime, handler = scripted(REPLY)
    endpoint = ArtifactEndpoint(runtime, ARTIFACT)
    endpoint.closed.set()
    assert asyncio.run(endpoint.answer("/v1/chat/completions", dict(CHAT))) == closed_answer
    # Closed while the call waits for its admission: the admission goes back, and the handler gets no call.
    gated = GatedRuntime()
    gated.handler = handler
    endpoint = ArtifactEndpoint(gated, ARTIFACT)

    async def close_during_the_admission() -> tuple[int, Mapping[str, Any]]:
        answer = asyncio.create_task(endpoint.answer("/v1/chat/completions", dict(CHAT)))
        await gated.waiting.wait()
        endpoint.closed.set()
        gated.admitted.set()
        return await answer

    assert asyncio.run(close_during_the_admission()) == closed_answer
    assert handler.payloads == [] and gated.inference_admission_status == {"open": True, "active": 0}


class SlowAdmissionRuntime(Runtime):
    """A runtime that admits each try after the next of ``admission_seconds``, and at once after them."""

    def __init__(self, *admission_seconds: float, inference_timeout_s: float) -> None:
        super().__init__(any_artifact=True, inference_timeout_s=inference_timeout_s)
        self.admission_seconds = list(admission_seconds)

    async def acquire_inference(self):
        if self.admission_seconds:
            await asyncio.sleep(self.admission_seconds.pop(0))
        return await super().acquire_inference()


@pytest.mark.unit
@pytest.mark.parametrize(("start_by", "timeout"), [(0.05, 300.0), (60.0, 0.05)])
def test_a_retry_admitted_after_its_time_sends_nothing(short_waits, monkeypatch, start_by, timeout) -> None:
    # The second try's admission comes after its latest start, or after the runtime's timeout of the call.
    monkeypatch.setattr(artifact_endpoint, "RATE_LIMIT_SECONDS", start_by)
    runtime = SlowAdmissionRuntime(0.0, 0.12, inference_timeout_s=timeout)
    runtime.handler = handler = ScriptedHandler(rate_limited(), REPLY)
    status, body = asyncio.run(ArtifactEndpoint(runtime, ARTIFACT).answer("/v1/chat/completions", dict(CHAT)))
    # The call gets the provider's refusal, and the late try reached no provider.
    assert (status, body["error"]["code"], len(handler.payloads)) == (429, "rate_limit", 1)
    assert runtime.inference_admission_status == {"open": True, "active": 0}


@pytest.mark.unit
def test_answer_gives_a_status_and_a_body_and_only_a_refusal_gets_a_retry_after_header() -> None:
    runtime, _ = scripted(REPLY)
    status, body = asyncio.run(ArtifactEndpoint(runtime, ARTIFACT).answer("/v1/chat/completions", dict(CHAT)))
    assert (status, body) == (200, REPLY)
    for outcome, header in (
        # A 200 or a 5xx whose body names a wait gets no header.
        ({**REPLY, "error": {"retry_after_seconds": 5}}, None),
        (refused(500, retry_after_seconds=5.0), None),
        (refused(402, limit=UpstreamLimit.PAYMENT, retry_after_seconds=5.0), "5"),
    ):
        runtime, _ = scripted(outcome)
        with ArtifactEndpoint(runtime, ARTIFACT) as endpoint:
            assert answered(endpoint.url, CHAT)[2] == header


@pytest.mark.unit
def test_a_route_wait_keeps_no_handler_alive() -> None:
    cooldowns = ProviderCooldowns()
    handler, healthy = ScriptedHandler(REPLY), ScriptedHandler(REPLY)
    retired = weakref.ref(handler)
    cooldowns.hold(handler, "a", time.monotonic() + 600)
    assert cooldowns.resume_at(handler, "a") > time.monotonic()
    del handler
    gc.collect()
    assert retired() is None
    # A new handler gets no wait of the dead one, also with its id, and the next refusal drops the dead route.
    assert cooldowns.resume_at(ScriptedHandler(REPLY), "a") == 0.0
    assert all(cooldowns.resume_at(healthy, "b") == 0.0 for _ in range(100))
    cooldowns.hold(healthy, "b", time.monotonic() + 600)
    assert list(cooldowns.until) == [(id(healthy), "b")]
