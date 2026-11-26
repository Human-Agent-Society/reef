"""A loopback endpoint that answers model calls from one artifact, for the rollouts that evaluate it.

An artifact evaluation runs its rollouts against an artifact that the scenario may not serve. A runtime that
answers each call from the artifact that the call names (``InferenceRuntime.serves_any_artifact``) can
answer them without loading the artifact: ``ArtifactEndpoint`` hands each call to the runtime's handler with that
artifact, under the runtime's inference admission. Reef's serving takes no part, and no call is a record.

The endpoint reads each call whole and answers it whole, so no output of a call reaches the agent before the
answer. So this endpoint, and no other layer of the evaluation, sends a call again that a provider's rate limit
refused (``ArtifactEndpoint.answer``). The endpoints of one process share the waits after rate limits
(``ProviderCooldowns``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import InferenceUnavailableError, ReefError, UpstreamLimit, UpstreamStatusError
from reef.core.evaluation import EvaluationCall, EvaluationTrajectory
from reef.core.served_models import ServedArtifact
from reef.inference.http import upstream_limit
from reef.inference.interfaces import REQUEST_SESSION_KEY, InferenceArtifact, InferenceHandler, InferenceRuntime

logger = logging.getLogger(__name__)

#: The model call routes that the endpoint answers, as Reef's own inference routes do.
MODEL_CALL_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")

#: The most tries of one call that a provider's rate limit refuses (``UpstreamLimit.RATE``), the first one included.
RATE_LIMIT_TRIES = 5
#: The latest start of a try, in seconds after the call arrived. The runtime's ``inference_timeout_s`` from the call's
#: arrival also bounds every try. The endpoint makes no wait that ends later: it answers at once.
RATE_LIMIT_SECONDS = 60.0
#: The wait before the second try, in seconds. Each later wait doubles, up to ``RATE_LIMIT_MAX_WAIT_SECONDS``. Each
#: wait is a random 50 % to 100 % of that, so that the rollouts that one limit refused do not try again together.
RATE_LIMIT_FIRST_WAIT_SECONDS = 1.0
RATE_LIMIT_MAX_WAIT_SECONDS = 16.0
#: How often, in seconds, a wait between two tries checks whether the endpoint closed.
CLOSE_CHECK_SECONDS = 0.1
#: The HTTP statuses of a refusal that can name a limit: payment, a spending cap and a rate limit.
REFUSAL_STATUSES = (402, 403, 429)
#: The error of a call that came after its rollout ended, with no provider refusal before.
CLOSED_ERROR = "the evaluation endpoint closed: its rollout ended"

#: The request fields that ask an engine for the sampled tokens' log probabilities and token IDs, by route. Chat
#: Completions takes OpenAI's ``logprobs`` and the ``return_token_ids`` of vLLM, SGLang and Reef's own handlers, with
#: SGLang's ``return_prompt_token_ids``. Anthropic Messages has no such fields: Reef's token-native handlers take the
#: same two as an extension. OpenAI Responses asks for log probabilities in ``include``.
LOG_PROB_FIELDS: dict[str, dict[str, Any]] = {
    "/v1/chat/completions": {"logprobs": True, "return_token_ids": True, "return_prompt_token_ids": True},
    "/v1/messages": {"logprobs": True, "return_token_ids": True},
}
RESPONSES_LOG_PROBS = "message.output_text.logprobs"


def log_prob_request(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The call ``payload`` to ``path``, one of ``MODEL_CALL_PATHS``, that also asks for log probabilities and token
    IDs: the fields of ``LOG_PROB_FIELDS`` over the caller's, or for ``/v1/responses`` ``RESPONSES_LOG_PROBS`` added
    to its ``include``. The other fields keep the caller's values."""
    if path != "/v1/responses":
        return {**payload, **LOG_PROB_FIELDS[path]}
    include = payload.get("include")
    included = list(include) if isinstance(include, list) else []
    return {**payload, "include": included if RESPONSES_LOG_PROBS in included else [*included, RESPONSES_LOG_PROBS]}


def refusal_limit(status: int, body: Mapping[str, Any] | None) -> UpstreamLimit | None:
    """The limit that refused a call that an ``ArtifactEndpoint`` answered with ``status`` and the JSON object ``body``
    (None for any other body).

    Only a status of ``REFUSAL_STATUSES`` names a limit: the ``UpstreamLimit`` whose value is the error's ``code``.
    Without that code, as an older endpoint answered, a 429 is ``RATE``, a 402 ``PAYMENT`` and a 403 None. Any other
    answer is None, also a 200 or a 5xx whose body has such an error ``code``.
    """
    if status not in REFUSAL_STATUSES:
        return None
    error = body.get("error") if body is not None else None
    code = error.get("code") if isinstance(error, Mapping) else None
    if isinstance(code, str) and code in {limit.value for limit in UpstreamLimit}:
        return UpstreamLimit(code)
    return {429: UpstreamLimit.RATE, 402: UpstreamLimit.PAYMENT}.get(status)


def call_limit(refusal: UpstreamStatusError) -> UpstreamLimit | None:
    """The limit of a handler's refusal: the handler's own, else the default of its status and ``Retry-After``
    (``upstream_limit``), as for an older handler that names none. Only a status of ``REFUSAL_STATUSES`` has one."""
    if refusal.status not in REFUSAL_STATUSES:
        return None
    if refusal.limit is not None:
        return refusal.limit
    return upstream_limit(refusal.status, "", retry_after=refusal.retry_after_seconds)


class ProviderCooldowns:
    """The waits after rate limits, by route, that the evaluation calls of one process share.

    Values: ``until`` maps ``(id(handler), model)`` of each route that waits to a weak reference to the handler and
    the ``time.monotonic()`` time at which the wait ends. The weak reference keeps no handler or credential alive, and
    gives no wait to a new handler that got a dead one's ``id``. The handler holds one provider address and credential,
    so a route names no secret. ``hold`` drops each wait that ended or whose handler is gone; ``lock`` guards
    ``until``. Purpose: ``LocalCandidateEndpoints`` gives one to each ``ArtifactEndpoint`` that it opens. Each try
    reads its route's wait (``resume_at``), and each rate limit's refusal extends it (``hold``).
    """

    def __init__(self) -> None:
        self.until: dict[tuple[int, str], tuple[weakref.ref[InferenceHandler], float]] = {}
        self.lock = threading.Lock()

    def resume_at(self, handler: InferenceHandler, model: str) -> float:
        """The ``time.monotonic()`` time at which the wait of the route ``(handler, model)`` ends; 0.0 for a route
        that never waited."""
        with self.lock:
            entry = self.until.get((id(handler), model))
        return entry[1] if entry is not None and entry[0]() is handler else 0.0

    def hold(self, handler: InferenceHandler, model: str, until: float) -> None:
        """Make the route ``(handler, model)`` wait at least until the ``time.monotonic()`` time ``until``, and drop
        each wait that ended or whose handler is gone."""
        now = time.monotonic()
        key = (id(handler), model)
        with self.lock:
            self.until = {
                other: entry for other, entry in self.until.items() if entry[1] > now and entry[0]() is not None
            }
            current = self.until.get(key)
            if current is None or current[0]() is not handler or current[1] < until:
                self.until[key] = (weakref.ref(handler), until)


class ArtifactEndpoint:
    """An OpenAI endpoint on loopback that answers every call from ``artifact`` with ``runtime``'s handler.

    Values: ``runtime`` is an ``InferenceRuntime`` that serves artifacts on request, and ``artifact``
    the materialized artifact that every call samples. ``served`` is the value that reaches an engine that answers
    for the artifact, or None for the runtime's own handler. A call to a path outside ``MODEL_CALL_PATHS`` gets 404, a
    body that is not a JSON object 400, a runtime that admits no call 503, a handler refusal its own status, and
    any other handler failure 502 with the reason. A call that a rate limit refused is sent again (``answer``). With
    ``log_probs``, each call asks the engine for log probabilities and token IDs (``log_prob_request``); the engine
    must accept those fields, as Reef's token-native handlers and SGLang's and vLLM's OpenAI servers do.
    ``cooldowns`` holds the rate-limit waits that this endpoint shares with others, or its own when None.
    ``session_key`` is the session key of every call, which the handler reads (``REQUEST_SESSION_KEY``), or None. ``closed`` is
    set from the exit on, so that no call waits or is sent after its rollout ended. Purpose: the artifact
    evaluation service points its rollouts at ``url``. Use it as a context manager: it binds its loopback port and sets
    ``url`` when it is entered, and closes the port when it exits. Raise ``ValueError`` at construction when the
    runtime cannot serve ``artifact`` on request.
    """

    def __init__(
        self,
        runtime: InferenceRuntime,
        artifact: Artifact,
        served: ServedArtifact | None = None,
        *,
        log_probs: bool = False,
        cooldowns: ProviderCooldowns | None = None,
        session_key: str | None = None,
    ) -> None:
        if not runtime.serves_any_artifact:
            raise ValueError(f"the runtime serves only its current artifact, not {artifact.ref.revision!r}")
        self.runtime = runtime
        self.artifact = artifact
        self.served = served
        self.log_probs = log_probs
        self.cooldowns = ProviderCooldowns() if cooldowns is None else cooldowns
        self.session_key = session_key
        self.server: ThreadingHTTPServer | None = None
        self.url = ""
        self.closed = threading.Event()

    async def answer(self, path: str, payload: dict[str, Any]) -> tuple[int, Mapping[str, Any]]:
        """The status and the body that answer the call ``payload`` to ``path``.

        Each try waits first for its route (``ProviderCooldowns``). A call that a rate limit refused
        (``UpstreamLimit.RATE``) is sent again unchanged, after a jittered wait that doubles from
        ``RATE_LIMIT_FIRST_WAIT_SECONDS`` and is never shorter than the provider's ``Retry-After``; that wait also
        holds the route. Tries start at most ``RATE_LIMIT_TRIES`` times, within ``RATE_LIMIT_SECONDS`` and the
        runtime's ``inference_timeout_s`` of the call's arrival; a try after a wait gets only the time left, also when
        its admission comes late (``call_handler``). When a wait would pass these limits, the endpoint answers at once
        (``refused``). It sends no other failure again. The error body of a refusal names the limit as its ``code``
        (``call_limit``). No try starts once ``closed`` is set: the call then gets its last refusal, or 503 with
        ``CLOSED_ERROR`` when the provider refused none.
        """
        if self.log_probs:
            payload = log_prob_request(path, payload)
        try:
            handler, sent = self.route(payload)
        except ValueError as exc:
            return 502, {"error": {"message": f"the runtime could not answer from the artifact: {exc}"}}
        model = str(sent.get("model", ""))
        arrived = time.monotonic()
        latest_start, deadline = arrived + RATE_LIMIT_SECONDS, arrived + self.runtime.inference_timeout_s
        refusal: UpstreamStatusError | None = None
        limit: UpstreamLimit | None = None
        refused_at = own_resume_at = arrived
        tries, wait_seconds, waited = 0, RATE_LIMIT_FIRST_WAIT_SECONDS, False
        while True:
            resume_at = max(own_resume_at, self.cooldowns.resume_at(handler, model))
            if resume_at > time.monotonic():
                if resume_at > latest_start or resume_at >= deadline:
                    break
                waited = True
                while not self.closed.is_set() and (left_seconds := resume_at - time.monotonic()) > 0:
                    await asyncio.sleep(min(left_seconds, CLOSE_CHECK_SECONDS))
                if self.closed.is_set():
                    break
                # Read the route again: another call can have made it wait longer.
                continue
            now = time.monotonic()
            plain_first_try = tries == 0 and not waited
            # The rollout can have ended, and the loop can wake late: check both just before each try.
            if self.closed.is_set() or (not plain_first_try and (now > latest_start or now >= deadline)):
                break
            tries += 1
            bounds = None if plain_first_try else (latest_start, deadline)
            try:
                reply = await self.call_handler(handler, path, sent, bounds)
            except InferenceUnavailableError as exc:
                return 503, {"error": {"message": str(exc)}}
            except UpstreamStatusError as exc:
                refusal, limit, refused_at = exc, call_limit(exc), time.monotonic()
            except (ValueError, TimeoutError, OSError) as exc:
                return 502, {"error": {"message": f"the runtime could not answer from the artifact: {exc}"}}
            else:
                if reply is None:
                    # The try sent nothing: the rollout ended, or its time ran out while it waited for its admission.
                    break
                return 200, reply
            if limit is not UpstreamLimit.RATE:
                break
            # The route waits for the later of the jittered backoff and the provider's Retry-After, also after the
            # last try, so that no peer call reaches the provider before it.
            backoff_seconds = wait_seconds * random.uniform(0.5, 1.0)
            own_resume_at = refused_at + max(refusal.retry_after_seconds or 0.0, backoff_seconds)
            self.cooldowns.hold(handler, model, own_resume_at)
            wait_seconds = min(2 * wait_seconds, RATE_LIMIT_MAX_WAIT_SECONDS)
            logger.info("a rate limit refused try %d of a call to artifact %s", tries, self.artifact.artifact_id)
            if tries == RATE_LIMIT_TRIES:
                break
        if refusal is None and self.closed.is_set():
            return 503, {"error": {"message": CLOSED_ERROR}}
        return self.refused(refusal, limit, refused_at, handler, model)

    def route(self, payload: dict[str, Any]) -> tuple[InferenceHandler, dict[str, Any]]:
        """The handler that answers the call ``payload``, and the payload that it sends: the runtime's own handler and
        ``payload``, or for ``served`` the runtime's handler of that engine and ``payload`` with the deployment's
        values and the engine's model name. Raise ``ValueError`` when the runtime has no handler of ``served``."""
        if self.served is None:
            return self.runtime.inference_handler, payload
        # A runtime that started the engine in another process has no handler of its own here.
        reached = self.runtime.handler_for(self.served.url)
        if reached is None:
            raise ValueError(f"the runtime has no handler for {self.served.url}")
        # The candidate samples with the deployment's values, as a client's call to a served artifact does.
        body = reached.request_payload(payload, serving_config={})
        return reached, reached.address_model(body, self.served.model_name)

    async def call_handler(
        self, handler: InferenceHandler, path: str, payload: dict[str, Any], bounds: tuple[float, float] | None
    ) -> dict[str, Any] | None:
        """One try: the answer of ``handler`` to ``payload`` at ``path``, under one inference admission of the runtime.

        ``bounds`` is None for a first try without a wait, which keeps the handler's own timeout. A try after a wait
        gives its latest start and its deadline, ``time.monotonic()`` times: it waits for its admission until the
        deadline, and its call ends by then. None, with no call, when the rollout ended or a bound passed before the
        admission came. Raise as the admission and the handler do."""
        if bounds is None:
            admission = await self.runtime.acquire_inference()
        else:
            try:
                admission = await asyncio.wait_for(self.runtime.acquire_inference(), bounds[1] - time.monotonic())
            except TimeoutError:
                return None
        try:
            now = time.monotonic()
            if self.closed.is_set() or (bounds is not None and (now > bounds[0] or now >= bounds[1])):
                return None
            # The handler reads the key in this task, or in a copy of its context.
            session = REQUEST_SESSION_KEY.set(self.session_key)
            try:
                call = handler.inference(InferenceArtifact.from_artifact(self.artifact), path, payload)
                return await (call if bounds is None else asyncio.wait_for(call, bounds[1] - now))
            finally:
                REQUEST_SESSION_KEY.reset(session)
        finally:
            admission.release()

    def refused(
        self,
        refusal: UpstreamStatusError | None,
        limit: UpstreamLimit | None,
        refused_at: float,
        handler: InferenceHandler,
        model: str,
    ) -> tuple[int, Mapping[str, Any]]:
        """The answer to a call that ``refusal`` refused at the ``time.monotonic()`` time ``refused_at``, for
        ``limit`` (``call_limit``): its ``retry_after_seconds`` is what is left of the provider's ``Retry-After``, in
        whole seconds rounded up. For None, a call that the route ``(handler, model)`` held before its first try, it is
        a 429 whose wait is what is left of the route's wait."""
        until: float | None
        if refusal is None:
            status, limit, until = 429, UpstreamLimit.RATE, self.cooldowns.resume_at(handler, model)
            error: dict[str, Any] = {"message": "Reef holds the calls to this model after a rate limit refused one"}
        else:
            status, wait = refusal.status, refusal.retry_after_seconds
            until = None if wait is None else refused_at + wait
            error = {"message": str(refusal)}
        if limit is not None:
            error["code"] = limit.value
        if until is not None:
            error["retry_after_seconds"] = math.ceil(max(0.0, until - time.monotonic()))
        return status, {"error": error}

    def __enter__(self) -> ArtifactEndpoint:
        endpoint = self
        self.closed.clear()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path not in MODEL_CALL_PATHS:
                    self.reply(404, {"error": {"message": f"no route {self.path}"}})
                    return
                try:
                    payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except json.JSONDecodeError as exc:
                    self.reply(400, {"error": {"message": f"the request is not JSON: {exc}"}})
                    return
                if not isinstance(payload, dict):
                    self.reply(400, {"error": {"message": "the request is not a JSON object"}})
                    return
                # Each call runs on its own server thread, so it runs the handler in its own event loop.
                self.reply(*asyncio.run(endpoint.answer(self.path, payload)))

            def reply(self, status: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                # Only a refusal tells the agent when to try again: no 200 or 5xx body gets the header.
                error = body.get("error") if status in REFUSAL_STATUSES else None
                wait = error.get("retry_after_seconds") if isinstance(error, Mapping) else None
                if isinstance(wait, int) and not isinstance(wait, bool) and wait >= 0:
                    self.send_header("Retry-After", str(wait))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, name="reef-artifact-endpoint", daemon=True).start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        # A call that waits to try again ends at once: its rollout no longer reads the answer.
        self.closed.set()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


class CandidateArtifacts(ABC):
    """Finds the candidate of an evaluation rollout, and keeps the rollout's calls.

    ``candidate(scenario, artifact_id, component)`` returns the runtime of ``scenario`` and the materialized
    artifact that ``artifact_id`` names for ``component``. It raises ``ReefError`` for a scenario without a
    runtime, and as ``Scenario.resolve_artifact`` does for an unknown artifact or component.
    ``candidate_task(scenario, task_name)`` returns the directory of one of its tasks here.
    ``task_owner(scenario)`` returns the owner that the record of ``scenario`` stores, whose data holds its tasks.
    ``add_evaluation_call(scenario, evaluation_id, artifact_id, call)`` keeps one call of a rollout to the candidate
    ``artifact_id`` as a record item of the scenario's owner (``reef.storage.scenario.record_items.evaluation_call_item``).
    ``add_evaluation_trajectory(scenario, evaluation_id, trajectory)`` keeps one trajectory that a rollout's verifier
    wrote as a trajectory item of that owner (``reef.storage.scenario.record_items.evaluation_trajectory_item``).

    Purpose: the dispatcher implements it; ``LocalCandidateEndpoints`` calls it for each rollout.
    """

    @abstractmethod
    def candidate_task(self, scenario: str, task_name: str) -> Path:
        """The directory of the Harbor task ``task_name`` of ``scenario`` on this process's file system, which the task
        file store fetches when another process stored it. ``ReefError`` for a task that the scenario does not use."""

    def candidate_task_for_digest(self, scenario: str, task_name: str, digest: str) -> Path:
        """The directory of the pinned version of ``task_name`` in the scenario's owner's task file store."""
        raise ReefError("this candidate provider cannot read a task by files digest")

    def candidate_task_version(self, scenario: str, task_name: str) -> tuple[Path, str | None]:
        """The current task directory and its stored files digest, or None when the digest is unavailable."""
        return self.candidate_task(scenario, task_name), None

    def task_owner(self, scenario: str) -> str | None:
        """The owner that the record of ``scenario`` stores (``ScenarioSettings.owner``); None when the scenario does
        not exist or its record stores no owner. It never puts the scenario's name in place of a missing owner."""
        return None

    @abstractmethod
    def candidate(
        self, scenario: str, artifact_id: str, component: str | None
    ) -> tuple[InferenceRuntime, Artifact]: ...

    @abstractmethod
    def add_evaluation_call(
        self, scenario: str, evaluation_id: str, artifact_id: str, call: EvaluationCall
    ) -> None: ...

    @abstractmethod
    def add_evaluation_trajectory(
        self, scenario: str, evaluation_id: str, trajectory: EvaluationTrajectory
    ) -> None: ...


__all__ = [
    "CLOSED_ERROR",
    "ArtifactEndpoint",
    "CandidateArtifacts",
    "ProviderCooldowns",
    "call_limit",
    "log_prob_request",
    "refusal_limit",
]
