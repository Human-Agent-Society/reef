"""The inference service: answers model requests from ``served_models`` and the record store alone.

An inference replica holds no scenario. For each request it reads the ``ServedArtifact`` of the request's
``ScenarioModel`` from ``served_models``, keeps it for the whole request, sends the request to the engine at its URL
under its model name (``EngineClients``), appends the request's ``AgentRecord`` to the scenario's record store, and
returns the answer with the record's ID. Control is needed only to change ``served_models``. The one-process layout
runs the same routes in control's app.

Routes:

- ``/v1/chat/completions``, ``/v1/responses``, ``/v1/messages`` and ``/v1/messages/count_tokens``: a recorded call.
  ``x-reef-scenario`` names the scenario; the request's ``model`` names a row of the scenario (an artifact served
  beside the active one), and any other ``model`` means ``active``. ``x-reef-tag-*`` pairs go to the record's
  metadata. With ``x-reef-rollout``, the call is one of Reef's own rollouts: no record, and the answer keeps its
  ``training`` block.
- ``/reef/scenarios/{s}/evaluation/v1/...`` and ``/reef/scenarios/{s}/components/{c}/evaluation/v1/...``: an
  evaluation rollout's call to the active model; no record, and the answer keeps its ``training`` block.
- ``/reef/scenarios/{s}/artifacts/{a}/v1/...``: a recorded call to the row named ``a``;
  ``/reef/scenarios/{s}/artifacts/{a}/evaluation/v1/...``: the same call, unrecorded.

Rules:

- No row: 404. A replica that cannot read ``served_models`` and has no read from the last ``max_cache_age_seconds``:
  503.
- An engine that fails or cannot be reached: 502; one that does not answer within the timeout: 504.
- An answer that the engine aborted is retried once after the replica reads the row again; then 409.
- A record that cannot be appended: 503. A replica never answers a recorded request without its record. A stream goes
  out as the engine sends it, so its record is appended when it ends, and its receipt is the stream's last frame.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web
from sqlalchemy.exc import SQLAlchemyError

from reef.artifact.artifact import Artifact
from reef.core.artifact_ref import ArtifactRef
from reef.core.errors import ReefError
from reef.core.records_types import ROLLOUT_HEADER, AgentRecord, RequestType
from reef.core.served_models import ACTIVE, ServedArtifact
from reef.dispatcher.served import CACHE_SECONDS
from reef.runtime.interfaces import (
    INFERENCE_ABORT_ERROR,
    EngineClients,
    InferenceHandler,
    UpstreamStatusError,
    inference_aborted,
)
from reef.service.request_service import with_request_tags
from reef.service.streaming import (
    SSEFrameDecoder,
    chat_completion_chunk_identity,
    frame_for_codex,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)
from reef.service.web import service_application
from reef.service.wire import HeaderError, RequestHeaders, parse_request_headers
from reef.storage.scenario import ScenarioStorage
from reef.storage.served_models import ServedModels

logger = logging.getLogger(__name__)

#: The header that carries the ID of the request's record.
RECORD_HEADER = "x-reef-agent-record-id"
#: The header that carries the ID of the artifact that answered.
RELEASE_HEADER = "x-reef-release-id"
#: The model routes that an inference replica answers.
INFERENCE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens")


class NotServed(ReefError):
    """The scenario serves no model under the request's name."""


class StaleServedModels(ReefError):
    """The replica could not read ``served_models`` for longer than it may serve from its cache."""


@dataclass
class ServedModelsCache:
    """A replica's cache of ``served_models``, by scenario.

    Values: ``table`` is the table; ``cache_seconds`` is how long a read stays fresh while the table's
    ``changed_sequence`` stays the same, so a change reaches the next request at once; ``max_cache_age_seconds`` is
    how long the replica serves from an old read while the table cannot be read; ``rows`` holds each scenario's
    rows with the time and the table sequence of their read.

    Purpose: each request of a replica reads its ``ServedArtifact`` through it (``read``).
    """

    table: ServedModels
    cache_seconds: float = CACHE_SECONDS
    max_cache_age_seconds: float = 300.0
    rows: dict[str, tuple[float, int, Mapping[str, ServedArtifact]]] = field(default_factory=dict)

    def scenario_rows(self, scenario: str, *, refresh: bool = False) -> Mapping[str, ServedArtifact]:
        now = time.monotonic()
        held = self.rows.get(scenario)
        try:
            sequence = self.table.changed_sequence()
            if held is not None and not refresh and now - held[0] < self.cache_seconds and sequence == held[1]:
                return held[2]
            rows = self.table.read_scenario(scenario)
        except (SQLAlchemyError, ReefError, OSError) as exc:
            if held is not None and now - held[0] < self.max_cache_age_seconds:
                logger.warning("serving %s from a cached read of served_models: %s", scenario, exc)
                return held[2]
            raise StaleServedModels(f"served_models cannot be read: {exc}") from exc
        self.rows[scenario] = (now, sequence, rows)
        return rows

    def read(self, scenario: str, model: object, *, refresh: bool = False) -> ServedArtifact:
        """The ``ServedArtifact`` that answers a request for ``model`` of ``scenario``."""
        rows = self.scenario_rows(scenario, refresh=refresh)
        name = model if isinstance(model, str) and model in rows else ACTIVE
        served = rows.get(name)
        if served is None:
            raise NotServed(f"scenario {scenario!r} serves no model {name!r}")
        return served

    def read_named(self, scenario: str, name: str, *, refresh: bool = False) -> ServedArtifact:
        """The ``ServedArtifact`` of the row ``name`` of ``scenario``; ``NotServed`` when there is none."""
        served = self.scenario_rows(scenario, refresh=refresh).get(name)
        if served is None:
            raise NotServed(f"scenario {scenario!r} serves no artifact {name!r}")
        return served


class ArtifactRefs:
    """The release of each artifact that a replica serves, read once from the scenario's commit history.

    The release of ``artifact_id`` is the newest committed release that binds it: one whose manifest
    (``CommitRecord.components``) names it, or, with one component, the release whose ``release_id`` it is. An
    artifact that no commit binds, such as the artifact that a scenario started from, has no release here, and its
    records carry no ``artifact_ref``.

    Values: ``storage`` holds the commit histories; ``known`` maps each ``(scenario, artifact_id)`` read before to
    its release, or to None.

    Purpose: ``InferenceReplica`` reads the release of each request's artifact through it, for the record.
    """

    def __init__(self, storage: ScenarioStorage) -> None:
        self.storage = storage
        self.known: dict[tuple[str, str], ArtifactRef | None] = {}

    def ref(self, scenario: str, artifact_id: str) -> ArtifactRef | None:
        key = (scenario, artifact_id)
        if key in self.known:
            return self.known[key]
        store = self.storage.open(scenario)
        try:
            history = store.history()
        finally:
            store.close()
        found = next(
            (
                commit.artifact_ref
                for commit in reversed(history)
                if artifact_id in (commit.components or {}).values() or commit.artifact_ref.release_id == artifact_id
            ),
            None,
        )
        self.known[key] = found
        return found


async def read_object(request: web.Request) -> dict[str, Any]:
    """The request's JSON object; 400 for a body that is not one."""
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text="request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    return payload


def upstream_artifact(served: ServedArtifact, ref: ArtifactRef | None) -> Artifact:
    """The artifact that the engine handler gets for its request headers: the release, or the artifact by its ID."""
    return Artifact(ref or ArtifactRef(served.artifact_id, served.artifact_id, None), None)


#: The engine's own names for the weights that answered; they stay inside training and never reach a record.
ENGINE_LOAD_KEYS = ("runtime_load_id", "runtime_load_spans")


def recorded_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """``response`` as a record keeps it: its ``training`` block without the engine's load versions. The record
    names the artifact that answered by its ``artifact_id`` instead."""
    training = response.get("training")
    if not isinstance(training, Mapping):
        return dict(response)
    return {**response, "training": {key: value for key, value in training.items() if key not in ENGINE_LOAD_KEYS}}


def client_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """``response`` without Reef's private ``training`` block; the record keeps the block."""
    return {key: value for key, value in response.items() if key != "training"}


def rollout_response(response: Mapping[str, Any], artifact_id: str) -> dict[str, Any]:
    """The answer of a rollout's call: the ``training`` block stays, naming the artifact that answered."""
    training = response.get("training")
    if not isinstance(training, Mapping):
        return dict(response)
    return {
        **recorded_response(response),
        "training": {**recorded_response(response)["training"], "artifact_id": artifact_id},
    }


class SSERelay:
    """Relays an engine's SSE frames to the client, holding back the terminal event.

    Values: ``path`` is the route; ``response`` the client's stream; ``for_codex`` whether frames are rewritten for
    a Codex client; ``chat_identity`` the chat-completion identity of the first frame that carries one; ``terminal``
    the held-back terminal frame, once it arrived.

    Purpose: the replica writes the record's receipt frames before the terminal event.
    """

    def __init__(self, path: str, response: web.StreamResponse, *, for_codex: bool) -> None:
        self.path = path
        self.response = response
        self.for_codex = for_codex
        self.decoder = SSEFrameDecoder()
        self.chat_identity: dict[str, Any] = {}
        self.terminal: bytes | None = None

    async def feed(self, chunk: bytes) -> bool:
        """Write the frames that ``chunk`` completes; return whether the terminal event arrived."""
        return await self.write(self.decoder.feed(chunk))

    async def finish(self) -> bool:
        """Write what the decoder still holds; return whether the terminal event arrived."""
        frames, remainder = self.decoder.finalize()
        if await self.write(frames):
            return True
        if remainder:
            await self.response.write(remainder)
        return False

    async def write(self, frames: Iterable[bytes]) -> bool:
        for frame in frames:
            if self.for_codex:
                frame = frame_for_codex(frame)
            if is_terminal_sse_event(self.path, frame):
                self.terminal = frame
                return True
            if not self.chat_identity:
                self.chat_identity = chat_completion_chunk_identity(frame)
            await self.response.write(frame)
        return False


@dataclass
class InferenceReplica:
    """One inference replica.

    Values: ``cache`` reads ``served_models``; ``storage`` holds each scenario's records and commit history;
    ``engines`` gives the handler of each row's engine; ``timeout_seconds`` is the positive time that the replica
    waits for an engine; ``for_codex`` is whether streams are rewritten for a Codex client; ``refs`` resolves the
    release of each served artifact.

    Purpose: ``add_inference_routes`` routes the model requests to ``answer``.
    """

    cache: ServedModelsCache
    storage: ScenarioStorage
    engines: EngineClients
    timeout_seconds: float = 300.0
    for_codex: bool = False
    refs: ArtifactRefs = field(init=False)

    def __post_init__(self) -> None:
        if not self.timeout_seconds > 0:
            raise ValueError("timeout_seconds is positive")
        self.refs = ArtifactRefs(self.storage)

    def record(
        self, parsed: RequestHeaders, payload: Mapping[str, Any], response: Mapping[str, Any], ref: ArtifactRef | None
    ) -> str:
        """Append the record of one answered request; return its ID. ``payload`` names the artifact that answered
        (``artifact_id``)."""
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=RequestType.INFERENCE,
            payload={**with_request_tags(payload, parsed), "response": recorded_response(response)},
            artifact_ref=ref,
        )
        store = self.storage.open(parsed.scenario)
        try:
            store.records.append(item)
        finally:
            store.close()
        return item.agent_record_id

    async def resolve(
        self, scenario: str, model: object, *, name: str | None, refresh: bool
    ) -> tuple[ServedArtifact, ArtifactRef | None]:
        """The row that answers (``name``, or the one that ``model`` names) and its release, read off the event
        loop; 404 when the scenario serves no such row, 503 when ``served_models`` cannot be read."""
        try:
            if name is None:
                served = await asyncio.to_thread(self.cache.read, scenario, model, refresh=refresh)
            else:
                served = await asyncio.to_thread(self.cache.read_named, scenario, name, refresh=refresh)
            ref = await asyncio.to_thread(self.refs.ref, scenario, served.artifact_id)
        except NotServed as exc:
            raise web.HTTPNotFound(text=str(exc)) from exc
        except StaleServedModels as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc
        return served, ref

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        try:
            return self.engines.handler(served)
        except LookupError as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc

    async def forward(
        self, served: ServedArtifact, ref: ArtifactRef | None, path: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """The answer of the engine of ``served`` to ``payload`` under the row's model name: 502 for an error status,
        an unreachable engine or an answer that is no JSON object, 504 when the engine does not answer in time."""
        handler = self.handler(served)
        try:
            return await asyncio.wait_for(
                handler.inference(
                    upstream_artifact(served, ref), path, handler.address_model(payload, served.model_name)
                ),
                timeout=self.timeout_seconds,
            )
        except (UpstreamStatusError, aiohttp.ClientError, TypeError, json.JSONDecodeError) as exc:
            raise web.HTTPBadGateway(text=f"the engine at {served.url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise web.HTTPGatewayTimeout(text=f"the engine at {served.url} did not answer in time") from exc

    async def answer(
        self,
        request: web.Request,
        path: str,
        *,
        scenario: str | None = None,
        name: str | None = None,
        record: bool,
        keep_training: bool,
    ) -> web.StreamResponse:
        """Answer one model request; see the module rules. ``scenario`` overrides the scenario header (a route
        names it), ``name`` the row; ``record`` keeps a record; ``keep_training`` keeps the ``training`` block."""
        headers = dict(request.headers) if scenario is None else {**dict(request.headers), "x-reef-scenario": scenario}
        try:
            parsed = parse_request_headers(headers, RequestType.INFERENCE)
        except HeaderError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        payload = await read_object(request)
        served, ref = await self.resolve(parsed.scenario, payload.get("model"), name=name, refresh=False)
        if payload.get("stream") is True:
            if keep_training:
                raise web.HTTPBadRequest(text="a rollout's call does not stream; send it without stream")
            return await self.relay_stream(request, path, parsed, payload, served, ref, record=record)
        for attempt in range(2):
            response = await self.forward(served, ref, path, payload)
            if not inference_aborted(response):
                break
            if attempt == 1:
                raise web.HTTPConflict(text=INFERENCE_ABORT_ERROR)
            served, ref = await self.resolve(parsed.scenario, payload.get("model"), name=name, refresh=True)
        stamped = {**payload, "artifact_id": served.artifact_id}
        reply_headers = {RELEASE_HEADER: served.artifact_id}
        if record:
            try:
                reply_headers[RECORD_HEADER] = await asyncio.to_thread(self.record, parsed, stamped, response, ref)
            except Exception as exc:
                logger.exception("could not append the record of a request to %s", parsed.scenario)
                raise web.HTTPServiceUnavailable(text="the record store cannot take the request's record") from exc
        body = rollout_response(response, served.artifact_id) if keep_training else client_response(response)
        return web.json_response(body, headers=reply_headers)

    async def relay_stream(
        self,
        request: web.Request,
        path: str,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        served: ServedArtifact,
        ref: ArtifactRef | None,
        *,
        record: bool,
    ) -> web.StreamResponse:
        """Stream the engine's answer to the client and, for a recorded call, append its record at the end. An SSE
        stream gets the record's receipt before its terminal event."""
        handler = self.handler(served)
        try:
            upstream = await handler.inference_stream(
                upstream_artifact(served, ref), path, handler.address_model(payload, served.model_name)
            )
        except (UpstreamStatusError, aiohttp.ClientError) as exc:
            raise web.HTTPBadGateway(text=f"the engine at {served.url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise web.HTTPGatewayTimeout(text=f"the engine at {served.url} did not answer in time") from exc
        headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in (RECORD_HEADER, RELEASE_HEADER, "content-length", "transfer-encoding")
        }
        headers[RELEASE_HEADER] = served.artifact_id
        content_type = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
        response = web.StreamResponse(status=upstream.status, headers=headers)
        body = bytearray()
        complete = False
        error: str | None = None
        recorded = False
        try:
            await response.prepare(request)
            if "text/event-stream" in content_type.lower():
                relay = SSERelay(path, response, for_codex=self.for_codex)
                async for chunk in upstream.chunks:
                    body.extend(chunk)
                    if await relay.feed(chunk):
                        break
                if relay.terminal is None:
                    await relay.finish()
                if relay.terminal is None:
                    error = "the engine's SSE stream ended without a terminal event"
                elif upstream.record_response is not None and inference_aborted(upstream.record_response):
                    error = INFERENCE_ABORT_ERROR
                    await response.write(relay.terminal)
                else:
                    complete = True
                    frames: tuple[bytes, ...] = (relay.terminal,)
                    if record:
                        recorded = True
                        captured = stream_record(upstream, bytes(body), complete=True)
                        record_id = await asyncio.to_thread(
                            self.record, parsed, {**payload, "artifact_id": served.artifact_id}, captured, ref
                        )
                        frames = receipt_sse_events(path, relay.chat_identity, relay.terminal, record_id)
                    for frame in frames:
                        await response.write(frame)
            else:
                async for chunk in upstream.chunks:
                    body.extend(chunk)
                    await response.write(chunk)
                complete = not (upstream.record_response is not None and inference_aborted(upstream.record_response))
                if not complete:
                    error = INFERENCE_ABORT_ERROR
        except ConnectionResetError:
            error = "the client disconnected"
        finally:
            await upstream.close()
            if error is not None:
                logger.warning("stream of %s for %s ended early: %s", path, parsed.scenario, error)
            if record and not recorded:
                captured = stream_record(upstream, bytes(body), complete=complete, error=error)
                try:
                    await asyncio.to_thread(
                        self.record, parsed, {**payload, "artifact_id": served.artifact_id}, captured, ref
                    )
                except Exception:
                    logger.exception("could not append the record of a stream of %s", parsed.scenario)
        return response


def add_inference_routes(app: web.Application, replica: InferenceReplica) -> None:
    """Add the model routes of ``replica`` (see the module rules) to ``app``."""

    def checked(route: str) -> str:
        path = "/v1/" + route
        if path not in INFERENCE_PATHS:
            raise web.HTTPNotFound(text=f"{path} is not an inference route")
        return path

    def recorded(path: str) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
        async def inference(request: web.Request) -> web.StreamResponse:
            rollout = request.headers.get(ROLLOUT_HEADER, "").strip().lower() in ("1", "true")
            return await replica.answer(request, path, record=not rollout, keep_training=rollout)

        return inference

    async def evaluation(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(
            request, path, scenario=request.match_info["scenario"], record=False, keep_training=True
        )

    async def artifact(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(
            request,
            path,
            scenario=request.match_info["scenario"],
            name=request.match_info["artifact_id"],
            record=True,
            keep_training=False,
        )

    async def artifact_evaluation(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(
            request,
            path,
            scenario=request.match_info["scenario"],
            name=request.match_info["artifact_id"],
            record=False,
            keep_training=True,
        )

    for path in INFERENCE_PATHS:
        app.router.add_post(path, recorded(path))
    app.router.add_post("/reef/scenarios/{scenario}/evaluation/v1/{route:.+}", evaluation)
    app.router.add_post("/reef/scenarios/{scenario}/components/{component}/evaluation/v1/{route:.+}", evaluation)
    app.router.add_post("/reef/scenarios/{scenario}/artifacts/{artifact_id}/v1/{route:.+}", artifact)
    app.router.add_post(
        "/reef/scenarios/{scenario}/artifacts/{artifact_id}/evaluation/v1/{route:.+}", artifact_evaluation
    )


def build_inference_app(
    served_models: ServedModels,
    storage: ScenarioStorage,
    engines: EngineClients,
    *,
    tokens: str | tuple[str, ...] | None,
    evaluation_tokens: str | tuple[str, ...] | None = None,
    max_cache_age_seconds: float = 300.0,
    timeout_seconds: float = 300.0,
    for_codex: bool = False,
) -> web.Application:
    """The HTTP app of one separate inference replica: the model routes and ``/healthz``, behind the service
    tokens."""
    replica = InferenceReplica(
        ServedModelsCache(served_models, max_cache_age_seconds=max_cache_age_seconds),
        storage,
        engines,
        timeout_seconds,
        for_codex,
    )
    app = service_application(tokens, evaluation_tokens=evaluation_tokens)
    add_inference_routes(app, replica)
    return app


__all__ = [
    "INFERENCE_PATHS",
    "RECORD_HEADER",
    "RELEASE_HEADER",
    "ArtifactRefs",
    "InferenceReplica",
    "NotServed",
    "ServedModelsCache",
    "StaleServedModels",
    "add_inference_routes",
    "build_inference_app",
    "client_response",
    "recorded_response",
]
