"""The inference service: answers model requests and reports artifact startup.

An inference replica holds no scenario. For each request it reads the ``ServedArtifact`` of the request's
``ScenarioModel`` from ``served_models``, keeps it for the whole request, sends the request to the engine at its URL
under its model name (``EngineClients``), appends the request's ``AgentRecord`` to the scenario's record store, and
returns the answer with the record's ID. The request that it sends and records is the body that the engine's handler
gives for the client's body (``InferenceHandler.request_payload``). Control selects artifacts; the inference
worker starts them and writes ``served_models``. The one-process layout runs the same routes.

Routes:

- ``/v1/chat/completions``, ``/v1/responses``, ``/v1/messages`` and ``/v1/messages/count_tokens``: a recorded call.
  ``x-reef-scenario`` names the scenario; the request's ``model`` names a row of the scenario (an artifact served
  beside the active one), and any other ``model`` means ``active``. ``x-reef-tag-*`` pairs go to the record's
  metadata. With ``x-reef-rollout``, the call is one of Reef's own rollouts: no record.
- ``/reef/scenarios/{s}/evaluation/v1/...`` and ``/reef/scenarios/{s}/components/{c}/evaluation/v1/...``: an
  evaluation rollout's call to the active model; no record.
- ``/reef/scenarios/{s}/artifacts/{a}/v1/...``: a recorded call to the row named ``a``;
  ``/reef/scenarios/{s}/artifacts/{a}/evaluation/v1/...``: the same call, unrecorded.
- ``GET /reef/scenarios/{s}/inference/status?model=active``: the current artifact's startup state.

Rules:

- A body that the engine's handler refuses (``request_payload``): 400.
- No row: 503 while its artifact starts or after startup fails; 404 for an unknown model.
  A replica that cannot read ``served_models`` and has no recent cached read answers 503.
- An engine that fails or cannot be reached: 502; one that does not answer within the timeout: 504.
- An answer that the engine aborted is retried once after the replica reads the row again; then 409.
- A scenario's traffic split (``TrafficWeights``) selects the row of a request that means ``active``: the replica puts
  the conversation in a bucket, and each bucket belongs to one artifact. A request without an identified conversation,
  a rollout's call and a request that names a row skip the split. The replica selects the model. The engine of the
  artifact selects its own replica.
- A record that cannot be appended: 503. A replica never answers a recorded request without its record. A stream goes
  out as the engine sends it, so its record is appended when it ends, and its receipt is the stream's last frame.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import web
from sqlalchemy.exc import SQLAlchemyError

from reef.core.errors import INFERENCE_ABORT_ERROR, ReefError, UpstreamStatusError, inference_aborted
from reef.core.records_types import ROLLOUT_HEADER, AgentRecord, RequestType
from reef.core.served_models import ACTIVE, CACHE_SECONDS, ServedArtifact
from reef.core.values import JsonValue
from reef.inference.interfaces import EngineClients, EngineState, InferenceArtifact, InferenceHandler
from reef.inference.provider_protocol import SSEFrameDecoder
from reef.service.streaming import (
    chat_completion_chunk_identity,
    frame_for_codex,
    is_terminal_sse_event,
    receipt_sse_events,
    stream_record,
)
from reef.service.web import service_application
from reef.service.wire import HeaderError, RequestHeaders, parse_request_headers, with_request_tags
from reef.storage.scenario import ScenarioStorage
from reef.storage.served_models import ServedModels

logger = logging.getLogger(__name__)
#: A client's identifier of a conversation, which keeps the conversation on one artifact of a traffic split.
SESSION_KEY_HEADER = "x-reef-session-key"
#: The number of buckets for conversations. A weight is a number of buckets.
BUCKET_COUNT = 100

#: The header that carries the ID of the request's record.
RECORD_HEADER = "x-reef-agent-record-id"
#: The header that carries the ID of the artifact that answered.
ARTIFACT_HEADER = "x-reef-artifact-id"
# Kept for clients deployed before artifact identity became the public contract.
#: The model routes that an inference replica answers.
INFERENCE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens")


class NotServed(ReefError):
    """The scenario serves no model under the request's name."""


class StaleServedModels(ReefError):
    """The replica could not read ``served_models`` for longer than it may serve from its cache."""


@dataclass(frozen=True)
class TrafficWeights:
    """The number of buckets that each artifact of one scenario's traffic split gets.

    Values: ``by_artifact`` maps an artifact ID, a non-empty string, to an integer from 0 to ``BUCKET_COUNT``. It is
    empty, which means no split, or its values sum to ``BUCKET_COUNT``. Construction checks these conditions.

    Purpose: ``from_json`` makes it from a client's value, and the replica makes it from a stored value. ``is_split`` is
    true exactly when ``by_artifact`` is not empty. For a split, ``artifact_for`` gives the artifact of a bucket from 0
    to ``BUCKET_COUNT`` - 1. Artifact IDs in sorted order get consecutive buckets. So a changed weight moves only the
    buckets that go to another artifact.
    """

    by_artifact: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for artifact_id, weight in self.by_artifact.items():
            if not isinstance(artifact_id, str) or not artifact_id:
                raise ValueError("a traffic weight must name an artifact ID")
            if type(weight) is not int or not 0 <= weight <= BUCKET_COUNT:
                raise ValueError(f"the traffic weight of {artifact_id!r} must be an integer from 0 to {BUCKET_COUNT}")
        if self.by_artifact and sum(self.by_artifact.values()) != BUCKET_COUNT:
            raise ValueError(f"traffic_weights must sum to {BUCKET_COUNT}")

    @classmethod
    def from_json(cls, value: object) -> TrafficWeights:
        """The weights of ``value``: a mapping, or None for no split. Raise ``ValueError`` for any other value."""
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("traffic_weights must map each artifact ID to its weight, or be null")
        return cls(dict(value))

    @property
    def is_split(self) -> bool:
        return bool(self.by_artifact)

    def artifact_for(self, bucket: int) -> str:
        buckets = [
            artifact_id for artifact_id in sorted(self.by_artifact) for _ in range(self.by_artifact[artifact_id])
        ]
        return buckets[bucket]


@dataclass(frozen=True)
class TrafficChoice:
    """The artifact that a traffic split gives one request.

    Values: ``artifact_id`` is the artifact of ``bucket`` under ``weights``, which are a split. Purpose:
    ``InferenceReplica.route`` makes it, and ``InferenceReplica.record`` stores ``bucket`` and ``weights``.
    """

    artifact_id: str
    bucket: int
    weights: TrafficWeights


def message_text(content: object) -> str:
    """The text of a message's ``content``: a string, or the ``text`` of each of its blocks.

    Clients change other block fields between turns, for example ``cache_control``, so only the text counts.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block["text"] for block in content if isinstance(block, Mapping) and isinstance(block.get("text"), str)
    )


def conversation_bucket(scenario: str, payload: Mapping[str, object], session_key: str | None = None) -> int | None:
    """The bucket of a provider-native request, or None when Reef cannot identify its conversation.

    With ``session_key``, the bucket is a hash of ``scenario`` and that key. Without it, the bucket is a hash of
    ``scenario``, the system text, the first message and the first user message. The later turns of a conversation
    repeat these messages.

    The result is None for a request without a message. It is also None for a request with ``previous_response_id``
    and no ``session_key``, because that request holds only its new input. The system text comes from ``system``,
    ``instructions`` or the first system or developer message. The messages come from ``messages`` or ``input``.
    """
    if session_key:
        head = [scenario, session_key]
    else:
        if payload.get("previous_response_id"):
            return None
        entries = payload.get("messages", payload.get("input"))
        if isinstance(entries, str):
            entries = [{"role": "user", "content": entries}]
        messages = [entry for entry in entries if isinstance(entry, Mapping)] if isinstance(entries, list) else []
        if not messages:
            return None

        def first_text(*roles: str) -> str:
            found = next((message for message in messages if message.get("role") in roles), None)
            return "" if found is None else message_text(found.get("content"))

        system = message_text(payload.get("system", payload.get("instructions"))) or first_text("system", "developer")
        head = [scenario, system, message_text(messages[0].get("content")), first_text("user")]
    digest = hashlib.sha256(json.dumps(head, ensure_ascii=False).encode()).digest()
    return int.from_bytes(digest[:8], "big") % BUCKET_COUNT


@dataclass(frozen=True)
class ScenarioServing:
    """What a replica read of one scenario's serving state.

    Values: ``read_at`` is a ``time.monotonic`` reading and ``sequence`` is the table's ``changed_sequence`` at the
    read. ``rows``, ``serving_config`` and ``traffic_weights`` are the scenario's values at the read. Purpose:
    ``ServedModelsCache`` makes and keeps it.
    """

    read_at: float
    sequence: int
    rows: Mapping[str, ServedArtifact]
    serving_config: Mapping[str, JsonValue]
    traffic_weights: TrafficWeights


def row_for_model(scenario: str, rows: Mapping[str, ServedArtifact], model: object) -> ServedArtifact:
    """The row that answers a request for ``model``: the row of that name, else the active row. Raise ``NotServed``
    when that row does not exist."""
    name = model if isinstance(model, str) and model in rows else ACTIVE
    served = rows.get(name)
    if served is None:
        raise NotServed(f"scenario {scenario!r} serves no model {name!r}")
    return served


@dataclass
class ServedModelsCache:
    """A replica's cache of ``served_models``, by scenario.

    Values: ``table`` is the table; ``cache_seconds`` is how long a read stays fresh while the table's
    ``changed_sequence`` stays the same, so a change reaches the next request at once; ``max_cache_age_seconds`` is
    how long the replica serves from an old read while the table cannot be read; ``reads`` holds the last read of
    each scenario.

    Purpose: each request of a replica reads its ``ServedArtifact`` (``read``) and the serving config that its
    handler applies (``serving_config``) through it.
    """

    table: ServedModels
    cache_seconds: float = CACHE_SECONDS
    max_cache_age_seconds: float = 300.0
    reads: dict[str, ScenarioServing] = field(default_factory=dict)

    def serving(self, scenario: str, *, refresh: bool = False) -> ScenarioServing:
        """The serving state of ``scenario``, from the cache while it is fresh."""
        now = time.monotonic()
        held = self.reads.get(scenario)
        try:
            sequence = self.table.changed_sequence()
            if (
                held is not None
                and not refresh
                and now - held.read_at < self.cache_seconds
                and sequence == held.sequence
            ):
                return held
            rows = self.table.read_scenario(scenario)
            serving_config = self.table.read_serving_config(scenario)
            weights = TrafficWeights(self.table.read_traffic_weights(scenario))
        except (SQLAlchemyError, ReefError, OSError) as exc:
            if held is not None and now - held.read_at < self.max_cache_age_seconds:
                logger.warning("serving %s from a cached read of served_models: %s", scenario, exc)
                return held
            raise StaleServedModels(f"served_models cannot be read: {exc}") from exc
        self.reads[scenario] = ScenarioServing(now, sequence, rows, serving_config, weights)
        return self.reads[scenario]

    def scenario_rows(self, scenario: str, *, refresh: bool = False) -> Mapping[str, ServedArtifact]:
        return self.serving(scenario, refresh=refresh).rows

    def serving_config(self, scenario: str) -> Mapping[str, JsonValue]:
        """The override that ``scenario``'s routes serve with (``ServedModels.read_serving_config``)."""
        return self.serving(scenario).serving_config

    def read(self, scenario: str, model: object, *, refresh: bool = False) -> ServedArtifact:
        """The ``ServedArtifact`` that answers a request for ``model`` of ``scenario``."""
        return row_for_model(scenario, self.scenario_rows(scenario, refresh=refresh), model)

    def read_named(self, scenario: str, name: str, *, refresh: bool = False) -> ServedArtifact:
        """The ``ServedArtifact`` of the row ``name`` of ``scenario``; ``NotServed`` when there is none."""
        served = self.scenario_rows(scenario, refresh=refresh).get(name)
        if served is None:
            raise NotServed(f"scenario {scenario!r} serves no artifact {name!r}")
        return served


async def read_object(request: web.Request) -> dict[str, Any]:
    """The request's JSON object; 400 for a body that is not one."""
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text="request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise web.HTTPBadRequest(text="request body must be a JSON object")
    return payload


def provider_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """``response`` as the client gets it and the record keeps it: without a ``training`` block, which an engine
    handler may return. Serving returns no training tensors; a training component computes the ones it needs. The
    record names the artifact that answered by its ``artifact_id``."""
    return {key: value for key, value in response.items() if key != "training"}


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
    waits for an engine; ``for_codex`` is whether streams are rewritten for a Codex client.

    Purpose: ``add_inference_routes`` routes model requests to ``answer`` and status reads to ``startup_status``.
    """

    cache: ServedModelsCache
    storage: ScenarioStorage
    engines: EngineClients
    timeout_seconds: float = 300.0
    for_codex: bool = False

    def __post_init__(self) -> None:
        if not self.timeout_seconds > 0:
            raise ValueError("timeout_seconds is positive")

    def route(
        self, scenario: str, payload: Mapping[str, Any], session_key: str | None, *, refresh: bool
    ) -> tuple[ServedArtifact, TrafficChoice | None]:
        """The row that answers ``payload``, with the choice of the scenario's traffic split.

        The choice is None when the scenario has no split, the request names another row, Reef cannot identify the
        conversation, or the selected artifact has no route. Then ``row_for_model`` gives the row. Only a split computes
        a bucket.
        """
        serving = self.cache.serving(scenario, refresh=refresh)
        weights = serving.traffic_weights
        model = payload.get("model")
        names_row = isinstance(model, str) and model != ACTIVE and model in serving.rows
        bucket = conversation_bucket(scenario, payload, session_key) if weights.is_split and not names_row else None
        if bucket is not None:
            artifact_id = weights.artifact_for(bucket)
            selected = serving.rows.get(artifact_id)
            if selected is not None:
                return selected, TrafficChoice(artifact_id, bucket, weights)
            logger.warning("scenario %s: artifact %s has a traffic weight but no route", scenario, artifact_id)
        return row_for_model(scenario, serving.rows, model), None

    def record(
        self,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        response: Mapping[str, Any],
        artifact_id: str,
        traffic: TrafficChoice | None = None,
    ) -> str:
        """Append the record of one request that the artifact ``artifact_id`` answered; return its ID.

        The stored payload also names ``artifact_id``. With ``traffic``, ``metadata.traffic`` holds the bucket and the
        weights.
        """
        recorded = with_request_tags(payload, parsed)
        if traffic is not None:
            metadata = {
                **(recorded.get("metadata") or {}),
                "traffic": {"bucket": traffic.bucket, "weights": dict(traffic.weights.by_artifact)},
            }
            recorded = {**recorded, "metadata": metadata}
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=RequestType.INFERENCE,
            payload={
                **recorded,
                "artifact_id": artifact_id,
                "response": provider_response(response),
            },
            artifact_id=artifact_id,
        )
        store = self.storage.open(parsed.scenario)
        try:
            store.records.append(item)
        finally:
            store.close()
        return item.agent_record_id

    def startup_status(self, scenario: str, model: str = ACTIVE) -> dict[str, str | None]:
        """Read the current serving work's artifact, state, and safe error without changing it, and for a ready
        artifact its engine's state (``engine_state``, RFC #1269).

        The model is ``active``, an active artifact alias, or an artifact served beside it.
        Unknown models answer 404; an unreadable or invalid work item answers 503.
        """
        try:
            item = self.cache.table.current_request(scenario)
        except (SQLAlchemyError, ReefError, OSError) as exc:
            logger.warning("Could not read inference startup status for %s: %s", scenario, type(exc).__name__)
            raise web.HTTPServiceUnavailable(text="Inference startup status is unavailable.") from exc
        if item is None or not item.payload:
            raise web.HTTPNotFound(text=f"scenario {scenario!r} has no inference startup request")
        artifact_id = item.payload.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise web.HTTPServiceUnavailable(text="Inference startup status is unavailable.")
        aliases = item.payload.get("aliases", [])
        beside = item.payload.get("beside", {})
        if model not in (ACTIVE, artifact_id) and (not isinstance(aliases, list) or model not in aliases):
            if not isinstance(beside, dict) or model not in beside:
                raise web.HTTPNotFound(text=f"scenario {scenario!r} serves no model {model!r}")
            artifact_id = model
        if item.state == "finished":
            state = "ready"
        elif item.state == "failed":
            state = "failed"
        elif item.state == "queued" and item.started_at is None and not item.metadata.get("recovery"):
            state = "pending"
        else:
            state = "starting"
        engine, detail = self.engine_state(scenario, model) if state == "ready" else (None, None)
        return {
            "artifact_id": artifact_id,
            "state": state,
            "error": "The inference engine did not start. Check the service logs." if state == "failed" else None,
            "engine": engine,
            "engine_detail": detail,
        }

    def engine_state(self, scenario: str, model: str) -> tuple[EngineState | None, str | None]:
        """The state of the engine of ``model``'s route (``InferenceHandler.engine_state``), and for a starting engine
        what it does now (``InferenceHandler.engine_detail``); None without a route or a handler for it."""
        try:
            served = self.cache.read(scenario, model)
            handler = self.engines.handler(served)
        except (NotServed, StaleServedModels, LookupError):
            return None, None
        engine = handler.engine_state(served)
        return engine, handler.engine_detail(served) if engine == "starting" else None

    async def resolve(
        self,
        scenario: str,
        payload: Mapping[str, Any],
        *,
        name: str | None,
        refresh: bool,
        use_split: bool = False,
        session_key: str | None = None,
    ) -> tuple[ServedArtifact, TrafficChoice | None]:
        """The serving row for ``name``, else for ``payload``, read off the event loop, with the choice of the traffic
        split when ``use_split`` is true. A missing route answers 503 during startup or after failure, and 404 for an
        unknown model."""
        try:
            if name is not None:
                return await asyncio.to_thread(self.cache.read_named, scenario, name, refresh=refresh), None
            if use_split:
                return await asyncio.to_thread(self.route, scenario, payload, session_key, refresh=refresh)
            return await asyncio.to_thread(self.cache.read, scenario, payload.get("model"), refresh=refresh), None
        except NotServed as exc:
            status = await asyncio.to_thread(self.startup_status, scenario, ACTIVE if name is None else name)
            if status["state"] in ("pending", "starting"):
                raise web.HTTPServiceUnavailable(
                    text=json.dumps(
                        {"error": {"type": "inference_starting", "message": "The inference engine is starting."}}
                    ),
                    content_type="application/json",
                    headers={"Retry-After": "30"},
                ) from exc
            if status["state"] == "failed":
                raise web.HTTPServiceUnavailable(
                    text=json.dumps({"error": {"type": "inference_start_failed", "message": status["error"]}}),
                    content_type="application/json",
                ) from exc
            raise web.HTTPServiceUnavailable(text="The inference route is unavailable.") from exc
        except StaleServedModels as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc

    def handler(self, served: ServedArtifact) -> InferenceHandler:
        try:
            return self.engines.handler(served)
        except LookupError as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc

    def request_payload(self, scenario: str, served: ServedArtifact, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The body that the engine of ``served`` receives for the client's ``payload`` to ``scenario``, with the
        scenario's serving config; 400 when its handler refuses it."""
        try:
            serving_config = self.cache.serving_config(scenario)
        except StaleServedModels as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc
        try:
            return self.handler(served).request_payload(payload, serving_config=serving_config)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc

    async def forward(self, served: ServedArtifact, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """The answer of the engine of ``served`` to ``payload`` under the row's model name: 502 for an error status,
        an unreachable engine or an answer that is no JSON object, 504 when the engine does not answer in time."""
        handler = self.handler(served)
        try:
            return await asyncio.wait_for(
                handler.inference(
                    InferenceArtifact(served.artifact_id), path, handler.address_model(payload, served.model_name)
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
        rollout: bool = False,
    ) -> web.StreamResponse:
        """Answer one model request; see the module rules. ``scenario`` overrides the scenario header (a route
        names it), ``name`` the row; ``record`` keeps a record; ``rollout`` marks a call of Reef's own rollouts, which
        does not stream."""
        headers = dict(request.headers) if scenario is None else {**dict(request.headers), "x-reef-scenario": scenario}
        try:
            parsed = parse_request_headers(headers, RequestType.INFERENCE)
        except HeaderError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        payload = await read_object(request)
        served, traffic = await self.resolve(
            parsed.scenario,
            payload,
            name=name,
            refresh=False,
            use_split=not rollout,
            session_key=request.headers.get(SESSION_KEY_HEADER),
        )
        if payload.get("stream") is True:
            if rollout:
                raise web.HTTPBadRequest(text="a rollout's call does not stream; send it without stream")
            sent = self.request_payload(parsed.scenario, served, payload)
            return await self.relay_stream(request, path, parsed, sent, served, record=record, traffic=traffic)
        for attempt in range(2):
            sent = self.request_payload(parsed.scenario, served, payload)
            response = await self.forward(served, path, sent)
            if not inference_aborted(response):
                break
            if attempt == 1:
                raise web.HTTPConflict(text=INFERENCE_ABORT_ERROR)
            retry_name = name if traffic is None else traffic.artifact_id
            served, _ = await self.resolve(parsed.scenario, payload, name=retry_name, refresh=True)
        reply_headers = {ARTIFACT_HEADER: served.artifact_id}
        if record:
            try:
                reply_headers[RECORD_HEADER] = await asyncio.to_thread(
                    self.record, parsed, sent, response, served.artifact_id, traffic
                )
            except Exception as exc:
                logger.exception("could not append the record of a request to %s", parsed.scenario)
                raise web.HTTPServiceUnavailable(text="the record store cannot take the request's record") from exc
        return web.json_response(provider_response(response), headers=reply_headers)

    async def relay_stream(
        self,
        request: web.Request,
        path: str,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        served: ServedArtifact,
        *,
        record: bool,
        traffic: TrafficChoice | None = None,
    ) -> web.StreamResponse:
        """Stream the engine's answer to the client and, for a recorded call, append its record at the end. An SSE
        stream gets the record's receipt before its terminal event."""
        handler = self.handler(served)
        try:
            upstream = await handler.inference_stream(
                InferenceArtifact(served.artifact_id), path, handler.address_model(payload, served.model_name)
            )
        except (UpstreamStatusError, aiohttp.ClientError) as exc:
            raise web.HTTPBadGateway(text=f"the engine at {served.url} failed: {exc}") from exc
        except TimeoutError as exc:
            raise web.HTTPGatewayTimeout(text=f"the engine at {served.url} did not answer in time") from exc
        body = bytearray()
        complete = False
        error: str | None = None
        recorded = False
        response: web.StreamResponse | None = None
        try:
            headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower() not in (RECORD_HEADER, ARTIFACT_HEADER, "content-length", "transfer-encoding")
            }
            headers[ARTIFACT_HEADER] = served.artifact_id
            content_type = next((value for key, value in headers.items() if key.lower() == "content-type"), "")
            response = web.StreamResponse(status=upstream.status, headers=headers)
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
                            self.record,
                            parsed,
                            payload,
                            captured,
                            served.artifact_id,
                            traffic,
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
            if response is None:
                raise
            error = "the client disconnected"
        finally:
            await upstream.close()
            if error is not None:
                logger.warning("stream of %s for %s ended early: %s", path, parsed.scenario, error)
            if record and not recorded:
                captured = stream_record(upstream, bytes(body), complete=complete, error=error)
                try:
                    await asyncio.to_thread(
                        self.record,
                        parsed,
                        payload,
                        captured,
                        served.artifact_id,
                        traffic,
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
            return await replica.answer(request, path, record=not rollout, rollout=rollout)

        return inference

    async def startup_status(request: web.Request) -> web.Response:
        status = await asyncio.to_thread(
            replica.startup_status, request.match_info["scenario"], request.query.get("model", ACTIVE)
        )
        return web.json_response(status, headers={"Cache-Control": "no-store"})

    async def scenario_inference(request: web.Request) -> web.StreamResponse:
        """A call to ``/v1/*`` with the scenario in the path, answered and recorded as the same call to ``/v1/*``.

        A harness that a client installed sends no scenario header: its model binding names the scenario in its
        base URL (``reef.harness.sync``).
        """
        path = checked(request.match_info["route"])
        rollout = request.headers.get(ROLLOUT_HEADER, "").strip().lower() in ("1", "true")
        return await replica.answer(
            request, path, scenario=request.match_info["scenario"], record=not rollout, rollout=rollout
        )

    async def evaluation(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(request, path, scenario=request.match_info["scenario"], record=False, rollout=True)

    async def artifact(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(
            request,
            path,
            scenario=request.match_info["scenario"],
            name=request.match_info["artifact_id"],
            record=True,
        )

    async def artifact_evaluation(request: web.Request) -> web.StreamResponse:
        path = checked(request.match_info["route"])
        return await replica.answer(
            request,
            path,
            scenario=request.match_info["scenario"],
            name=request.match_info["artifact_id"],
            record=False,
            rollout=True,
        )

    for path in INFERENCE_PATHS:
        app.router.add_post(path, recorded(path))
    app.router.add_get("/reef/scenarios/{scenario}/inference/status", startup_status)
    app.router.add_post("/reef/scenarios/{scenario}/v1/{route:.+}", scenario_inference)
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
    """The inference HTTP app: model routes, startup status, and ``/healthz`` behind the service tokens."""
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
    "ARTIFACT_HEADER",
    "INFERENCE_PATHS",
    "RECORD_HEADER",
    "SESSION_KEY_HEADER",
    "InferenceReplica",
    "NotServed",
    "ServedModelsCache",
    "StaleServedModels",
    "TrafficChoice",
    "TrafficWeights",
    "add_inference_routes",
    "build_inference_app",
    "provider_response",
]
