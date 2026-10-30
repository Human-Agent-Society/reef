"""The inference service: answers model requests from ``served_models`` and the record store alone.

An inference replica holds no scenario. For each request it reads the ``ServedArtifact`` of the request's
``ScenarioModel`` from ``served_models``, keeps it for the whole request, sends the request to the engine at its URL
under its model name, appends the request's ``AgentRecord`` to the scenario's record store, and returns the answer
with the record's ID. Control restarts do not stop it: control is needed only to change ``served_models``.

Rules:

- ``x-reef-scenario`` names the scenario; ``x-reef-tag-*`` pairs go to the record's metadata, as in-process. The request's ``model`` names a row of the scenario (an artifact served
  beside the active one, or an auxiliary model); any other ``model`` means ``active``.
- No row: 404. A replica that could not read ``served_models`` for longer than ``max_cache_age_seconds``: 503.
- An answer whose ``model`` is not the row's ``model_name``: the replica reads the row again and retries once; then
  409.
- A record that cannot be appended: 503. A replica never answers a request without its record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from reef.artifact.artifact import Artifact
from reef.core.artifact_ref import ArtifactRef
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.served_models import ACTIVE, ServedArtifact
from reef.inference.http import HttpInferenceHandler
from reef.runtime.interfaces import UpstreamStatusError
from reef.service.auth import create_authentication_middleware
from reef.service.request_service import with_request_tags
from reef.service.streaming import stream_record
from reef.service.wire import HeaderError, RequestHeaders, parse_request_headers
from reef.storage.scenario import ScenarioStorage
from reef.storage.served_models import ServedModels

logger = logging.getLogger(__name__)

#: The header that carries the ID of the request's record.
RECORD_HEADER = "x-reef-agent-record-id"
#: The header that carries the ID of the artifact that answered.
RELEASE_HEADER = "x-reef-release-id"
#: The model routes that an inference replica answers.
INFERENCE_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")


class NotServed(ReefError):
    """The scenario serves no model under the request's name."""


class StaleServedModels(ReefError):
    """The replica could not read ``served_models`` for longer than it may serve from its cache."""


@dataclass
class ServedModelsCache:
    """A replica's cache of ``served_models``, by scenario.

    Values: ``table`` is the table; ``cache_seconds`` is how long a read stays fresh; ``max_cache_age_seconds`` is
    how long the replica serves from an old read while the table cannot be read; ``rows`` holds each scenario's
    rows and the time of their read.

    Purpose: each request of a replica reads its ``ServedArtifact`` through it (``read``).
    """

    table: ServedModels
    cache_seconds: float = 5.0
    max_cache_age_seconds: float = 300.0
    rows: dict[str, tuple[float, Mapping[str, ServedArtifact]]] = field(default_factory=dict)

    def scenario_rows(self, scenario: str, *, refresh: bool = False) -> Mapping[str, ServedArtifact]:
        now = time.monotonic()
        held = self.rows.get(scenario)
        if held is not None and not refresh and now - held[0] < self.cache_seconds:
            return held[1]
        try:
            rows = self.table.read_scenario(scenario)
        except Exception as exc:
            if held is not None and now - held[0] < self.max_cache_age_seconds:
                logger.warning("serving %s from a cached read of served_models: %s", scenario, exc)
                return held[1]
            raise StaleServedModels(f"served_models cannot be read: {exc}") from exc
        self.rows[scenario] = (now, rows)
        return rows

    def read(self, scenario: str, model: object, *, refresh: bool = False) -> ServedArtifact:
        """The ``ServedArtifact`` that answers a request for ``model`` of ``scenario``."""
        rows = self.scenario_rows(scenario, refresh=refresh)
        name = model if isinstance(model, str) and model in rows else ACTIVE
        served = rows.get(name)
        if served is None:
            raise NotServed(f"scenario {scenario!r} serves no model {name!r}")
        return served


class ArtifactRefs:
    """The release of each artifact that a replica serves, read once from the scenario's commit history.

    The release of ``artifact_id`` is the newest committed release that binds it: one whose manifest
    (``CommitRecord.components``) names it, or, with one component, the release whose ``release_id`` it is. An
    artifact that no commit binds, such as the artifact that a scenario started from, has no release here, and its
    records carry no ``artifact_ref``.
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


@dataclass
class InferenceReplica:
    """One inference replica: ``served_models``, the record store, and the engine timeout.

    Purpose: ``build_inference_app`` routes the model requests to ``answer``.
    """

    cache: ServedModelsCache
    storage: ScenarioStorage
    timeout_seconds: float = 300.0
    refs: ArtifactRefs = field(init=False)

    def __post_init__(self) -> None:
        self.refs = ArtifactRefs(self.storage)

    def record(
        self, parsed: RequestHeaders, payload: Mapping[str, Any], response: Mapping[str, Any], ref: ArtifactRef | None
    ) -> str:
        """Append the record of one answered request, in the form that the in-process path writes."""
        item = AgentRecord.create(
            scenario=parsed.scenario,
            request_type=RequestType.INFERENCE,
            payload={**with_request_tags(payload, parsed), "response": dict(response)},
            artifact_ref=ref,
        )
        store = self.storage.open(parsed.scenario)
        try:
            store.records.append(item)
        finally:
            store.close()
        return item.agent_record_id

    async def answer(self, request: web.Request, path: str) -> web.StreamResponse:
        try:
            parsed = parse_request_headers(request.headers, RequestType.INFERENCE)
        except HeaderError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        scenario = parsed.scenario
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise web.HTTPBadRequest(text="the request body is a JSON object") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(text="the request body is a JSON object")
        try:
            served = self.cache.read(scenario, payload.get("model"))
            ref = await asyncio.to_thread(self.refs.ref, scenario, served.artifact_id)
        except NotServed as exc:
            raise web.HTTPNotFound(text=str(exc)) from exc
        except StaleServedModels as exc:
            raise web.HTTPServiceUnavailable(text=str(exc)) from exc
        if payload.get("stream") is True:
            return await self.relay_stream(request, path, parsed, payload, served, ref)
        for attempt in range(2):
            handler = HttpInferenceHandler(served.url, timeout_s=self.timeout_seconds, error_label=served.url)
            body = {**payload, "model": served.model_name}
            try:
                response = await handler.inference(upstream_artifact(served, ref), path, body)
            except UpstreamStatusError as exc:
                raise web.HTTPBadGateway(text=str(exc)) from exc
            if response.get("model") in (None, served.model_name):
                break
            if attempt == 1:
                raise web.HTTPConflict(
                    text=f"the engine answered as {response.get('model')!r}, not {served.model_name!r}"
                )
            served = self.cache.read(scenario, payload.get("model"), refresh=True)
            ref = await asyncio.to_thread(self.refs.ref, scenario, served.artifact_id)
        try:
            record_id = await asyncio.to_thread(self.record, parsed, payload, response, ref)
        except Exception as exc:
            logger.exception("could not append the record of a request to %s", scenario)
            raise web.HTTPServiceUnavailable(text="the record store cannot take the request's record") from exc
        return web.json_response(response, headers={RECORD_HEADER: record_id, RELEASE_HEADER: served.artifact_id})

    async def relay_stream(
        self,
        request: web.Request,
        path: str,
        parsed: RequestHeaders,
        payload: Mapping[str, Any],
        served: ServedArtifact,
        ref: ArtifactRef | None,
    ) -> web.StreamResponse:
        handler = HttpInferenceHandler(served.url, timeout_s=self.timeout_seconds, error_label=served.url)
        try:
            upstream = await handler.inference_stream(
                upstream_artifact(served, ref), path, {**payload, "model": served.model_name}
            )
        except UpstreamStatusError as exc:
            raise web.HTTPBadGateway(text=str(exc)) from exc
        response = web.StreamResponse(
            status=upstream.status, headers={**upstream.headers, RELEASE_HEADER: served.artifact_id}
        )
        body = bytearray()
        complete = False
        try:
            await response.prepare(request)
            async for chunk in upstream.chunks:
                body.extend(chunk)
                await response.write(chunk)
            complete = True
        finally:
            await upstream.close()
            captured = stream_record(upstream, bytes(body), complete=complete)
            try:
                await asyncio.to_thread(self.record, parsed, payload, captured, ref)
            except Exception:
                logger.exception("could not append the record of a stream of %s", parsed.scenario)
        return response


def upstream_artifact(served: ServedArtifact, ref: ArtifactRef | None) -> Artifact:
    """The artifact that the engine handler gets for its request headers: the release, or the artifact by its ID."""
    return Artifact(ref or ArtifactRef(served.artifact_id, served.artifact_id, None), None)


def build_inference_app(
    served_models: ServedModels,
    storage: ScenarioStorage,
    *,
    tokens: str | tuple[str, ...] | None,
    cache_seconds: float = 5.0,
    max_cache_age_seconds: float = 300.0,
    timeout_seconds: float = 300.0,
) -> web.Application:
    """The HTTP app of one inference replica: the model routes and ``/healthz``, behind the service tokens."""
    replica = InferenceReplica(
        ServedModelsCache(served_models, cache_seconds, max_cache_age_seconds), storage, timeout_seconds
    )
    app = web.Application(middlewares=[create_authentication_middleware(tokens)])

    async def healthz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def route(path: str) -> Callable[[web.Request], Awaitable[web.StreamResponse]]:
        async def answer(request: web.Request) -> web.StreamResponse:
            return await replica.answer(request, path)

        return answer

    app.router.add_get("/healthz", healthz)
    for path in INFERENCE_PATHS:
        app.router.add_post(path, route(path))
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
    "build_inference_app",
]
