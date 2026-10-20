from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from urllib.parse import urlsplit

from aiohttp import web

from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.runtime.interfaces import InferenceHandler
from reef.service.auth import create_authentication_middleware, normalize_tokens
from reef.service.cors import configure_browser_access
from reef.service.errors import translate_errors
from reef.service.lifecycle import (
    LifecycleSettings,
    PendingWrites,
    RequestAcceptance,
    RoutingPublisher,
    configure_owner_watch,
    create_acceptance_middleware,
    create_control_app,
    register_admission,
)
from reef.service.request_service import InferenceRetryPolicy, RequestService
from reef.service.routes import register_routes
from reef.service.routing import RoutingAdapter, RoutingRequestError, RoutingSnapshot, check_policy_change
from reef.service.routing_preview import PreviewSettings, RoutingPreviewClient
from reef.storage.records import RecordRetention

logger = logging.getLogger(__name__)
_RECORD_RETENTION_INTERVAL_SECONDS = 60.0
#: The largest request body the service reads. aiohttp's default is 1 MiB, and a coding agent's first call carries
#: its whole context: Claude Code with a dozen MCP servers and a few hundred skills sends several MiB, which the
#: default answered with 413 and the agent reported as a request too large.
MAX_REQUEST_BYTES = 64 * 1024 * 1024
REQUEST_SERVICE_KEY = web.AppKey("reef_request_service", RequestService)
#: The private lifecycle application of a managed child; ``serve_managed`` serves it on the control socket.
CONTROL_APPLICATION = web.AppKey("reef_control_application", web.Application)
#: The only public routes of a routed deployment: chat completions, feedback, and liveness.
ROUTED_PUBLIC_ROUTES = frozenset({("POST", "/v1/chat/completions"), ("POST", "/reef/report"), ("GET", "/healthz")})


def get_dispatcher(app: web.Application) -> Dispatcher:
    """The dispatcher that a Reef app serves; extension routes read scenario state through it.

    Raises ``KeyError`` for an app that ``create_app`` did not build.
    """
    return app[REQUEST_SERVICE_KEY].dispatcher


@web.middleware
async def routed_routes_only(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    """Answer 404 for every route a routed deployment does not serve, before the route runs."""
    if (request.method, request.path) not in ROUTED_PUBLIC_ROUTES:
        raise web.HTTPNotFound(text=f"{request.method} {request.path} is not served by a routed deployment")
    return await handler(request)


async def _maintain_records(dispatcher: Dispatcher, retention: RecordRetention, stopped: asyncio.Event) -> None:
    while not stopped.is_set():
        try:
            purged = await asyncio.to_thread(dispatcher.prune_record_archives, retention)
            if purged:
                logger.warning("record capacity limit evicted %d bodies; training data may be incomplete", purged)
        except Exception:
            logger.exception("record retention failed; will retry on the next sweep")
        try:
            await asyncio.wait_for(stopped.wait(), timeout=_RECORD_RETENTION_INTERVAL_SECONDS)
        except TimeoutError:
            continue


class RoutingPublication(RoutingPublisher):
    """Replace the service's routing policy and router runtime together, keeping the per-target handlers.

    It holds ``request_service`` and reads and replaces only that service's ``routing`` adapter.
    ``current`` returns it, None without routing. ``publish`` refuses when routing is off, loads the
    manifest, and refuses one whose digest is not ``snapshot_version`` or that ``check_policy_change``
    refuses (a changed catalog, automatic model or set of Scenario names). Otherwise it assigns a new
    adapter with the old handlers, credentials and readiness timeout, whose constructor checks the router
    runtime hash; a refusal leaves the adapter unchanged. ``check_ready``
    refuses a missing or unpublished runtime, then checks every provider and previews every Scenario.
    """

    def __init__(self, request_service: RequestService) -> None:
        self.request_service = request_service

    def current(self) -> RoutingAdapter | None:
        return self.request_service.routing

    async def check_ready(self) -> None:
        routing = self.request_service.routing
        preview = self.request_service.preview
        if routing is None or routing.router_config_hash is None or preview is None:
            raise RoutingRequestError("routing has no verified router runtime", code="routing_unpublished", status=503)
        await routing.check_ready()
        await preview.check_ready(routing)

    def publish(self, manifest_path: str, snapshot_version: str, router_config_hash: str) -> None:
        current = self.request_service.routing
        if current is None:
            raise ValueError("routing is not enabled")
        snapshot = RoutingSnapshot.load(manifest_path)
        if snapshot.version != snapshot_version:
            raise ValueError("manifest does not have the requested snapshot version")
        # The owner checked this before changing the router; the child keeps its own boundary.
        check_policy_change(current.snapshot, snapshot)
        # One assignment: a request admitted later captures the new manifest and runtime hash together.
        self.request_service.routing = RoutingAdapter(
            snapshot, router_config_hash, current.handlers, current.api_keys, current.readiness_timeout_s
        )


def create_app(
    dispatcher: Dispatcher,
    *,
    tokens: str | Iterable[str] | None = None,
    evaluation_tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
    inference_handler: InferenceHandler | None = None,
    inference_retry_policy: InferenceRetryPolicy | None = None,
    close_dispatcher: bool = False,
    record_retention: RecordRetention | None = None,
    open_local_cycles_at: str | None = None,
    routing: RoutingAdapter | None = None,
    lifecycle: LifecycleSettings | None = None,
    preview: PreviewSettings | None = None,
) -> web.Application:
    """Build the HTTP app around an existing dispatcher; close it only when requested.

    ``open_local_cycles_at``: the address this app answers at; once it accepts
    a connection, the dispatcher's held local cycles open (see
    ``Dispatcher(hold_local_cycles=True)``). ``evaluation_tokens`` open the
    evaluation routes only (see ``create_authentication_middleware``).
    ``routing`` serves chat completions through scenario-scoped targets; its
    automatic requests are selected by the Semantic Router's routing preview
    that ``preview`` names, and the app owns that client's session. A routed
    app serves only chat completions, reports and ``/healthz``.

    Without ``lifecycle``, startup checks every enabled target and one preview
    per Scenario. A managed child (``lifecycle``) instead starts closed, with
    no verified router runtime; its owner publishes one and opens admission
    through the control application stored under ``CONTROL_APPLICATION``.
    """
    if routing is not None:
        if not normalize_tokens(tokens):
            raise ValueError("routing requires a service token")
        if type(dispatcher.recipe) is not Recipe or dispatcher.recipe.training_runtime is not None:
            raise ValueError("routing requires the base record-only Recipe")
        if preview is None:
            raise ValueError("routing requires the Semantic Router preview settings")
    elif preview is not None:
        raise ValueError("Semantic Router preview settings require a routing manifest")
    if lifecycle is not None and (
        routing is None
        or lifecycle.snapshot_version != routing.snapshot.version
        or (lifecycle.router_config_hash or None) != routing.router_config_hash
    ):
        raise ValueError("managed lifecycle requires its exact routing snapshot and router runtime")
    if lifecycle is None and routing is not None and routing.router_config_hash is None:
        raise ValueError("an unmanaged routed app needs a verified router runtime hash")
    writes = None if lifecycle is None else PendingWrites()
    preview_client = None if preview is None else RoutingPreviewClient(preview)
    request_service = RequestService(
        dispatcher, retry_policy=inference_retry_policy, routing=routing, writes=writes, preview=preview_client
    )
    app = web.Application(
        middlewares=[create_authentication_middleware(tokens, evaluation_tokens=evaluation_tokens), translate_errors],
        client_max_size=MAX_REQUEST_BYTES,
    )
    app[REQUEST_SERVICE_KEY] = request_service
    if routing is not None:
        app.middlewares.insert(1, routed_routes_only)
    if lifecycle is not None and writes is not None:
        managed = RequestAcceptance(lifecycle, writes)
        managed.publisher = RoutingPublication(request_service)
        # Inside authentication, outside error translation, so generated error
        # responses for admitted requests also complete within their root lease.
        app.middlewares.insert(len(app.middlewares) - 1, create_acceptance_middleware(managed))
        register_admission(app, managed)
        app[CONTROL_APPLICATION] = create_control_app(managed)
        owner_fd = os.environ.get("REEF_ROUTING_OWNER_FD")
        if owner_fd is not None:
            configure_owner_watch(app, managed, int(owner_fd))

        async def finalize_requests(app: web.Application) -> None:
            managed.owner_lost()
            if not await writes.wait(lifecycle.finalization_timeout_seconds):
                logger.error(
                    "managed request finalization deadline exceeded with %d pending writes", len(writes.tasks)
                )

        app.on_shutdown.append(finalize_requests)
    configure_browser_access(app, console_origins)
    register_routes(
        app,
        request_service=request_service,
        inference_handler=inference_handler,
    )
    if routing is not None and preview_client is not None:

        async def run_selection(app: web.Application) -> AsyncIterator[None]:
            # One reusable router session for the app's lifetime, closed after requests have drained.
            await preview_client.start()
            try:
                if lifecycle is None:
                    await routing.check_ready()
                    await preview_client.check_ready(routing)
                # A managed child checks both later, before each open, under the runtime its owner publishes.
                yield
            finally:
                await preview_client.close()

        app.cleanup_ctx.append(run_selection)
    if record_retention is not None:

        async def maintain_records(app: web.Application):
            stopped = asyncio.Event()
            task = asyncio.create_task(_maintain_records(request_service.dispatcher, record_retention, stopped))
            try:
                yield
            finally:
                # Let an in-flight retention sweep finish before closing the dispatcher.
                stopped.set()
                await task

        app.cleanup_ctx.append(maintain_records)
    if open_local_cycles_at is not None:
        address = urlsplit(open_local_cycles_at)
        host, port = address.hostname or "127.0.0.1", address.port or 80

        async def open_local_cycles(app: web.Application):
            async def once_listening() -> None:
                while True:
                    try:
                        _reader, writer = await asyncio.open_connection(host, port)
                    except OSError:
                        await asyncio.sleep(0.1)
                        continue
                    writer.close()
                    break
                await asyncio.to_thread(app[REQUEST_SERVICE_KEY].dispatcher.open_local_cycles)

            # on_startup runs before the site listens: wait for the socket, off the startup path.
            task = asyncio.create_task(once_listening())
            try:
                yield
            finally:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        app.cleanup_ctx.append(open_local_cycles)
    if close_dispatcher:

        async def stop_local_cycles(app: web.Application) -> None:
            # The listening socket is closed by now: a harness cycle's calls through Reef fail from here on.
            app[REQUEST_SERVICE_KEY].dispatcher.stop_local_cycles()

        async def cleanup(app: web.Application) -> None:
            await asyncio.to_thread(app[REQUEST_SERVICE_KEY].dispatcher.close)

        app.on_shutdown.append(stop_local_cycles)
        app.on_cleanup.append(cleanup)
    return app


__all__ = [
    "InferenceRetryPolicy",
    "RequestService",
    "create_app",
    "get_dispatcher",
]
