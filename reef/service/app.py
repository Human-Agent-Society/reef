from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from aiohttp import web

from reef.dispatcher import Dispatcher
from reef.runtime.interfaces import InferenceHandler
from reef.service.auth import create_authentication_middleware
from reef.service.cors import configure_browser_access
from reef.service.errors import translate_errors
from reef.service.request_service import InferenceRetryPolicy, RequestService
from reef.service.routes import register_routes
from reef.storage.records import RecordRetention

logger = logging.getLogger(__name__)
_RECORD_RETENTION_INTERVAL_SECONDS = 60.0
#: The largest request body the service reads. aiohttp's default is 1 MiB, and a coding agent's first call carries
#: its whole context: Claude Code with a dozen MCP servers and a few hundred skills sends several MiB, which the
#: default answered with 413 and the agent reported as a request too large.
MAX_REQUEST_BYTES = 64 * 1024 * 1024


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


def create_app(
    dispatcher: Dispatcher,
    *,
    tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
    inference_handler: InferenceHandler | None = None,
    inference_retry_policy: InferenceRetryPolicy | None = None,
    close_dispatcher: bool = False,
    record_retention: RecordRetention | None = None,
) -> web.Application:
    """Build the HTTP app around an existing dispatcher; close it only when requested."""
    request_service = RequestService(dispatcher, retry_policy=inference_retry_policy)
    request_service_key = web.AppKey("reef_request_service", RequestService)
    app = web.Application(
        middlewares=[create_authentication_middleware(tokens), translate_errors], client_max_size=MAX_REQUEST_BYTES
    )
    configure_browser_access(app, console_origins)
    app[request_service_key] = request_service
    register_routes(
        app,
        request_service=request_service,
        inference_handler=inference_handler,
    )
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
    if close_dispatcher:

        async def cleanup(app: web.Application) -> None:
            await asyncio.to_thread(app[request_service_key].dispatcher.close)

        app.on_cleanup.append(cleanup)
    return app


__all__ = [
    "InferenceRetryPolicy",
    "RequestService",
    "create_app",
]
