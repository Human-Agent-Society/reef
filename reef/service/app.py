from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from aiohttp import web

from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.records import RecordRetention
from reef.runtime.inference import InferenceBackend
from reef.service.auth import create_authentication_middleware
from reef.service.cors import configure_browser_access
from reef.service.errors import translate_errors
from reef.service.request_service import InferenceRetryPolicy, RequestService
from reef.service.routes import register_routes

logger = logging.getLogger(__name__)
_RECORD_RETENTION_INTERVAL_SECONDS = 60.0


async def _maintain_records(dispatcher: Dispatcher, retention: RecordRetention, stopped: asyncio.Event) -> None:
    while not stopped.is_set():
        try:
            purged = await asyncio.to_thread(dispatcher.prune_record_archives, retention)
            if purged:
                logger.info("purged %d compacted trace bodies under record retention limits", purged)
        except Exception:
            logger.exception("record retention failed; will retry on the next sweep")
        try:
            await asyncio.wait_for(stopped.wait(), timeout=_RECORD_RETENTION_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            continue


def create_app(
    dispatcher: Dispatcher | None = None,
    *,
    tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
    inference_backend: InferenceBackend | None = None,
    inference_retry_policy: InferenceRetryPolicy | None = None,
    close_dispatcher: bool = False,
    record_retention: RecordRetention | None = None,
):
    request_service = RequestService(
        dispatcher or build_default_dispatcher(),
        retry_policy=inference_retry_policy,
    )
    request_service_key = web.AppKey("reef_request_service", RequestService)
    app = web.Application(middlewares=[create_authentication_middleware(tokens), translate_errors])
    configure_browser_access(app, console_origins)
    app[request_service_key] = request_service
    register_routes(
        app,
        request_service=request_service,
        inference_backend=inference_backend,
    )
    if record_retention is not None:

        async def maintain_records(app: web.Application):
            stopped = asyncio.Event()
            task = asyncio.create_task(_maintain_records(request_service.dispatcher, record_retention, stopped))
            try:
                yield
            finally:
                # Let an in-flight SQLite batch finish before closing the dispatcher.
                stopped.set()
                await task

        app.cleanup_ctx.append(maintain_records)
    if dispatcher is None or close_dispatcher:

        async def cleanup(app: web.Application) -> None:
            await asyncio.to_thread(app[request_service_key].dispatcher.close)

        app.on_cleanup.append(cleanup)
    return app


__all__ = [
    "InferenceRetryPolicy",
    "RequestService",
    "create_app",
]
