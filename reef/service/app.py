from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterable
from urllib.parse import urlsplit

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
    evaluation_tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
    inference_handler: InferenceHandler | None = None,
    inference_retry_policy: InferenceRetryPolicy | None = None,
    close_dispatcher: bool = False,
    record_retention: RecordRetention | None = None,
    open_local_cycles_at: str | None = None,
) -> web.Application:
    """Build the HTTP app around an existing dispatcher; close it only when requested.

    ``open_local_cycles_at``: the address this app answers at; once it accepts
    a connection, the dispatcher's held local cycles open (see
    ``Dispatcher(hold_local_cycles=True)``). ``evaluation_tokens`` open the
    evaluation routes only (see ``create_authentication_middleware``).
    """
    request_service = RequestService(dispatcher, retry_policy=inference_retry_policy)
    request_service_key = web.AppKey("reef_request_service", RequestService)
    app = web.Application(
        middlewares=[create_authentication_middleware(tokens, evaluation_tokens=evaluation_tokens), translate_errors],
        client_max_size=MAX_REQUEST_BYTES,
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
                await asyncio.to_thread(app[request_service_key].dispatcher.open_local_cycles)

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
            app[request_service_key].dispatcher.stop_local_cycles()

        async def cleanup(app: web.Application) -> None:
            await asyncio.to_thread(app[request_service_key].dispatcher.close)

        app.on_shutdown.append(stop_local_cycles)
        app.on_cleanup.append(cleanup)
    return app


__all__ = [
    "InferenceRetryPolicy",
    "RequestService",
    "create_app",
]
