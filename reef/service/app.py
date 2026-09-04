from __future__ import annotations

import asyncio
from collections.abc import Iterable

from aiohttp import web

from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.observability import InferenceObserver
from reef.runtime.inference import InferenceBackend
from reef.service.auth import create_authentication_middleware
from reef.service.errors import translate_errors
from reef.service.request_service import InferenceRetryPolicy, RequestService
from reef.service.routes import register_routes


def create_app(
    dispatcher: Dispatcher | None = None,
    *,
    tokens: str | Iterable[str] | None = None,
    inference_backend: InferenceBackend | None = None,
    inference_retry_policy: InferenceRetryPolicy | None = None,
    inference_observer: InferenceObserver | None = None,
    close_dispatcher: bool = False,
):
    request_service = RequestService(
        dispatcher or build_default_dispatcher(),
        retry_policy=inference_retry_policy,
        inference_observer=inference_observer,
    )
    request_service_key = web.AppKey("reef_request_service", RequestService)
    app = web.Application(middlewares=[create_authentication_middleware(tokens), translate_errors])
    app[request_service_key] = request_service
    register_routes(
        app,
        request_service=request_service,
        inference_backend=inference_backend,
    )

    async def cleanup_observer(app: web.Application) -> None:
        await asyncio.to_thread(app[request_service_key].close)

    app.on_cleanup.append(cleanup_observer)
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
