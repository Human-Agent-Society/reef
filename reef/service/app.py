from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

from aiohttp import web

from reef.artifact.sources import ArtifactSources
from reef.dispatcher import Dispatcher
from reef.dispatcher.served import PublishedEngines
from reef.runtime.interfaces import FixedEngineClients, InferenceHandler
from reef.service.artifact_evaluations import ArtifactEvaluations
from reef.service.artifact_imports import ArtifactImports
from reef.service.cors import configure_browser_access
from reef.service.inference_service import InferenceReplica, ServedModelsCache
from reef.service.request_service import RequestService
from reef.service.routes import register_data_service_routes, register_routes
from reef.service.web import service_application
from reef.storage.records import RecordRetention

logger = logging.getLogger(__name__)
_RECORD_RETENTION_INTERVAL_SECONDS = 60.0
REQUEST_SERVICE_KEY = web.AppKey("reef_request_service", RequestService)


def get_dispatcher(app: web.Application) -> Dispatcher:
    """The dispatcher that a Reef app serves; extension routes read scenario state through it.

    Raises ``KeyError`` for an app that ``create_app`` did not build.
    """
    return app[REQUEST_SERVICE_KEY].dispatcher


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


def create_data_app(
    dispatcher: Dispatcher,
    *,
    tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
) -> web.Application:
    """The HTTP app of the data service: the data routes and the reads of what each scenario stores.

    ``dispatcher`` runs the data service and no control (``reef serve --component data``): it loads each scenario
    on demand, for the checks of a data write and for the reads. The app closes it on cleanup.
    """
    if "data" not in dispatcher.components:
        raise ValueError("the data app runs the data service: build its dispatcher with the data service")
    request_service = RequestService(dispatcher)
    app = service_application(tokens)
    configure_browser_access(app, console_origins)
    app[REQUEST_SERVICE_KEY] = request_service
    register_data_service_routes(app, request_service=request_service, evaluations=ArtifactEvaluations(dispatcher))

    async def cleanup(app: web.Application) -> None:
        await asyncio.to_thread(dispatcher.close)

    app.on_cleanup.append(cleanup)
    return app


def create_app(
    dispatcher: Dispatcher,
    *,
    tokens: str | Iterable[str] | None = None,
    evaluation_tokens: str | Iterable[str] | None = None,
    console_origins: Iterable[str] = (),
    inference_handler: InferenceHandler | None = None,
    close_dispatcher: bool = False,
    record_retention: RecordRetention | None = None,
    open_local_cycles_at: str | None = None,
    evaluations: ArtifactEvaluations | None = None,
    artifact_imports: ArtifactImports | None = None,
) -> web.Application:
    """Build the HTTP app around an existing dispatcher; close it only when requested.

    ``open_local_cycles_at``: the address this app answers at; once it accepts
    a connection, the dispatcher's held local cycles open (see
    ``Dispatcher(hold_local_cycles=True)``). ``evaluation_tokens`` open the
    evaluation routes only (see ``create_authentication_middleware``).
    ``evaluations`` runs the artifact evaluations of the evaluation routes, and
    the automatic ones while the app runs; None runs the routes' evaluations
    with reef-eval's Harbor lab and evaluates no new artifact.
    ``artifact_imports`` reads and holds artifact imports; None reads the
    built-in sources into the system's temporary directory.

    The app runs the inference routes in the same process (``reef.service.inference_service``): they read
    ``served_models`` and reach each row's engine through the runtime that control serves it with, or through
    ``inference_handler`` when the caller gives one.
    """
    publisher = dispatcher.publisher
    if publisher is None:
        raise ValueError("the Reef app runs control: build its dispatcher with the control service")
    runtime = dispatcher.recipe.runtime
    replica = InferenceReplica(
        ServedModelsCache(dispatcher.coordination.served_models),
        dispatcher.record_storage,
        PublishedEngines(publisher) if inference_handler is None else FixedEngineClients(inference_handler),
        timeout_seconds=300.0 if runtime is None else runtime.inference_timeout_s,
        for_codex=dispatcher.recipe.harness_adapter == "codex",
    )
    request_service = RequestService(dispatcher)
    app = service_application(tokens, evaluation_tokens=evaluation_tokens)
    configure_browser_access(app, console_origins)
    app[REQUEST_SERVICE_KEY] = request_service
    artifact_evaluations = ArtifactEvaluations(dispatcher) if evaluations is None else evaluations
    register_routes(
        app,
        request_service=request_service,
        replica=replica,
        evaluations=artifact_evaluations,
        imports=(
            ArtifactImports(ArtifactSources.from_setting(()), Path(tempfile.gettempdir()) / "reef-artifact-imports")
            if artifact_imports is None
            else artifact_imports
        ),
    )

    async def evaluate_new_artifacts(app: web.Application):
        artifact_evaluations.open()
        try:
            yield
        finally:
            await asyncio.to_thread(artifact_evaluations.close)

    app.cleanup_ctx.append(evaluate_new_artifacts)
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
    "RequestService",
    "create_app",
    "create_data_app",
    "get_dispatcher",
]
