"""The remote inference component: this host's engines, their control operations and inference requests over HTTP."""

from __future__ import annotations

import asyncio
import hmac
import logging
import shutil
import tarfile
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, web

from reef.inference.deployment import inference_service_for
from reef.inference.http import HOP_BY_HOP_HEADERS
from reef.inference.remote.config import CONTROL_PATH
from reef.runtime.deployment import InferenceResources, InferenceService
from reef.runtime.executor.placement import ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.interfaces import InferenceBackend, WeightFiles

logger = logging.getLogger(__name__)

#: Request headers that describe the connection to this component, not the request to the engines.
FORWARD_EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {"host", "authorization"}
#: Largest inference request body; weight uploads stream and are not bound by it.
MAX_REQUEST_BYTES = 64 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


class LocalEngineResources(InferenceResources):
    """The GPUs that one engine start reserves on this host, in the component's Ray session.

    ``num_gpus`` is positive. ``start`` reserves that many GPU bundles for
    inference and ``close`` releases them; the Ray session outlives both.
    """

    def __init__(self, num_gpus: int) -> None:
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
            raise ValueError("a remote engine start needs a positive GPU count")
        self.num_gpus = num_gpus
        self.reservation: ModelGpuReservation | None = None

    @property
    def inference_placement(self) -> Any:
        return None if self.reservation is None else self.reservation.inference

    def start(self) -> None:
        self.reservation = reserve_model_gpus(ModelGpuLayout(0, self.num_gpus))

    def close(self) -> None:
        reservation, self.reservation = self.reservation, None
        if reservation is not None:
            reservation.release()


class RemoteInferenceServer:
    """Serve this host's engines to one Reef control plane over HTTP.

    Values: ``control_token`` and ``inference_token`` are different non-empty
    strings; ``engine`` is the non-empty name of the one inference backend this
    component may start, fixed when the component starts; ``work_dir`` holds
    the uploaded weight files. ``resources``, ``service``, ``backend`` and
    ``inference_url`` are all None (no engines run) or all set (one started
    engine stack). Each ``adapters`` entry maps the name of an adapter that the
    running engines loaded to the directory under ``work_dir`` it was loaded
    from; the map is empty when no engines run.

    Routes under ``CONTROL_PATH`` need ``Authorization: Bearer <control_token>``;
    every other path is forwarded to the engines and needs the inference token.
    ``PUT engines`` starts ``engine`` with the input that the control plane
    sends, replacing running engines, and refuses any other backend name. The
    other control routes call the engines' ``InferenceBackend``, one at a time.
    ``GET health`` never waits behind such a call: it observes failures through
    ``InferenceService.poll``. Full weight files are removed after the load, and
    an adapter's files when it is unloaded or replaced.
    """

    def __init__(self, *, engine: str, control_token: str, inference_token: str, work_dir: Path) -> None:
        if not engine:
            raise ValueError("the remote inference component needs the engine backend it may start")
        if not control_token or not inference_token or control_token == inference_token:
            raise ValueError("the remote inference component needs two different non-empty tokens")
        self.engine = engine
        self.control_token = control_token
        self.inference_token = inference_token
        self.work_dir = work_dir
        self.lock = asyncio.Lock()
        self.session: ClientSession | None = None
        self.resources: LocalEngineResources | None = None
        self.service: InferenceService | None = None
        self.backend: InferenceBackend | None = None
        self.inference_url: str | None = None
        self.adapters: dict[str, Path] = {}

    def application(self) -> web.Application:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES, middlewares=[self.report_errors])
        control = [
            ("PUT", "engines", self.start_engines),
            ("DELETE", "engines", self.stop_engines),
            ("GET", "health", self.health),
            ("GET", "runtime-load-ids", self.runtime_load_ids),
            ("POST", "initialize-version", self.initialize_version),
            ("POST", "{operation:pause|resume|recover|abort}", self.receiver_operation),
            ("PUT", "weights", self.load_weights),
            ("DELETE", "adapters/{name}", self.unload_adapter),
        ]
        for method, route, handler in control:
            app.router.add_route(method, f"{CONTROL_PATH}/{route}", handler)
        app.router.add_route("*", "/{path:.*}", self.forward)
        app.on_startup.append(self.open_session)
        app.on_cleanup.append(self.close_application)
        return app

    # -- Lifecycle

    async def open_session(self, app: web.Application) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = ClientSession(timeout=ClientTimeout(total=None), auto_decompress=False)

    async def close_application(self, app: web.Application) -> None:
        async with self.lock:
            await asyncio.to_thread(self.stop_stack)
        if self.session is not None:
            await self.session.close()

    def start_stack(self, backend_name: str, config: Mapping[str, Any], num_gpus: int) -> None:
        self.stop_stack()
        resources = LocalEngineResources(num_gpus)
        service: InferenceService | None = None
        try:
            resources.start()
            service = inference_service_for(backend_name, config)
            connection = service.start(resources)
            service.check_health()
            # The engines start fenced; Reef's coordinator decides when they serve.
            service.prepare_weight_transfer(connection)
            backend = service.backend(connection)
            inference_url = backend.inference_url().rstrip("/")
        except BaseException:
            if service is not None:
                with suppress(Exception):
                    service.close()
            with suppress(Exception):
                resources.close()
            raise
        self.resources, self.service, self.backend, self.inference_url = resources, service, backend, inference_url
        logger.info("started %s engines on %d GPUs; serving %s", backend_name, num_gpus, inference_url)

    def stop_stack(self) -> None:
        service, resources = self.service, self.resources
        self.service = self.resources = self.backend = self.inference_url = None
        try:
            if service is not None:
                service.close()
        finally:
            if resources is not None:
                resources.close()
            for directory in self.adapters.values():
                shutil.rmtree(directory, ignore_errors=True)
            self.adapters.clear()

    # -- Control routes

    @web.middleware
    async def report_errors(
        self, request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as exc:
            logger.exception("remote inference request %s %s failed", request.method, request.path)
            return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def authorize(self, request: web.Request, token: str) -> None:
        if not hmac.compare_digest(request.headers.get("Authorization", ""), f"Bearer {token}"):
            raise web.HTTPUnauthorized(text="invalid or missing bearer token")

    def running_backend(self) -> InferenceBackend:
        if self.backend is None:
            raise web.HTTPConflict(text="no engines are running; the control plane starts them with PUT engines")
        return self.backend

    async def start_engines(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        body = await request.json()
        if not isinstance(body, Mapping):
            raise web.HTTPBadRequest(text="the engine request must be a JSON object")
        backend_name, config, num_gpus = body.get("backend"), body.get("config"), body.get("num_gpus")
        if not isinstance(backend_name, str) or not backend_name or not isinstance(config, Mapping):
            raise web.HTTPBadRequest(text="the engine request needs a backend name and its config object")
        if backend_name != self.engine:
            # The control plane chooses engine input, never which code this host imports.
            raise web.HTTPBadRequest(text=f"this component starts only {self.engine!r} engines, not {backend_name!r}")
        if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus <= 0:
            raise web.HTTPBadRequest(text="the engine request needs a positive num_gpus")
        async with self.lock:
            await asyncio.to_thread(self.start_stack, backend_name, config, num_gpus)
        return web.json_response({"ok": True})

    async def stop_engines(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        async with self.lock:
            await asyncio.to_thread(self.stop_stack)
        return web.json_response({"ok": True})

    async def health(self, request: web.Request) -> web.StreamResponse:
        """Report failed engines; busy engines, such as ones loading weights, are healthy."""
        self.authorize(request, self.control_token)
        self.running_backend()
        service = self.service
        if service is not None:
            await asyncio.to_thread(service.poll)
        return web.json_response({"ok": True})

    async def runtime_load_ids(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        backend = self.running_backend()
        async with self.lock:
            versions = await asyncio.to_thread(backend.runtime_load_ids)
        return web.json_response({"runtime_load_ids": [str(version) for version in versions]})

    async def initialize_version(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        backend = self.running_backend()
        body = await request.json()
        runtime_load_id = body.get("runtime_load_id") if isinstance(body, Mapping) else None
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise web.HTTPBadRequest(text="initialize-version needs a runtime_load_id")
        async with self.lock:
            await asyncio.to_thread(backend.initialize_version, runtime_load_id)
        return web.json_response({"ok": True})

    async def receiver_operation(self, request: web.Request) -> web.StreamResponse:
        """Pause, resume, recover or abort the engines, as the route names."""
        self.authorize(request, self.control_token)
        backend = self.running_backend()
        operations = {
            "pause": backend.pause,
            "resume": backend.resume,
            "recover": backend.recover,
            "abort": backend.abort,
        }
        async with self.lock:
            await asyncio.to_thread(operations[request.match_info["operation"]])
        return web.json_response({"ok": True})

    async def load_weights(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        backend = self.running_backend()
        layout = request.query.get("layout")
        adapter_name = request.query.get("adapter_name")
        runtime_load_id = request.query.get("runtime_load_id")
        if layout not in ("full", "adapter") or (adapter_name is None) != (layout == "full"):
            raise web.HTTPBadRequest(text="weights need layout full, or layout adapter with an adapter_name")
        directory = self.work_dir / uuid4().hex
        archive = directory.with_suffix(".tar")
        try:
            with archive.open("wb") as stream:
                # Write in a thread: a multi-gigabyte upload must not stall the requests this component forwards.
                async for chunk in request.content.iter_chunked(UPLOAD_CHUNK_BYTES):
                    await asyncio.to_thread(stream.write, chunk)
            await asyncio.to_thread(unpack_archive, archive, directory)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            archive.unlink(missing_ok=True)
        files = WeightFiles(directory, "full" if layout == "full" else "adapter")
        loaded = False
        try:
            async with self.lock:
                await asyncio.to_thread(
                    backend.load_weight_files, files, adapter_name=adapter_name, runtime_load_id=runtime_load_id
                )
            loaded = True
        finally:
            # Engines read full weights once; an adapter's files stay until it is unloaded.
            if adapter_name is None or not loaded:
                shutil.rmtree(directory, ignore_errors=True)
        if adapter_name is not None:
            previous = self.adapters.get(adapter_name)
            self.adapters[adapter_name] = directory
            if previous is not None:
                shutil.rmtree(previous, ignore_errors=True)
        return web.json_response({"ok": True})

    async def unload_adapter(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        backend = self.running_backend()
        name = request.match_info["name"]
        async with self.lock:
            await asyncio.to_thread(backend.unload_adapter, name)
        directory = self.adapters.pop(name, None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        return web.json_response({"ok": True})

    # -- Inference requests

    async def forward(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.inference_token)
        upstream_url = self.inference_url
        session = self.session
        if upstream_url is None or session is None:
            raise web.HTTPServiceUnavailable(text="no engines are running")
        headers = {
            name: value for name, value in request.headers.items() if name.lower() not in FORWARD_EXCLUDED_HEADERS
        }
        body = await request.read()
        async with session.request(
            request.method, upstream_url + request.rel_url.path_qs, headers=headers, data=body
        ) as upstream:
            response = web.StreamResponse(
                status=upstream.status,
                headers={
                    name: value for name, value in upstream.headers.items() if name.lower() not in HOP_BY_HOP_HEADERS
                },
            )
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
        return response


def unpack_archive(archive: Path, directory: Path) -> None:
    """Unpack an uploaded tar; the ``data`` filter refuses absolute paths and links that leave ``directory``."""
    directory.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")
