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

from reef.core.weight_layout import WEIGHT_LAYOUTS
from reef.inference.deployment import inference_service_for
from reef.inference.http import HOP_BY_HOP_HEADERS
from reef.inference.remote.config import CONTROL_PATH, SHARED_GPU_CONTROL_CAPABILITY
from reef.runtime.deployment import BorrowedGpuResources, InferenceResources, InferenceService
from reef.runtime.executor.placement import ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.interfaces import InferenceBackend, MemoryManagedInferenceBackend, WeightFiles

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
        self.reservation = reserve_model_gpus(ModelGpuLayout(training_gpus=0, inference_gpus=self.num_gpus))

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
    ``instance_id`` names this process: ``PUT engines`` and ``GET health``
    report it, so the control plane sees a restarted component, which runs no
    engines. ``GET health`` reports whether engines run and never waits behind
    a control call: it observes their failures through ``InferenceService.poll``. Full weight files are removed after the load, and
    an adapter's files when it is unloaded or replaced.

    Shared starts borrow a live Ray reservation and identify it with a unique
    ``engine_instance``. All control mutations must name that instance.
    ``memory_state`` is stopped without engines; otherwise it is resident,
    draining, paused, releasing, released, restoring, or failed. Only resident
    admits requests. ``active_requests`` counts forwarded responses through
    their final chunk; ``drained`` is set exactly when that count is zero.
    Release requires paused engines. Restore requires released engines and
    leaves them paused. A failed transition blocks requests until restart.
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
        self.resources: InferenceResources | None = None
        self.service: InferenceService | None = None
        self.backend: InferenceBackend | None = None
        self.inference_url: str | None = None
        self.adapters: dict[str, Path] = {}
        self.instance_id = uuid4().hex
        self.engine_instance: str | None = None
        self.memory_state = "stopped"
        self.active_requests = 0
        self.drained = asyncio.Event()
        self.drained.set()

    def application(self) -> web.Application:
        app = web.Application(client_max_size=MAX_REQUEST_BYTES, middlewares=[self.report_errors])
        control = [
            ("PUT", "engines", self.start_engines),
            ("DELETE", "engines", self.stop_engines),
            ("GET", "health", self.health),
            ("GET", "runtime-load-ids", self.runtime_load_ids),
            ("POST", "initialize-version", self.initialize_version),
            ("POST", "{operation:pause|resume|recover|abort|release-memory|resume-memory}", self.receiver_operation),
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

    def start_stack(
        self,
        backend_name: str,
        config: Mapping[str, Any],
        num_gpus: int,
        shared_allocation: Mapping[str, Any] | None = None,
        engine_instance: str | None = None,
    ) -> None:
        self.stop_stack()
        resources: InferenceResources
        if shared_allocation is None:
            resources = LocalEngineResources(num_gpus)
        else:
            resources = BorrowedGpuResources.from_config(shared_allocation)
            if backend_name != "sglang" or len(resources.bundles.gpu_ids) != num_gpus:
                raise ValueError("shared inference requires SGLang and the exact borrowed GPU count")
            width = resources.node_width(config["gpus_per_engine"])
            if config["gpus_per_node"] != width:
                raise ValueError("shared inference topology disagrees with its live GPU placement")
            options = {name.replace("-", "_"): value for name, value in config.get("options", {}).items()}
            if options.get("enable_memory_saver") is not True or options.get("enable_weights_cpu_backup") is not True:
                raise ValueError("shared inference needs memory saver and incumbent weight backup")
        service: InferenceService | None = None
        try:
            resources.start()
            service = inference_service_for(backend_name, config)
            connection = service.start(resources)
            service.check_health()
            backend = service.backend(connection)
            if shared_allocation is not None and not isinstance(backend, MemoryManagedInferenceBackend):
                raise ValueError("shared inference backend does not control GPU memory")
            # The engines start paused; the control plane decides when they serve.
            backend.pause()
            inference_url = backend.inference_url().rstrip("/")
        except BaseException:
            if service is not None:
                with suppress(Exception):
                    service.close()
            with suppress(Exception):
                resources.close()
            raise
        self.resources, self.service, self.backend, self.inference_url = resources, service, backend, inference_url
        self.engine_instance = engine_instance or uuid4().hex
        self.memory_state = "paused" if shared_allocation is not None else "resident"
        logger.info("started %s engines on %d GPUs; serving %s", backend_name, num_gpus, inference_url)

    def stop_stack(self) -> None:
        service, resources = self.service, self.resources
        self.service = self.resources = self.backend = self.inference_url = None
        self.memory_state = "stopped"
        self.engine_instance = None
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

    def require_engine_instance(self, request: web.Request) -> None:
        """Bind shared control commands to one engine start, after taking the control lock."""
        expected = request.query.get("engine_instance")
        if (
            expected is not None or isinstance(self.resources, BorrowedGpuResources)
        ) and expected != self.engine_instance:
            raise web.HTTPConflict(text="shared inference engine instance changed; restart the deployment")

    def require_resident_memory(self) -> None:
        if isinstance(self.resources, BorrowedGpuResources) and self.memory_state not in ("resident", "paused"):
            raise web.HTTPConflict(text=f"shared inference memory is {self.memory_state}")

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
        shared_allocation, engine_instance = body.get("shared_allocation"), body.get("engine_instance")
        if shared_allocation is not None and (
            not isinstance(shared_allocation, Mapping) or not isinstance(engine_instance, str) or not engine_instance
        ):
            raise web.HTTPBadRequest(text="shared inference needs an allocation object and an engine instance ID")
        async with self.lock:
            if isinstance(self.resources, BorrowedGpuResources):
                raise web.HTTPConflict(text="stop the existing shared engines before replacing their allocation")
            await asyncio.to_thread(
                self.start_stack, backend_name, config, num_gpus, shared_allocation, engine_instance
            )
        return web.json_response({"ok": True, "instance": self.instance_id, "engine_instance": self.engine_instance})

    async def stop_engines(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        async with self.lock:
            if self.backend is not None:
                self.require_engine_instance(request)
            await asyncio.to_thread(self.stop_stack)
        return web.json_response({"ok": True})

    async def health(self, request: web.Request) -> web.StreamResponse:
        """Report this process and whether engines run; raise for failed engines, not for busy ones."""
        self.authorize(request, self.control_token)
        service = self.service
        if service is not None:
            await asyncio.to_thread(service.poll)
        return web.json_response(
            {
                "ok": True,
                "instance": self.instance_id,
                "engine_instance": self.engine_instance,
                "engines": self.backend is not None,
                "memory_state": self.memory_state,
                "capabilities": [SHARED_GPU_CONTROL_CAPABILITY],
            }
        )

    async def runtime_load_ids(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        async with self.lock:
            self.require_engine_instance(request)
            backend = self.running_backend()
            versions = await asyncio.to_thread(backend.runtime_load_ids)
        return web.json_response({"runtime_load_ids": [str(version) for version in versions]})

    async def initialize_version(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        body = await request.json()
        runtime_load_id = body.get("runtime_load_id") if isinstance(body, Mapping) else None
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise web.HTTPBadRequest(text="initialize-version needs a runtime_load_id")
        async with self.lock:
            self.require_engine_instance(request)
            backend = self.running_backend()
            self.require_resident_memory()
            await asyncio.to_thread(backend.initialize_version, runtime_load_id)
        return web.json_response({"ok": True})

    async def receiver_operation(self, request: web.Request) -> web.StreamResponse:
        """Serialize engine control and acknowledge every shared GPU transition."""
        self.authorize(request, self.control_token)
        operation = request.match_info["operation"]
        async with self.lock:
            self.require_engine_instance(request)
            backend = self.running_backend()
            shared = isinstance(self.resources, BorrowedGpuResources)
            result: int | None = None
            if operation in ("release-memory", "resume-memory"):
                if not shared or not isinstance(backend, MemoryManagedInferenceBackend):
                    raise web.HTTPConflict(text="this engine start does not share GPU memory")
                required = "paused" if operation == "release-memory" else "released"
                if self.memory_state != required:
                    raise web.HTTPConflict(text=f"{operation} requires memory state {required}")
                self.memory_state = "releasing" if operation == "release-memory" else "restoring"
                try:
                    if operation == "release-memory":
                        await asyncio.to_thread(backend.release_memory)
                        self.memory_state = "released"
                    else:
                        await asyncio.to_thread(backend.resume_memory)
                        self.memory_state = "paused"
                except BaseException:
                    self.memory_state = "failed"
                    raise
            else:
                self.require_resident_memory()
                operations = {
                    "pause": backend.pause,
                    "resume": backend.resume,
                    "recover": backend.recover,
                    "abort": backend.abort,
                }
                try:
                    if shared and operation == "pause":
                        self.memory_state = "draining"
                        await asyncio.wait_for(self.drained.wait(), timeout=300)
                    result = await asyncio.to_thread(operations[operation])
                    if shared and operation in ("pause", "resume"):
                        self.memory_state = "paused" if operation == "pause" else "resident"
                except BaseException:
                    if shared:
                        self.memory_state = "failed"
                    raise
        if operation == "recover":
            return web.json_response({"ok": True, "replaced": result})
        return web.json_response({"ok": True})

    async def load_weights(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        layout = request.query.get("layout")
        adapter_name = request.query.get("adapter_name")
        runtime_load_id = request.query.get("runtime_load_id")
        if layout not in WEIGHT_LAYOUTS or (adapter_name is None) != (layout == "full"):
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
                self.require_engine_instance(request)
                backend = self.running_backend()
                self.require_resident_memory()
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
        name = request.match_info["name"]
        async with self.lock:
            self.require_engine_instance(request)
            backend = self.running_backend()
            self.require_resident_memory()
            await asyncio.to_thread(backend.unload_adapter, name)
        directory = self.adapters.pop(name, None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        return web.json_response({"ok": True})

    # -- Inference requests

    async def forward(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.inference_token)
        if isinstance(self.resources, BorrowedGpuResources) and self.memory_state != "resident":
            raise web.HTTPServiceUnavailable(text=f"shared inference memory is {self.memory_state}")
        self.active_requests += 1
        self.drained.clear()
        try:
            return await self.forward_request(request)
        finally:
            self.active_requests -= 1
            if self.active_requests == 0:
                self.drained.set()

    async def forward_request(self, request: web.Request) -> web.StreamResponse:
        """Forward one admitted request, retaining its active count through the entire response stream."""
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
