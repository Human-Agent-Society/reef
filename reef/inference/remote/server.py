"""The remote inference component: this host's engines, their control operations and inference requests over HTTP."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import shutil
import tarfile
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from reef.core.weight_layout import WEIGHT_LAYOUTS, WeightFiles
from reef.inference.deployment import InferenceResources, InferenceService, inference_service_for
from reef.inference.http import HOP_BY_HOP_HEADERS, SERVED_MODEL_HEADER
from reef.inference.interfaces import InferenceBackend
from reef.inference.remote.config import CONTROL_PATH, OWNER_SEQUENCE_HEADER, require_no_external_kv_cache
from reef.inference.remote.pins import RequestPins, generation_stopped

logger = logging.getLogger(__name__)

#: Request headers that describe the connection to this component, not the request to the engines.
FORWARD_EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {"host", "authorization"}
#: Largest inference request body; weight uploads stream and are not bound by it.
MAX_REQUEST_BYTES = 64 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
#: Longest wait, in seconds, of a pinned or owner-fenced control call for an earlier one.
CONTROL_WAIT_S = 30.0


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
    ``owner_sequence`` is the greatest positive serving-owner sequence accepted by this process, initially zero.
    Older owners cannot mutate its engines; the mark survives engine restarts but not a component restart.
    ``pins`` tracks acknowledged bindings and admitted requests on this event loop. ``session`` forwards requests
    while pinning is off. ``pinned_session`` forwards every admitted pinned request, on connections that close after
    their response (``force_close``), so no upstream connection outlives its request's pin; ``forward`` keeps reading
    an admitted upstream response after its client disconnects. Both sessions are None before HTTP startup, created by
    ``open_session``, and closed when the component shuts down.

    Routes under ``CONTROL_PATH`` need ``Authorization: Bearer <control_token>``;
    every other path is forwarded to the engines and needs the inference token.
    ``PUT engines`` starts ``engine`` with the input that the control plane
    sends, replacing running engines, and refuses any other backend name. The
    other control routes call the engines' ``InferenceBackend``, one at a time.
    ``instance_id`` names this process: ``PUT engines`` and ``GET health``
    report it, so the control plane sees a restarted component, which runs no
    engines. ``GET health`` reports whether engines run and never waits behind
    a control call: it observes their failures through ``InferenceService.poll``.
    Full weight files are removed after the load, and an adapter's files when
    it is unloaded or replaced.
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
        self.pinned_session: ClientSession | None = None
        self.resources: InferenceResources | None = None
        self.service: InferenceService | None = None
        self.backend: InferenceBackend | None = None
        self.inference_url: str | None = None
        self.adapters: dict[str, Path] = {}
        self.instance_id = uuid4().hex
        self.owner_sequence = 0
        self.pins = RequestPins()

    def check_owner(self, request: web.Request, *, required: bool = False) -> None:
        """Fence delayed calls before waits and again before mutation, using the durable owner take sequence.

        A new high mark immediately closes admission. Its owner must restart/reload before publishing routes.
        The mark lives with the engine stack; a component restart loses both. Resetting the coordination store
        requires a component restart because its sequence may then be below this high mark.
        """
        value = request.headers.get(OWNER_SEQUENCE_HEADER)
        if value is None and not (required or self.pins.enabled):
            return
        if value is None or not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise web.HTTPPreconditionFailed(text="a serving-owner sequence is required")
        sequence = int(value)
        if sequence < self.owner_sequence:
            raise web.HTTPPreconditionFailed(text="the inference owner has been superseded")
        if sequence > self.owner_sequence:
            self.owner_sequence = sequence
            if self.pins.enabled:
                self.pins.quarantine()

    async def control_call[**P, T](self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Keep a blocking control operation inside its caller's lock even if the HTTP handler is cancelled."""
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
                if self.pins.enabled:
                    self.pins.quarantine()
            except Exception:
                break
        if cancelled:
            # Retrieve a thread exception without allowing another mutation to overtake the thread.
            with suppress(Exception):
                task.result()
            raise asyncio.CancelledError
        return task.result()

    @asynccontextmanager
    async def control_lock(self, request: web.Request) -> AsyncIterator[None]:
        """Bound waiting callers while an earlier mutation retains exclusive ownership of its worker thread.

        A timed-out waiter cannot kill or overtake that mutation. Health remains readable; an indefinitely
        wedged native call requires operator recovery of the component, not an unsafe overlapping restart.
        """
        try:
            if self.pins.enabled or OWNER_SEQUENCE_HEADER in request.headers:
                await asyncio.wait_for(self.lock.acquire(), min(CONTROL_WAIT_S, self.pins.drain_timeout_s))
            else:
                await self.lock.acquire()
        except TimeoutError as exc:
            raise web.HTTPServiceUnavailable(
                text="an earlier control operation is still running", headers={"Retry-After": "1"}
            ) from exc
        try:
            self.check_owner(request)
            yield
        finally:
            self.lock.release()

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
        self.pinned_session = ClientSession(
            timeout=ClientTimeout(total=None), connector=TCPConnector(force_close=True), auto_decompress=False
        )

    async def close_application(self, app: web.Application) -> None:
        await self.pins.interrupt()
        async with self.lock:
            await self.control_call(self.stop_stack)
        if self.session is not None:
            await self.session.close()
        if self.pinned_session is not None:
            await self.pinned_session.close()

    def start_stack(self, backend_name: str, config: Mapping[str, Any], num_gpus: int) -> None:
        self.stop_stack()
        resources = InferenceResources(num_gpus)
        service: InferenceService | None = None
        try:
            resources.start()
            service = inference_service_for(backend_name, config)
            connection = service.start(resources)
            service.check_health()
            backend = service.backend(connection)
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
            if request.transport is None or request.transport.is_closing():
                raise
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
        self.check_owner(request)
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
        pinning = body.get("request_pinning", False)
        if not isinstance(pinning, bool):
            raise web.HTTPBadRequest(text="request_pinning must be a bool")
        if self.pins.enabled and not pinning:
            raise web.HTTPConflict(text="restart the component before disabling request pinning")
        layout = body.get("weight_layout", "full")
        drain_timeout_s = body.get("drain_timeout_s", 300.0)
        if pinning:
            if backend_name not in ("sglang", "vllm") or layout not in WEIGHT_LAYOUTS:
                raise web.HTTPBadRequest(text="request pinning requires a built-in engine and a weight layout")
            if (
                isinstance(drain_timeout_s, bool)
                or not isinstance(drain_timeout_s, (int, float))
                or not math.isfinite(drain_timeout_s)
                or drain_timeout_s <= 0
            ):
                raise web.HTTPBadRequest(text="drain_timeout_s must be positive and finite")
            # Every pause follows a fresh start or a completed drain, so a retracting pause finds no request to
            # retract: it only clears the engine's own prefix cache, and its failure quarantines the engines. The
            # native configuration decides whether that cache is safe, and no pause clears KV outside the engines.
            try:
                require_no_external_kv_cache(backend_name, config)
            except ValueError as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc
        self.check_owner(request, required=pinning)
        async with self.control_lock(request):
            if self.pins.enabled and not pinning:
                # A pinned start can finish while this request waits for the lock.
                raise web.HTTPConflict(text="restart the component before disabling request pinning")
            await self.pins.interrupt()
            self.check_owner(request, required=pinning)
            if pinning:
                # Only a pinned start checked these values. Pinning stays on until the component restarts.
                self.pins.enabled = True
                self.pins.layout = layout
                self.pins.drain_timeout_s = drain_timeout_s
            try:
                await self.control_call(self.start_stack, backend_name, config, num_gpus)
                self.check_owner(request)
                self.pins.reset()
            except BaseException:
                self.pins.quarantine()
                raise
        return web.json_response({"ok": True, "instance": self.instance_id, "request_pinning": pinning})

    async def stop_engines(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        self.check_owner(request)
        async with self.control_lock(request):
            await self.pins.interrupt()
            self.check_owner(request)
            try:
                await self.control_call(self.stop_stack)
                self.check_owner(request)
                self.pins.reset()
            except BaseException:
                self.pins.quarantine()
                raise
        return web.json_response({"ok": True})

    async def health(self, request: web.Request) -> web.StreamResponse:
        """Report this process and whether engines run; raise for failed engines, not for busy ones."""
        self.authorize(request, self.control_token)
        self.check_owner(request)
        service = self.service
        if service is not None:
            try:
                await asyncio.to_thread(service.poll)
            except Exception:
                if not self.pins.enabled:
                    raise
                self.pins.quarantine()
        return web.json_response(
            {
                "ok": True,
                "instance": self.instance_id,
                "engines": self.backend is not None,
                **({"request_pinning": True, "binding_uncertain": self.pins.uncertain} if self.pins.enabled else {}),
            }
        )

    async def runtime_load_ids(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        self.check_owner(request)
        async with self.control_lock(request):
            versions = await self.control_call(self.running_backend().runtime_load_ids)
        return web.json_response({"runtime_load_ids": [str(version) for version in versions]})

    async def initialize_version(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        self.check_owner(request)
        body = await request.json()
        runtime_load_id = body.get("runtime_load_id") if isinstance(body, Mapping) else None
        if not isinstance(runtime_load_id, str) or not runtime_load_id:
            raise web.HTTPBadRequest(text="initialize-version needs a runtime_load_id")
        async with self.control_lock(request):
            if self.pins.enabled and self.pins.loaded:
                raise web.HTTPConflict(text="a loaded binding cannot be relabelled")
            try:
                await self.control_call(self.running_backend().initialize_version, runtime_load_id)
                self.check_owner(request)
                if self.pins.enabled and self.pins.layout == "adapter":
                    self.pins.base_load_id = runtime_load_id
            except BaseException:
                if self.pins.enabled:
                    self.pins.quarantine()
                raise
        return web.json_response({"ok": True})

    async def receiver_operation(self, request: web.Request) -> web.StreamResponse:
        """Pause, resume, recover or abort the engines, as the route names."""
        self.authorize(request, self.control_token)
        self.check_owner(request)
        operation = request.match_info["operation"]
        async with self.control_lock(request):
            backend = self.running_backend()
            try:
                if self.pins.enabled:
                    if operation == "abort":
                        await self.pins.interrupt()
                    elif operation in ("pause", "recover"):
                        await self.pins.drain()
                    elif self.pins.uncertain:
                        raise web.HTTPServiceUnavailable(text="engine recovery is required before resume")
                    if operation == "recover" and self.pins.uncertain:
                        raise web.HTTPServiceUnavailable(
                            text="replace the engine stack to restore an uncertain binding"
                        )
                    if operation == "resume" and self.pins.layout == "full":
                        versions = await self.control_call(backend.runtime_load_ids)
                        if self.pins.full_load_id is None or set(versions) != {self.pins.full_load_id}:
                            raise web.HTTPServiceUnavailable(text="no verified full-weight binding to resume")
                self.check_owner(request)
                if self.pins.enabled and operation == "pause" and self.pins.uncertain:
                    # Recovery replaces the whole stack; pausing a dead engine again would prevent reaching it.
                    result = None
                elif self.pins.enabled and operation == "abort":
                    await self.control_call(self.stop_stack)
                    result = None
                else:
                    operations = {
                        "pause": backend.pause,
                        "resume": backend.resume,
                        "recover": backend.recover,
                        "abort": backend.abort,
                    }
                    result = await self.control_call(operations[operation])
                self.check_owner(request)
                if self.pins.enabled:
                    if operation == "abort" or (operation == "recover" and result):
                        self.pins.reset()
                    if operation == "recover":
                        self.inference_url = (await self.control_call(backend.inference_url)).rstrip("/")
                    if operation == "resume":
                        self.pins.accepting = True
            except BaseException:
                if self.pins.enabled:
                    self.pins.quarantine()
                raise
        if operation == "recover":
            return web.json_response({"ok": True, "replaced": result})
        return web.json_response({"ok": True})

    async def load_weights(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.control_token)
        self.check_owner(request)
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
            async with self.control_lock(request):
                backend = self.running_backend()
                if self.pins.enabled:
                    if self.pins.uncertain or self.pins.accepting or self.pins.active:
                        raise web.HTTPConflict(
                            text="pause and drain before loading weights; recover uncertain engines"
                        )
                    if not runtime_load_id or layout != self.pins.layout:
                        raise web.HTTPBadRequest(text="a pinned load needs its runtime ID and the stack's layout")
                    if adapter_name is None:
                        self.pins.full_load_id = None
                    elif adapter_name in self.pins.adapters:
                        raise web.HTTPConflict(text="an adapter name is immutable; load a fresh name")
                try:
                    await self.control_call(
                        backend.load_weight_files, files, adapter_name=adapter_name, runtime_load_id=runtime_load_id
                    )
                    self.check_owner(request)
                    if self.pins.enabled:
                        versions = await self.control_call(backend.runtime_load_ids)
                        self.check_owner(request)
                        if set(versions) != {runtime_load_id}:
                            raise RuntimeError("the engines did not acknowledge the loaded revision")
                        self.pins.loaded = True
                        if adapter_name is None:
                            self.pins.full_load_id = runtime_load_id
                        else:
                            self.pins.adapters.add(adapter_name)
                    loaded = True
                except BaseException:
                    if self.pins.enabled:
                        self.pins.quarantine()
                    raise
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
        self.check_owner(request)
        name = request.match_info["name"]
        async with self.control_lock(request):
            backend = self.running_backend()
            resume = self.pins.enabled and self.pins.accepting
            if self.pins.enabled:
                await self.pins.drain()
                self.check_owner(request)
                if self.pins.uncertain:
                    raise web.HTTPServiceUnavailable(text="recover uncertain engines before unloading adapters")
                self.pins.adapters.discard(name)
            try:
                if self.pins.enabled:
                    await self.control_call(backend.pause)
                self.check_owner(request)
                await self.control_call(backend.unload_adapter, name)
                self.check_owner(request)
                if resume:
                    await self.control_call(backend.resume)
                    self.check_owner(request)
                    self.pins.accepting = True
            except BaseException:
                if self.pins.enabled:
                    self.pins.quarantine()
                raise
        directory = self.adapters.pop(name, None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
        return web.json_response({"ok": True})

    # -- Inference requests

    async def forward(self, request: web.Request) -> web.StreamResponse:
        self.authorize(request, self.inference_token)
        body = await request.read()
        pin = None
        if self.pins.enabled:
            path = "/generate" if self.engine == "sglang" else "/inference/v1/generate"
            if request.method != "POST" or request.path != path or request.query_string:
                raise web.HTTPNotFound(text="not a pinned generation route")
            try:
                payload = json.loads(body)
            except ValueError as exc:
                raise web.HTTPBadRequest(text="generation needs a JSON object") from exc
            if not isinstance(payload, Mapping):
                raise web.HTTPBadRequest(text="generation needs a JSON object")
            pin = self.pins.admit(request.headers.get(SERVED_MODEL_HEADER, ""), payload)
        upstream_url = self.inference_url
        session = self.pinned_session if pin is not None else self.session
        if upstream_url is None or session is None:
            if pin is not None:
                self.pins.release(pin)
            raise web.HTTPServiceUnavailable(text="no engines are running")
        headers = {
            name: value for name, value in request.headers.items() if name.lower() not in FORWARD_EXCLUDED_HEADERS
        }
        if pin is not None:
            headers["Accept-Encoding"] = "identity"
        response: web.StreamResponse | None = None
        captured = bytearray()
        connected = True
        try:
            async with session.request(
                request.method, upstream_url + request.rel_url.path_qs, headers=headers, data=body
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        name: value
                        for name, value in upstream.headers.items()
                        if name.lower() not in HOP_BY_HOP_HEADERS
                    },
                )
                try:
                    await response.prepare(request)
                except ConnectionResetError:
                    if pin is None:
                        raise
                    connected = False
                async for chunk in upstream.content.iter_any():
                    if pin is not None:
                        captured.extend(chunk)
                    if connected:
                        try:
                            await response.write(chunk)
                        except ConnectionResetError:
                            if pin is None:
                                raise
                            # The engine may still generate: retain its pin and drain the upstream response.
                            connected = False
                if pin is not None and not 400 <= upstream.status < 500:
                    if upstream.status >= 500:
                        raise ValueError("the upstream failed without establishing whether generation stopped")
                    if not generation_stopped(bytes(captured), upstream.headers.get("Content-Type", ""), self.engine):
                        raise ValueError("the upstream ended without a verified native generation")
                if connected:
                    try:
                        await response.write_eof()
                    except ConnectionResetError:
                        # Completion was verified above; losing the client now cannot invalidate the weights.
                        if pin is None:
                            raise
            return response
        except BaseException:
            if pin is not None:
                self.pins.quarantine()
            if response is not None and response.prepared and request.transport is not None:
                # A second HTTP response or a clean EOF would turn an upstream failure into apparent success.
                request.transport.abort()
            raise
        finally:
            if pin is not None:
                self.pins.release(pin)


def unpack_archive(archive: Path, directory: Path) -> None:
    """Unpack an uploaded tar; the ``data`` filter refuses absolute paths and links that leave ``directory``."""
    directory.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")
