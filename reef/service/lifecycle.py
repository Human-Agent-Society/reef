"""Managed deployment admission, private lifecycle control, and bounded request finalization.

Admission covers a whole HTTP request: body upload, routing selection, the
model call, and the response up to its local EOF, including streamed
receipts. Admission and the routing policy a request keeps are both decided
when its headers arrive, before any body is read. Runtime admission remains
responsible for individual model attempts and can therefore outlive neither
this boundary nor the response's local EOF.

The owner controls admission through ``/reef/lifecycle`` on a separate
application served only on a private Unix socket (``serve_managed``); the
public TCP application has no lifecycle route.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import secrets
import signal
import stat
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar, cast

from aiohttp import web
from aiohttp.typedefs import Middleware

from reef.service.deploy.routing.routing_control import MAX_SOCKET_PATH_BYTES
from reef.service.routing import RoutingAdapter, RoutingRequestError, is_router_config_hash

WriteResult = TypeVar("WriteResult")
logger = logging.getLogger(__name__)
INFERENCE_FINALIZATION = ContextVar("reef_inference_finalization", default=False)
RESPONSE_STARTED = web.RequestKey("reef_lifecycle_response_started", bool)
#: The routing policy an inference request keeps, captured when it is admitted and before its body is read.
ADMITTED_ROUTING = web.RequestKey("reef_admitted_routing", RoutingAdapter)
#: The reply to an ``open`` that reached its deadline before admission opened; the owner may send a new one.
OPEN_DEADLINE_PASSED = "lifecycle open deadline passed before admission opened"


@dataclass(frozen=True)
class LifecycleSettings:
    """Private child settings assembled by the deployment controller."""

    deployment_id: str
    generation: str
    control_token: str = field(repr=False)
    snapshot_version: str = ""
    body_timeout_seconds: float = 30.0
    finalization_timeout_seconds: float = 5.0
    #: The Semantic Router credential that grants only ``classify.invoke``; empty without routing.
    selection_token: str = field(default="", repr=False)
    #: The router runtime the owner verified and published; empty until its first publication.
    router_config_hash: str = ""
    #: The private Unix socket the control application listens on; empty for in-process tests.
    control_socket: str = ""

    def __post_init__(self) -> None:
        if not self.deployment_id or not self.generation or not self.control_token:
            raise ValueError("managed lifecycle requires deployment ID, generation, and a control token")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.body_timeout_seconds, self.finalization_timeout_seconds)
        ):
            raise ValueError("managed lifecycle timeouts must be positive")
        if self.router_config_hash and not is_router_config_hash(self.router_config_hash):
            raise ValueError("lifecycle router_config_hash must be 64 lowercase hexadecimal characters")
        if self.control_socket and (
            not Path(self.control_socket).is_absolute()
            or len(os.fsencode(self.control_socket)) > MAX_SOCKET_PATH_BYTES
        ):
            raise ValueError("lifecycle control_socket must be a short absolute path")

    @classmethod
    def load(cls, path: str | Path) -> LifecycleSettings:
        """Read the controller's private runtime file without logging its credential."""
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, encoding="utf-8") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                raise ValueError(
                    "lifecycle control file must be a private regular file owned by this user, not group or other users"
                )
            encoded = source.read(65537)
            if len(encoded) > 65536:
                raise ValueError("lifecycle control file exceeds its size limit")
            raw = json.loads(encoded)
        fields = {
            "deployment_id",
            "generation",
            "control_token",
            "snapshot_version",
            "body_timeout_seconds",
            "finalization_timeout_seconds",
            "selection_token",
            "router_config_hash",
            "control_socket",
        }
        if not isinstance(raw, dict) or raw.keys() - fields:
            raise ValueError("invalid lifecycle control settings")
        for name in ("deployment_id", "generation", "control_token", "snapshot_version"):
            if not isinstance(raw.get(name), str) or not raw[name]:
                raise ValueError(f"lifecycle {name} must be a nonempty string")
        for name in ("selection_token", "router_config_hash", "control_socket"):
            if name in raw and (not isinstance(raw[name], str) or not raw[name]):
                raise ValueError(f"lifecycle {name} must be a nonempty string")
        for name, default in (("body_timeout_seconds", 30.0), ("finalization_timeout_seconds", 5.0)):
            value = raw.get(name, default)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"lifecycle {name} must be a positive finite number")
        return cls(**raw)


@dataclass(frozen=True)
class WriteAdmission:
    sequence: int
    inference: bool


class PendingWrites:
    """Keep once-submitted writes alive and observable after request cancellation."""

    def __init__(self) -> None:
        self.tasks: dict[asyncio.Task[object], WriteAdmission] = {}
        self.sequence = 0
        self.idle = asyncio.Event()
        self.idle.set()
        self.failed_count = 0

    async def run(self, write: Awaitable[WriteResult]) -> WriteResult:
        async def execute() -> object:
            return await write

        task = asyncio.create_task(execute())
        self.sequence += 1
        self.tasks[task] = WriteAdmission(self.sequence, INFERENCE_FINALIZATION.get())
        self.idle.clear()
        task.add_done_callback(self.finished)
        # Cancelling a request cannot cancel its already submitted record write.
        return cast(WriteResult, await asyncio.shield(task))

    def finished(self, task: asyncio.Task[object]) -> None:
        self.tasks.pop(task, None)
        if task.cancelled() or task.exception() is not None:
            self.failed_count += 1
        if not self.tasks:
            self.idle.set()

    def count_before(self, sequence: int | None) -> int:
        if sequence is None:
            return len(self.tasks)
        return sum(admission.inference or admission.sequence <= sequence for admission in self.tasks.values())

    async def wait(self, timeout_seconds: float) -> bool:
        try:
            await asyncio.wait_for(self.idle.wait(), timeout_seconds)
        except TimeoutError:
            return False
        return True


class RoutingPublisher(ABC):
    """The service's routing policy: read when a request is admitted, replaced in one event-loop step."""

    @abstractmethod
    def current(self) -> RoutingAdapter | None:
        """The policy a request admitted now keeps until it completes."""

    @abstractmethod
    def publish(self, manifest_path: str, snapshot_version: str, router_config_hash: str) -> None:
        """Load and publish a compatible manifest whose digest is ``snapshot_version``, selected by the router
        runtime ``router_config_hash``.

        Raises:
            ValueError: The manifest is not that version or changes the target catalog.
        """

    @abstractmethod
    async def check_ready(self) -> None:
        """Check the published policy's providers and preview one message per Scenario under its runtime.

        Any selection valid for a Scenario passes; the message need not match the default rule.

        Raises:
            RoutingRequestError: No runtime is published, or a check failed.
        """


class RequestAcceptance:
    """An event-loop-local admission barrier, response lease, and control state."""

    def __init__(self, settings: LifecycleSettings, writes: PendingWrites) -> None:
        self.settings = settings
        self.writes = writes
        self.accepting = False
        self.stopping = False
        self.control_revision = -1
        self.last_action: tuple[str, bool, str, str, float] | None = None
        self.publisher: RoutingPublisher | None = None
        self.active: set[asyncio.Task[object]] = set()
        self.forced_requests = 0
        self.drain_sequence: int | None = None

    def check(self) -> None:
        if not self.accepting or self.stopping:
            raise web.HTTPServiceUnavailable(text="routing deployment is not accepting requests")

    def register(self) -> asyncio.Task[object]:
        # There is no await between the final check and insertion, or in control().
        # Both operations run on the same aiohttp event loop.
        self.check()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("request admission requires an asyncio task")
        self.active.add(task)
        return task

    def activate(
        self, manifest_path: str, snapshot_version: str, router_config_hash: str, expected_snapshot_version: str
    ) -> None:
        """Publish a new routing policy and its router runtime while closed and drained; admission stays closed.

        The check and the publication run without an await, so no request can
        be admitted between them on this event loop.
        """
        if self.publisher is None:
            raise web.HTTPConflict(text="this service has no routing publication")
        if self.accepting:
            raise web.HTTPConflict(text="routing policy activation requires closed admission")
        if self.active or self.writes.count_before(self.drain_sequence):
            raise web.HTTPConflict(text="routing policy activation requires drained requests and writes")
        if self.settings.snapshot_version != expected_snapshot_version:
            raise web.HTTPConflict(text="routing policy activation expected another current snapshot")
        try:
            self.publisher.publish(manifest_path, snapshot_version, router_config_hash)
        except (OSError, ValueError) as exc:
            raise web.HTTPConflict(text=f"routing policy rejected: {exc}") from exc
        self.settings = dataclasses.replace(
            self.settings, snapshot_version=snapshot_version, router_config_hash=router_config_hash
        )

    def cancel_active(self) -> None:
        for task in self.active:
            if not task.done() and not task.cancelling():
                self.forced_requests += 1
                task.cancel()

    def owner_lost(self) -> None:
        """Close permanently when the parent lease disappears."""
        self.accepting = False
        self.stopping = True
        self.cancel_active()

    def status(self) -> dict[str, object]:
        return {
            "deployment_id": self.settings.deployment_id,
            "generation": self.settings.generation,
            "snapshot_version": self.settings.snapshot_version,
            "router_config_hash": self.settings.router_config_hash,
            "accepting": self.accepting,
            "stopping": self.stopping,
            "control_revision": self.control_revision,
            "active_requests": len(self.active),
            "pending_writes": len(self.writes.tasks),
            "drain_pending_writes": self.writes.count_before(None if self.stopping else self.drain_sequence),
            "failed_writes": self.writes.failed_count,
            "forced_requests": self.forced_requests,
        }

    def readiness_deadline(self, body: Mapping[str, object]) -> float | None:
        """The deadline of ``body`` if it is a valid new ``open`` of a routed deployment, which needs a readiness
        check first; None for any other valid command.

        Raises:
            web.HTTPException: The command is invalid, stale, or for another generation.
        """
        action, revision, command = self.validated(body)
        if action == "open" and revision > self.control_revision and self.publisher is not None:
            return command[4]
        return None

    def validated(self, body: Mapping[str, object]) -> tuple[str, int, tuple[str, bool, str, str, float]]:
        """The action, revision and comparable command of a control body, after every check that changes nothing.

        Raises:
            web.HTTPException: The command is invalid, stale, or for another generation.
        """
        required = {"deployment_id", "generation", "control_revision", "action"}
        publication = {"manifest_path", "snapshot_version", "router_config_hash", "expected_snapshot_version"}
        allowed = required | {"cancel_active"}
        if body.get("action") == "activate":
            allowed |= publication
        elif body.get("action") == "open":
            allowed.add("open_deadline_monotonic")
        if not required <= body.keys() or body.keys() - allowed:
            raise web.HTTPBadRequest(text="invalid lifecycle control fields")
        if body["deployment_id"] != self.settings.deployment_id or body["generation"] != self.settings.generation:
            raise web.HTTPConflict(text="lifecycle deployment identity mismatch")
        revision = body["control_revision"]
        action = body["action"]
        cancel = body.get("cancel_active", False)
        if type(revision) is not int or revision < 0:
            raise web.HTTPBadRequest(text="control_revision must be a nonnegative integer")
        if (
            action not in ("open", "close", "stop", "activate")
            or type(cancel) is not bool
            or (action in ("open", "activate") and cancel)
        ):
            raise web.HTTPBadRequest(text="invalid lifecycle action")
        if action == "activate" and not (
            publication <= body.keys()
            and all(isinstance(body[name], str) and body[name] for name in publication)
            and is_router_config_hash(body["router_config_hash"])
        ):
            raise web.HTTPBadRequest(
                text="activation requires a manifest path, both snapshot versions and a router runtime hash"
            )
        if action == "open":
            # The owner's time.monotonic() value after which it no longer waits; the owner shares this host's clock.
            requested_deadline = body.get("open_deadline_monotonic")
            if (
                not isinstance(requested_deadline, (int, float))
                or isinstance(requested_deadline, bool)
                or not math.isfinite(requested_deadline)
                or requested_deadline <= 0
            ):
                raise web.HTTPBadRequest(text="open requires a finite positive open_deadline_monotonic")
            open_deadline = float(requested_deadline)
        else:
            open_deadline = 0.0
        command = (
            str(action),
            cancel,
            str(body.get("snapshot_version", "")),
            str(body.get("router_config_hash", "")),
            open_deadline,
        )
        if revision < self.control_revision or (revision == self.control_revision and command != self.last_action):
            raise web.HTTPConflict(text="stale or conflicting lifecycle control revision")
        return str(action), revision, command

    def control(self, body: Mapping[str, object]) -> dict[str, object]:
        action, revision, command = self.validated(body)
        cancel = command[1]
        if revision == self.control_revision:
            return self.status()
        if action in ("open", "activate") and self.stopping:
            raise web.HTTPConflict(text="stopping is terminal for this generation")
        if action == "open" and time.monotonic() >= command[4]:
            # No await separates this check from the change below, so a late readiness result cannot open.
            raise web.HTTPServiceUnavailable(text=OPEN_DEADLINE_PASSED)
        if action == "open" and self.drain_sequence is not None and self.writes.count_before(self.drain_sequence):
            raise web.HTTPConflict(text="previously admitted writes have not finalized")
        if action == "open" and self.publisher is not None and not self.settings.router_config_hash:
            raise web.HTTPConflict(text="routing has no verified router runtime; publish one before opening")
        if action == "activate":
            self.activate(
                str(body["manifest_path"]),
                str(body["snapshot_version"]),
                str(body["router_config_hash"]),
                str(body["expected_snapshot_version"]),
            )
        self.control_revision = revision
        self.last_action = command
        if action == "activate":
            return self.status()
        self.accepting = action == "open"
        if self.accepting:
            self.drain_sequence = None
        elif self.drain_sequence is None:
            # Retain this boundary through later forced-close commands. New
            # reports may proceed while closed without extending the old drain.
            self.drain_sequence = self.writes.sequence
        if action == "stop":
            self.stopping = True
        if cancel:
            self.cancel_active()
        return self.status()


REQUEST_ACCEPTANCE = web.AppKey("reef_request_acceptance", RequestAcceptance)


def create_acceptance_middleware(acceptance: RequestAcceptance) -> Middleware:
    @web.middleware
    async def manage_request(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        mutation = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        resource = request.match_info.route.resource
        inference = (
            request.method == "POST"
            and resource is not None
            and (resource.canonical.startswith("/v1/") or resource.canonical.endswith("/evaluation/v1/{route}"))
        )
        if mutation and acceptance.stopping:
            raise web.HTTPServiceUnavailable(text="routing deployment is stopping")
        if not inference:
            return await handler(request)
        # Admission and the routing policy are fixed together, before any body byte is awaited: the upload
        # counts toward the drain, and a policy published meanwhile cannot change what this request uses.
        # The body itself is read once, with its deadline, by the route (routes/payload.py).
        task = acceptance.register()
        routing = None if acceptance.publisher is None else acceptance.publisher.current()
        if routing is not None:
            request[ADMITTED_ROUTING] = routing
        scope = INFERENCE_FINALIZATION.set(True)
        try:
            try:
                response = await handler(request)
            except web.HTTPException as exc:
                if request.get(RESPONSE_STARTED, False):
                    # A failure mapped to a status after the stream's headers cannot become a second response.
                    if request.transport is not None:
                        request.transport.close()
                    raise
                response = web.Response(status=exc.status, reason=exc.reason, body=exc.body, headers=exc.headers)
                response.cookies.update(exc.cookies)
            except Exception:
                logger.exception("managed inference handler failed")
                if request.get(RESPONSE_STARTED, False):
                    # A stream with sent headers cannot be replaced by a 500.
                    if request.transport is not None:
                        request.transport.close()
                    raise
                response = web.Response(status=500, text="500: Internal Server Error")
            # Returning a buffered Response from a handler is not the HTTP EOF.
            # Do the normal aiohttp send here while the root lease is still held.
            await response.prepare(request)
            await response.write_eof()
            return response
        except (asyncio.CancelledError, ConnectionError):
            if request.transport is not None:
                request.transport.close()
            raise
        finally:
            acceptance.active.discard(task)
            INFERENCE_FINALIZATION.reset(scope)

    return manage_request


def register_admission(app: web.Application, acceptance: RequestAcceptance) -> None:
    """Attach managed admission to the public application; it has no lifecycle route."""

    async def response_started(request: web.Request, response: web.StreamResponse) -> None:
        request[RESPONSE_STARTED] = True

    app[REQUEST_ACCEPTANCE] = acceptance
    app.on_response_prepare.append(response_started)


def create_control_app(acceptance: RequestAcceptance) -> web.Application:
    """The private lifecycle application; ``serve_managed`` serves it only on the control socket.

    Only the control token opens it, and it has no other route. Commands are
    serialized: an ``open`` of a routed deployment first checks every provider
    and previews every Scenario under the published runtime, and no other
    command can run until it finishes. That check stops at the open's
    deadline, and an open whose deadline has passed never opens admission.
    """
    control_digest = hashlib.sha256(acceptance.settings.control_token.encode("utf-8")).digest()
    commands = asyncio.Lock()

    @web.middleware
    async def authenticate(
        request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
    ) -> web.StreamResponse:
        scheme, separator, credential = request.headers.get("Authorization", "").partition(" ")
        presented = hashlib.sha256(credential.encode("utf-8")).digest()
        if not separator or scheme.lower() != "bearer" or not secrets.compare_digest(presented, control_digest):
            raise web.HTTPUnauthorized(text="invalid lifecycle control token")
        return await handler(request)

    async def status(request: web.Request) -> web.Response:
        return web.json_response(acceptance.status())

    async def control(request: web.Request) -> web.Response:
        try:
            body = await asyncio.wait_for(request.json(), acceptance.settings.body_timeout_seconds)
        except TimeoutError as exc:
            raise web.HTTPRequestTimeout(text="lifecycle body deadline exceeded") from exc
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        async with commands:
            if acceptance.publisher is not None and (deadline := acceptance.readiness_deadline(body)) is not None:
                if not acceptance.settings.router_config_hash:
                    raise web.HTTPConflict(text="routing has no verified router runtime; publish one before opening")
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise web.HTTPServiceUnavailable(text=OPEN_DEADLINE_PASSED)
                try:
                    # A blocked event loop can run this timer late; control() checks the clock again.
                    async with asyncio.timeout(remaining_seconds):
                        await acceptance.publisher.check_ready()
                except TimeoutError as exc:
                    raise web.HTTPServiceUnavailable(text=OPEN_DEADLINE_PASSED) from exc
                except RoutingRequestError as exc:
                    raise web.HTTPServiceUnavailable(text=f"routing readiness failed: {exc.code}") from exc
            return web.json_response(acceptance.control(body))

    app = web.Application(middlewares=[authenticate])
    app.router.add_get("/reef/lifecycle", status)
    app.router.add_post("/reef/lifecycle", control)
    return app


async def serve_managed(
    public_app: web.Application,
    control_app: web.Application,
    *,
    host: str,
    port: int,
    control_socket: Path,
    shutdown_timeout_seconds: float,
) -> None:
    """Serve the control socket first, then the public listener, until SIGTERM or SIGINT; stop in reverse.

    The control application stays reachable while the public listener drains and shuts down. The public
    server cancels a request's handler when its client disconnects, so a selection in progress never
    dispatches a provider call for a client that is gone. Only the socket this call created is unlinked.
    """
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopped.set)
    directory = os.lstat(control_socket.parent)
    if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid() or directory.st_mode & 0o077:
        raise ValueError("control socket directory must be a private directory owned by this user")
    control_runner = web.AppRunner(control_app)
    await control_runner.setup()
    identity: os.stat_result | None = None
    try:
        await web.UnixSite(control_runner, str(control_socket)).start()
        os.chmod(control_socket, 0o600)
        identity = os.lstat(control_socket)
        public_runner = web.AppRunner(public_app, handler_cancellation=True, shutdown_timeout=shutdown_timeout_seconds)
        await public_runner.setup()
        try:
            await web.TCPSite(public_runner, host, port).start()
            await stopped.wait()
        finally:
            await public_runner.cleanup()
    finally:
        await control_runner.cleanup()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)
        if identity is not None:
            with contextlib.suppress(FileNotFoundError):
                current = os.lstat(control_socket)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    control_socket.unlink()


def configure_owner_watch(app: web.Application, acceptance: RequestAcceptance, owner_fd: int) -> None:
    """Exit a managed HTTP child if the parent's inherited pipe closes.

    Abrupt owner loss cannot promise completed records. The final exit deadline
    also bounds a Python executor thread stuck in a cancelled request's write.
    """
    if owner_fd < 0:
        raise ValueError("owner pipe descriptor must be nonnegative")

    async def watch_owner(app: web.Application) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        lost = asyncio.Event()
        os.set_blocking(owner_fd, False)

        def owner_readable() -> None:
            try:
                data = os.read(owner_fd, 1)
            except BlockingIOError:
                return
            if not data:
                loop.remove_reader(owner_fd)
                lost.set()

        async def stop_when_lost() -> None:
            await lost.wait()
            acceptance.owner_lost()
            # SIGTERM takes aiohttp's normal path; hard exit bounds teardown if
            # record storage or dispatcher cleanup cannot finish after owner loss.
            deadline = threading.Timer(acceptance.settings.finalization_timeout_seconds, os._exit, args=(1,))
            deadline.daemon = True
            deadline.start()
            os.kill(os.getpid(), signal.SIGTERM)

        loop.add_reader(owner_fd, owner_readable)
        task = asyncio.create_task(stop_when_lost())
        try:
            yield
        finally:
            loop.remove_reader(owner_fd)
            os.close(owner_fd)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app.cleanup_ctx.append(watch_owner)
