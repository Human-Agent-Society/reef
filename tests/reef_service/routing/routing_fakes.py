"""Shared fakes for routing tests: a fake Semantic Router preview listener and fake lifecycle collaborators.

``FakeSemanticRouter`` is not the Semantic Router: it applies the manifest's keyword rules to the last user
message and answers in the pinned router's reply shape. The controller fakes record what the deployment
controller asks of Docker, the REEF child and the router's management API. Nothing here starts Docker.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestServer

from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.service.deploy.routing.containers import (
    ContainerCommandError,
    ContainerEvent,
    ContainerEventStream,
    ContainerInventory,
    ContainerRuntime,
    ContainerState,
)
from reef.service.deploy.routing.deployment_controller import RoutingDeploymentController
from reef.service.deploy.routing.policy import RouterManagementClient, RouterPolicyError, RouterRuntime, document_etag
from reef.service.deploy.routing.reef_lifecycle_client import LifecycleStatus, ReefLifecycleClient, RoutingControlError
from reef.service.deploy.routing.registry import DeploymentOwnership
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceConfig
from reef.service.routing import DEFAULT_RULE, ENTRYPOINT_PREFIX, RoutingAdapter, RoutingSnapshot
from reef.service.routing_preview import EXPECTED_HASH_HEADER, READINESS_TEXT, RETURNED_HASH_HEADER, PreviewSettings
from reef.storage.sqlite import SQLiteScenarioStorage

ROUTER_HASH = "1" * 64
RUNTIME_HASH = "6" * 64
SELECTION_TOKEN = "selection-test-token"
SERVICE_TOKEN = "gateway-test-token"
#: An ``open_deadline_monotonic`` no test reaches; it is fixed, so a repeated open is the same command.
OPEN_DEADLINE = time.monotonic() + 24 * 3600


def routing_manifest(upstream_url: str, *, teacher_url: str | None = None) -> dict[str, object]:
    return {
        "schema_version": 2,
        "automatic_model": "auto",
        "scenarios": {
            "coding": {
                "default_target": "student",
                "targets": ["student", "teacher"],
                "rules": [{"name": "teacher_requested", "keywords": ["reef_teacher_required"], "target": "teacher"}],
            }
        },
        "targets": {
            "student": {
                "provider_model": "student-v1",
                "revision": "revision-1",
                "role": "student",
                "url": upstream_url,
            },
            "teacher": {
                "provider_model": "teacher-v2",
                "revision": "revision-2",
                "role": "teacher",
                "url": upstream_url if teacher_url is None else teacher_url,
            },
        },
        "deployment": {
            "semantic_router_image": "semantic-router@sha256:" + "a" * 64,
        },
    }


def routing_adapter(
    tmp_path: Path,
    upstream_url: str,
    *,
    teacher_url: str | None = None,
    timeout_s: float = 2.0,
    router_config_hash: str = ROUTER_HASH,
) -> RoutingAdapter:
    path = tmp_path / "routing.yaml"
    path.write_text(json.dumps(routing_manifest(upstream_url, teacher_url=teacher_url)), encoding="utf-8")
    return RoutingAdapter.from_manifest(path, router_config_hash=router_config_hash, environ={}, timeout_s=timeout_s)


def routing_headers(scenario: str = "coding") -> dict[str, str]:
    """What a client sends: its service token and its Scenario."""
    return {"Authorization": f"Bearer {SERVICE_TOKEN}", "x-reef-scenario": scenario}


async def provider_models(request: web.Request) -> web.Response:
    return web.json_response({"data": [{"id": "student-v1"}, {"id": "teacher-v2"}]})


def routing_dispatcher(tmp_path: Path) -> Dispatcher:
    return build_default_dispatcher(local_artifact_dir=tmp_path / "local", scenario_storage=SQLiteScenarioStorage())


def preview_reply(scenario: str, rule: str, target: str, runtime_hash: str) -> dict[str, object]:
    """A selection shaped like the pinned router's EvalResponse for a static, single-target decision."""
    signals = None if rule == DEFAULT_RULE else {"keywords": [rule]}
    return {
        "config_hash": runtime_hash,
        "original_text": "",
        "requested_model": ENTRYPOINT_PREFIX + scenario,
        "recipe": scenario,
        "decision_result": {
            "decision_name": rule,
            "algorithm": "static",
            "used_signals": signals,
            "matched_signals": signals,
            "unmatched_signals": None,
        },
        "recommended_models": [target],
        "selected_model": target,
        "selection_status": "selected",
        "selection_method": "single",
        "selection_reason": "selected by the live runtime selector",
        "selection_provenance": {
            "mode": "stateless",
            "config_hash": runtime_hash,
            "state_dependent": False,
            "sampled": False,
        },
        "routing_decision": rule,
        "metrics": None,
    }


def last_user_text(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") != "user":
            continue
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
            if text.strip():
                return text
    return ""


class FakeSemanticRouter:
    """A fake routing preview listener (not the Semantic Router) that records what Reef sends it.

    ``previews`` lists every request in arrival order, repeats included. ``hold``, when set, delays each
    answer until it is set, after setting ``arrived``; ``disconnected`` is set when Reef drops a held
    preview. Then the first failing check answers with an error body: a token other than the selection
    token is 401, an expected hash other than ``runtime_hash`` is 412, ``failure`` gives its status and
    code, and a model outside ``snapshot``'s entrypoints is 400. Otherwise it selects the first rule with a
    keyword in the last user text, ignoring case, or the default, and answers under ``runtime_hash``;
    ``reply_changes`` overwrites reply fields, so a test can send replies that Reef must refuse.
    """

    def __init__(self, snapshot: RoutingSnapshot, runtime_hash: str = ROUTER_HASH) -> None:
        self.snapshot = snapshot
        self.runtime_hash = runtime_hash
        #: How Reef reaches this router; set once it listens.
        self.settings: PreviewSettings | None = None
        self.previews: list[tuple[dict[str, str], dict[str, object]]] = []
        self.failure: tuple[int, str] | None = None
        self.reply_changes: dict[str, object] = {}
        self.hold: asyncio.Event | None = None
        self.arrived = asyncio.Event()
        self.disconnected = asyncio.Event()

    @property
    def selections(self) -> list[tuple[dict[str, str], dict[str, object]]]:
        """Previews of client requests, without the readiness previews Reef sends when it starts."""
        return [
            (headers, body)
            for headers, body in self.previews
            if body.get("messages") != [{"role": "user", "content": READINESS_TEXT}]
        ]

    async def preview(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.previews.append((dict(request.headers), body))
        if self.hold is not None:
            self.arrived.set()
            try:
                await self.hold.wait()
            except asyncio.CancelledError:
                # TestServer cancels a handler only after its client's connection is lost.
                self.disconnected.set()
                raise
        error = None
        if request.headers.get("Authorization") != f"Bearer {SELECTION_TOKEN}":
            error = (401, "UNAUTHORIZED")
        elif request.headers.get(EXPECTED_HASH_HEADER) != self.runtime_hash:
            error = (412, "CONFIG_HASH_MISMATCH")
        elif self.failure is not None:
            error = self.failure
        scenario = self.snapshot.scenarios.get(str(body.get("model", "")).removeprefix(ENTRYPOINT_PREFIX))
        if scenario is None:
            error = error or (400, "INVALID_ROUTING_MODEL")
        if error is not None or scenario is None:
            status, code = error
            return web.json_response({"error": {"code": code, "message": "fake router error"}}, status=status)
        text = last_user_text(body["messages"]).casefold()
        rule = next(
            (rule.name for rule in scenario.rules if any(word.casefold() in text for word in rule.keywords)),
            DEFAULT_RULE,
        )
        target = scenario.target_for_rule(rule)
        reply = {**preview_reply(scenario.name, rule, str(target), self.runtime_hash), **self.reply_changes}
        return web.json_response(reply, headers={RETURNED_HASH_HEADER: self.runtime_hash})


@contextlib.asynccontextmanager
async def running_router(
    snapshot: RoutingSnapshot, runtime_hash: str = ROUTER_HASH, *, timeout_seconds: int = 5
) -> AsyncIterator[FakeSemanticRouter]:
    """A fake Semantic Router listening on loopback; its ``settings`` reach it."""
    router = FakeSemanticRouter(snapshot, runtime_hash)
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app.router.add_post("/api/v1/routing/preview", router.preview)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    router.settings = PreviewSettings(f"http://127.0.0.1:{server.port}", SELECTION_TOKEN, timeout_seconds)
    try:
        yield router
    finally:
        await server.close()


class FakeEvents(ContainerEventStream):
    """Events a test queues in ``pending``: ``poll`` returns the oldest at once, or None, never waiting.

    ``close`` sets ``closed`` and then, with ``close_error``, raises ``ContainerCommandError``.
    """

    def __init__(self) -> None:
        self.pending: deque[ContainerEvent] = deque()
        self.closed = False
        self.close_error = False

    def poll(self, timeout_seconds: float = 0) -> ContainerEvent | None:
        return self.pending.popleft() if self.pending else None

    def close(self) -> None:
        self.closed = True
        if self.close_error:
            raise ContainerCommandError("event reader failed to stop")


class FakeRuntime(ContainerRuntime):
    """One owned router container per start, with no Docker.

    ``inventory`` is what ``inspect`` returns; tests may replace it with an empty, stopped or unhealthy one.
    Each ``start`` raises ``start_error`` if set, else makes container ``router-<n>`` with the next ``n``,
    healthy unless ``healthy_after_start`` is false; ``remove`` empties the inventory. ``ensure_available``,
    ``start``, ``inspect``, ``remove`` and ``events`` append ``available``, ``start``, ``inspect``,
    ``remove`` and ``subscribe`` to the shared ``actions`` log before they act. Each ``events`` stream is a
    new ``FakeEvents`` kept in ``streams``, already holding a disconnect when ``new_stream_disconnected``.
    """

    def __init__(self, actions: list[str]) -> None:
        self.compose_environment = {}
        self.actions = actions
        self.group_number = 0
        self.inventory = self.healthy_inventory()
        self.streams: list[FakeEvents] = []
        self.start_error: ContainerCommandError | None = None
        self.healthy_after_start = True
        self.new_stream_disconnected = False

    def healthy_inventory(self, health: str = "healthy") -> ContainerInventory:
        return ContainerInventory((ContainerState(f"router-{self.group_number}", "semantic-router", True, health),))

    def ensure_available(self) -> None:
        self.actions.append("available")

    def start(self, compose_file: Path, ownership: DeploymentOwnership, timeout_seconds: float) -> None:
        self.actions.append("start")
        if self.start_error is not None:
            raise self.start_error
        self.group_number += 1
        self.inventory = self.healthy_inventory("healthy" if self.healthy_after_start else "unhealthy")

    def inspect(self, ownership: DeploymentOwnership, timeout_seconds: float = 5) -> ContainerInventory:
        self.actions.append("inspect")
        return self.inventory

    def remove(self, ownership: DeploymentOwnership, stop_timeout_seconds: float) -> None:
        self.actions.append("remove")
        self.inventory = ContainerInventory()

    def events(self, ownership: DeploymentOwnership) -> ContainerEventStream:
        self.actions.append("subscribe")
        stream = FakeEvents()
        if self.new_stream_disconnected:
            stream.pending.append(ContainerEvent("", "stream_disconnected"))
        self.streams.append(stream)
        return stream


class FakeControl(ReefLifecycleClient):
    """The REEF child's lifecycle: closed and unpublished until the controller publishes a runtime.

    It keeps the client's ``request`` contract (``LifecycleStatus``, ``RoutingControlError``) but answers in
    memory, so an ``open`` never previews anything. With ``available`` False every request raises
    ``RoutingControlError``, as for a lost child, before it is logged. Otherwise ``actions`` and
    ``deadlines`` log it, and each command advances ``revision`` even if it then fails: an ``activate``
    raises ``activation_error`` if set, or a non-retryable 409 when its expected snapshot is not the current
    one; an ``open`` raises a retryable 409 before any runtime is published, then ``open_error`` if set.
    A command that does not fail becomes ``control_revision``, and ``open``, ``close`` and ``stop`` set
    ``accepting`` as the child does. As in the child, ``stopping`` closes admission, a forced close counts the
    requests it cancels in ``forced_requests``, and the drain counts every pending write.
    """

    def __init__(self, actions: list[str]) -> None:
        super().__init__(Path("/unused/control.sock"), "unit-routing", "generation-one", "unit-control-token")
        self.actions = actions
        self.active_requests = 0
        self.pending_writes = 0
        self.failed_writes = 0
        self.forced_requests = 0
        self.snapshot_version = "snapshot-one"
        self.router_config_hash = ""
        self.accepting = False
        self.stopping = False
        self.control_revision = -1
        self.available = True
        self.open_error: RoutingControlError | None = None
        self.activation_error: RoutingControlError | None = None
        self.publications: list[dict[str, str]] = []
        #: The action and phase deadline of every request that reached the child.
        self.deadlines: list[tuple[str | None, float | None]] = []

    def request(
        self,
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> LifecycleStatus:
        if not self.available:
            raise RoutingControlError("REEF child exited")
        self.actions.append("control:" + str(action))
        self.deadlines.append((action, deadline))
        if action is not None:
            self.revision += 1
        if action == "activate" and publication is not None:
            if self.activation_error is not None:
                raise self.activation_error
            if publication["expected_snapshot_version"] != self.snapshot_version:
                raise RoutingControlError("REEF lifecycle returned HTTP 409", retryable=False, status_code=409)
            self.publications.append(publication)
            self.snapshot_version = publication["snapshot_version"]
            self.router_config_hash = publication["router_config_hash"]
        if action == "open":
            if not self.router_config_hash:
                raise RoutingControlError("REEF lifecycle returned HTTP 409", status_code=409)
            if self.open_error is not None:
                raise self.open_error
        if action is not None:
            self.control_revision = self.revision
        if action in {"open", "close", "stop"}:
            self.accepting = action == "open"
        if cancel_active:
            self.forced_requests += self.active_requests
            self.active_requests = 0
        if action == "stop":
            self.stopping = True
        return self.status()

    def status(self) -> LifecycleStatus:
        return LifecycleStatus(
            deployment_id=self.deployment_id,
            generation=self.generation,
            snapshot_version=self.snapshot_version,
            router_config_hash=self.router_config_hash,
            accepting=self.accepting and not self.stopping,
            stopping=self.stopping,
            control_revision=self.control_revision,
            active_requests=self.active_requests,
            pending_writes=self.pending_writes,
            drain_pending_writes=self.pending_writes,
            failed_writes=self.failed_writes,
            forced_requests=self.forced_requests,
        )


class FakeManagement(RouterManagementClient):
    """The router's management API serving one stored document and its runtime.

    Only three reads are faked, in memory. ``config_etag`` returns ``etag``, and ``runtime_hashes`` reads the two
    runtime hashes and ``activation_status`` the way the real client reads a reply that holds them; with
    ``available`` False both raise ``RouterPolicyError``.
    ``ready`` logs ``router:ready``, requests the stop of ``stop_on_ready`` if set, and returns
    ``available and is_ready``, so an unavailable router is not ready rather than an error. All three accept
    and ignore ``deadline``. The inherited ``call``, ``plan`` and ``replace`` are the real client's, aimed at
    port 1.
    """

    def __init__(self, actions: list[str], document: Path) -> None:
        super().__init__(1, "unit-management-token")
        self.actions = actions
        self.etag = document_etag(document.read_bytes())
        self.active = RUNTIME_HASH
        self.generated = RUNTIME_HASH
        self.activation_status = "active"
        self.is_ready = True
        self.available = True
        self.stop_on_ready: RoutingDeploymentController | None = None

    def config_etag(self, *, deadline: float | None = None) -> str:
        if not self.available:
            raise RouterPolicyError("router unavailable")
        return self.etag

    def runtime_hashes(self, *, deadline: float | None = None) -> RouterRuntime:
        if not self.available:
            raise RouterPolicyError("router unavailable")
        return RouterRuntime.from_reply(
            {
                "active_runtime_hash": self.active,
                "generated_runtime_hash": self.generated,
                "activation_status": self.activation_status,
            }
        )

    def ready(self, *, deadline: float | None = None) -> bool:
        self.actions.append("router:ready")
        if self.stop_on_ready is not None:
            self.stop_on_ready.request_stop()
        return self.available and self.is_ready


@dataclass
class ControllerCase:
    """What ``controller_case`` assembles: ``controller`` uses ``runtime``, ``control`` and ``management``.

    ``actions`` is the one ordered log, repeats included, that those three fakes append their calls to;
    the controller writes its own lifecycle events to its event file, not here.
    """

    controller: RoutingDeploymentController
    runtime: FakeRuntime
    control: FakeControl
    management: FakeManagement
    actions: list[str]


@contextlib.contextmanager
def controller_case(tmp_path: Path) -> Iterator[ControllerCase]:
    """A controller past preparation, with a healthy fake router container, before its first opening."""
    actions: list[str] = []
    runtime = FakeRuntime(actions)
    policy = RoutingDeploymentSettings(
        deployment_id="unit-routing",
        restart_policy="bounded",
        max_restarts=2,
        restart_backoff_seconds=(0,),
        ready_timeout_seconds=1,
        drain_timeout_seconds=0.03,
        finalization_timeout_seconds=0.03,
        container_stop_timeout_seconds=0.03,
    )
    controller = RoutingDeploymentController(
        ServiceConfig(recipe="base", routing_deployment=policy), runtime, registry_root=tmp_path / "registry"
    )
    controller.registry.acquire()
    controller.directory = tmp_path / "generation"
    controller.directory.mkdir()
    (controller.directory / "router.yaml").write_text("version: v0.3\n", encoding="utf-8")
    (controller.directory / "manifest.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    controller.ownership = controller.registry.prepare("generation-one", controller.directory)
    controller.snapshot_version = "snapshot-one"
    control = FakeControl(actions)
    management = FakeManagement(actions, controller.directory / "router.yaml")
    controller.control = control
    controller.management = management
    controller.event_stream = runtime.events(controller.ownership)
    controller.inspect_ready(runtime.inventory)
    controller.ready_deadline = time.monotonic() + policy.ready_timeout_seconds
    try:
        yield ControllerCase(controller, runtime, control, management, actions)
    finally:
        controller.registry.release()


def opened(case: ControllerCase) -> ControllerCase:
    """Bring the case through its first verified opening."""
    case.controller.wait_ready()
    case.actions.clear()
    case.controller.ready_deadline = time.monotonic() + case.controller.policy.ready_timeout_seconds
    return case
