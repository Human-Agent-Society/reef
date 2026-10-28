"""Compatibility and startup ownership boundaries across the real orchestrator."""

from __future__ import annotations

import asyncio
import signal
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from tests.reef_service.routing.routing_fakes import (
    ControllerCase,
    FakeControl,
    FakeRuntime,
    opened,
    routing_dispatcher,
)

from reef.service.app import create_app
from reef.service.deploy import orchestrator
from reef.service.deploy.routing.containers import ContainerEvent
from reef.service.deploy.routing.deployment_controller import RoutingDeploymentController, RoutingDeploymentError
from reef.service.deploy.routing.reef_lifecycle_client import LifecycleStatus, RoutingControlError
from reef.service.deploy.routing.registry import DeploymentOwnership
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceConfig
from reef.service.lifecycle import REQUEST_ACCEPTANCE

pytestmark = pytest.mark.unit


def deployment_config(directory: Path, *, managed: bool = False) -> Path:
    reef: dict[str, object] = {"host": "127.0.0.1", "tokens": ["test-gateway"], "run_dir": str(directory / "stack")}
    if managed:
        # A routing manifest always means a managed routing deployment.
        reef["routing_manifest"] = str(directory / "manifest.yaml")
        reef["routing_deployment"] = {"deployment_id": "test-managed"}
    path = directory / "reef.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "reef": reef,
                "recipe": {"implementation": "recipe"},
                "inference": {"upstream_url": "http://127.0.0.1:8101", "upstream_model": "student"},
            }
        )
    )
    return path


def test_unrouted_deployment_does_not_construct_controller_or_touch_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions = []

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("a deployment without routing reached a managed Docker/controller boundary")

    class NativeStack:
        """An ordinary stack: checks no service gets a routing control file and logs start, block and shutdown."""

        def __init__(
            self,
            config: dict[str, object],
            services: Sequence[dict[str, object]],
            run_dir: Path,
            timeout: int,
            path: str | Path,
            *,
            source_root: Path | None,
        ) -> None:
            assert all("REEF_ROUTING_CONTROL_FILE" not in service.get("env", {}) for service in services)
            self.exit_code = 0

        def start(self) -> None:
            actions.append("start")

        def block(self) -> None:
            actions.append("block")

        def shutdown(self) -> None:
            actions.append("shutdown")

    monkeypatch.setattr(orchestrator, "DockerComposeRuntime", forbidden)
    monkeypatch.setattr(orchestrator, "RoutingDeploymentController", forbidden)
    monkeypatch.setattr(orchestrator, "_Stack", NativeStack)
    assert orchestrator._run_orchestrator(deployment_config(tmp_path)) == 0
    assert actions == ["start", "block", "shutdown"]


def test_unrouted_app_does_not_register_private_lifecycle(tmp_path: Path) -> None:
    dispatcher = routing_dispatcher(tmp_path)

    async def run() -> None:
        app = create_app(dispatcher, tokens="test-service")
        assert REQUEST_ACCEPTANCE not in app
        async with TestClient(TestServer(app)) as client:
            for method in (client.get, client.post):
                response = await method("/reef/lifecycle", headers={"Authorization": "Bearer test-service"})
                assert response.status == 404
            response = await client.get("/healthz")
            assert response.status == 200

    try:
        asyncio.run(run())
    finally:
        dispatcher.close()


def test_startup_stop_joins_slow_create_before_cleanup_and_second_signal_is_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions = []
    runtime = FakeRuntime(actions)
    policy = RoutingDeploymentSettings(
        deployment_id="test-managed", drain_timeout_seconds=0.03, finalization_timeout_seconds=0.03
    )
    controller = RoutingDeploymentController(
        ServiceConfig(recipe="base", routing_deployment=policy), runtime, registry_root=tmp_path / "registry"
    )
    controller.registry.acquire()
    controller.directory = tmp_path / "generation"
    controller.directory.mkdir()
    controller.ownership = controller.registry.prepare("generation-one", controller.directory)
    controller.snapshot_version = "snapshot-one"
    controller.control = FakeControl(actions)
    controller.control.deployment_id = "test-managed"
    created = threading.Event()
    released = threading.Event()
    original_start, original_remove = runtime.start, runtime.remove
    previous_term, previous_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)

    def slow_create(compose_file: Path, ownership: DeploymentOwnership, timeout_seconds: float) -> None:
        actions.append("create-entered")
        created.set()
        if not released.wait(5):
            raise TimeoutError("test did not release container creation")
        original_start(compose_file, ownership, timeout_seconds)
        actions.append("create-completed")

    def second_signal_then_remove(ownership: DeploymentOwnership, stop_timeout_seconds: float) -> None:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
        actions.append("second-signal")
        assert controller.force_stop_requested
        original_remove(ownership, stop_timeout_seconds)

    def signal_during_create() -> None:
        if not created.wait(5):
            return
        try:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            actions.append("first-signal")
            assert "remove" not in actions
        finally:
            released.set()

    class RoutingOnlyStack(orchestrator._Stack):
        """The real stack with its routing controller but no native services."""

        def __init__(
            self,
            config: dict[str, object],
            services: Sequence[dict[str, object]],
            run_dir: Path,
            timeout: int,
            path: str | Path,
            *,
            source_root: Path | None,
            routing_controller: RoutingDeploymentController,
        ) -> None:
            # Exercise real startup/rollback, while native child readiness is
            # represented by the fixture's private control endpoint.
            super().__init__(
                config, [], run_dir, timeout, path, source_root=source_root, routing_controller=routing_controller
            )

    monkeypatch.setattr(controller, "prepare", lambda: None)
    monkeypatch.setattr(runtime, "start", slow_create)
    monkeypatch.setattr(runtime, "remove", second_signal_then_remove)
    monkeypatch.setattr(orchestrator, "DockerComposeRuntime", lambda: runtime)
    monkeypatch.setattr(orchestrator, "RoutingDeploymentController", lambda settings, selected: controller)
    monkeypatch.setattr(orchestrator, "_Stack", RoutingOnlyStack)
    sender = threading.Thread(target=signal_during_create)
    sender.start()
    try:
        with pytest.raises(orchestrator.DeployStartupError, match="startup stopped"):
            orchestrator._run_orchestrator(deployment_config(tmp_path, managed=True))
    finally:
        released.set()
        sender.join(5)
        controller.registry.release()
    assert not sender.is_alive()
    assert actions.index("first-signal") < actions.index("create-completed") < actions.index("remove")
    assert actions.count("remove") == 1
    assert "control:open" not in actions
    assert actions[-1] == "control:stop"
    assert controller.closed and controller.cleanup_complete
    assert signal.getsignal(signal.SIGTERM) is previous_term
    assert signal.getsignal(signal.SIGINT) is previous_int


@pytest.mark.parametrize("open_race", [False, True])
def test_event_reconnect_waits_for_write_completion_and_rechecks_open_conflict(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, open_race: bool
) -> None:
    opened(case)
    case.controller.policy = replace(case.controller.policy, ready_timeout_seconds=1)
    previous_ids = case.controller.container_ids.copy()
    original_request = case.control.request
    release_at = time.monotonic() + 0.03
    attempts = []
    status_after_conflict = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> LifecycleStatus:
        if action is None:
            if attempts:
                status_after_conflict.append(True)
            case.control.pending_writes = int(not open_race and time.monotonic() < release_at)
        if action == "open":
            attempts.append(True)
            if open_race and len(attempts) == 1:
                # The control client reports a 409 as retryable: identity and status are checked again first.
                raise RoutingControlError("write submitted after status snapshot", status_code=409)
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    case.controller.tick()
    assert case.controller.state == "ready"
    assert case.controller.restart_count == 0
    assert case.controller.container_ids == previous_ids
    assert "start" not in case.actions and "remove" not in case.actions
    assert len(attempts) == (2 if open_race else 1)
    if open_race:
        assert status_after_conflict
    else:
        assert time.monotonic() >= release_at


def test_reconnect_open_conflict_never_retries_a_changed_generation(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened(case)
    case.controller.policy = replace(case.controller.policy, ready_timeout_seconds=1)
    original_request = case.control.request
    attempts = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> LifecycleStatus:
        if action == "open":
            attempts.append(True)
            case.control.generation = "different-generation"
            raise RoutingControlError("generation conflict", status_code=409)
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    with pytest.raises(RoutingDeploymentError, match="mismatch"):
        case.controller.tick()
    assert len(attempts) == 1
    assert "control:open" not in case.actions and "start" not in case.actions
