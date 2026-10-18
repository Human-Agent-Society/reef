"""State-machine acceptance tests independent of Docker, the router and model services.

The runtime, the REEF child's control client and the router's management API
are in-process fakes that record the controller's calls.
"""

from __future__ import annotations

import socket
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.reef_service.routing_fakes import RUNTIME_HASH, ControllerCase, FakeRuntime, controller_case, opened

from reef.service.deploy.routing_config import RoutingDeploymentSettings
from reef.service.deploy.routing_containers import ContainerCommandError, ContainerEvent
from reef.service.deploy.routing_control import RoutingControlError
from reef.service.deploy.routing_controller import (
    RoutingDeploymentController,
    RoutingDeploymentError,
    RoutingReadinessTimeout,
)
from reef.service.deploy.service_config import ServiceConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def case(tmp_path: Path) -> Iterator[ControllerCase]:
    with controller_case(tmp_path) as built:
        yield built


def test_first_opening_publishes_the_verified_runtime_while_closed(case: ControllerCase) -> None:
    case.controller.wait_ready()
    assert case.control.publications == [
        {
            "manifest_path": str(case.controller.directory / "manifest.yaml"),
            "snapshot_version": "snapshot-one",
            "router_config_hash": RUNTIME_HASH,
            "expected_snapshot_version": "snapshot-one",
        }
    ]
    assert case.actions.index("control:activate") < case.actions.index("control:open")
    assert case.controller.state == "ready"


@pytest.mark.parametrize(
    "change",
    [
        {"activation_status": "pending"},
        {"generated": "7" * 64},
        {"is_ready": False},
        {"available": False},
    ],
)
def test_no_opening_until_the_router_runtime_is_active_ready_and_consistent(
    case: ControllerCase, change: dict[str, object]
) -> None:
    for name, value in change.items():
        setattr(case.management, name, value)
    with pytest.raises(RoutingReadinessTimeout):
        case.controller.wait_ready()
    assert "control:activate" not in case.actions and "control:open" not in case.actions


def test_a_router_running_another_document_is_fatal(case: ControllerCase) -> None:
    case.management.etag = '"' + "0" * 64 + '"'
    with pytest.raises(RoutingDeploymentError, match="configuration"):
        case.controller.wait_ready()
    assert "control:open" not in case.actions


def test_a_failed_readiness_preview_is_retried_until_the_deadline(case: ControllerCase) -> None:
    case.control.open_error = RoutingControlError("REEF lifecycle returned HTTP 503", status_code=503)
    with pytest.raises(RoutingReadinessTimeout):
        case.controller.wait_ready()
    assert case.actions.count("control:open") >= 1 and case.controller.state != "ready"
    case.control.open_error = None
    case.controller.ready_deadline = time.monotonic() + 1
    case.controller.wait_ready()
    assert case.controller.state == "ready"


def test_bounded_recovery_closes_before_remove_and_opens_new_container_ids(case: ControllerCase) -> None:
    opened(case)
    previous_ids = case.controller.container_ids.copy()
    case.runtime.streams[-1].pending.append(ContainerEvent("router-0", "die"))
    case.controller.tick()
    assert case.controller.state == "ready"
    assert case.controller.restart_count == 1
    assert previous_ids.isdisjoint(case.controller.container_ids)
    assert case.actions.index("control:close") < case.actions.index("remove") < case.actions.index("start")
    assert case.actions.index("start") < case.actions.index("control:open")


def test_recovery_publishes_a_changed_runtime_hash_before_reopening(case: ControllerCase) -> None:
    opened(case)
    # The restarted router reports a new runtime for the same committed document and policy.
    committed = case.controller.policy_store.load()["committed"]
    case.management.active = case.management.generated = "8" * 64
    case.controller.policy_store.write({"committed": {**committed, "runtime_hash": "8" * 64}})
    case.controller.recover("router-failure")
    assert case.control.publications[-1]["router_config_hash"] == "8" * 64
    assert case.control.publications[-1]["snapshot_version"] == "snapshot-one"
    assert case.actions.index("control:activate") < case.actions.index("control:open")


def test_periodic_checks_detect_runtime_drift_without_selection_previews(case: ControllerCase) -> None:
    opened(case)
    # Another writer's consistent runtime: active and generated agree, but not with the committed pair.
    case.management.active = case.management.generated = "9" * 64
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="router does not run the committed policy"):
        case.controller.tick()
    # Admission closed before the container was replaced, and it never reopened under the foreign runtime.
    assert case.actions.index("control:close") < case.actions.index("remove")
    assert "control:open" not in case.actions and "control:activate" not in case.actions


def test_a_pending_runtime_is_not_drift(case: ControllerCase) -> None:
    opened(case)
    case.management.active, case.management.activation_status = "9" * 64, "pending"
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="budget"):
        case.controller.tick()
    # Recovery waited for an active runtime and never opened under a pending one.
    assert "control:open" not in case.actions


def test_repeated_readiness_failures_count_as_router_failure(case: ControllerCase) -> None:
    opened(case)
    case.management.is_ready = False
    for _ in range(2):
        case.controller.next_probe = 0
        case.controller.tick()
    assert "remove" not in case.actions
    case.controller.next_probe = 0
    # The third failure starts recovery; a router that stays unready exhausts the restart budget.
    with pytest.raises(RoutingDeploymentError, match="budget"):
        case.controller.tick()
    assert case.actions.index("control:close") < case.actions.index("remove")
    assert "control:open" not in case.actions


def test_late_old_container_event_does_not_restart_replacement(case: ControllerCase) -> None:
    opened(case)
    case.controller.recover("first-failure")
    case.runtime.streams[-1].pending.append(ContainerEvent("router-0", "die"))
    case.controller.tick()
    assert case.controller.restart_count == 1


def test_restart_budget_never_resets_between_successful_recoveries(case: ControllerCase) -> None:
    opened(case)
    case.controller.recover("first-failure")
    case.controller.recover("second-failure")
    with pytest.raises(RoutingDeploymentError, match="budget"):
        case.controller.recover("third-failure")
    assert case.controller.restart_count == 2
    assert case.actions.count("start") == 2


def test_unknown_mutation_is_not_retried(case: ControllerCase) -> None:
    case.runtime.start_error = ContainerCommandError("unknown create", outcome_unknown=True)
    with pytest.raises(RoutingDeploymentError, match="unknown"):
        case.controller.recover("router-failure")
    assert case.controller.restart_count == 1
    assert case.actions.count("start") == 1
    assert "control:open" not in case.actions


def test_pending_record_finalization_blocks_recovery(case: ControllerCase) -> None:
    case.control.pending_writes = 1
    with pytest.raises(RoutingDeploymentError, match="finalization"):
        case.controller.recover("router-failure")
    assert "start" not in case.actions
    assert "control:open" not in case.actions


def test_drain_waits_for_admitted_requests_then_forces_them(case: ControllerCase) -> None:
    case.control.active_requests = 1
    case.controller.drain(recovery=False)
    # The admitted request (an upload counts) outlasts the drain window, so a second close cancels it.
    assert case.actions.count("control:close") == 2
    assert case.controller.state == "finalizing"
    assert case.control.active_requests == 0


def test_stop_during_readiness_never_reopens_acceptance(case: ControllerCase) -> None:
    case.management.stop_on_ready = case.controller
    with pytest.raises(InterruptedError):
        case.controller.wait_ready()
    assert "control:open" not in case.actions


def test_stop_before_tick_never_starts_recovery(case: ControllerCase) -> None:
    opened(case)
    case.runtime.streams[-1].pending.append(ContainerEvent("router-0", "die"))
    case.controller.request_stop()
    case.controller.tick()
    assert "remove" not in case.actions and "start" not in case.actions


def test_stream_reconnect_inspects_without_recreating_healthy_containers(case: ControllerCase) -> None:
    opened(case)
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    case.controller.tick()
    assert case.controller.state == "ready"
    assert case.controller.restart_count == 0
    assert "remove" not in case.actions and "start" not in case.actions
    assert case.actions.index("control:close") < case.actions.index("control:open")
    assert case.actions.count("subscribe") == 1
    assert "inspect" in case.actions[case.actions.index("control:close") :]
    # The verified runtime is unchanged, so it is not published again.
    assert "control:activate" not in case.actions


def test_child_control_loss_does_not_restart_sr(case: ControllerCase) -> None:
    opened(case)
    case.control.available = False
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="REEF"):
        case.controller.tick()
    assert "start" not in case.actions


def test_generation_mismatch_never_opens(case: ControllerCase) -> None:
    case.control.generation = "stale-generation"
    with pytest.raises(RoutingDeploymentError, match="mismatch"):
        case.controller.wait_ready()
    assert "control:open" not in case.actions


def test_shutdown_is_idempotent_and_stops_control_after_cleanup(case: ControllerCase) -> None:
    case.controller.shutdown()
    case.controller.shutdown()
    assert case.controller.cleanup_complete
    assert case.actions.count("remove") == 1
    assert case.actions.index("remove") < case.actions.index("control:stop")


def test_disconnected_stream_cannot_open_initial_acceptance(case: ControllerCase) -> None:
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    with pytest.raises(RoutingDeploymentError):
        case.controller.wait_ready()
    assert "control:open" not in case.actions


def test_immediately_failed_reconnection_cannot_reopen(case: ControllerCase) -> None:
    opened(case)
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    case.runtime.new_stream_disconnected = True
    with pytest.raises(RoutingDeploymentError):
        case.controller.tick()
    assert "control:close" in case.actions and "control:open" not in case.actions
    assert "start" not in case.actions


def test_fatal_generation_mismatch_is_not_counted_as_retryable_readiness(case: ControllerCase) -> None:
    case.control.generation = "stale-generation"
    with pytest.raises(RoutingDeploymentError, match="mismatch"):
        case.controller.recover("router-failure")
    assert case.actions.count("start") == 0
    assert "control:open" not in case.actions


def test_event_reader_shutdown_failure_does_not_skip_resource_cleanup(case: ControllerCase) -> None:
    case.runtime.streams[-1].close_error = True
    case.controller.shutdown()
    assert case.controller.failed
    assert case.controller.cleanup_complete
    assert "remove" in case.actions and "control:stop" in case.actions


def test_failed_write_diagnostic_does_not_stickily_disable_completed_requests(case: ControllerCase) -> None:
    case.control.failed_writes = 1
    case.controller.wait_ready()
    assert "control:open" in case.actions


def test_partial_prepare_failure_keeps_lock_until_safe_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reef.service.deploy.routing_controller as controller_module

    def broken_renderer(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid renderer input")

    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    monkeypatch.setattr(controller_module, "generate", broken_renderer)
    actions: list[str] = []
    runtime = FakeRuntime(actions)
    settings = ServiceConfig(
        recipe="base",
        host="127.0.0.1",
        tokens=("unit-token",),
        routing_manifest=str(tmp_path / "manifest.yaml"),
        routing_deployment=RoutingDeploymentSettings(deployment_id="partial"),
    )
    controller = RoutingDeploymentController(settings, runtime, registry_root=tmp_path / "registry")
    monkeypatch.setattr(controller, "validate_candidate", lambda: "already-validated")
    with pytest.raises(ValueError, match="renderer"):
        controller.prepare()
    assert controller.registry.lock_descriptor is not None
    controller.shutdown()
    assert "remove" in actions
    assert controller.registry.lock_descriptor is None
    assert controller.cleanup_complete
    controller.release_control_directory()
    assert controller.control_directory is None


def test_readiness_failure_consumes_budget_even_without_opening(case: ControllerCase) -> None:
    case.runtime.healthy_after_start = False
    with pytest.raises(RoutingDeploymentError, match="budget"):
        case.controller.recover("router-failure")
    assert case.controller.restart_count == 2
    assert case.actions.count("start") == 2
    assert "control:open" not in case.actions


@pytest.mark.parametrize("failure", [ValueError("bad response"), OSError("storage failure")])
def test_unexpected_drain_error_still_removes_containers_and_stops_child(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail_drain(*, recovery: bool) -> None:
        raise failure

    monkeypatch.setattr(case.controller, "drain", fail_drain)
    case.controller.shutdown()
    assert case.controller.failed and case.controller.cleanup_complete
    assert "remove" in case.actions and "control:stop" in case.actions
    assert case.controller.registry.lock_descriptor is None


def test_shutdown_diagnostics_error_cannot_skip_cleanup(case: ControllerCase, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_event(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(case.controller, "event", fail_event)
    case.controller.shutdown()
    assert case.controller.failed and case.controller.cleanup_complete
    assert "remove" in case.actions and "control:stop" in case.actions


def test_child_loss_during_recovery_does_not_spend_second_restart(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_start = case.runtime.start

    def lose_child(*args: object, **kwargs: object) -> None:
        original_start(*args, **kwargs)
        case.control.available = False

    monkeypatch.setattr(case.runtime, "start", lose_child)
    with pytest.raises(RoutingDeploymentError, match="REEF lifecycle unavailable"):
        case.controller.recover("router-failure")
    assert case.actions.count("start") == 1
    assert "control:open" not in case.actions


def test_operator_stop_during_recovery_is_not_a_failure(case: ControllerCase) -> None:
    case.management.stop_on_ready = case.controller
    case.controller.recover("router-failure")
    assert case.controller.stop_requested
    assert not case.controller.failed
    assert "control:open" not in case.actions


def test_control_directory_release_removes_only_its_socket(case: ControllerCase, tmp_path: Path) -> None:
    # A short directory like the controller's own: pytest's tmp_path can exceed the AF_UNIX path limit.
    directory = Path(tempfile.mkdtemp(prefix="reef-ctl-"))
    directory.chmod(0o700)
    leftover = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        leftover.bind(str(directory / "control.sock"))
    finally:
        leftover.close()
    case.controller.control_directory = directory
    case.controller.release_control_directory()
    assert not directory.exists()
    unexpected = tmp_path / "other"
    unexpected.mkdir(mode=0o700)
    (unexpected / "control.sock").write_text("not a socket")
    case.controller.control_directory = unexpected
    with pytest.raises(RoutingDeploymentError, match="unexpected file"):
        case.controller.release_control_directory()
    assert (unexpected / "control.sock").read_text() == "not a socket"


def test_invalid_candidate_cannot_touch_an_existing_generation(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    import reef.service.deploy.routing_controller as controller_module

    def invalid_candidate() -> str:
        raise ValueError("invalid routing candidate")

    case.controller.service_settings = ServiceConfig(
        recipe="base",
        host="127.0.0.1",
        tokens=("test-token",),
        routing_manifest="invalid.yaml",
        routing_deployment=case.controller.policy,
    )
    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    monkeypatch.setattr(case.controller, "validate_candidate", invalid_candidate)
    before = case.actions.copy()
    with pytest.raises(ValueError, match="candidate"):
        case.controller.prepare()
    assert case.actions == before


def test_routing_requires_a_deployment_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deployment_id"):
        RoutingDeploymentController(ServiceConfig(recipe="base"), FakeRuntime([]), registry_root=tmp_path)
