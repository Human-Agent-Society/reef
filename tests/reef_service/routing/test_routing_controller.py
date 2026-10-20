"""State-machine acceptance tests independent of Docker, the router and model services.

The runtime, the REEF child's control client and the router's management API
are in-process fakes that record the controller's calls.
"""

from __future__ import annotations

import json
import socket
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
from tests.reef_service.routing.routing_fakes import (
    RUNTIME_HASH,
    SELECTION_TOKEN,
    ControllerCase,
    FakeEvents,
    FakeRuntime,
    opened,
    routing_manifest,
)

from reef.service.deploy.routing.routing_config import RoutingDeploymentSettings
from reef.service.deploy.routing.routing_containers import (
    ContainerCommandError,
    ContainerEvent,
    ContainerInventory,
    ContainerState,
)
from reef.service.deploy.routing.routing_control import RoutingControlError, RoutingControlTimeout
from reef.service.deploy.routing.routing_controller import (
    RoutingDeploymentController,
    RoutingDeploymentError,
    RoutingReadinessTimeout,
)
from reef.service.deploy.routing.routing_registry import DeploymentOwnership, RoutingOwnershipError
from reef.service.deploy.service_config import ServiceConfig
from reef.service.routing import RoutingAdapter
from reef.service.routing_preview import PreviewSettings

pytestmark = pytest.mark.unit


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
    # Every readiness request, including the open, ends by the readiness deadline.
    assert {action for action, _ in case.control.deadlines} == {None, "activate", "open"}
    assert {deadline for _, deadline in case.control.deadlines} == {case.controller.ready_deadline}


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
    started = time.monotonic()
    case.controller.drain(recovery=False)
    # The admitted request (an upload counts) outlasts the drain window, so a second close cancels it.
    assert case.actions.count("control:close") == 2
    assert case.controller.state == "finalizing"
    assert case.control.active_requests == 0
    # The first close starts the drain phase, so it has no phase deadline. Each status read of the drain
    # shares the drain deadline; the forced close and the reads after it share the finalization deadline.
    phases = case.control.deadlines
    forced = [action for action, _ in phases].index("close", 1)
    assert phases[0] == ("close", None)
    assert {action for action, _ in phases[1:forced]} == {None}
    [drain_deadline] = {deadline for _, deadline in phases[1:forced]}
    [finalization_deadline] = {deadline for _, deadline in phases[forced:]}
    assert drain_deadline is not None and finalization_deadline is not None
    assert started + case.controller.policy.drain_timeout_seconds <= drain_deadline < finalization_deadline


@pytest.mark.parametrize("phase_left", [False, True])
def test_a_drain_read_that_outlasts_the_drain_phase_moves_on_to_the_forced_close(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, phase_left: bool
) -> None:
    case.control.active_requests = 1
    case.controller.policy = replace(case.controller.policy, drain_timeout_seconds=5 if phase_left else 0.05)
    original_request = case.control.request
    timed_out: list[float | None] = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> dict[str, object]:
        if action is None and not timed_out:
            timed_out.append(deadline)
            if not phase_left and deadline is not None:
                time.sleep(max(0, deadline - time.monotonic()))
            raise RoutingControlTimeout("REEF lifecycle request timed out")
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    if phase_left:
        # Three attempts failed while the drain had time: the child is not answering.
        with pytest.raises(RoutingControlTimeout):
            case.controller.drain(recovery=False)
        assert case.actions.count("control:close") == 1
    else:
        case.controller.drain(recovery=False)
        assert case.actions.count("control:close") == 2 and case.controller.state == "finalizing"
        assert case.control.active_requests == 0
    assert len(timed_out) == 1 and timed_out[0] is not None


@pytest.mark.parametrize("phase_left", [False, True])
def test_a_recovery_status_read_cut_short_by_the_readiness_deadline_is_a_readiness_timeout(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, phase_left: bool
) -> None:
    opened(case)
    case.controller.ready_deadline = time.monotonic() + (5 if phase_left else 0.05)
    original_request = case.control.request
    deadlines: list[float | None] = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> dict[str, object]:
        if action is None:
            deadlines.append(deadline)
            if not phase_left and deadline is not None:
                time.sleep(max(0, deadline - time.monotonic()))
            raise RoutingControlTimeout("REEF lifecycle request timed out")
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    with pytest.raises(RoutingDeploymentError) as error:
        case.controller.wait_ready(initial=False)
    # Bounded recovery may try again after a readiness timeout, but not after the child stopped answering.
    assert isinstance(error.value, RoutingReadinessTimeout) is not phase_left
    assert deadlines == [case.controller.ready_deadline]
    assert "control:open" not in case.actions


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


@pytest.mark.parametrize("router", ["missing", "stopped", "unhealthy"])
def test_a_failed_router_found_after_observation_returns_gets_bounded_recovery(
    case: ControllerCase, router: str
) -> None:
    opened(case)
    case.runtime.inventory = {
        "missing": ContainerInventory(),
        "stopped": ContainerInventory((ContainerState("router-0", "semantic-router", False, "healthy"),)),
        "unhealthy": case.runtime.healthy_inventory("unhealthy"),
    }[router]
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    case.controller.tick()
    # The same recovery a periodic inspection starts: one restart, then a verified reopening.
    assert case.controller.state == "ready" and case.controller.restart_count == 1
    assert case.actions.index("inspect") < case.actions.index("remove") < case.actions.index("start")
    assert case.actions.index("start") < case.actions.index("control:open")


@pytest.mark.parametrize("policy", ["fail_stop", "budget_spent"])
def test_a_failed_router_found_after_observation_returns_follows_the_restart_policy(
    case: ControllerCase, policy: str
) -> None:
    opened(case)
    if policy == "fail_stop":
        case.controller.policy = replace(case.controller.policy, restart_policy="fail_stop")
    else:
        case.controller.restart_count = case.controller.policy.max_restarts
    case.runtime.inventory = ContainerInventory()
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    with pytest.raises(RoutingDeploymentError, match="fail_stop" if policy == "fail_stop" else "budget"):
        case.controller.tick()
    assert "start" not in case.actions and "control:open" not in case.actions


@pytest.mark.parametrize("router", ["healthy", "unhealthy", "missing", "queued_failure"])
def test_a_restored_event_stream_that_disconnects_again_is_fatal_before_any_recovery(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, router: str
) -> None:
    opened(case)
    if router == "unhealthy":
        case.runtime.inventory = case.runtime.healthy_inventory("unhealthy")
    elif router != "healthy":
        case.runtime.inventory = ContainerInventory()
    if router == "queued_failure":
        create_stream = case.runtime.events

        def events(ownership: DeploymentOwnership) -> FakeEvents:
            # The old container's failure is queued first, then the new stream's own disconnect.
            stream = create_stream(ownership)
            stream.pending.extend([ContainerEvent("router-0", "die"), ContainerEvent("", "stream_disconnected")])
            return stream

        monkeypatch.setattr(case.runtime, "events", events)
    else:
        case.runtime.new_stream_disconnected = True
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    with pytest.raises(RoutingDeploymentError):
        case.controller.tick()
    # Admission was closed and never reopened; nothing replaced the router or spent the restart budget.
    assert "control:close" in case.actions and "control:open" not in case.actions
    assert "remove" not in case.actions and "start" not in case.actions
    assert case.controller.restart_count == 0


@pytest.mark.parametrize("fault", ["foreign_owner", "events_unavailable", "inventory_unavailable", "control_lost"])
def test_a_reconnect_without_trusted_observation_stays_fatal(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    opened(case)

    def fail(*args: object, **kwargs: object) -> NoReturn:
        if fault == "foreign_owner":
            raise RoutingOwnershipError("refusing to adopt or change a foreign routing container")
        raise ContainerCommandError("Docker CLI deadline expired")

    if fault == "control_lost":
        case.control.available = False
    else:
        monkeypatch.setattr(case.runtime, "events" if fault == "events_unavailable" else "inspect", fail)
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    expected = RoutingOwnershipError if fault == "foreign_owner" else RoutingDeploymentError
    with pytest.raises(expected):
        case.controller.tick()
    # Nothing restarts SR when the observer, the ownership or REEF control cannot be trusted.
    assert case.controller.restart_count == 0
    assert "remove" not in case.actions and "start" not in case.actions and "control:open" not in case.actions


class SteppingClock:
    """A monotonic clock that only ``sleep`` advances, standing in for the controller module's ``time``.

    ``now`` starts at 1000.0. ``monotonic`` returns it; ``sleep(seconds)`` adds ``seconds`` at once, without
    waiting or checking the sign. It provides only these two functions of ``time`` and owns no resource.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.mark.parametrize("ready_timeout_seconds", [30, 11])
def test_reconnect_readiness_keeps_the_configured_deadline(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, ready_timeout_seconds: float
) -> None:
    import reef.service.deploy.routing.routing_controller as controller_module

    opened(case)
    clock = SteppingClock()
    monkeypatch.setattr(controller_module, "time", clock)
    case.controller.policy = replace(case.controller.policy, ready_timeout_seconds=ready_timeout_seconds)
    original_request = case.control.request
    open_deadlines: list[float | None] = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> dict[str, object]:
        if action == "open":
            open_deadlines.append(deadline)
            if len(open_deadlines) == 1:
                # A slow readiness check that fails after 12 seconds, longer than the old ten-second cap.
                clock.sleep(12)
                raise RoutingControlError("REEF lifecycle returned HTTP 503", status_code=503)
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    started = clock.now
    case.runtime.streams[-1].pending.append(ContainerEvent("", "stream_disconnected"))
    if ready_timeout_seconds == 30:
        case.controller.tick()
        assert case.controller.state == "ready"
        # Every open of the phase carries the configured deadline.
        assert open_deadlines == [started + 30, started + 30]
    else:
        with pytest.raises(RoutingReadinessTimeout):
            case.controller.tick()
        assert open_deadlines == [started + 11]
    assert case.controller.restart_count == 0


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
    import reef.service.deploy.routing.routing_controller as controller_module

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
    import reef.service.deploy.routing.routing_controller as controller_module

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


@pytest.mark.parametrize("inference_timeout_s", [6.0, 300.0])
def test_the_open_budget_covers_each_provider_check_and_preview_with_the_childs_own_timeouts(
    tmp_path: Path, inference_timeout_s: float
) -> None:
    """Two enabled targets on one URL (one check in the child, two in this upper bound), one disabled target."""
    manifest = routing_manifest("http://127.0.0.1:9")
    manifest["targets"]["retired"] = {
        "provider_model": "old-v0",
        "revision": "r0",
        "role": "student",
        "url": "http://127.0.0.1:10",
        "enabled": False,
    }
    path = tmp_path / "routing.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = RoutingDeploymentSettings(deployment_id="unit-routing", selection_timeout_seconds=7)
    controller = RoutingDeploymentController(
        ServiceConfig(recipe="base", routing_deployment=policy, inference_timeout_s=inference_timeout_s),
        FakeRuntime([]),
        registry_root=tmp_path / "registry",
    )
    # The timeouts the child itself uses: a provider check, then each Scenario's preview.
    child = RoutingAdapter.from_manifest(path, router_config_hash=None, environ={}, timeout_s=inference_timeout_s)
    preview = PreviewSettings("http://127.0.0.1:1", SELECTION_TOKEN, policy.selection_timeout_seconds)
    # aiohttp may end each request up to one second late; the reply then gets five seconds.
    slowest = 2 * (child.readiness_timeout_s + 1) + len(child.snapshot.scenarios) * (preview.http_timeout_seconds + 1)
    assert controller.open_timeout_seconds(child.snapshot) == slowest + 5
    assert child.readiness_timeout_s == min(inference_timeout_s, 10)


def test_routing_requires_a_deployment_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="deployment_id"):
        RoutingDeploymentController(ServiceConfig(recipe="base"), FakeRuntime([]), registry_root=tmp_path)
