"""State-machine acceptance tests independent of Docker, the router and model services.

The runtime, the REEF child's control client and the router's management API
are in-process fakes that record the controller's calls.
"""

from __future__ import annotations

import errno
import io
import json
import logging
import socket
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest
import yaml
from tests.reef_service.routing.routing_fakes import (
    RUNTIME_HASH,
    SELECTION_TOKEN,
    ControllerCase,
    FakeEvents,
    FakeRuntime,
    opened,
    routing_manifest,
)

from reef.service.deploy.routing.containers import (
    ContainerCommandError,
    ContainerEvent,
    ContainerInventory,
    ContainerState,
)
from reef.service.deploy.routing.deployment_controller import (
    IMAGE_BUILD_LOG,
    RoutingDeploymentController,
    RoutingDeploymentError,
    RoutingReadinessTimeoutError,
)
from reef.service.deploy.routing.policy import PolicyState
from reef.service.deploy.routing.reef_lifecycle_client import (
    LifecycleStatus,
    RoutingControlError,
    RoutingControlTimeoutError,
)
from reef.service.deploy.routing.registry import DeploymentOwnership, DeploymentRegistry, RoutingOwnershipError
from reef.service.deploy.routing.settings import RoutingDeploymentSettings
from reef.service.deploy.service_config import ServiceSettings
from reef.service.routing import RoutingAdapter, RoutingSnapshot
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
    with pytest.raises(RoutingReadinessTimeoutError):
        case.controller.wait_ready()
    assert "control:activate" not in case.actions and "control:open" not in case.actions


def test_a_router_running_another_document_is_fatal(case: ControllerCase) -> None:
    case.management.etag = '"' + "0" * 64 + '"'
    with pytest.raises(RoutingDeploymentError, match="configuration"):
        case.controller.wait_ready()
    assert "control:open" not in case.actions


def test_a_failed_readiness_preview_is_retried_until_the_deadline(case: ControllerCase) -> None:
    case.control.open_error = RoutingControlError("REEF lifecycle returned HTTP 503", status_code=503)
    with pytest.raises(RoutingReadinessTimeoutError):
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
    committed = case.controller.committed_policy()
    case.management.active = case.management.generated = "8" * 64
    case.controller.policy_store.write(PolicyState(replace(committed, runtime_hash="8" * 64)))
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


def test_a_runtime_report_without_the_active_hash_is_drift_at_the_first_check(case: ControllerCase) -> None:
    opened(case)
    case.management.active = None
    case.controller.next_probe = 0
    # Unlike a failed readiness read, which counts only after three in a row, one report without the committed
    # runtime starts recovery; a router that keeps reporting it never reopens.
    with pytest.raises(RoutingDeploymentError, match="budget"):
        case.controller.tick()
    assert case.actions.index("control:close") < case.actions.index("remove")
    assert "control:open" not in case.actions


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
    ) -> LifecycleStatus:
        if action is None and not timed_out:
            timed_out.append(deadline)
            if not phase_left and deadline is not None:
                time.sleep(max(0, deadline - time.monotonic()))
            raise RoutingControlTimeoutError("REEF lifecycle request timed out")
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    if phase_left:
        # Three attempts failed while the drain had time: the child is not answering.
        with pytest.raises(RoutingControlTimeoutError):
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
    ) -> LifecycleStatus:
        if action is None:
            deadlines.append(deadline)
            if not phase_left and deadline is not None:
                time.sleep(max(0, deadline - time.monotonic()))
            raise RoutingControlTimeoutError("REEF lifecycle request timed out")
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)
    with pytest.raises(RoutingDeploymentError) as error:
        case.controller.wait_ready(initial=False)
    # Bounded recovery may try again after a readiness timeout, but not after the child stopped answering.
    assert isinstance(error.value, RoutingReadinessTimeoutError) is not phase_left
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
    import reef.service.deploy.routing.deployment_controller as controller_module

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
    ) -> LifecycleStatus:
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
        with pytest.raises(RoutingReadinessTimeoutError):
            case.controller.tick()
        assert open_deadlines == [started + 11]
    assert case.controller.restart_count == 0


@pytest.mark.parametrize("outcome", ["finished", "unfinished_request", "unfinished_write", "stopped"])
def test_a_late_open_reply_is_drained_again_before_the_next_recovery_removes_sr(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    import reef.service.deploy.routing.deployment_controller as controller_module

    opened(case)
    clock = SteppingClock()
    monkeypatch.setattr(controller_module, "time", clock)
    # Long enough for several status reads before the forced close; the stepping clock makes it instant.
    case.controller.policy = replace(case.controller.policy, drain_timeout_seconds=1)
    original_request, original_remove = case.control.request, case.runtime.remove
    accepting = [True]
    late_open = [True]
    #: At each SR removal: whether admission was open, and REEF's admitted requests and pending record writes.
    removals: list[tuple[bool, int, int]] = []

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> LifecycleStatus:
        if action is None and not late_open[0] and outcome != "unfinished_request":
            if case.control.active_requests:
                # The request admitted after the late OPEN finishes; its record write starts and is pending.
                case.control.active_requests, case.control.pending_writes = 0, 1
                if outcome == "stopped":
                    case.controller.request_stop()
            elif case.control.pending_writes and outcome != "unfinished_write":
                # On a later status read, the record write has finished too.
                case.control.pending_writes = 0
        status = original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)
        if cancel_active and outcome == "unfinished_request":
            # Even the forced close cannot finish it.
            case.control.active_requests = 1
        if action is not None:
            accepting[0] = action == "open"
        if action == "open" and late_open[0]:
            # REEF opened before the deadline, admitted a request, and the reply reached the owner too late.
            late_open[0] = False
            case.control.active_requests = 1
            clock.now = (deadline or clock.now) + 0.001
            raise RoutingControlTimeoutError("REEF lifecycle replied after the caller's deadline")
        return status

    def remove(ownership: DeploymentOwnership, stop_timeout_seconds: float) -> None:
        removals.append((accepting[0], case.control.active_requests, case.control.pending_writes))
        original_remove(ownership, stop_timeout_seconds)

    monkeypatch.setattr(case.control, "request", request)
    monkeypatch.setattr(case.runtime, "remove", remove)
    if outcome.startswith("unfinished"):
        # The finalization deadline passes with work left, which prohibits another removal.
        with pytest.raises(RoutingDeploymentError, match="recovery prohibited"):
            case.controller.recover("router-failure")
    else:
        case.controller.recover("router-failure")
    # The first attempt removes SR while REEF is closed and idle; after the late OPEN, only a drained REEF may.
    expected = [(False, 0, 0), (False, 0, 0)] if outcome == "finished" else [(False, 0, 0)]
    assert removals == expected
    final_state = {"finished": "ready", "stopped": "stopping"}.get(outcome, "finalizing")
    assert case.controller.state == final_state


def test_child_control_loss_does_not_restart_sr(case: ControllerCase) -> None:
    opened(case)
    case.control.available = False
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="REEF"):
        case.controller.tick()
    assert "start" not in case.actions


def time_out_status_reads(case: ControllerCase, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """Make the next ``count`` status reads time out, as when the child's event loop is busy."""
    original_request = case.control.request
    remaining = count

    def request(
        action: str | None = None,
        *,
        cancel_active: bool = False,
        publication: dict[str, str] | None = None,
        deadline: float | None = None,
    ) -> LifecycleStatus:
        nonlocal remaining
        if action is None and remaining:
            remaining -= 1
            raise RoutingControlTimeoutError("REEF lifecycle request timed out")
        return original_request(action, cancel_active=cancel_active, publication=publication, deadline=deadline)

    monkeypatch.setattr(case.control, "request", request)


def test_periodic_status_timeouts_keep_serving_and_a_later_reply_is_still_checked(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened(case)
    time_out_status_reads(case, monkeypatch, 3)
    for _ in range(3):
        case.controller.next_probe = case.controller.next_inspect = 0
        case.controller.tick()
    assert case.controller.state == "ready" and case.controller.restart_count == 0
    # The router and container checks still ran after every timeout; nothing closed, stopped or restarted.
    assert case.actions.count("router:ready") == case.actions.count("inspect") == 3
    assert not {"control:close", "control:stop", "remove", "start"} & set(case.actions)
    events = (case.controller.directory / "lifecycle-events.jsonl").read_text().splitlines()
    assert [json.loads(line)["state"] for line in events].count("status_timeout") == 3
    case.controller.next_probe = 0
    case.controller.tick()
    assert case.actions.count("control:None") == 1
    assert case.controller.state == "ready" and case.controller.restart_count == 0
    assert not {"control:close", "control:stop", "remove", "start"} & set(case.actions)
    case.control.stopping = True
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="stopping"):
        case.controller.tick()


@pytest.mark.parametrize(
    ("generation", "snapshot_version"),
    [("stale-generation", "snapshot-one"), ("generation-one", "other-snapshot")],
)
def test_a_periodic_status_for_another_generation_or_snapshot_is_fatal(
    case: ControllerCase, generation: str, snapshot_version: str
) -> None:
    opened(case)
    case.control.generation = generation
    case.control.snapshot_version = snapshot_version
    case.controller.next_probe = 0
    with pytest.raises(RoutingDeploymentError, match="mismatch"):
        case.controller.tick()
    assert "start" not in case.actions


@pytest.mark.parametrize("failure", ["runtime", "container"])
def test_a_periodic_status_timeout_does_not_hide_a_router_failure(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    opened(case)
    time_out_status_reads(case, monkeypatch, 1)
    case.controller.next_probe = case.controller.next_inspect = 0
    if failure == "runtime":
        # Another writer's consistent runtime, as in the drift test above.
        case.management.active = case.management.generated = "9" * 64
        with pytest.raises(RoutingDeploymentError, match="router does not run the committed policy"):
            case.controller.tick()
        assert case.actions.index("control:close") < case.actions.index("remove")
    else:
        case.runtime.inventory = case.runtime.healthy_inventory("unhealthy")
        case.controller.tick()
        assert case.controller.restart_count == 1 and case.controller.state == "ready"


@pytest.mark.parametrize("failure", ["open", "write", "chmod", "stderr", "every_sink", "closed_logger"])
def test_a_diagnostic_write_failure_leaves_a_healthy_deployment_serving(
    case: ControllerCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    import reef.service.deploy.routing.deployment_controller as controller_module

    opened(case)
    diagnostic = case.controller.directory / "lifecycle-events.jsonl"
    original_open, original_chmod = Path.open, Path.chmod

    class FullFile(io.StringIO):
        """The event file on a full disk: it opens, and every write fails."""

        def write(self, text: str) -> int:
            raise OSError(errno.ENOSPC, "no space left for diagnostics")

    class BrokenStream:
        """A standard error whose every write fails, as after its reader exited."""

        def write(self, text: str) -> int:
            raise OSError(errno.EPIPE, "standard error is broken")

        def flush(self) -> None:
            raise OSError(errno.EPIPE, "standard error is broken")

    def path_open(path: Path, *args: object, **kwargs: object) -> object:
        if path == diagnostic and failure == "open":
            raise OSError(errno.ENOSPC, "no space left for diagnostics")
        if path == diagnostic and failure in {"write", "every_sink", "closed_logger"}:
            return FullFile()
        return original_open(path, *args, **kwargs)

    def path_chmod(path: Path, mode: int, **kwargs: object) -> None:
        if path == diagnostic and failure == "chmod":
            raise OSError(errno.EPERM, "diagnostics are read-only")
        original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(Path, "open", path_open)
    monkeypatch.setattr(Path, "chmod", path_chmod)
    if failure == "stderr":
        monkeypatch.setattr(sys, "stderr", BrokenStream())
    if failure in {"every_sink", "closed_logger"}:
        # The fallback handler is unavailable too, and logging's own error output uses closed stderr.
        closed = io.StringIO()
        closed.close()
        handler: logging.Handler
        if failure == "every_sink":
            handler = logging.FileHandler(tmp_path / "missing-directory" / "reef.log", delay=True)
        else:
            handler = logging.StreamHandler(closed)
        monkeypatch.setattr(controller_module.logger, "handlers", [handler])
        monkeypatch.setattr(controller_module.logger, "propagate", False)
        monkeypatch.setattr(logging, "raiseExceptions", True)
        monkeypatch.setattr(sys, "stderr", closed)
        assert controller_module.logger.isEnabledFor(logging.WARNING)
    time_out_status_reads(case, monkeypatch, 1)
    case.controller.next_probe = case.controller.next_inspect = 0
    case.controller.tick()
    # A reload this case cannot render is rejected; the serving policy stays.
    case.controller.request_reload()
    case.controller.tick()
    assert case.controller.state == "ready" and not case.controller.reload_requested
    # The router and container checks after the timed-out status read still ran; nothing closed or restarted.
    assert case.actions.count("router:ready") == 1 and "inspect" in case.actions
    assert not {"control:close", "control:stop", "remove", "start"} & set(case.actions)


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
    import reef.service.deploy.routing.deployment_controller as controller_module

    def broken_renderer(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid renderer input")

    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    monkeypatch.setattr(controller_module, "generate", broken_renderer)
    actions: list[str] = []
    runtime = FakeRuntime(actions)
    # A source that names its image: preparation reads it, and the render then fails.
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps(routing_manifest("http://127.0.0.1:9")), encoding="utf-8")
    settings = ServiceSettings(
        recipe="base",
        host="127.0.0.1",
        tokens=("unit-token",),
        routing_manifest=str(manifest),
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


def preparing_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: FakeRuntime, image: str | None
) -> RoutingDeploymentController:
    """An unprepared controller over ``runtime`` whose source manifest names ``image``, or omits it when None.

    A previous generation of the deployment is on record, so ``prepare`` removes its resources.
    """
    import reef.service.deploy.routing.deployment_controller as controller_module

    monkeypatch.setattr(controller_module.sys, "platform", "linux")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(json.dumps(routing_manifest("http://127.0.0.1:9", image=image)), encoding="utf-8")
    settings = ServiceSettings(
        recipe="base",
        host="127.0.0.1",
        tokens=("unit-token",),
        routing_manifest=str(manifest),
        routing_deployment=RoutingDeploymentSettings(deployment_id="resolved", image_build_timeout_seconds=12.5),
    )
    with DeploymentRegistry("resolved", root=tmp_path / "registry") as registry:
        registry.prepare("previous", tmp_path / "previous")
    return RoutingDeploymentController(settings, runtime, registry_root=tmp_path / "registry")


@pytest.mark.parametrize("image", [None, "sha256:" + "f" * 64, "registry.example/sr@sha256:" + "e" * 64])
def test_prepare_resolves_an_omitted_image_once_under_ownership_and_freezes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image: str | None
) -> None:
    actions: list[str] = []
    runtime = FakeRuntime(actions)
    controller = preparing_controller(tmp_path, monkeypatch, runtime, image)
    try:
        controller.prepare()
        expected = runtime.resolved_image if image is None else image
        if image is None:
            # Resolved under the lock, before the previous generation's resources are removed and before any token
            # exists: the build environment holds none.
            assert actions[:3] == ["available", "resolve", "remove"]
            assert runtime.resolutions == [(controller.registry.directory / IMAGE_BUILD_LOG, 12.5, {})]
        else:
            # A named image, a local ID included, is used as it is: nothing resolves or builds one.
            assert "resolve" not in actions and runtime.resolutions == []
        assert controller.semantic_router_image == expected
        compose = yaml.safe_load((controller.directory / "compose.yaml").read_text(encoding="utf-8"))
        assert compose["services"]["semantic-router"]["image"] == expected
        generated = RoutingSnapshot.load(controller.directory / "manifest.yaml")
        assert generated.semantic_router_image == expected
        assert generated.version == controller.snapshot_version
        control_data = json.loads((controller.directory / "control.json").read_text(encoding="utf-8"))
        assert control_data["snapshot_version"] == controller.snapshot_version
        events = (controller.directory / "lifecycle-events.jsonl").read_text(encoding="utf-8").splitlines()
        prepared = json.loads(events[-1])
        assert prepared["state"] == "prepared"
        assert (prepared["semantic_router_image"], prepared["snapshot_version"]) == (expected, generated.version)
        assert "unit-token" not in json.dumps(prepared)
        # Validation stays offline: without a named image it cannot know the version, and says so.
        assert controller.validate_candidate() == (None if image is None else controller.snapshot_version)
    finally:
        controller.registry.release()
        controller.release_control_directory()


@pytest.mark.parametrize("edit", ["named_image", "malformed_yaml", "invalid_manifest", "removed"])
def test_a_source_edited_during_the_image_build_does_not_reach_the_generation_being_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, edit: str
) -> None:
    class EditingRuntime(FakeRuntime):
        """The public source manifest changes while the image builds."""

        def resolve_image(self, build_log: Path, timeout_seconds: float, stop_requested: Callable[[], bool]) -> str:
            source = Path(str(controller.service_settings.routing_manifest))
            if edit == "named_image":
                named = routing_manifest("http://127.0.0.1:9", image="registry.example/sr@sha256:" + "e" * 64)
                source.write_text(json.dumps(named), encoding="utf-8")
            elif edit == "malformed_yaml":
                source.write_text("deployment: [", encoding="utf-8")
            elif edit == "invalid_manifest":
                source.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
            else:
                source.unlink()
            return super().resolve_image(build_log, timeout_seconds, stop_requested)

    actions: list[str] = []
    runtime = EditingRuntime(actions)
    controller = preparing_controller(tmp_path, monkeypatch, runtime, None)
    startup = routing_manifest("http://127.0.0.1:9", image=None)
    try:
        controller.prepare()
        built = runtime.resolved_image
        assert actions[:3] == ["available", "resolve", "remove"]
        # One generation: the controller, the Compose file, the generated manifest, the child's control file and the
        # prepared event all name the built image and the version of the document read before the build began.
        assert controller.semantic_router_image == built
        compose = yaml.safe_load((controller.directory / "compose.yaml").read_text(encoding="utf-8"))
        assert compose["services"]["semantic-router"]["image"] == built
        generated = yaml.safe_load((controller.directory / "manifest.yaml").read_text(encoding="utf-8"))
        assert generated["deployment"]["semantic_router_image"] == built
        unchanged = {key: value for key, value in startup.items() if key != "deployment"}
        assert {key: value for key, value in generated.items() if key != "deployment"} == unchanged
        version = RoutingSnapshot.from_mapping(generated).version
        assert version == controller.snapshot_version
        control_data = json.loads((controller.directory / "control.json").read_text(encoding="utf-8"))
        assert control_data["snapshot_version"] == version
        events = (controller.directory / "lifecycle-events.jsonl").read_text(encoding="utf-8").splitlines()
        prepared = json.loads(events[-1])
        assert (prepared["state"], prepared["semantic_router_image"], prepared["snapshot_version"]) == (
            "prepared",
            built,
            version,
        )
        # The private copy that the generation rendered is gone; the edit reaches it only as a later update.
        assert not list(controller.registry.directory.glob("startup-*"))
    finally:
        controller.registry.release()
        controller.release_control_directory()


@pytest.mark.parametrize("moment", ["during", "after"])
def test_a_stop_while_the_image_is_resolved_creates_and_removes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, moment: str
) -> None:
    class StoppedRuntime(FakeRuntime):
        """Stop is requested while the image is resolved; a build that sees it stops."""

        def resolve_image(self, build_log: Path, timeout_seconds: float, stop_requested: Callable[[], bool]) -> str:
            image = super().resolve_image(build_log, timeout_seconds, stop_requested)
            controller.request_stop()
            if moment == "during" and stop_requested():
                raise InterruptedError("the build was stopped")
            return image

    actions: list[str] = []
    runtime = StoppedRuntime(actions)
    controller = preparing_controller(tmp_path, monkeypatch, runtime, None)
    controller.prepare()
    assert controller.stop_requested and not controller.failed
    assert actions == ["available", "resolve"]
    assert controller.ownership is None and controller.control_directory is None
    assert not (controller.registry.directory / "generations").exists()
    controller.shutdown()
    assert actions == ["available", "resolve"]
    assert controller.registry.lock_descriptor is None


def test_a_failed_image_build_fails_preparation_before_any_resource_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    actions: list[str] = []
    runtime = FakeRuntime(actions)
    runtime.resolve_error = ContainerCommandError("the Semantic Router image build failed with status 1; see a log")
    controller = preparing_controller(tmp_path, monkeypatch, runtime, None)
    with pytest.raises(ContainerCommandError, match="image build failed"):
        controller.prepare()
    assert controller.failed and controller.ownership is None
    assert not (controller.registry.directory / "generations").exists()
    controller.shutdown()
    assert actions == ["available", "resolve"]
    assert controller.registry.lock_descriptor is None


def test_recovery_restarts_the_frozen_generation_without_resolving_an_image(case: ControllerCase) -> None:
    opened(case)
    case.controller.semantic_router_image = "sha256:" + "c" * 64
    case.controller.recover("router-failure")
    assert "start" in case.actions and "control:open" in case.actions
    assert "resolve" not in case.actions and case.runtime.resolutions == []


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
    import reef.service.deploy.routing.deployment_controller as controller_module

    def invalid_candidate() -> str:
        raise ValueError("invalid routing candidate")

    case.controller.service_settings = ServiceSettings(
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
        ServiceSettings(recipe="base", routing_deployment=policy, inference_timeout_s=inference_timeout_s),
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
        RoutingDeploymentController(ServiceSettings(recipe="base"), FakeRuntime([]), registry_root=tmp_path)
