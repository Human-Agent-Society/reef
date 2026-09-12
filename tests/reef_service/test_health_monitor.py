"""Deterministic drain barriers for asynchronous engine health probes."""

from threading import Event, Thread
from time import monotonic

import pytest

from reef.runtime.health_monitor import EngineHealthMonitor, HealthMonitorConfig


class Target:
    def __init__(self):
        self.entered = Event()
        self.release = Event()
        self.retiring = Event()
        self.retire_release = Event()
        self.retire_release.set()
        self.retired = False
        self.fail = True

    def check(self, timeout):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("test probe was not released")
        if self.fail:
            raise RuntimeError("old engine failed")

    def retire(self, timeout):
        self.retiring.set()
        if not self.retire_release.wait(5):
            raise TimeoutError("test retirement was not released")
        self.retired = True


class Checks:
    def __init__(self, target):
        self.target = target

    def targets(self):
        return [self.target]


@pytest.fixture
def running():
    target = Target()
    checks = Checks(target)
    monitor = EngineHealthMonitor(checks, HealthMonitorConfig(interval=60, timeout=0.5))
    monitor.start()
    try:
        yield monitor, target, checks
    finally:
        target.release.set()
        target.retire_release.set()
        monitor.stop()


def wait_disabled(monitor):
    deadline = monotonic() + 5
    tick = Event()
    while monitor.is_checking_enabled():
        assert monotonic() < deadline, "pause did not disable checks"
        tick.wait(0.001)


def test_pause_drains_probe_and_discards_late_failure_before_replacement(running):
    monitor, old, checks = running
    replacement = Target()
    monitor.resume()
    assert old.entered.wait(5)
    replaced = Event()

    def replace_after_pause():
        monitor.pause()
        checks.target = replacement
        replaced.set()

    owner = Thread(target=replace_after_pause)
    owner.start()
    try:
        wait_disabled(monitor)
        assert not replaced.is_set()
        old.release.set()
        assert replaced.wait(5)
        assert checks.target is replacement
        assert not old.retired and not replacement.retired
        assert not replacement.entered.is_set()
    finally:
        old.release.set()
        owner.join(5)
    assert not owner.is_alive()


def test_pause_waits_for_retirement_already_in_progress(running):
    monitor, target, _ = running
    target.retire_release.clear()
    target.release.set()
    monitor.resume()
    assert target.retiring.wait(5)
    paused = Event()

    def pause():
        monitor.pause()
        paused.set()

    owner = Thread(target=pause)
    owner.start()
    try:
        wait_disabled(monitor)
        assert not paused.is_set()
        target.retire_release.set()
        assert paused.wait(5)
        assert target.retired
    finally:
        target.retire_release.set()
        owner.join(5)


def test_pause_timeout_stays_disabled_and_can_be_retried(running):
    monitor, target, _ = running
    monitor.resume()
    assert target.entered.wait(5)
    with pytest.raises(TimeoutError, match="in-flight"):
        monitor.pause(timeout=0.01)
    assert not monitor.is_checking_enabled()
    target.release.set()
    monitor.pause()
    assert not target.retired
    target.fail = False
    monitor.resume()
    monitor.pause()


def test_stop_timeout_preserves_drain_state_and_retries(running):
    monitor, target, _ = running
    monitor.resume()
    assert target.entered.wait(5)
    with pytest.raises(TimeoutError, match="in-flight"):
        monitor.stop(timeout=0.01)
    assert not monitor.is_checking_enabled()
    with pytest.raises(RuntimeError, match="not running"):
        monitor.resume()
    target.release.set()
    monitor.stop()
    monitor.stop()
    assert not target.retired
    with pytest.raises(RuntimeError, match="stopped"):
        monitor.start()


def test_failed_probe_retires_captured_target_when_not_paused(running):
    monitor, target, _ = running
    target.release.set()
    monitor.resume()
    assert target.retiring.wait(5)
    monitor.pause()
    assert target.retired


def test_healthy_probe_does_not_retire(running):
    monitor, target, _ = running
    target.fail = False
    target.release.set()
    monitor.resume()
    assert target.entered.wait(5)
    monitor.pause()
    assert not target.retired
    monitor.check_health()


def test_stop_interrupts_initial_wait_without_scheduling_probe():
    target = Target()
    monitor = EngineHealthMonitor(Checks(target), HealthMonitorConfig(interval=60, timeout=1, first_wait=60))
    assert monitor.start()
    assert not monitor.start()
    monitor.resume()
    monitor.stop()
    assert not target.entered.is_set()


def test_internal_monitor_failure_surfaces_in_health_and_resume(running):
    monitor, target, _ = running
    target.release.set()

    def fail_retire(timeout):
        target.retiring.set()
        raise RuntimeError("retirement failed")

    target.retire = fail_retire
    monitor.resume()
    assert target.retiring.wait(5)
    monitor.pause()
    with pytest.raises(RuntimeError, match="monitor failed"):
        monitor.check_health()
    with pytest.raises(RuntimeError, match="monitor failed"):
        monitor.resume()


@pytest.mark.parametrize(
    "options",
    [{"interval": 0}, {"timeout": 0}, {"first_wait": -1}, {"timeout": float("inf")}, {"interval": float("nan")}],
)
def test_invalid_monitor_timings_fail_before_thread_start(options):
    with pytest.raises(ValueError):
        HealthMonitorConfig(**{"interval": 1, "timeout": 1, **options})
