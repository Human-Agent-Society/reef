"""Backend-independent inference pause, recovery and termination contracts."""

import pytest

from reef.runtime.recovery import InferenceControl, InferenceEngineGroup, InferenceMonitor


class MemoryEngines(InferenceEngineGroup):
    owned = True

    def __init__(self, events):
        self.events = events
        self.failure = None
        self.paused = False
        self.dead_engines = 0

    def event(self, name):
        self.events.append(name)
        if self.failure == name:
            raise RuntimeError(f"failed {name}")

    def pause(self):
        self.event("pause_engines")
        self.paused = True
        return ["paused"]

    def resume(self):
        self.event("resume_engines")
        self.paused = False
        return ["resumed"]

    def recover(self):
        self.event("recover_engines")
        # Replacement engines start without the previous controller's barrier.
        self.paused = False
        replaced, self.dead_engines = self.dead_engines, 0
        return replaced

    def terminate(self):
        self.event("terminate_engines")
        return 2


class MemoryMonitor(InferenceMonitor):
    def __init__(self, events):
        self.events = events
        self.paused = False

    def pause(self):
        self.events.append("pause_monitor")
        self.paused = True

    def resume(self):
        self.events.append("resume_monitor")
        self.paused = False


@pytest.fixture
def control():
    events = []
    engines = MemoryEngines(events)
    monitor = MemoryMonitor(events)
    return InferenceControl(engines, monitor), engines, monitor, events


def test_recovery_reapplies_pause_and_keeps_monitor_stopped_until_commit_resume(control):
    controller, engines, monitor, events = control
    assert controller.pause() == ["paused"]
    events.clear()
    controller.recover()
    assert events == ["pause_monitor", "recover_engines", "pause_engines"]
    assert engines.paused and monitor.paused and controller.paused
    assert controller.resume() == ["resumed"]
    assert events[-2:] == ["resume_engines", "resume_monitor"]
    assert not monitor.paused and not controller.paused


def test_partial_pause_failure_preserves_recovery_barrier(control):
    controller, engines, monitor, _ = control
    engines.failure = "pause_engines"
    with pytest.raises(RuntimeError, match="failed pause_engines"):
        controller.pause()
    assert controller.paused and monitor.paused
    engines.failure = None
    controller.recover()
    assert engines.paused and monitor.paused


@pytest.mark.parametrize("failure", ["recover_engines", "pause_engines"])
def test_recovery_failure_never_resumes_monitor(control, failure):
    controller, engines, monitor, events = control
    controller.pause()
    events.clear()
    engines.failure = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        controller.recover()
    assert controller.paused and monitor.paused
    assert "resume_monitor" not in events


def test_failure_on_unpaused_recovery_latches_pause_for_retry(control):
    controller, engines, monitor, events = control
    engines.failure = "recover_engines"
    with pytest.raises(RuntimeError, match="failed recover_engines"):
        controller.recover()
    assert controller.paused and monitor.paused
    engines.failure = None
    events.clear()
    controller.recover()
    assert events[-1] == "pause_engines"
    assert engines.paused and monitor.paused


def test_recovery_of_running_engines_resumes_the_monitor(control):
    controller, engines, monitor, events = control
    assert controller.recover() == 0
    assert events == ["pause_monitor", "recover_engines", "resume_monitor"]
    assert not controller.paused and not monitor.paused and not engines.paused


def test_recovery_returns_the_number_of_dead_engines_it_refilled(control):
    controller, engines, _, _ = control
    engines.dead_engines = 2
    assert controller.recover() == 2
    assert controller.recover() == 0


def test_external_engines_are_never_terminated(control):
    controller, engines, monitor, events = control
    engines.owned = False
    assert controller.terminate() == 0
    assert "terminate_engines" not in events
    assert monitor.paused


def test_owned_engine_termination_preserves_pause_for_replacement(control):
    controller, engines, monitor, _ = control
    assert controller.terminate() == 2
    controller.recover()
    assert engines.paused and monitor.paused


def test_failed_resume_keeps_monitor_paused(control):
    controller, engines, monitor, events = control
    controller.pause()
    engines.failure = "resume_engines"
    with pytest.raises(RuntimeError, match="failed resume_engines"):
        controller.resume()
    assert controller.paused and monitor.paused
    assert "resume_monitor" not in events
