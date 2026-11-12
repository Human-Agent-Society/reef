"""Backend-independent inference pause, recovery and termination contracts."""

import pytest

from reef.inference.recovery import InferenceControl, InferenceEngineGroup


class MemoryEngines(InferenceEngineGroup):
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


@pytest.fixture
def control():
    events = []
    engines = MemoryEngines(events)
    return InferenceControl(engines), engines, events


def test_recovery_reapplies_pause_until_commit_resume(control):
    controller, engines, events = control
    assert controller.pause() == ["paused"]
    events.clear()
    controller.recover()
    assert events == ["recover_engines", "pause_engines"]
    assert engines.paused and controller.paused
    assert controller.resume() == ["resumed"]
    assert not engines.paused and not controller.paused


def test_partial_pause_failure_preserves_recovery_barrier(control):
    controller, engines, _ = control
    engines.failure = "pause_engines"
    with pytest.raises(RuntimeError, match="failed pause_engines"):
        controller.pause()
    assert controller.paused
    engines.failure = None
    controller.recover()
    assert engines.paused


@pytest.mark.parametrize("failure", ["recover_engines", "pause_engines"])
def test_recovery_failure_keeps_the_pause(control, failure):
    controller, engines, _ = control
    controller.pause()
    engines.failure = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        controller.recover()
    assert controller.paused


def test_failure_on_unpaused_recovery_latches_pause_for_retry(control):
    controller, engines, events = control
    engines.failure = "recover_engines"
    with pytest.raises(RuntimeError, match="failed recover_engines"):
        controller.recover()
    assert controller.paused
    engines.failure = None
    events.clear()
    controller.recover()
    assert events == ["recover_engines", "pause_engines"]
    assert engines.paused


def test_recovery_of_running_engines_leaves_them_running(control):
    controller, engines, events = control
    assert controller.recover() == 0
    assert events == ["recover_engines"]
    assert not controller.paused and not engines.paused


def test_recovery_returns_the_number_of_dead_engines_it_refilled(control):
    controller, engines, _ = control
    engines.dead_engines = 2
    assert controller.recover() == 2
    assert controller.recover() == 0


def test_engine_termination_preserves_pause_for_replacement(control):
    controller, engines, _ = control
    assert controller.terminate() == 2
    controller.recover()
    assert engines.paused


def test_failed_resume_keeps_the_pause(control):
    controller, engines, _ = control
    controller.pause()
    engines.failure = "resume_engines"
    with pytest.raises(RuntimeError, match="failed resume_engines"):
        controller.resume()
    assert controller.paused
