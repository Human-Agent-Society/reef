"""Backend-independent inference recovery and update-lock contracts."""

import pytest

from reef.runtime.inference_control import InferenceControl
from reef.runtime.weight_update import WeightUpdateLock


class MemoryEngines:
    owned = True

    def __init__(self, events):
        self.events = events
        self.failure = None
        self.paused = False

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

    def terminate(self):
        self.event("terminate_engines")
        return 2


class MemoryConnection:
    def __init__(self, events):
        self.events = events
        self.usable = True
        self.failure = None

    def is_usable(self):
        self.events.append("lock_status")
        if self.failure == "status":
            raise RuntimeError("lock unavailable")
        return self.usable

    def replace(self):
        self.events.append("replace_lock")
        if self.failure == "replace":
            raise RuntimeError("failed replace")
        self.usable = True


class MemoryMonitor:
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
    connection = MemoryConnection(events)
    monitor = MemoryMonitor(events)
    return InferenceControl(engines, connection, monitor), engines, connection, monitor, events


def test_recovery_reapplies_pause_and_keeps_monitor_stopped_until_commit_resume(control):
    controller, engines, _, monitor, events = control
    assert controller.pause() == ["paused"]
    events.clear()
    controller.recover()
    assert events == ["pause_monitor", "lock_status", "recover_engines", "pause_engines"]
    assert engines.paused and monitor.paused and controller.paused
    assert controller.resume() == ["resumed"]
    assert events[-2:] == ["resume_engines", "resume_monitor"]
    assert not monitor.paused and not controller.paused


def test_partial_pause_failure_preserves_recovery_barrier(control):
    controller, engines, _, monitor, _ = control
    engines.failure = "pause_engines"
    with pytest.raises(RuntimeError, match="failed pause_engines"):
        controller.pause()
    assert controller.paused and monitor.paused
    engines.failure = None
    controller.recover()
    assert engines.paused and monitor.paused


@pytest.mark.parametrize("failure", ["recover_engines", "pause_engines"])
def test_recovery_failure_never_resumes_monitor(control, failure):
    controller, engines, _, monitor, events = control
    controller.pause()
    events.clear()
    engines.failure = failure
    with pytest.raises(RuntimeError, match=f"failed {failure}"):
        controller.recover()
    assert controller.paused and monitor.paused
    assert "resume_monitor" not in events


def test_failure_on_unpaused_recovery_latches_pause_for_retry(control):
    controller, engines, _, monitor, events = control
    engines.failure = "recover_engines"
    with pytest.raises(RuntimeError, match="failed recover_engines"):
        controller.recover()
    assert controller.paused and monitor.paused
    engines.failure = None
    events.clear()
    controller.recover()
    assert events[-1] == "pause_engines"
    assert engines.paused and monitor.paused


@pytest.mark.parametrize("unknown_status", [False, True])
def test_uncertain_lock_forces_reconnect_until_worker_acknowledgement(control, unknown_status):
    controller, _, connection, _, events = control
    connection.usable = False
    if unknown_status:
        connection.failure = "status"
    controller.recover()
    assert events == ["pause_monitor", "lock_status", "replace_lock", "recover_engines", "resume_monitor"]
    assert controller.reconnect_required
    connection.failure = None
    events.clear()
    controller.recover()
    assert "replace_lock" not in events
    assert controller.reconnect_required
    controller.acknowledge_reconnect()
    assert not controller.reconnect_required


def test_failed_replacement_keeps_old_connection_fenced(control):
    controller, _, connection, monitor, events = control
    connection.usable = False
    connection.failure = "replace"
    with pytest.raises(RuntimeError, match="failed replace"):
        controller.recover()
    assert controller.paused and monitor.paused
    assert not controller.reconnect_required
    assert "recover_engines" not in events


def test_external_engines_are_never_replaced_or_terminated(control):
    controller, engines, connection, monitor, events = control
    engines.owned = False
    connection.usable = False
    with pytest.raises(RuntimeError, match="restarting the external deployment"):
        controller.recover()
    assert "replace_lock" not in events
    assert "recover_engines" not in events
    assert controller.terminate() == 0
    assert "terminate_engines" not in events
    assert monitor.paused


def test_owned_engine_termination_preserves_pause_for_replacement(control):
    controller, engines, _, monitor, _ = control
    assert controller.terminate() == 2
    controller.recover()
    assert engines.paused and monitor.paused


def test_failed_resume_keeps_monitor_paused(control):
    controller, engines, _, monitor, events = control
    controller.pause()
    engines.failure = "resume_engines"
    with pytest.raises(RuntimeError, match="failed resume_engines"):
        controller.resume()
    assert controller.paused and monitor.paused
    assert "resume_monitor" not in events


def test_weight_update_lock_fences_poisoned_transport_and_preserves_phase_errors():
    lock = WeightUpdateLock()
    assert lock.acquire()
    assert not lock.acquire()
    lock.complete_phase("send", "connection lost")
    assert lock.phase_status("send") == {"error": "connection lost"}
    assert lock.phase_status("missing") is None
    lock.poison()
    assert lock.status() == {"locked": True, "poisoned": True}
    for operation in (lock.acquire, lock.release, lock.clear_phases):
        with pytest.raises(RuntimeError):
            operation()
    assert lock.phase_status("send") == {"error": "connection lost"}


def test_weight_update_lock_can_clear_finished_phases_only_when_idle():
    lock = WeightUpdateLock()
    with pytest.raises(RuntimeError, match="not acquired"):
        lock.release()
    lock.acquire()
    lock.complete_phase("send")
    with pytest.raises(RuntimeError, match="active"):
        lock.clear_phases()
    lock.release()
    lock.clear_phases()
    assert lock.phase_status("send") is None
    assert lock.acquire()


@pytest.mark.parametrize("phase,error", [("", None), (None, None), ("send", ""), ("send", 5)])
def test_weight_update_lock_rejects_invalid_phase_records(phase, error):
    with pytest.raises(ValueError):
        WeightUpdateLock().complete_phase(phase, error)


@pytest.mark.parametrize("committed", [False, True])
def test_republication_restores_engine_and_monitor_pause_through_commit_gate(control, tmp_path, committed):
    from reef.runtime.training_job.marker import read_marker, write_marker
    from reef.runtime.training_job.publication import TrainingPublication

    controller, engines, connection, monitor, events = control
    path = tmp_path / "job.json"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    write_marker(
        path,
        {
            "status": "COMPLETE" if committed else "READY_TO_COMMIT",
            "job_id": "job",
            "rollout_id": 0,
            "checkpoint_path": str(checkpoint),
            "runtime_load_id": "engine:1",
        },
    )
    # The bridge may still remember a pause from a previous controller. The
    # new owner starts unpaused and needs to observe the barrier again.
    connection.usable = False

    class Publisher:
        def pause(self):
            controller.pause()

        def recover(self, marker):
            controller.recover()

        def republish(self, runtime_load_id, marker):
            assert engines.paused and monitor.paused and controller.paused
            assert controller.reconnect_required
            events.append("transfer")
            controller.acknowledge_reconnect()
            return runtime_load_id

        def resume(self):
            assert read_marker(path)["status"] in {"HEAD_COMMITTED", "COMPLETE"}
            controller.resume()

        def abort(self):
            controller.terminate()

    publication = TrainingPublication(path, Publisher())
    assert publication.republish("engine:1") == "engine:1"
    assert events[:6] == [
        "pause_monitor",
        "pause_engines",
        "pause_monitor",
        "lock_status",
        "replace_lock",
        "recover_engines",
    ]
    assert events[6:8] == ["pause_engines", "transfer"]
    if not committed:
        assert controller.paused and engines.paused and monitor.paused
        assert "resume_monitor" not in events
        publication.acknowledge("job")
    assert not controller.paused and not engines.paused and not monitor.paused
    assert events[-2:] == ["resume_engines", "resume_monitor"]
