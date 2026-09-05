import pickle
import threading
from concurrent.futures import Future

import pytest

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFailedError, ExecutorFailure, WorkerSpec
from reef.runtime.executor.failure import FailureState


class Listener:
    def __init__(self):
        self.events = []

    def on_executor_failure(self, failure):
        self.events.append(failure)


def test_failure_is_terminal_notifies_once_and_fails_pending_requests(caplog):
    class BadListener:
        def on_executor_failure(self, failure):
            raise ValueError("bad observer")

    state = FailureState()
    listener = Listener()
    state.register(BadListener())
    state.register(listener)
    state.register(listener)
    source = Future()
    pending = state.track(source)
    failure = ExecutorFailure("test", "worker died", 2)
    assert state.fail(failure)
    assert not state.fail(ExecutorFailure("test", "second failure"))
    assert listener.events == [failure]
    assert "bad observer" in caplog.text
    with pytest.raises(ExecutorFailedError) as raised:
        pending.result(timeout=0)
    assert raised.value.failure == failure
    assert pickle.loads(pickle.dumps(raised.value)).failure == failure
    source.set_result("too late")
    state.close()
    with pytest.raises(ExecutorFailedError):
        state.check()
    late = Listener()
    state.register(late)
    assert late.events == [failure]


def test_normal_shutdown_and_business_error_do_not_report_worker_failure():
    state = FailureState()
    listener = Listener()
    state.register(listener)
    source = Future()
    pending = state.track(source)
    source.set_exception(ValueError("scorer failed"))
    with pytest.raises(ValueError):
        pending.result()
    state.check()
    unfinished = state.track(Future())
    state.close()
    with pytest.raises(RuntimeError, match="shut down"):
        unfinished.result(timeout=0)
    assert not state.fail(ExecutorFailure("test", "normal teardown"))
    assert state.failure is None
    assert listener.events == []


def test_pickled_failure_state_does_not_copy_process_local_listeners():
    state = FailureState()
    listener = Listener()
    state.register(listener)
    restored = pickle.loads(pickle.dumps(state))
    restored.fail(ExecutorFailure("test", "failed elsewhere"))
    assert listener.events == []
    assert state.failure is None


def test_local_failure_unblocks_pending_and_rejects_queued_work():
    release = threading.Event()
    started = threading.Event()

    class Worker:
        def __init__(self):
            self.calls = 0

        def run(self):
            self.calls += 1
            started.set()
            release.wait(5)

    executor = Executor.create(ExecutorConfig(backend="uni", workers=(WorkerSpec(Worker),)))
    listener = Listener()
    executor.register_failure_listener(listener)
    try:
        running = executor.rpc(0, "run", non_block=True)
        assert started.wait(2)
        queued = executor.rpc(0, "run", non_block=True)
        executor._fail("runtime reported failure")
        for future in (running, queued):
            with pytest.raises(ExecutorFailedError):
                future.result(timeout=0)
        with pytest.raises(ExecutorFailedError):
            executor.rpc(0, "run")
        assert len(listener.events) == 1
    finally:
        release.set()
        executor.shutdown()
    assert executor.workers[0].calls == 1
