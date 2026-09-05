from __future__ import annotations

import pickle
import subprocess
import sys
import threading
from concurrent.futures import CancelledError, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec, resolve
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import ConcurrentExecutorFuture, UniProcExecutor

pytestmark = pytest.mark.unit


def test_concurrent_future_normalizes_only_wait_timeouts():
    pending = Future()
    wrapped = ConcurrentExecutorFuture([pending], single=True)
    with pytest.raises(TimeoutError, match="executor RPC timed out"):
        wrapped.result(timeout=0)
    assert not pending.cancelled()
    error = FutureTimeoutError("worker deadline")
    pending.set_exception(error)
    with pytest.raises(FutureTimeoutError) as caught:
        wrapped.result(timeout=0)
    assert caught.value is error


def test_concurrent_future_preserves_cancellation():
    pending = Future()
    pending.cancel()
    with pytest.raises(CancelledError):
        ConcurrentExecutorFuture([pending]).result(timeout=0)


class Worker:
    def __init__(self, rank=0, *, offset=0):
        self.rank = rank
        self.offset = offset
        self.stopped = 0
        self.calls = 0

    def compute(self, value, *, scale=1):
        self.calls += 1
        return (self.rank, (value + self.offset) * scale)

    def fail(self):
        self.calls += 1
        raise ValueError(f"rank {self.rank} failed")

    def shutdown(self):
        self.stopped += 1


class CustomExecutor(UniProcExecutor):
    pass


class RendezvousWorker(Worker):
    def __init__(self, rank, barrier):
        super().__init__(rank)
        self.barrier = barrier

    def rendezvous(self):
        self.barrier.wait(timeout=2)
        return self.rank


class WaitingWorker(Worker):
    def __init__(self, event):
        super().__init__()
        self.event = event

    def wait(self):
        self.calls += 1
        self.event.wait(timeout=2)
        return "done"


@dataclass
class FakeRef:
    value: object = None
    error: Exception | None = None


class FakeTimeout(Exception):
    pass


class FakeActorDeath(Exception):
    pass


class FakeMethod:
    def __init__(self, actor, method):
        self.actor = actor
        self.method = method

    def remote(self, *args, **kwargs):
        self.actor.ray.events.append(("submit", self.actor.worker.rank, self.method))
        if self.method == "__ray_ready__":
            return FakeRef(error=self.actor.init_error)
        try:
            return FakeRef(getattr(self.actor.worker, self.method)(*args, **kwargs))
        except Exception as exc:
            return FakeRef(error=exc)


class FakeActor:
    def __init__(self, ray, worker, init_error=None):
        self.ray = ray
        self.worker = worker
        self.init_error = init_error

    def __getattr__(self, method):
        if method.startswith("__") and method != "__ray_ready__":
            raise AttributeError(method)
        if method != "__ray_ready__":
            getattr(self.worker, method)
        return FakeMethod(self, method)


class FakeActorBuilder:
    def __init__(self, ray, worker_cls):
        self.ray = ray
        self.worker_cls = worker_cls

    def options(self, **options):
        self.ray.options.append(options)
        return self

    def remote(self, *args, **kwargs):
        actor = FakeActor(self.ray, self.worker_cls(*args, **kwargs))
        self.ray.created.append(actor)
        return actor


class FakeRay:
    exceptions = SimpleNamespace(GetTimeoutError=FakeTimeout, RayActorError=FakeActorDeath)

    def __init__(self):
        self.events = []
        self.created = []
        self.killed = []
        self.options = []

    def remote(self, worker_cls):
        return FakeActorBuilder(self, worker_cls)

    def is_initialized(self):
        return True

    def get_runtime_context(self):
        return SimpleNamespace(gcs_address="127.0.0.1:6379")

    def get(self, refs, *, timeout=None):
        self.events.append(("wait", timeout))
        if isinstance(refs, list):
            return [self._value(ref) for ref in refs]
        return self._value(refs)

    @staticmethod
    def _value(ref):
        if ref.error is not None:
            raise ref.error
        return ref.value

    def kill(self, worker, *, no_restart):
        assert no_restart is True
        self.killed.append(worker)


@pytest.fixture
def fake_ray(monkeypatch):
    ray = FakeRay()
    monkeypatch.setattr("reef.runtime.executor.ray._require_ray", lambda: ray)
    monkeypatch.setattr("reef.runtime.executor.ray_runtime._require_ray", lambda: ray)
    # This synchronous transport fake has no liveness service. Real monitoring
    # is exercised by the opt-in Ray integration tests.
    monkeypatch.setattr(RayExecutor, "_start_monitor", lambda self: None)
    return ray


def test_factory_auto_launches_multiple_process_workers_and_dispatches_ranked_rpc():
    executor = Executor.create(
        ExecutorConfig(
            backend="auto",
            workers=(WorkerSpec(Worker, (0,)), WorkerSpec(Worker, (1,), {"offset": 3})),
        )
    )
    try:
        executor.check_health()
        assert executor.collective_rpc("compute", args=(2,), kwargs={"scale": 4}) == [(0, 8), (1, 20)]
        assert executor.rpc(1, "compute", args=(0,)) == (1, 3)
        with pytest.raises(NotImplementedError, match="distributed reinitialization"):
            executor.reinitialize_distributed({"world_size": 1})
    finally:
        executor.shutdown()


@pytest.mark.parametrize("backend", [CustomExecutor, f"{__name__}:CustomExecutor", f"{__name__}.CustomExecutor"])
def test_factory_accepts_custom_executor_class_and_import_paths(backend):
    executor = Executor.create(ExecutorConfig(backend=backend))
    assert isinstance(executor, CustomExecutor)
    executor.shutdown()


@pytest.mark.parametrize(
    ("backend", "error"),
    [(int, TypeError), ("builtins:int", TypeError), ("missing", ValueError), (None, TypeError), ("", TypeError)],
)
def test_factory_rejects_invalid_backends(backend, error):
    with pytest.raises(error):
        Executor.create(ExecutorConfig(backend=backend))


@pytest.mark.parametrize(
    "fields",
    [{"worker_cls": None}, {"args": "bad"}, {"kwargs": None}, {"options": 1}],
)
def test_worker_spec_rejects_malformed_config(fields):
    with pytest.raises(TypeError):
        WorkerSpec(**{"worker_cls": Worker, **fields})


@pytest.mark.parametrize("fields", [{"options": None}, {"workers": "bad"}, {"workers": [{}]}])
def test_executor_config_rejects_malformed_config(fields):
    with pytest.raises(TypeError):
        ExecutorConfig(**fields)


def test_configs_snapshot_launch_arguments():
    args = [1]
    kwargs = {"offset": 2}
    options = {"num_cpus": 1}
    spec = WorkerSpec(Worker, args=args, kwargs=kwargs, options=options)
    workers = [spec]
    config = ExecutorConfig(workers=workers, options=options)
    args.append(3)
    kwargs["offset"] = 10
    options["num_cpus"] = 100
    workers.clear()
    assert spec.args == (1,)
    assert spec.kwargs == {"offset": 2}
    assert spec.options == {"num_cpus": 1}
    assert config.workers == (spec,)
    assert config.options == {"num_cpus": 1}


def test_local_executor_never_imports_ray():
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'ray' or name.startswith('ray.'):
        raise AssertionError('local executor imported Ray')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
class Worker:
    def value(self):
        return 42
executor = Executor.create(ExecutorConfig(backend='uni', workers=(WorkerSpec(Worker),)))
assert executor.collective_rpc('value') == [42]
executor.shutdown()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_uni_rejects_multiple_attached_workers():
    with pytest.raises(ValueError, match="at most one worker"):
        UniProcExecutor.from_workers([Worker(), Worker()])


def test_local_timeout_does_not_retry_or_cancel_worker():
    event = threading.Event()
    worker = WaitingWorker(event)
    executor = UniProcExecutor.from_workers([worker])
    try:
        future = executor.rpc(0, "wait", non_block=True)
        assert isinstance(future, ExecutorFuture)
        with pytest.raises(TimeoutError):
            future.result(timeout=0.01)
        event.set()
        assert future.result(timeout=2) == "done"
        assert worker.calls == 1
        with pytest.raises(ValueError, match="rank 0 failed"):
            resolve(executor.rpc(0, "fail", non_block=True))
        assert worker.calls == 2
    finally:
        event.set()
        executor.shutdown()


def test_resolve_preserves_values_shapes_and_shares_timeout_deadline(monkeypatch):
    timestamps = iter([5.0, 5.25, 5.75])
    monkeypatch.setattr("reef.runtime.executor.base.monotonic", lambda: next(timestamps))
    timeouts = []

    class CapturedFuture(ExecutorFuture):
        def result(self, timeout=None):
            timeouts.append(timeout)
            return "resolved"

    opaque = FakeRef("unresolved")
    payload = {"opaque": CapturedFuture()}
    assert resolve([CapturedFuture(), (CapturedFuture(), opaque), payload], timeout=1) == [
        "resolved",
        ("resolved", opaque),
        payload,
    ]
    assert timeouts == [0.75, 0.25]
    assert resolve(opaque) is opaque


@pytest.mark.parametrize("owned", [False, True])
def test_local_attachment_ownership_and_shutdown_are_explicit(owned):
    worker = Worker()
    executor = UniProcExecutor.from_workers([worker], owned=owned)
    executor.shutdown()
    executor.shutdown()
    assert worker.stopped == int(owned)
    with pytest.raises(RuntimeError, match="shut down"):
        executor.check_health()
    with pytest.raises(RuntimeError, match="shut down"):
        executor.rpc(0, "compute", args=(0,))


def test_uni_rejects_multiple_specs_before_launch():
    with pytest.raises(ValueError, match="at most one worker"):
        Executor.create(ExecutorConfig(backend="uni", workers=(WorkerSpec(Worker), WorkerSpec(Worker))))


def test_local_rejects_gpu_resource_options():
    with pytest.raises(ValueError, match="resource or backend options"):
        Executor.create(ExecutorConfig(backend="uni", workers=(WorkerSpec(Worker, options={"num_gpus": 1}),)))


def test_ray_launch_passes_resource_options_and_collective_fans_out(fake_ray):
    executor = Executor.create(
        ExecutorConfig(
            workers=(WorkerSpec(Worker, (0,)), WorkerSpec(f"{__name__}:Worker", (1,), options={"num_gpus": 2})),
            options={"num_cpus": 2, "num_gpus": 1},
        )
    )
    assert fake_ray.options == [
        {"num_cpus": 2, "num_gpus": 1, "max_restarts": 0, "max_task_retries": 0},
        {"num_cpus": 2, "num_gpus": 2, "max_restarts": 0, "max_task_retries": 0},
    ]
    fake_ray.events.clear()
    future = executor.collective_rpc("compute", args=(4,), non_block=True)
    assert fake_ray.events == [("submit", 0, "compute"), ("submit", 1, "compute")]
    assert resolve(future, timeout=5) == [(0, 4), (1, 4)]
    assert fake_ray.events[-1][0] == "wait"
    assert executor.rpc(1, "compute", args=(3,), kwargs={"scale": 2}) == (1, 6)
    executor.shutdown()
    assert fake_ray.killed == list(executor.workers)


@pytest.mark.parametrize("owned", [False, True])
def test_ray_attachment_preserves_ownership_and_is_pickleable(fake_ray, owned):
    actors = [FakeActor(fake_ray, Worker(0)), FakeActor(fake_ray, Worker(1))]
    executor = RayExecutor.from_workers(actors, owned=owned)
    restored = pickle.loads(pickle.dumps(executor))
    assert len(restored.workers) == 2
    executor.check_health()
    executor.shutdown()
    executor.shutdown()
    assert fake_ray.killed == (actors if owned else [])
    with pytest.raises(RuntimeError, match="shut down"):
        executor.collective_rpc("compute", args=(0,))


def test_ray_launch_rolls_back_partial_allocation(fake_ray):
    class BrokenWorker:
        def __init__(self):
            raise ValueError("launch failed")

    with pytest.raises(ValueError, match="launch failed"):
        Executor.create(ExecutorConfig(backend="ray", workers=(WorkerSpec(Worker), WorkerSpec(BrokenWorker))))
    assert len(fake_ray.created) == 1
    assert fake_ray.killed == fake_ray.created


def test_ray_launch_rolls_back_async_constructor_failure(fake_ray, monkeypatch):
    original_remote = FakeActorBuilder.remote

    def fail_ready(self, *args, **kwargs):
        actor = original_remote(self, *args, **kwargs)
        actor.init_error = ValueError("async constructor failed")
        return actor

    monkeypatch.setattr(FakeActorBuilder, "remote", fail_ready)
    with pytest.raises(ValueError, match="async constructor failed"):
        Executor.create(ExecutorConfig(backend="ray", workers=(WorkerSpec(Worker), WorkerSpec(Worker))))
    assert len(fake_ray.created) == 2
    assert fake_ray.killed == fake_ray.created


def test_ray_errors_and_timeouts_are_exposed_without_retry(fake_ray):
    actor = FakeActor(fake_ray, Worker(7))
    executor = RayExecutor.from_workers([actor])
    with pytest.raises(ValueError, match="rank 7 failed"):
        executor.rpc(0, "fail")
    assert actor.worker.calls == 1
    with pytest.raises(AttributeError):
        executor.rpc(0, "missing")
    actor.init_error = FakeTimeout()
    with pytest.raises(TimeoutError, match="may still be running"):
        executor.check_health(timeout=0)
    executor.shutdown()


def test_ray_failure_polling_preserves_total_timeout_budget(fake_ray, monkeypatch):
    now = [0.0]
    waits = []

    def get(refs, *, timeout):
        waits.append(timeout)
        now[0] += timeout
        raise FakeTimeout

    monkeypatch.setattr("reef.runtime.executor.ray.monotonic", lambda: now[0])
    monkeypatch.setattr(fake_ray, "get", get)
    actor = FakeActor(fake_ray, Worker(0))
    executor = RayExecutor.from_workers([actor])
    try:
        with pytest.raises(TimeoutError, match="may still be running"):
            executor.rpc(0, "compute", args=(2,), timeout=0.25)
        assert waits == [0.1, 0.1, pytest.approx(0.05)]
        assert sum(event[0] == "submit" for event in fake_ray.events) == 1
        assert executor.failure is None
    finally:
        executor.shutdown()


def test_ray_does_not_confuse_business_timeout_with_get_timeout(fake_ray):
    actor = FakeActor(fake_ray, Worker(0), init_error=TimeoutError("worker method timed out"))
    executor = RayExecutor.from_workers([actor])
    try:
        with pytest.raises(TimeoutError, match="worker method timed out"):
            executor.check_health(timeout=1)
        assert executor.failure is None
        assert sum(event[0] == "wait" for event in fake_ray.events) == 1
    finally:
        executor.shutdown()


def test_ray_rpc_observed_death_records_failure_without_waiting_for_monitor(fake_ray):
    from reef.runtime.executor import ExecutorFailedError

    actor = FakeActor(fake_ray, Worker(0), init_error=FakeActorDeath())
    executor = RayExecutor.from_workers([actor])
    try:
        with pytest.raises(ExecutorFailedError):
            executor.rpc(0, "__ray_ready__")
        assert executor.failure.rank == 0
        with pytest.raises(ExecutorFailedError):
            executor.rpc(0, "compute", args=(1,))
    finally:
        executor.shutdown()


def test_ray_rejects_automatic_mutation_retries(fake_ray):
    with pytest.raises(ValueError, match="max_task_retries=0"):
        Executor.create(ExecutorConfig(workers=(WorkerSpec(Worker),), options={"max_task_retries": 2}))
    assert fake_ray.created == []


@pytest.mark.parametrize("executor_cls", [UniProcExecutor, RayExecutor])
@pytest.mark.parametrize("rank", [-1, 1, True, "0"])
def test_rpc_rejects_invalid_rank(executor_cls, rank):
    executor = executor_cls.from_workers([Worker()])
    try:
        with pytest.raises(IndexError, match="outside the executor group"):
            executor.rpc(rank, "compute", args=(0,))
    finally:
        executor.shutdown()
