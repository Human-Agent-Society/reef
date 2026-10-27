"""Real Ray control attachment with CPU engines and no model dependencies."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.control import SGLangControl
from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


class CpuEngine:
    def __init__(self):
        self.weight = 0

    def update_weight(self, value):
        self.weight = value

    def current_weight(self):
        return self.weight


class CpuServingWorker:
    def __init__(self):
        import ray

        self.engine = ray.remote(CpuEngine).remote()

    def check_health(self):
        import ray

        ray.get(self.engine.__ray_ready__.remote(), timeout=30)

    def updatable_engines(self):
        return [self.engine]

    def shutdown(self):
        import ray

        ray.kill(self.engine, no_restart=True)


class CpuServingExecutor(DelegatingExecutor):
    def _init_executor(self):
        self._rpc = UniProcExecutor.from_workers([CpuServingWorker()], owned=True)


class BorrowingControl:
    """An actor that borrows the serving control connection, as an inference backend does."""

    def __init__(self, serving):
        self._serving = serving

    def check_health(self):
        self._serving.rpc(0, "check_health", timeout=30)

    def updatable_engines(self):
        return self._serving.rpc(0, "updatable_engines", timeout=30)

    def shutdown(self):
        self._serving.shutdown()


def test_a_borrowed_control_connection_reaches_the_engines_and_leaves_them_to_their_owner(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(root)}},
    )
    owner = None
    borrower = None
    try:
        owner = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                options={"num_cpus": 1, "num_gpus": 0},
                workers=(
                    WorkerSpec(
                        SGLangControl,
                        args=(SGLangConfig("model", 1, 1, 1, executor=CpuServingExecutor), None),
                    ),
                ),
            )
        )
        borrowed = RayExecutor.from_workers(owner.workers)
        borrower = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                options={"num_cpus": 1, "num_gpus": 0},
                workers=(WorkerSpec(BorrowingControl, args=(borrowed,)),),
            )
        )
        borrower.rpc(0, "check_health", timeout=30)
        [engine] = borrower.rpc(0, "updatable_engines", timeout=30)
        # The connection yields actual engine handles: engine calls need no control relay.
        ray.get(engine.update_weight.remote(7), timeout=30)
        assert ray.get(engine.current_weight.remote(), timeout=30) == 7
        borrower.rpc(0, "shutdown", timeout=30)
        borrower.shutdown()
        borrower = None
        assert ray.get(engine.current_weight.remote(), timeout=30) == 7
        owner.rpc(0, "check_health", timeout=30)
        owner.rpc(0, "shutdown", timeout=30)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(engine.current_weight.remote(), timeout=30)
    finally:
        if borrower is not None:
            borrower.shutdown()
        if owner is not None:
            try:
                owner.rpc(0, "shutdown", timeout=30)
            finally:
                owner.shutdown()
        ray.shutdown()


def test_real_ray_probe_timeout_does_not_wait_for_engine_response(monkeypatch):
    from threading import Event

    from reef.inference.sglang.health import SGLangEngineHealthChecks

    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    ray.init(address="local", num_cpus=1, include_dashboard=False)

    @ray.remote(max_concurrency=3)
    class BlockedEngine:
        def __init__(self):
            self.entered = Event()
            self.release = Event()

        def health_generate(self, timeout):
            self.entered.set()
            self.release.wait(60)
            return True

        def probe_started(self):
            return self.entered.wait(5)

        def shutdown(self):
            self.release.set()

    engine = None
    try:
        engine = BlockedEngine.remote()
        ray.get(engine.__ray_ready__.remote(), timeout=30)
        group = SimpleNamespace(all_engines=[engine], nodes_per_engine=1)
        target = SGLangEngineHealthChecks(group).targets()[0]
        # Even an engine ignoring its HTTP timeout cannot hold the monitor's
        # Ray wait forever. The queued probe remains live until retirement.
        with pytest.raises(ray.exceptions.GetTimeoutError):
            target.check(0.1)
        assert ray.get(engine.probe_started.remote(), timeout=10)
        target.retire(5)
        assert group.all_engines == [None]
    finally:
        if engine is not None:
            ray.kill(engine, no_restart=True)
        ray.shutdown()
