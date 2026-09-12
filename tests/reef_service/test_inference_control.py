"""Real Ray control attachment with CPU engines and no model dependencies."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.train.slime_backend.reef_adapters.inference import SlimeInferenceControl, SlimeInferenceWorker

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

    def get_updatable_engines_and_lock(self):
        return [self.engine], None, 0, [1], [0], [{}]

    def shutdown(self):
        import ray

        ray.kill(self.engine, no_restart=True)


class CpuServingExecutor(DelegatingExecutor):
    def _init_executor(self):
        self._rpc = UniProcExecutor.from_workers([CpuServingWorker()], owned=True)


class TrainingControl(SlimeInferenceControl):
    def shutdown(self):
        self._serving.shutdown()


def test_remote_training_borrows_inference_and_transfers_directly_to_engine(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join((str(root), str(root / "tests")))}},
    )
    owner = None
    training = None
    try:
        owner = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                options={"num_cpus": 1, "num_gpus": 0},
                workers=(
                    WorkerSpec(
                        SlimeInferenceWorker,
                        args=(SimpleNamespace(reef_rollout_executor_backend=CpuServingExecutor), None),
                    ),
                ),
            )
        )
        borrowed = RayExecutor.from_workers(owner.workers)
        training = RayExecutor(
            ExecutorConfig(
                backend=RayExecutor,
                options={"num_cpus": 1, "num_gpus": 0},
                workers=(WorkerSpec(TrainingControl, args=(borrowed,)),),
            )
        )
        training.rpc(0, "check_health", timeout=30)
        engines, *_ = training.rpc(0, "get_updatable_engines_and_lock", timeout=30)
        engine = engines[0]
        # The connection yields actual engine handles: tensors need no control relay.
        ray.get(engine.update_weight.remote(7), timeout=30)
        assert ray.get(engine.current_weight.remote(), timeout=30) == 7
        training.rpc(0, "shutdown", timeout=30)
        training.shutdown()
        assert ray.get(engine.current_weight.remote(), timeout=30) == 7
        owner.rpc(0, "check_health", timeout=30)
        owner.rpc(0, "shutdown", timeout=30)
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(engine.current_weight.remote(), timeout=30)
    finally:
        if training is not None:
            training.shutdown()
        if owner is not None:
            try:
                owner.rpc(0, "shutdown", timeout=30)
            finally:
                owner.shutdown()
        ray.shutdown()
