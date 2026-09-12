"""Real Ray control attachment with CPU engines and no model dependencies."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.runtime.executor import ExecutorConfig, WorkerSpec
from reef.runtime.executor.delegating import DelegatingExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor
from reef.runtime.sglang.config import SGLangConfig
from reef.runtime.sglang.control import SGLangControl

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


class TrainingControl:
    def __init__(self, serving):
        self._serving = serving

    def check_health(self):
        self._serving.rpc(0, "check_health", timeout=30)

    def get_updatable_engines_and_lock(self):
        return self._serving.rpc(0, "get_updatable_engines_and_lock", timeout=30)

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
                        SGLangControl,
                        args=(SGLangConfig("model", 1, 1, 1, executor=CpuServingExecutor), None),
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


def test_reef_driver_owns_real_ray_components_and_training_borrows_engines(monkeypatch):
    from reef.runtime.deployment import ModelDeploymentPlan
    from reef.runtime.sglang.config import SGLangConfig
    from reef.runtime.sglang.service import INFERENCE_PROTOCOL, SGLangInferenceService
    from reef.service.training_driver import ModelDeployment
    from reef.train.slime_backend.resources import SlimeDeploymentResources

    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    original_init = ray.init

    def init(**kwargs):
        return original_init(**kwargs, num_cpus=4, include_dashboard=False)

    monkeypatch.setattr(ray, "init", init)
    root = Path(__file__).resolve().parents[2]
    args = SimpleNamespace(reef_rollout_executor_backend=CpuServingExecutor)

    class CpuReservations(SlimeDeploymentResources):
        def start(self):
            super().start()
            self.placement_groups["rollout"] = (None, [], [])

    class CpuTraining:
        inference_protocol = INFERENCE_PROTOCOL
        actor = None
        engine = None
        inference_survived = False

        def start(self, resources, inference):
            self.actor = RayExecutor(
                ExecutorConfig(
                    backend=RayExecutor,
                    options={"num_cpus": 1, "num_gpus": 0},
                    workers=(WorkerSpec(TrainingControl, args=(inference.control,)),),
                )
            )
            engines, *_ = self.actor.rpc(0, "get_updatable_engines_and_lock", timeout=30)
            self.engine = engines[0]
            ray.get(self.engine.update_weight.remote(7), timeout=30)

        def check_health(self):
            self.actor.rpc(0, "check_health", timeout=30)

        def close(self):
            if self.actor is not None:
                self.actor.rpc(0, "shutdown", timeout=30)
                self.actor.shutdown()
                if self.engine is not None:
                    self.inference_survived = ray.get(self.engine.current_weight.remote(), timeout=30) == 7

    training = CpuTraining()
    owner = ModelDeployment(
        ModelDeploymentPlan(
            CpuReservations(
                args,
                ray_address="local",
                namespace="reef-model-owner-test",
                allocate_models=False,
                runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join((str(root), str(root / "tests")))}},
            ),
            SGLangInferenceService(SGLangConfig("model", 1, 1, 1, executor=args.reef_rollout_executor_backend)),
            training,
        )
    )
    try:
        owner.start()
        assert ray.get(training.engine.current_weight.remote(), timeout=30) == 7
    finally:
        owner.close()
    assert training.inference_survived is True
    assert not ray.is_initialized()


def test_real_ray_update_lock_replacement_forces_trainer_reconnect(monkeypatch):
    from reef.runtime.inference_control import InferenceControl
    from reef.runtime.sglang.lock import ReefRolloutLock

    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": str(root)}},
    )
    locks = []
    events = []

    class Engines:
        owned = True

        def pause(self):
            events.append("pause_engines")

        def resume(self):
            events.append("resume_engines")

        def recover(self):
            events.append("recover_engines")

        def terminate(self):
            return 0

    class Monitor:
        def pause(self):
            events.append("pause_monitor")

        def resume(self):
            events.append("resume_monitor")

    class Connection:
        def __init__(self):
            self.actor = None
            self.replace()

        def is_usable(self):
            return ray.get(self.actor.status.remote(), timeout=30) == {"locked": False, "poisoned": False}

        def replace(self):
            old = self.actor
            self.actor = ReefRolloutLock.options(num_cpus=0).remote()
            locks.append(self.actor)
            if old is not None:
                ray.kill(old, no_restart=True)

    try:
        connection = Connection()
        old_lock = connection.actor
        assert ray.get(old_lock.acquire.remote(), timeout=30)
        ray.get(old_lock.complete_phase.remote("weights", "failed fan-out"), timeout=30)
        ray.get(old_lock.poison.remote(), timeout=30)
        with pytest.raises(ray.exceptions.RayTaskError, match="poisoned"):
            ray.get(old_lock.acquire.remote(), timeout=30)
        owner = InferenceControl(Engines(), connection, Monitor())
        owner.pause()
        events.clear()
        owner.recover()
        assert owner.reconnect_required
        assert events == ["pause_monitor", "recover_engines", "pause_engines"]
        # Ray kill is asynchronous; until it completes the poisoned actor must
        # still refuse acquisition. Either outcome fences the old transport.
        with pytest.raises((ray.exceptions.RayActorError, ray.exceptions.RayTaskError)):
            ray.get(old_lock.acquire.remote(), timeout=30)
        assert ray.get(connection.actor.acquire.remote(), timeout=30)
        ray.get(connection.actor.release.remote(), timeout=30)
        owner.acknowledge_reconnect()
        assert not owner.reconnect_required
        owner.resume()
        assert events[-2:] == ["resume_engines", "resume_monitor"]
    finally:
        for actor in locks:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


def test_real_ray_probe_timeout_does_not_wait_for_engine_response(monkeypatch):
    from threading import Event

    from reef.runtime.sglang.health import SGLangEngineHealthChecks

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
