"""Process replacement over real Ray with CPU weights and durable job markers."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reef.runtime.executor.ray import RayExecutor
from reef.runtime.inference_control import InferenceControl
from reef.runtime.training_job.marker import read_marker, write_marker
from reef.runtime.training_job.publication import TrainingPublication
from reef.train.slime_backend.reef_adapters.inference import SlimeInferenceControl

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


class RestartEngine:
    def __init__(self):
        self.paused = False
        self.version = "boot"
        self.weight = 0

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def load(self, weight, version):
        if not self.paused:
            raise RuntimeError("transfer requires a paused engine")
        self.weight, self.version = weight, version

    def status(self):
        return {"pid": os.getpid(), "paused": self.paused, "weight": self.weight, "version": self.version}


class AttachedEngine:
    owned = False

    def __init__(self, engine):
        self.engine = engine

    def pause(self):
        import ray

        ray.get(self.engine.pause.remote(), timeout=10)

    def resume(self):
        import ray

        ray.get(self.engine.resume.remote(), timeout=10)

    def recover(self):
        import ray

        ray.get(self.engine.__ray_ready__.remote(), timeout=10)

    def terminate(self):
        raise RuntimeError("test engine belongs to deployment owner")


class RestartMonitor:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class StableConnection:
    def is_usable(self):
        return True

    def replace(self):
        raise RuntimeError("healthy connection must stay attached")


class RestartInference:
    def __init__(self, engine):
        self.engine = engine
        self.monitor = RestartMonitor()
        self.control = InferenceControl(AttachedEngine(engine), StableConnection(), self.monitor)

    def prepare_training_connection(self):
        self.control.prepare_training_connection()

    def pause_generation_for_update(self):
        self.control.pause()

    def continue_generation_after_update(self):
        self.control.resume()

    def get_updatable_engines_and_lock(self):
        return [self.engine], None, int(self.control.reconnect_required), [1], [0], [{}]

    def clear_updatable_num_new_engines(self):
        self.control.acknowledge_reconnect()

    def terminate_updatable_engines(self):
        # Keep the owner's CPU engine available for checking the failure fence.
        self.control.pause()

    def status(self):
        return {
            "paused": self.control.paused,
            "monitor_paused": self.monitor.paused,
            "reconnect": self.control.reconnect_required,
        }


class CheckpointPublisher:
    def __init__(self, control):
        self.control = control

    def pause(self):
        self.control.pause_generation_for_update()

    def resume(self):
        self.control.continue_generation_after_update()

    def abort(self):
        self.control.terminate_updatable_engines()

    def restore(self, marker):
        import ray

        engines, _, reconnect, *_ = self.control.get_updatable_engines_and_lock()
        if not reconnect:
            raise RuntimeError("new trainer was not asked to attach")
        # The fixture represents backend checkpoint I/O. The actual transport
        # uses the borrowed engine handle, never the serving control actor.
        weights = json.loads((Path(marker["checkpoint_path"]) / "weights.json").read_text())
        self.control.clear_updatable_num_new_engines()
        ray.get(engines[0].load.remote(weights["weight"], weights["version"]), timeout=10)
        return ray.get(engines[0].status.remote(), timeout=10)["version"]


class RestartCoordinator:
    def __init__(self, marker_path, serving):
        self.control = SlimeInferenceControl(serving)
        self.control.prepare_training_connection()
        publisher = CheckpointPublisher(self.control)
        self.publication = TrainingPublication(Path(marker_path), publisher)
        marker = read_marker(Path(marker_path))
        with self.publication.recovery(marker):
            version = publisher.restore(marker)
            self.publication.finish_recovery(marker, version)

    def status(self):
        return {"pid": os.getpid(), "phase": self.publication.phase}

    def acknowledge(self):
        self.publication.acknowledge("restart-job")


@pytest.fixture
def cluster(monkeypatch):
    ray = pytest.importorskip("ray")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=3,
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join((str(root), str(root / "tests")))}},
    )
    engine = ray.remote(RestartEngine).remote()
    inference = ray.remote(RestartInference).remote(engine)
    try:
        yield SimpleNamespace(ray=ray, engine=engine, inference=inference)
    finally:
        ray.shutdown()


@pytest.mark.parametrize("status", ["READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"])
def test_new_coordinator_process_reconnects_surviving_engine_and_preserves_commit_gate(cluster, tmp_path, status):
    ray, engine, inference = cluster.ray, cluster.engine, cluster.inference
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.json").write_text(json.dumps({"weight": 7, "version": "checkpoint:1"}))
    path = tmp_path / "job.json"
    write_marker(
        path,
        {
            "status": status,
            "job_id": "restart-job",
            "rollout_id": 0,
            "checkpoint_path": str(checkpoint),
            "runtime_load_id": "checkpoint:1",
            "commit_acknowledged": status != "READY_TO_COMMIT",
        },
    )
    actor_class = ray.remote(RestartCoordinator)
    connection = RayExecutor.from_workers([inference])
    old = actor_class.remote(str(path), connection)
    original = ray.get(old.status.remote(), timeout=30)
    engine_before = ray.get(engine.status.remote(), timeout=10)
    ray.kill(old, no_restart=True)
    new = actor_class.remote(str(path), connection)
    try:
        replacement = ray.get(new.status.remote(), timeout=30)
        assert replacement["pid"] != original["pid"]
        engine_after = ray.get(engine.status.remote(), timeout=10)
        assert engine_after == engine_before
        assert engine_after["weight"] == 7 and engine_after["version"] == "checkpoint:1"
        paused = status == "READY_TO_COMMIT"
        assert ray.get(inference.status.remote(), timeout=10) == {
            "paused": paused,
            "monitor_paused": paused,
            "reconnect": False,
        }
        assert replacement["phase"] == ("awaiting_commit" if paused else "serving")
        if paused:
            assert read_marker(path)["status"] == "READY_TO_COMMIT"
            ray.get(new.acknowledge.remote(), timeout=10)
        assert read_marker(path)["status"] == "COMPLETE"
        assert not ray.get(engine.status.remote(), timeout=10)["paused"]
    finally:
        ray.kill(new, no_restart=True)


def test_new_process_with_invalid_checkpoint_never_reopens_surviving_engine(cluster, tmp_path):
    ray, engine, inference = cluster.ray, cluster.engine, cluster.inference
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.json"
    weights.write_text(json.dumps({"weight": 7, "version": "checkpoint:1"}))
    path = tmp_path / "job.json"
    marker = {
        "status": "COMPLETE",
        "job_id": "restart-job",
        "rollout_id": 0,
        "checkpoint_path": str(checkpoint),
        "runtime_load_id": "checkpoint:1",
    }
    write_marker(path, marker)
    actors = ray.remote(RestartCoordinator)
    connection = RayExecutor.from_workers([inference])
    old = actors.remote(str(path), connection)
    ray.get(old.status.remote(), timeout=30)
    ray.kill(old, no_restart=True)
    weights.write_text("invalid checkpoint")
    failed = actors.remote(str(path), connection)
    try:
        with pytest.raises(ray.exceptions.RayActorError):
            ray.get(failed.status.remote(), timeout=30)
        assert ray.get(engine.status.remote(), timeout=10)["paused"]
        assert ray.get(inference.status.remote(), timeout=10) == {
            "paused": True,
            "monitor_paused": True,
            "reconnect": True,
        }
        assert read_marker(path) == marker
    finally:
        ray.kill(failed, no_restart=True)
