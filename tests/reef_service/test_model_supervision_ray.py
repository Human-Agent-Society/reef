"""Real Ray, process leases, HTTP reconnection and durable publication recovery."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from reef.runtime.recovery import FileTrainingJobStore
from reef.train.algos import StepScheduling

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from reef.inference.sglang.service import RayHealthProbe
from reef.runtime.deployment import (
    ComponentHealth,
    DeploymentResources,
    InferenceConnection,
    InferenceService,
    ModelDeploymentPlan,
    ModelPlanSource,
    TrainingService,
)
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import InferenceBackend, TrainingBackend
from reef.runtime.publication import TrainingPublication, WeightPublisher
from reef.runtime.recovery import read_marker, write_marker
from reef.service.runtime import connect_ray_runtime
from reef.service.training_driver import run_deployment
from reef.train.runtime_backend import RuntimeCandidateBackend
from reef.train.slime_backend.resources import SlimeDeploymentResources

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


def wait_until(predicate, *, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("condition did not become true")


def write_state(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    os.replace(temporary, path)


class ModelController:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.state = {"paused": True, "version": "boot"}
        self.state_path = self.directory / "serving.json"
        write_state(self.state_path, self.state)
        endpoint = self.directory / "endpoint.json"
        endpoint.unlink(missing_ok=True)
        self.child = subprocess.Popen(
            [sys.executable, "-m", "reef_service.test_model_supervision_ray", "engine", str(self.directory)]
        )
        self.endpoint = wait_until(lambda: json.loads(endpoint.read_text()) if endpoint.exists() else None)

    def health(self):
        if self.child.poll() is not None:
            raise RuntimeError("engine process died")
        return {**self.endpoint, "controller_pid": os.getpid()}

    def pause(self):
        self.state["paused"] = True
        write_state(self.state_path, self.state)

    def resume(self):
        self.state["paused"] = False
        write_state(self.state_path, self.state)

    def load(self, version):
        assert self.state["paused"]
        self.state["version"] = version
        write_state(self.state_path, self.state)


class Publisher(WeightPublisher):
    def __init__(self, controller):
        self.controller = controller

    def pause(self):
        import ray

        ray.get(self.controller.pause.remote())

    def resume(self):
        import ray

        ray.get(self.controller.resume.remote())

    def abort(self):
        self.pause()

    def recover(self, marker):
        raise AssertionError("startup restores its checkpoint explicitly")

    def publish(self, marker, *, force_full):
        raise AssertionError("startup must not publish a new candidate")

    def republish(self, runtime_load_id, marker):
        raise AssertionError("startup restores its checkpoint explicitly")

    def restore_incumbent(self):
        raise AssertionError("startup must not reject a candidate")


class Coordinator:
    def __init__(self, directory, controller):
        import ray

        self.path = Path(directory) / "job.json"
        self.endpoint = ray.get(controller.health.remote())["url"]
        self.publication = TrainingPublication(FileTrainingJobStore(self.path), Publisher(controller))
        marker = read_marker(self.path)
        with self.publication.recovery(marker):
            self.version = json.loads((Path(directory) / "checkpoint.json").read_text())["version"]
            ray.get(controller.load.remote(self.version))
            self.publication.finish_recovery(marker, self.version)

    def health(self):
        marker = read_marker(self.path)
        return {
            "ok": True,
            "phase": self.publication.phase,
            "inference_url": self.endpoint,
            "pid": os.getpid(),
            "training_job": {**marker, "deferred_weight_update": True, "training_job_id": marker["job_id"]},
        }

    def serving_runtime_load_id(self):
        return self.version

    def acknowledge_training_commit(self, job_id):
        self.publication.acknowledge(job_id)


class Inference(InferenceService):
    connection_protocol = "cpu-test-v1"

    def __init__(self, directory, namespace):
        self.directory, self.namespace = directory, namespace
        self.actor = None
        self.probe = RayHealthProbe()

    def start(self, resources):
        import ray

        self.actor = (
            ray.remote(num_cpus=0, max_restarts=0)(ModelController)
            .options(name="inference", namespace=self.namespace)
            .remote(str(self.directory))
        )
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers([self.actor]))

    def prepare_weight_transfer(self, connection):
        connection.control.rpc(0, "pause", timeout=30)

    def check_health(self):
        import ray

        ray.get(self.actor.health.remote(), timeout=30)

    def poll(self):
        self.probe.poll(self.actor, "health")

    def close(self):
        import ray

        if self.actor is not None:
            # Deliberately skip child cleanup: the process lease must complete it.
            ray.kill(self.actor, no_restart=True)

    def backend(self, connection):
        raise AssertionError("unexpected backend in this fixture")


class Training(TrainingService):
    weight_transfer_protocol = "cpu-test-v1"

    def __init__(self, directory, namespace):
        self.directory, self.namespace = directory, namespace
        self.actor = None
        self.probe = RayHealthProbe()

    def start(self, resources):
        pass

    def attach_weight_transport(self, session):
        import ray

        self.actor = (
            ray.remote(num_cpus=0, max_restarts=0)(Coordinator)
            .options(name="training", namespace=self.namespace)
            .remote(str(self.directory), session.receiver.workers[0])
        )

    def check_health(self):
        import ray

        ray.get(self.actor.health.remote(), timeout=30)

    def poll(self):
        self.probe.poll(self.actor, "health")

    def close(self):
        import ray

        if self.actor is not None:
            ray.kill(self.actor, no_restart=True)

    def backend(self):
        raise AssertionError("unexpected backend in this fixture")


class Source(ModelPlanSource):
    def __init__(self, address, directory, namespace):
        self.address, self.directory, self.namespace = address, directory, namespace

    def create(self):
        marker = read_marker(self.directory / "job.json")
        if marker["status"] == "RUNNING":
            raise RuntimeError("ambiguous RUNNING optimizer step; operator recovery required")
        resources = SlimeDeploymentResources(
            SimpleNamespace(),
            ray_address=self.address,
            namespace=self.namespace,
            allocate_models=False,
            runtime_env={"env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"]}},
        )
        # Exercise the production process owner without allocating model GPUs.
        resources._process_lease = uuid4().hex
        inference, training = Inference(self.directory, self.namespace), Training(self.directory, self.namespace)
        return ModelDeploymentPlan(resources, inference, training, ComponentHealth(inference, training))


def engine(directory):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state = json.loads((directory / "serving.json").read_text())
            self.send_response(503 if state["paused"] else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({**state, "payload": payload}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    write_state(directory / "endpoint.json", {"url": f"http://127.0.0.1:{server.server_port}", "pid": os.getpid()})
    server.serve_forever()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    import ray
    from ray.cluster_utils import Cluster

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    namespace = "supervision-" + uuid4().hex
    cluster = Cluster()
    cluster.add_node(num_cpus=2, include_dashboard=False)
    ray.init(address=cluster.address, namespace=namespace)
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests"))), "NO_PROXY": "*"}
    (tmp_path / "checkpoint.json").write_text(json.dumps({"version": "checkpoint:1"}))
    write_marker(
        tmp_path / "job.json",
        {
            "status": "COMPLETE",
            "job_id": "job",
            "rollout_id": 0,
            "scenario_step": 0,
            "runtime_load_id": "checkpoint:1",
            "checkpoint_path": str(tmp_path),
            "commit_acknowledged": True,
        },
    )
    with (tmp_path / "driver.log").open("w") as log:
        driver = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "reef_service.test_model_supervision_ray",
                "driver",
                cluster.address,
                str(tmp_path),
                namespace,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:

            def started():
                assert driver.poll() is None, (tmp_path / "driver.log").read_text()
                return (tmp_path / "ready").exists()

            wait_until(started)
            yield SimpleNamespace(ray=ray, namespace=namespace, directory=tmp_path, driver=driver)
        finally:
            driver.terminate()
            try:
                driver.wait(timeout=60)
            except subprocess.TimeoutExpired:
                driver.kill()
                driver.wait()
            ray.shutdown()
            cluster.shutdown()


def test_controller_and_training_crashes_recover_without_recreating_http_runtime(deployment):
    import psutil

    ray, namespace = deployment.ray, deployment.namespace
    training, runtime = connect_ray_runtime(actor_name="training", namespace=namespace, inference_timeout_s=30)
    RuntimeCandidateBackend(training, "sft", StepScheduling(), inference_runtime=runtime)
    backend = runtime.inference_handler

    async def infer():
        lease = await runtime.acquire_inference()
        try:
            artifact = SimpleNamespace(ref=SimpleNamespace(release_id="committed"), local_path=None)
            return await backend.inference(artifact, "/v1/chat/completions", {"messages": [], "custom": 7})
        finally:
            lease.release()

    async def exercise():
        assert (await infer())["version"] == "checkpoint:1"
        for component in ("training", "inference"):
            old = ray.get_actor("inference", namespace=namespace)
            before = ray.get(old.health.remote())
            victim = ray.get_actor(component, namespace=namespace)
            pid = ray.get(victim.health.remote())["pid" if component == "training" else "controller_pid"]
            os.kill(pid, signal.SIGKILL)
            await asyncio.to_thread(wait_until, lambda: not (deployment.directory / "ready").exists())
            request = asyncio.create_task(infer())
            await asyncio.sleep(0.05)
            assert not request.done()
            await asyncio.to_thread(wait_until, lambda: (deployment.directory / "ready").exists())
            after = ray.get(ray.get_actor("inference", namespace=namespace).health.remote())
            assert after["controller_pid"] != before["controller_pid"]
            assert after["pid"] != before["pid"]
            assert (
                not psutil.pid_exists(before["pid"]) or psutil.Process(before["pid"]).status() == psutil.STATUS_ZOMBIE
            )
            result = await request
            assert result == {"version": "checkpoint:1", "paused": False, "payload": {"messages": [], "custom": 7}}
            assert runtime.base_url == after["url"]
            assert runtime.inference_handler is backend
            assert ray.is_initialized()
        runtime.shutdown()
        training.shutdown()

    asyncio.run(exercise())


def test_running_marker_stops_automatic_recovery_and_clears_readiness(deployment):
    ray = deployment.ray
    path = deployment.directory / "job.json"
    marker = read_marker(path)
    write_marker(path, {**marker, "status": "RUNNING", "commit_acknowledged": False})
    ray.kill(ray.get_actor("training", namespace=deployment.namespace), no_restart=True)
    wait_until(lambda: deployment.driver.poll() is not None)
    assert deployment.driver.returncode != 0
    assert not (deployment.directory / "ready").exists()
    assert read_marker(path)["status"] == "RUNNING"
    assert "ambiguous RUNNING" in (deployment.directory / "driver.log").read_text()


def test_rebuilt_deployment_keeps_pending_candidate_paused_until_commit(deployment):
    training, runtime = connect_ray_runtime(
        actor_name="training", namespace=deployment.namespace, inference_timeout_s=30
    )
    coordinator = RuntimeCandidateBackend(training, "sft", StepScheduling(), inference_runtime=runtime)
    path = deployment.directory / "job.json"
    marker = read_marker(path)
    (deployment.directory / "checkpoint.json").write_text(json.dumps({"version": "checkpoint:2"}))
    write_marker(
        path,
        {
            **marker,
            "status": "READY_TO_COMMIT",
            "commit_acknowledged": False,
            "job_id": "next-job",
            "rollout_id": 1,
            "runtime_load_id": "checkpoint:2",
        },
    )
    deployment.ray.kill(deployment.ray.get_actor("inference", namespace=deployment.namespace), no_restart=True)
    wait_until(lambda: not (deployment.directory / "ready").exists())
    wait_until(lambda: (deployment.directory / "ready").exists())

    async def check_gate():
        pending = asyncio.create_task(runtime.acquire_inference())
        await asyncio.sleep(0.2)
        assert not pending.done()
        assert read_marker(path)["status"] == "READY_TO_COMMIT"
        serving = json.loads((deployment.directory / "serving.json").read_text())
        assert serving == {"paused": True, "version": "checkpoint:2"}
        await asyncio.to_thread(coordinator.recover_pending_step, 2, committed_training_job_id="next-job")
        (await asyncio.wait_for(pending, timeout=10)).release()
        assert read_marker(path)["status"] == "COMPLETE"
        assert json.loads((deployment.directory / "serving.json").read_text())["paused"] is False
        assert runtime.current_runtime_load_id() == "checkpoint:2"
        runtime.shutdown()
        training.shutdown()

    asyncio.run(check_gate())


class ScheduledReceiver:
    def __init__(self):
        self.paused = False
        self.value = 0
        self.version = "boot"

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def load(self, value, version):
        if not self.paused:
            raise RuntimeError("weights require a fenced receiver")
        self.value, self.version = value, version

    def status(self):
        return {"paused": self.paused, "value": self.value, "version": self.version, "pid": os.getpid()}


class ScheduledSender:
    def send(self, receiver, runtime_load_id):
        import ray

        ray.get(receiver.load.remote(7, runtime_load_id), timeout=30)
        return runtime_load_id

    def pid(self):
        return os.getpid()


class ScheduledTrainingBackend(TrainingBackend):
    def __init__(self, sender, receiver, directory):
        from reef.runtime.interfaces import TrainingContext, TrainingCoordinationConfig

        self.sender = sender
        self.receiver = receiver
        self.directory = Path(directory)
        self._config = TrainingCoordinationConfig(save_hf_template=str(self.directory / "hf/{rollout_id}"))
        self._context = TrainingContext()

    def start(self):
        pass

    def check_health(self):
        pass

    def prepare_weights(self, runtime_load_id, *, force_full):
        # Preparing sender state must not load the receiver before Reef has
        # restored its resources. Keep the exact target for the later send.
        self.prepared_version = runtime_load_id

    def send_weights(self, runtime_load_id, *, force_full):
        import ray

        assert runtime_load_id == self.prepared_version
        return ray.get(self.sender.send.remote(self.receiver, runtime_load_id), timeout=30)

    def close(self):
        import ray

        (self.directory / "operations-closed").write_text(str(os.getpid()))
        ray.kill(self.sender, no_restart=True)

    @property
    def config(self):
        return self._config

    @property
    def context(self):
        return self._context

    def prepare_training_step(self, batch, objective, algorithm_state, scheduling):
        raise AssertionError("unexpected prepare_training_step in this fixture")

    def prepare(self, payload, *, job_id, scenario_step, prior_marker):
        raise AssertionError("unexpected prepare in this fixture")

    def initialize_version(self, runtime_load_id):
        raise AssertionError("unexpected initialize_version in this fixture")

    def activate_scenario(self, scenario):
        raise AssertionError("unexpected activate_scenario in this fixture")

    def send_adapter(self, scenario, name):
        raise AssertionError("unexpected send_adapter in this fixture")


class ScheduledInferenceBackend(InferenceBackend):
    def __init__(self, receiver):
        self.receiver = receiver

    def inference_url(self):
        return "http://cpu-inference"

    def runtime_load_ids(self):
        import ray

        return [ray.get(self.receiver.status.remote(), timeout=30)["version"]]

    def pause(self):
        import ray

        ray.get(self.receiver.pause.remote(), timeout=30)

    def resume(self):
        import ray

        ray.get(self.receiver.resume.remote(), timeout=30)

    def abort(self):
        self.pause()

    def initialize_version(self, runtime_load_id):
        raise AssertionError("unexpected initialize_version in this fixture")

    def recover(self):
        raise AssertionError("unexpected recover in this fixture")

    def offload(self, tags):
        raise AssertionError("unexpected offload in this fixture")

    def onload_weights(self):
        raise AssertionError("unexpected onload_weights in this fixture")

    def onload_kv(self):
        raise AssertionError("unexpected onload_kv in this fixture")

    def unload_adapter(self, name):
        raise AssertionError("unexpected unload_adapter in this fixture")


class ScheduledInferenceService(InferenceService):
    connection_protocol = "scheduled-cpu-v1"

    def __init__(self):
        self.receiver = None

    def start(self, resources):
        import ray

        self.receiver = ray.remote(num_cpus=0)(ScheduledReceiver).remote()
        return InferenceConnection(self.connection_protocol, RayExecutor.from_workers([self.receiver]))

    def prepare_weight_transfer(self, connection):
        connection.control.rpc(0, "pause", timeout=30)

    def backend(self, connection):
        return ScheduledInferenceBackend(connection.control.workers[0])

    def check_health(self):
        import ray

        ray.get(self.receiver.status.remote(), timeout=30)

    def close(self):
        import ray

        ray.kill(self.receiver, no_restart=True)

    def poll(self):
        raise AssertionError("unexpected poll in this fixture")


class ScheduledTrainingService(TrainingService):
    weight_transfer_protocol = "scheduled-cpu-v1"

    def __init__(self, directory):
        self.directory = directory
        self.sender = None
        self.receiver = None

    def start(self, resources):
        import ray

        assert self.receiver is None
        self.sender = ray.remote(num_cpus=0)(ScheduledSender).remote()

    def attach_weight_transport(self, session):
        self.receiver = session.receiver.workers[0]

    def backend(self):
        return ScheduledTrainingBackend(self.sender, self.receiver, self.directory)

    def check_health(self):
        import ray

        ray.get(self.sender.pid.remote(), timeout=30)

    def close(self):
        import ray

        ray.kill(self.sender, no_restart=True)

    def poll(self):
        raise AssertionError("unexpected poll in this fixture")


class BorrowedRayResources(DeploymentResources):
    def start(self):
        pass

    def close(self):
        pass


def test_generic_named_coordinator_serializes_operations_and_retires_owned_workers(tmp_path, monkeypatch):
    import ray

    from reef.runtime.deployment import CoordinatorConfig, ModelDeployment

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    namespace = "generic-coordinator-" + uuid4().hex
    root = Path(__file__).resolve().parents[2]
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        namespace=namespace,
        runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join((str(root), str(root / "tests")))}},
    )
    inference = ScheduledInferenceService()
    training = ScheduledTrainingService(str(tmp_path))
    owner = ModelDeployment(
        ModelDeploymentPlan(
            BorrowedRayResources(),
            inference,
            training,
            coordinator=CoordinatorConfig(
                backend="ray",
                options={"name": "generic-training", "namespace": namespace, "max_concurrency": 64, "num_cpus": 0},
            ),
        )
    )
    try:
        owner.start()
        coordinator = ray.get_actor("generic-training", namespace=namespace)
        assert ray.get(coordinator.health.remote(), timeout=30)["ok"]
        state = ray.get(inference.receiver.status.remote(), timeout=30)
        assert state["value"] == 7
        assert state["paused"] is False
        assert state["version"] == ray.get(coordinator.serving_runtime_load_id.remote(), timeout=30)
        owner.close()
        assert int((tmp_path / "operations-closed").read_text()) != os.getpid()
        for actor, method in ((coordinator, "health"), (training.sender, "pid"), (inference.receiver, "status")):

            def retired(actor=actor, method=method):
                try:
                    ray.get(getattr(actor, method).remote(), timeout=5)
                except ray.exceptions.RayActorError:
                    return True
                return False

            wait_until(retired, timeout=30)
        assert ray.is_initialized()
    finally:
        owner.close()
        ray.shutdown()


if __name__ == "__main__":
    if sys.argv[1] == "engine":
        engine(Path(sys.argv[2]))
    else:
        source = Source(sys.argv[2], Path(sys.argv[3]), sys.argv[4])
        raise SystemExit(run_deployment(source.create(), source.directory / "ready", source=source))
