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

from reef.runtime.adapters.ray_runtime import connect_ray_runtime
from reef.runtime.deployment import InferenceConnection, ModelDeploymentPlan
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.sglang.service import RayHealthProbe
from reef.runtime.training_job.marker import read_marker, write_marker
from reef.runtime.training_job.publication import TrainingPublication
from reef.service.training_driver import run_deployment
from reef.train.runtime_backend import RuntimeTrainingBackend
from reef.train.slime_backend.resources import SlimeDeploymentHealth, SlimeDeploymentResources

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


class Publisher:
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


class Coordinator:
    def __init__(self, directory, controller):
        import ray

        self.path = Path(directory) / "job.json"
        self.endpoint = ray.get(controller.health.remote())["url"]
        self.publication = TrainingPublication(self.path, Publisher(controller))
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


class Inference:
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


class Training:
    inference_protocol = "cpu-test-v1"

    def __init__(self, directory, namespace):
        self.directory, self.namespace = directory, namespace
        self.actor = None
        self.probe = RayHealthProbe()

    def start(self, resources, inference):
        import ray

        self.actor = (
            ray.remote(num_cpus=0, max_restarts=0)(Coordinator)
            .options(name="training", namespace=self.namespace)
            .remote(str(self.directory), inference.control.workers[0])
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


class Source:
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
        return ModelDeploymentPlan(resources, inference, training, SlimeDeploymentHealth(inference, training))


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
    RuntimeTrainingBackend(training, "sft", inference_runtime=runtime)
    backend = runtime.inference_backend

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
            assert runtime.inference_backend is backend
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
    coordinator = RuntimeTrainingBackend(training, "sft", inference_runtime=runtime)
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


if __name__ == "__main__":
    if sys.argv[1] == "engine":
        engine(Path(sys.argv[2]))
    else:
        source = Source(sys.argv[2], Path(sys.argv[3]), sys.argv[4])
        raise SystemExit(run_deployment(source.create(), source.directory / "ready", source=source))
