"""Opt-in real Ray/HTTP integration: REEF_TEST_RAY=1 pytest this file."""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request

import pytest

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFailedError, WorkerSpec
from reef.runtime.executor.local import LocalExecutor
from reef.runtime.executor.ray import RayExecutor
from reef.service.deploy.config import validate_services
from reef.service.deploy.orchestrator import _Stack
from reef.service.deploy.process import RayProcessWorker

pytestmark = pytest.mark.skipif(os.environ.get("REEF_TEST_RAY") != "1", reason="opt-in real Ray integration")


@pytest.fixture
def ray_cluster():
    ray = pytest.importorskip("ray")
    ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        yield ray
    finally:
        ray.shutdown()


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.parametrize("owned", [True, False])
def test_ray_idle_worker_death_fails_busy_peer_without_retries(ray_cluster, owned):
    class Worker:
        def run(self, delay=0):
            time.sleep(delay)
            return "done"

        def bad(self):
            raise ValueError("ordinary scorer failure")

    class Listener:
        def __init__(self):
            self.events = []
            self.received = threading.Event()

        def on_executor_failure(self, failure):
            self.events.append(failure)
            self.received.set()

    ray = ray_cluster
    actors = [ray.remote(Worker).options(num_cpus=1).remote() for _ in range(2)]
    executor = RayExecutor.from_workers(actors, owned=owned)
    executor.check_health(timeout=15)
    listener = Listener()
    executor.register_failure_listener(listener)
    try:
        with pytest.raises(ValueError, match="ordinary scorer failure"):
            executor.rpc(0, "bad")
        assert executor.failure is None
        pending = executor.rpc(0, "run", args=(2,), non_block=True)
        ray.kill(actors[1], no_restart=True)
        assert listener.received.wait(10)
        assert listener.events[0].rank == 1
        with pytest.raises(ExecutorFailedError):
            pending.result(timeout=1)
        with pytest.raises(ExecutorFailedError):
            executor.rpc(0, "run")
        assert len(listener.events) == 1
        if not owned:
            assert ray.get(actors[0].run.remote(), timeout=10) == "done"
    finally:
        executor.shutdown()
        for actor in actors:
            ray.kill(actor, no_restart=True)


@pytest.mark.parametrize("backend", ["ray", "auto"])
def test_ray_service_publishes_address_and_passes_it_to_local_consumer(tmp_path, ray_cluster, backend):
    port = _port()
    output = tmp_path / "seen-url"
    config = {
        "services": [
            {
                "name": "prm",
                "executor": backend,
                "resources": {"num_cpus": 1},
                "command": [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0"],
                "endpoint": f"http://{{host}}:{port}",
                "ready": f"curl -sf http://127.0.0.1:{port}/",
                "ready_timeout": 30,
            },
            {
                "name": "consumer",
                "depends_on": ["prm"],
                "env": {"PRM_URL": "${endpoints.prm}"},
                "command": [
                    sys.executable,
                    "-c",
                    f"import os,time; from pathlib import Path; Path({str(output)!r}).write_text(os.environ['PRM_URL']); time.sleep(120)",
                ],
                "ready": f"test -f {output}",
            },
        ]
    }
    stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 30, tmp_path / "input.yaml")
    try:
        stack.start()
        assert isinstance(stack._executors["prm"], RayExecutor)
        assert isinstance(stack._executors["consumer"], LocalExecutor)
        endpoint = stack.config["endpoints"]["prm"]
        assert output.read_text() == endpoint
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(endpoint, timeout=5) as response:
            assert response.status == 200
        assert stack._is_alive("prm")
    finally:
        stack.shutdown(grace=2)
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


def test_killing_ray_owner_retires_its_service_process_group(tmp_path, ray_cluster):
    port = _port()
    service = {
        "name": "prm",
        "command": [sys.executable, "-m", "http.server", str(port)],
        "ready": f"curl -sf http://127.0.0.1:{port}/",
    }
    executor = Executor.create(
        ExecutorConfig(
            backend="ray",
            workers=(
                WorkerSpec(
                    RayProcessWorker,
                    args=({}, [service], tmp_path, 30, tmp_path / "input.yaml"),
                    options={"num_cpus": 1},
                ),
            ),
        )
    )
    try:
        executor.rpc(0, "start")
        deadline = time.monotonic() + 15
        while not executor.rpc(0, "probe", args=("prm",)):
            if time.monotonic() > deadline:
                pytest.fail("server did not start")
            time.sleep(0.1)
        executor.shutdown()  # Deliberately skip the worker shutdown RPC.
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) != 0:
                    break
            time.sleep(0.1)
        else:
            pytest.fail("service survived its owner")
    finally:
        executor.shutdown()
