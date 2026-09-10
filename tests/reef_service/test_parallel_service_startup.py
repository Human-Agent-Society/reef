"""Independent services initialize together; only explicit readiness edges gate launch."""

import json
import os
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from reef.runtime.executor import Executor
from reef.runtime.executor.ray import RayExecutor
from reef.service.deploy import orchestrator
from reef.service.deploy.config import validate_services
from reef.service.deploy.orchestrator import _Stack
from reef.service.deploy.process import ProcessWorker


def service(name, marker, **options):
    return {
        "name": name,
        "command": [
            sys.executable,
            "-c",
            "import os,time; from pathlib import Path; "
            f"Path({str(marker)!r}).write_text(os.environ.get('PEER_URL', 'started')); time.sleep(120)",
        ],
        "ready": f"test -f {marker}",
        "ready_timeout": 10,
        **options,
    }


def stack_for(tmp_path, services, backend="uni"):
    config = {"execution": {"services": backend}, "services": services}
    run_dir = tmp_path / "stack"
    run_dir.mkdir()
    return _Stack(config, validate_services(config, "test.yaml"), run_dir, 10, tmp_path / "input.yaml")


def assert_processes_stopped(tmp_path):
    for path in (tmp_path / "stack").glob("*.worker.json"):
        for pid in json.loads(path.read_text())["pids"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


@pytest.mark.parametrize("backend", ["uni", "mp"])
def test_independent_services_publish_all_endpoints_then_start_before_waiting(tmp_path, backend):
    left, right = tmp_path / "left-started", tmp_path / "right-started"
    stack = stack_for(
        tmp_path,
        [
            service(
                "left",
                left,
                endpoint="http://{host}:8001",
                env={"PEER_URL": "${endpoints.right}"},
                ready=f"test -f {left} && test -f {right}",
            ),
            service(
                "right",
                right,
                endpoint="http://{host}:8002",
                env={"PEER_URL": "${endpoints.left}"},
                ready=f"test -f {left} && test -f {right}",
            ),
        ],
        backend,
    )
    try:
        stack.start()
        assert left.read_text() == "http://127.0.0.1:8002"
        assert right.read_text() == "http://127.0.0.1:8001"
        for name in ("left", "right"):
            config = yaml.safe_load((tmp_path / "stack" / name / "runtime.yaml").read_text())
            assert config["endpoints"] == stack.config["endpoints"]
    finally:
        stack.shutdown(grace=1)
    assert_processes_stopped(tmp_path)


def test_start_rpcs_are_dispatched_concurrently(tmp_path, monkeypatch):
    rendezvous = threading.Barrier(2)
    original = ProcessWorker.start

    def start(self):
        rendezvous.wait(timeout=5)
        original(self)

    monkeypatch.setattr(ProcessWorker, "start", start)
    stack = stack_for(tmp_path, [service(name, tmp_path / name) for name in ("left", "right")])
    try:
        stack.start()
    finally:
        stack.shutdown(grace=1)


def test_ready_branch_unlocks_dependents_without_waiting_for_unrelated_service(tmp_path):
    first, slow, child = (tmp_path / name for name in ("first", "slow", "child"))
    stack = stack_for(
        tmp_path,
        [
            service("first", first),
            service("slow", slow, ready=f"test -f {child}"),
            service("child", child, depends_on=["first"]),
        ],
    )
    try:
        stack.start()
        assert all(path.exists() for path in (first, slow, child))
    finally:
        stack.shutdown(grace=1)


def test_failed_peer_cancels_other_readiness_waits_and_never_launches_dependents(tmp_path):
    child = tmp_path / "child"
    stack = stack_for(
        tmp_path,
        [
            service("slow", tmp_path / "slow", ready="false", ready_timeout=30),
            service("bad", tmp_path / "bad", command=[sys.executable, "-c", "raise SystemExit(7)"]),
            service("child", child, depends_on=["bad"]),
        ],
    )
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=r"bad.*exited"):
        stack.start()
    assert time.monotonic() - started < 10
    assert not child.exists()
    assert stack._closed
    assert_processes_stopped(tmp_path)


def test_deadline_is_per_service_and_timeout_cleans_up_all_peers(tmp_path):
    stack = stack_for(
        tmp_path,
        [
            service("slow", tmp_path / "slow", ready="false", ready_timeout=30),
            service("timeout", tmp_path / "timeout", ready="false", ready_timeout=0.2),
        ],
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError, match=r"timeout.*did not become ready"):
        stack.start()
    assert time.monotonic() - started < 10
    assert_processes_stopped(tmp_path)


def test_previously_ready_service_is_monitored_while_other_services_initialize(tmp_path, monkeypatch):
    stack = stack_for(
        tmp_path,
        [service("ready", tmp_path / "ready"), service("slow", tmp_path / "slow", ready="false", ready_timeout=30)],
    )
    original = stack._is_alive

    def is_alive(name):
        alive = original(name)
        if name == "ready" and alive:
            stack._executors[name].rpc(0, "request_stop")
        return alive

    monkeypatch.setattr(stack, "_is_alive", is_alive)
    with pytest.raises(RuntimeError, match=r"ready.*exited during startup"):
        stack.start()
    assert_processes_stopped(tmp_path)


def test_preparation_failure_releases_workers_before_any_process_is_started(tmp_path, monkeypatch):
    original = ProcessWorker.describe
    starts = []

    def describe(self):
        if self.services[0]["name"] == "bad":
            raise RuntimeError("cannot describe placement")
        return original(self)

    monkeypatch.setattr(ProcessWorker, "describe", describe)
    monkeypatch.setattr(ProcessWorker, "start", lambda self: starts.append(self))
    stack = stack_for(tmp_path, [service(name, tmp_path / name) for name in ("first", "bad")])
    with pytest.raises(RuntimeError, match="cannot describe placement"):
        stack.start()
    assert starts == []
    assert stack._closed
    assert all(executor._closed for executor in stack._executors.values())


def test_in_flight_launch_finishes_before_failure_cleanup(tmp_path, monkeypatch):
    stack = stack_for(tmp_path, [service(name, tmp_path / name) for name in ("late", "bad")])
    rendezvous = threading.Barrier(2)
    original = ProcessWorker.start

    def start(self):
        rendezvous.wait(timeout=5)
        if self.services[0]["name"] == "bad":
            raise RuntimeError("launch failed")
        assert stack._stopping.wait(5)
        # Simulate a start RPC completing after another worker failed.
        original(self)

    monkeypatch.setattr(ProcessWorker, "start", start)
    with pytest.raises(RuntimeError, match="launch failed"):
        stack.start()
    assert (stack.run_dir / "late.worker.json").exists()
    assert_processes_stopped(tmp_path)


def test_manual_ray_head_is_ready_before_connecting_and_preparing_ray_dependents(tmp_path, monkeypatch):
    head_ready = []
    runtime_events = []
    create = Executor.create
    probe = ProcessWorker.probe

    def acquire(address):
        assert head_ready
        assert address == "127.0.0.1:6321"
        runtime_events.append("connect")
        return SimpleNamespace(address=address, close=lambda: runtime_events.append("disconnect"))

    def create_executor(config):
        if config.backend is RayExecutor:
            # Exercise orchestration without starting an actual cluster.
            config = replace(
                config,
                backend="uni",
                workers=tuple(replace(spec, worker_cls=ProcessWorker) for spec in config.workers),
            )
        return create(config)

    def check_ready(self, name, timeout=5):
        result = probe(self, name, timeout)
        if name == "ray-head" and result:
            head_ready.append(True)
        return result

    monkeypatch.setattr(orchestrator, "acquire_ray_runtime", acquire)
    monkeypatch.setattr(Executor, "create", staticmethod(create_executor))
    monkeypatch.setattr(ProcessWorker, "probe", check_ready)
    marker = tmp_path / "head-ready"
    child = tmp_path / "child"
    stack = stack_for(
        tmp_path,
        [
            service("ray-head", marker),
            service(
                "worker",
                child,
                executor="ray",
                depends_on=["ray-head"],
                env={"PEER_URL": "${reef.ray_address}"},
            ),
        ],
    )
    stack.config["reef"] = {"ray_address": "127.0.0.1:6321"}
    stack.config["execution"]["training"] = "ray"
    try:
        stack.start()
        assert child.read_text() == "127.0.0.1:6321"
    finally:
        stack.shutdown(grace=1)
    assert runtime_events == ["connect", "disconnect"]


@pytest.mark.parametrize("role", ["training", "rollout"])
@pytest.mark.parametrize("backend", ["ray", "auto", "model-workers"])
def test_local_training_driver_gets_managed_runtime_address(tmp_path, monkeypatch, role, backend):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    events = []

    def acquire(address):
        assert address is None
        events.append("start")
        return SimpleNamespace(address="127.0.0.1:12345", close=lambda: events.append("stop"))

    monkeypatch.setattr(orchestrator, "acquire_ray_runtime", acquire)
    marker = tmp_path / "driver"
    stack = stack_for(tmp_path, [service("driver", marker, env={"PEER_URL": "${reef.ray_address}"})])
    stack.config["execution"][role] = backend
    stack.config["executors"] = {"model-workers": {"backend": "ray"}}
    try:
        stack.start()
        assert marker.read_text() == "127.0.0.1:12345"
        snapshot = yaml.safe_load((stack.run_dir / "driver" / "runtime.yaml").read_text())
        assert snapshot["reef"]["ray_address"] == "127.0.0.1:12345"
        assert events == ["start"]
    finally:
        stack.shutdown(grace=1)
    assert events == ["start", "stop"]


@pytest.mark.parametrize("source", ["config", "environment", "both", "none"])
def test_local_controllers_pass_external_address_without_starting_ray(tmp_path, monkeypatch, source):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)

    def acquire(address):
        pytest.fail("local controllers must not start Ray for an external address or a CPU-only stack")

    monkeypatch.setattr(orchestrator, "acquire_ray_runtime", acquire)
    marker = tmp_path / "driver"
    process = service("driver", marker)
    process["command"][2] = process["command"][2].replace("PEER_URL", "RAY_ADDRESS")
    stack = stack_for(tmp_path, [process])
    if source != "none":
        stack.config["execution"]["training"] = "ray"
    if source in ("config", "both"):
        stack.config["reef"] = {"ray_address": "127.0.0.1:12345"}
    if source in ("environment", "both"):
        monkeypatch.setenv("RAY_ADDRESS", "127.0.0.1:23456")
    try:
        stack.start()
        expected = {"none": "started", "config": "127.0.0.1:12345"}.get(source, "127.0.0.1:23456")
        assert marker.read_text() == expected
        if source != "none":
            assert stack.config["reef"]["ray_address"] == expected
    finally:
        stack.shutdown(grace=1)
