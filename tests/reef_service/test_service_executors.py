from __future__ import annotations

import json
import os
import socket
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from reef.runtime.executor import ExecutorConfig
from reef.runtime.executor.config import executor_settings, role_executor_settings
from reef.runtime.executor.local import LocalExecutor
from reef.service.deploy.config import DeployConfigError, validate_services
from reef.service.deploy.execution import service_executor_config
from reef.service.deploy.orchestrator import _Stack
from reef.service.deploy.process import ProcessWorker, RayProcessWorker


class RecordingExecutor(LocalExecutor):
    created: list[ExecutorConfig] = []

    def _init_executor(self):
        type(self).created.append(self.config)
        super()._init_executor()


def _port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _server(port, **extra):
    return {
        "name": "prm",
        "command": [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0"],
        "endpoint": f"http://{{host}}:{port}",
        "ready": f"curl -sf http://127.0.0.1:{port}/",
        "ready_timeout": 15,
        **extra,
    }


def test_all_services_use_the_selected_executor_and_discover_endpoints(tmp_path):
    RecordingExecutor.created.clear()
    port = _port()
    config = {
        "executors": {"testing": {"backend": f"{__name__}:RecordingExecutor"}},
        "execution": {"services": "testing"},
        "reef": {"prm_url": "${endpoints.prm}"},
        "services": [
            {
                "name": "reef",
                "command": [sys.executable, "-c", "import time; time.sleep(120)"],
                "depends_on": ["prm"],
            },
            _server(port),
        ],
    }
    stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 20, tmp_path / "input.yaml")
    try:
        stack.start()
        assert len(RecordingExecutor.created) == 2
        assert stack.config["endpoints"]["prm"] == f"http://127.0.0.1:{port}"
        child_config = yaml.safe_load((tmp_path / "reef" / "runtime.yaml").read_text())
        assert child_config["endpoints"]["prm"] == f"http://127.0.0.1:{port}"
        from reef.service.deploy.config import interpolate_config

        assert interpolate_config(child_config, child_config["reef"]["prm_url"]) == f"http://127.0.0.1:{port}"
        assert stack._is_alive("prm") and stack._is_alive("reef")
        assert (tmp_path / "prm.worker.json").exists()
    finally:
        stack.shutdown(grace=1)
    for config_created in RecordingExecutor.created:
        name = config_created.workers[0].args[1][0]["name"]
        info = json.loads((tmp_path / f"{name}.worker.json").read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(info["pids"][name], 0)
    stack.shutdown()


def test_startup_failure_rolls_back_healthy_dependencies(tmp_path):
    config = {
        "services": [
            {"name": "prm", "command": [sys.executable, "-c", "import time; time.sleep(120)"]},
            {"name": "bad", "command": ["/no/such/reef-test-executable"], "depends_on": ["prm"]},
        ]
    }
    stack = _Stack(config, validate_services(config, "test.yaml"), tmp_path, 5, tmp_path / "input.yaml")
    with pytest.raises(FileNotFoundError):
        stack.start()
    info = json.loads((tmp_path / "prm.worker.json").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(info["pids"]["prm"], 0)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"name": "../escape"}, "invalid service name"),
        ({"depends_on": ["missing"]}, "unknown dependency"),
        ({"executor": "missing"}, "invalid class import path"),
        ({"executor": "ray", "cuda": "0"}, "num_gpus"),
        ({"executor": "ray", "env": {"CUDA_VISIBLE_DEVICES": "0"}}, "num_gpus"),
        ({"executor": "local", "resources": {"num_gpus": 1}}, "reservations require"),
        ({"ready_timeout": 0}, "positive"),
        ({"executor": {"backend": "ray", "options": {"max_restarts": -1}}}, "replay"),
    ],
)
def test_invalid_executor_and_placement_fail_before_launch(change, message):
    with pytest.raises(DeployConfigError, match=message):
        validate_services({"services": [{"name": "worker", "command": "true", **change}]}, "test.yaml")


def test_service_dependency_cycles_are_rejected():
    with pytest.raises(DeployConfigError, match="cycle"):
        validate_services(
            {
                "services": [
                    {"name": "a", "command": "true", "depends_on": ["b"]},
                    {"name": "b", "command": "true", "depends_on": ["a"]},
                ]
            },
            "test.yaml",
        )


def test_ray_placement_is_carried_by_worker_options(tmp_path):
    config = {"executors": {"gpu": {"backend": "ray", "options": {"num_cpus": 2}}}}
    service = {"name": "prm", "command": "serve", "executor": "gpu", "resources": {"num_gpus": 2}}
    selected = service_executor_config(config, service, tmp_path, 30, tmp_path / "config.yaml")
    assert selected.workers[0].worker_cls is RayProcessWorker
    assert selected.options == {"num_cpus": 2}
    assert selected.workers[0].options == {"num_gpus": 2}
    assert selected.launch_timeout_s == 30


def test_named_role_selectors_share_one_configuration_contract():
    config = {
        "executors": {"custom": {"backend": "my_package:Executor", "options": {"queue": "gpu"}}},
        "execution": {"training": "custom", "rollout": "custom"},
    }
    assert role_executor_settings(config, "training", "ray") == executor_settings(config, "custom")
    assert role_executor_settings(config, "rollout", "ray").options == {"queue": "gpu"}
    assert role_executor_settings(config, "services", "local").backend == "local"


@pytest.mark.parametrize("cli", [False, True])
def test_slime_driver_honors_yaml_roles_and_explicit_flag_precedence(cli):
    pytest.importorskip("ray")
    from reef.train.slime_backend.reef_adapters.driver import _configure_executors

    config = {
        "executors": {"special": {"backend": "custom:Training", "options": {"queue": "gpu"}}},
        "execution": {"training": "special", "rollout": {"backend": "custom:Serving"}},
    }
    args = SimpleNamespace(reef_executor_backend="ray", reef_rollout_executor_backend="ray")
    _configure_executors(args, config, ["--reef-executor-backend=ray"] if cli else [])
    assert args.reef_executor_backend == ("ray" if cli else "custom:Training")
    assert args.reef_train_executor_options == ({} if cli else {"queue": "gpu"})
    assert args.reef_rollout_executor_backend == "custom:Serving"


def test_nested_endpoint_interpolation_supports_service_names_with_hyphens():
    from reef.service.deploy.config import interpolate_config

    config = {"reef": {"prm_url": "${endpoints.prm-sglang}"}, "endpoints": {"prm-sglang": "http://node:23001"}}
    assert interpolate_config(config, "--url=${reef.prm_url}") == "--url=http://node:23001"
    with pytest.raises(DeployConfigError, match="cyclic"):
        interpolate_config({"a": {"url": "${b.url}"}, "b": {"url": "${a.url}"}}, "${a.url}")


def test_probe_runs_with_service_environment_and_cwd_and_has_a_deadline(tmp_path):
    marker = tmp_path / "marker"
    marker.touch()
    service = {
        "name": "worker",
        "command": [sys.executable, "-c", "import time; time.sleep(120)"],
        "cwd": str(tmp_path),
        "env": {"EXPECTED": "yes"},
        "ready": 'test "$EXPECTED" = yes && test -f marker',
    }
    worker = ProcessWorker({}, [service], tmp_path, 10, tmp_path / "config.yaml")
    try:
        worker.start()
        assert worker.probe("worker")
        service["ready"] = "sleep 60"
        start = time.monotonic()
        assert not worker.probe("worker", timeout=0.1)
        assert time.monotonic() - start < 2
    finally:
        worker.shutdown(grace=0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX owner leases")
def test_remote_service_dies_when_its_owner_lease_closes(tmp_path):
    pid_file = tmp_path / "service-child.pid"
    command = [
        sys.executable,
        "-c",
        f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)",
    ]
    worker = RayProcessWorker({}, [{"name": "prm", "command": command}], tmp_path, 10, tmp_path / "config.yaml")
    try:
        worker.start()
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        os.close(worker._leases.pop())
        while worker.status()["prm"] is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert worker.status()["prm"] is not None
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        worker.shutdown(grace=0)


def test_shutdown_reaches_all_executors_even_if_one_fails(tmp_path):
    events = []

    class Worker:
        def __init__(self, name):
            self.name = name

        def request_stop(self):
            events.append((self.name, "stop"))
            if self.name == "dependent":
                raise RuntimeError("lost worker")

        def shutdown(self, grace=0):
            events.append((self.name, "shutdown"))

        def read_log(self, *args):
            return "", 0

    stack = _Stack({}, [], tmp_path, 5, tmp_path / "config.yaml")
    for name in ("dependency", "dependent"):
        stack._executors[name] = LocalExecutor.from_workers([Worker(name)])
    stack.shutdown(grace=0)
    assert events == [
        ("dependent", "stop"),
        ("dependency", "stop"),
        ("dependent", "shutdown"),
        ("dependency", "shutdown"),
    ]
