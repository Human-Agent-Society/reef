"""Independent component configuration and process lifecycle contracts."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

from reef.cli import main
from reef.record2dataset import __main__ as generator_main
from reef.service.deploy import orchestrator
from reef.service.deploy.config_utils import DeployConfigError
from reef.service.deploy.generator import generator_settings
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping


def provider_deployment(tmp_path: Path) -> dict:
    return {
        "schema-version": 2,
        "reef": {"host": "0.0.0.0", "port": 8900},
        "inference": {"upstream-url": "http://provider:8000", "upstream-model": "model"},
        "generator": {"tasks-root": str(tmp_path / "tasks")},
        "endpoints": {"reef": "http://reef:8900", "generator": "http://generator:8910"},
    }


@pytest.mark.parametrize("component", ["reef", "generator"])
def test_selects_only_one_component_and_keeps_declared_endpoints(tmp_path: Path, component: str) -> None:
    deployment = provider_deployment(tmp_path)
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml", component=component)
    assert [service["name"] for service in resolved["services"]] == [component]
    assert resolved["services"][0]["depends_on"] == []
    assert resolved["endpoints"] == deployment["endpoints"]
    assert "depends_on" not in deployment["reef"]


def test_generator_uses_declared_reef_for_designer_and_task_player(tmp_path: Path) -> None:
    deployment = provider_deployment(tmp_path)
    deployment["reef"]["token"] = "test-token"
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml", component="generator")
    settings = service_config_from_mapping(resolved)
    generator = generator_settings(settings.generator_settings)
    built = generator_main.generator_service(settings, generator)
    assert built.plays.reef_url == "http://reef:8900"
    assert built.plays.token == "test-token"


@pytest.mark.parametrize("component,missing", [("reef", "generator"), ("generator", "reef")])
def test_missing_peer_addresses_fail_before_startup(tmp_path: Path, component: str, missing: str) -> None:
    deployment = provider_deployment(tmp_path)
    deployment["endpoints"].pop(missing)
    with pytest.raises(DeployConfigError, match=f"requires declared endpoints: {missing}"):
        resolve_deployment_config(deployment, None, tmp_path / "serve.yaml", component=component)


@pytest.mark.parametrize(
    "endpoints",
    [
        None,
        [],
        {"reef": 1},
        {"reef": ""},
        {"reef": "file:///tmp/a"},
        {"reef": "http://0.0.0.0:8900"},
        {"reef": "http://reef:70000"},
    ],
)
def test_invalid_public_endpoints_fail_resolution(tmp_path: Path, endpoints: object) -> None:
    deployment = provider_deployment(tmp_path)
    deployment["endpoints"] = endpoints
    with pytest.raises((DeployConfigError, ValueError), match="endpoints"):
        resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")


def test_endpoints_support_environment_config_references_and_cli_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REEF_TEST_HOST", "remote-reef")
    deployment = provider_deployment(tmp_path)
    deployment["endpoints"]["reef"] = "http://${REEF_TEST_HOST}:${reef.port}"
    resolved, _ = resolve_deployment_config(
        deployment,
        {"endpoints.generator": "http://remote-generator:8001"},
        tmp_path / "serve.yaml",
        component="generator",
    )
    assert resolved["endpoints"] == {"reef": "http://remote-reef:8900", "generator": "http://remote-generator:8001"}
    resolved, _ = resolve_deployment_config(
        deployment, {"endpoints": '{"reef": "http://other:9000"}'}, tmp_path / "serve.yaml", component="generator"
    )
    assert resolved["endpoints"] == {"reef": "http://other:9000"}


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("component", ["reef", "generator", "engine"])
def test_separate_inference_uses_fixed_bind_and_declared_peer(tmp_path: Path, backend: str, component: str) -> None:
    deployment = provider_deployment(tmp_path)
    deployment["inference"] = {"model-path": "org/model", "backend": backend, "host": "0.0.0.0", "port": 8000}
    deployment["endpoints"][backend] = f"http://{backend}:8000"
    component = backend if component == "engine" else component
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml", component=component)
    assert resolved["reef"]["upstream_url"] == f"http://{backend}:8000"
    service = resolved["services"][0]
    assert service["name"] == component
    if component == backend:
        assert service["command"][service["command"].index("--host") + 1] == "0.0.0.0"
        assert service["command"][service["command"].index("--port") + 1] == "8000"
        assert service["ready"][-1] == "http://127.0.0.1:8000/health"


def test_separate_engine_requires_fixed_port(tmp_path: Path) -> None:
    deployment = {
        "schema-version": 2,
        "inference": {"model-path": "org/model"},
        "endpoints": {"sglang": "http://sglang:8000"},
    }
    with pytest.raises(DeployConfigError, match=r"fixed inference\.port"):
        resolve_deployment_config(deployment, None, tmp_path / "serve.yaml", component="sglang")


def test_complete_stack_keeps_dependency_order_and_automatic_inference_port(tmp_path: Path) -> None:
    deployment = provider_deployment(tmp_path)
    deployment["inference"] = {"model-path": "org/model"}
    resolved, _ = resolve_deployment_config(deployment, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in resolved["services"]] == ["generator", "sglang", "reef"]
    assert resolved["services"][2]["depends_on"] == ["sglang", "generator"]
    assert resolved["reef"]["upstream_url"].startswith("http://127.0.0.1:")


def test_service_publication_preserves_declared_address(tmp_path: Path) -> None:
    service = {
        "name": "reef",
        "command": [sys.executable, "-c", "import time; time.sleep(120)"],
        "endpoint": "http://127.0.0.1:8900",
    }
    config = {"endpoints": {"reef": "http://remote-reef:8900"}, "services": [service]}
    stack = orchestrator._Stack(config, [service], tmp_path, 5, tmp_path / "serve.yaml")
    try:
        stack.start()
        assert stack.config["endpoints"]["reef"] == "http://remote-reef:8900"
    finally:
        stack.shutdown()


def test_unknown_component_and_training_split_fail_explicitly(tmp_path: Path) -> None:
    for component, message in [
        ("missing", "unknown component"),
        ("slime-driver", "weight-training"),
        ("tinker-driver", "weight-training"),
    ]:
        with pytest.raises(DeployConfigError, match=message):
            resolve_deployment_config(
                provider_deployment(tmp_path), None, tmp_path / "serve.yaml", component=component
            )


def test_unselected_engine_does_not_download_models(tmp_path: Path, monkeypatch) -> None:
    deployment = {
        "schema-version": 2,
        "inference": {"model-path": "org/model"},
        "endpoints": {"sglang": "http://sglang:8000"},
    }
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(deployment))
    monkeypatch.chdir(tmp_path)

    def must_not_download(config):
        pytest.fail("the Reef component must not download the external engine's model")

    class Stack:
        exit_code = 0

        def __init__(self, config, services, *args, **kwargs):
            assert [service["name"] for service in services] == ["reef"]

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "resolve_model_paths", must_not_download)
    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    with pytest.raises(SystemExit) as result:
        main(["serve", "-c", str(path), "--component", "reef"])
    assert result.value.code == 0


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for_health(process: subprocess.Popen, url: str, log_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        assert process.poll() is None, log_path.read_text()
        try:
            with urlopen(url + "/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.05)
    pytest.fail(f"service did not become healthy: {log_path.read_text()}")


@pytest.mark.integration
def test_reef_and_generator_run_and_restart_independently(tmp_path: Path, fake_git_lfs) -> None:
    """Run real service entrypoints; fake only Docker readiness and the model provider."""

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "invalid task reply"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    binaries = tmp_path / "bin"
    docker = binaries / "docker"
    docker.write_text('#!/bin/sh\n[ "$1" = version ]\n')
    docker.chmod(0o755)
    deployment = provider_deployment(tmp_path)
    reef_port, generator_port = unused_port(), unused_port()
    reef_url, generator_url = f"http://127.0.0.1:{reef_port}", f"http://127.0.0.1:{generator_port}"
    deployment["reef"].update(port=reef_port, token="test-token")
    deployment["generator"].update(
        host="0.0.0.0", port=generator_port, harbor=str(Path(sys.executable).parent / "harbor")
    )
    deployment["inference"]["upstream-url"] = f"http://127.0.0.1:{provider.server_port}"
    deployment["endpoints"] = {"reef": reef_url, "generator": generator_url}
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(deployment))
    env = {key: value for key, value in os.environ.items() if not key.startswith("REEF_")}
    processes: list[subprocess.Popen] = []
    try:
        for component in ("reef", "generator"):
            log_path = tmp_path / f"{component}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "reef", "serve", "-c", str(path), "--component", component],
                    cwd=tmp_path,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            processes.append(process)
            wait_for_health(process, reef_url if component == "reef" else generator_url, log_path)
        # The generator reaches Reef using the declared address and Reef's bearer token.
        request = Request(
            generator_url + "/proposals",
            data=json.dumps(
                {"scenario": "test", "request": {"target": "write a small task", "asset_files": {}}}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            job_id = json.load(response)["job"]
        deadline = time.monotonic() + 15
        while True:
            with urlopen(generator_url + f"/jobs/{job_id}", timeout=5) as response:
                job = json.load(response)
            if job["state"] in {"done", "failed"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert job["state"] == "done", job
        assert job["result"]["record_id"]
        assert job["result"]["refusal"]
        runtime_path = tmp_path / ".reef/run/generator/runtime.yaml"
        assert yaml.safe_load(runtime_path.read_text())["endpoints"] == deployment["endpoints"]
        reef_process, generator_process = processes
        worker_path = tmp_path / ".reef/run/generator.worker.json"
        child_pid = json.loads(worker_path.read_text())["pids"]["generator"]
        os.kill(child_pid, signal.SIGTERM)
        assert generator_process.wait(timeout=15) == 1
        wait_for_health(reef_process, reef_url, tmp_path / "reef.log")
        with (tmp_path / "generator-restart.log").open("w") as log:
            restarted = subprocess.Popen(
                [sys.executable, "-m", "reef", "serve", "-c", str(path), "--component", "generator"],
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        processes.append(restarted)
        wait_for_health(restarted, generator_url, tmp_path / "generator-restart.log")
        reef_process.terminate()
        assert reef_process.wait(timeout=15) == 0
        wait_for_health(restarted, generator_url, tmp_path / "generator-restart.log")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        provider.shutdown()
        provider.server_close()
        provider_thread.join(timeout=5)
