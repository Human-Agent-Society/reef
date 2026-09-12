"""External-provider startup needs neither YAML nor optional training packages."""

from __future__ import annotations

import json
import os
import re
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

import reef
from reef.cli import main
from reef.service.deploy import orchestrator
from reef.service.deploy.config import DeployConfigError, load_config, validate_services
from reef.service.deploy.provider import assemble_provider_services
from reef.service.deploy.settings import service_settings_from_config


@pytest.fixture
def provider_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("REEF_"):
            monkeypatch.delenv(key)


@pytest.fixture
def captured_stack(monkeypatch):
    captured = {}

    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            captured.update(config=load_config(config_path), path=config_path, run_dir=run_dir.resolve())
            assert config_path.stat().st_mode & 0o077 == 0

        def start(self):
            captured["started"] = True

        def block(self):
            pass

        def shutdown(self):
            captured["stopped"] = True

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    return captured


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize(
    "spelling", ["--inference.upstream-model", "--upstream-model", "--upstream_model", "--reef.upstream-model"]
)
def test_provider_settings_use_shared_types_and_reach_the_child(tmp_path, monkeypatch, captured_stack, spelling):
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "env-model")
    monkeypatch.setenv("REEF_UPSTREAM_API_KEY", "test-provider-secret")
    monkeypatch.setenv("REEF_TOKEN", "test-reef-secret")
    with pytest.raises(SystemExit) as result:
        main(
            [
                "serve",
                "--inference.upstream-url",
                "http://localhost:8000",
                spelling,
                "00123",
                "--reef.port",
                "9001",
                "--no-reef.allow-implicit-scenario-creation",
                "--reef.tokens",
                "[]",
            ]
        )
    assert result.value.code == 0
    config = captured_stack["config"]
    settings = service_settings_from_config(config)
    assert settings.upstream_model == "00123"
    assert settings.upstream_api_key == "test-provider-secret"
    assert settings.tokens == ("test-reef-secret",)
    assert settings.port == 9001 and settings.host == "127.0.0.1"
    assert settings.allow_implicit_scenario_creation is False
    assert settings.recipe == "recipe" and settings.model_path is None
    assert config["services"][0]["ready"][-1] == "http://127.0.0.1:9001/healthz"
    assert captured_stack["run_dir"] == tmp_path / ".reef/run"
    assert captured_stack["started"] and captured_stack["stopped"]
    assert not captured_stack["path"].exists()
    assert not (tmp_path / "reef.yaml").exists()


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize("source", ["environment", "directory"])
def test_without_c_ignores_files_and_config_environment(tmp_path, monkeypatch, captured_stack, source):
    (tmp_path / "reef.yaml").write_text("invalid: [yaml")
    if source == "environment":
        monkeypatch.setenv("REEF_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(SystemExit) as result:
        main(["serve", "--model", "ollama/my-model"])
    assert result.value.code == 0
    settings = service_settings_from_config(captured_stack["config"])
    assert settings.upstream_url == "http://127.0.0.1:11434"
    assert settings.upstream_model == "my-model"


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize("filename", ["", "missing.yaml", "reef.yaml"])
def test_selected_unreadable_file_does_not_fall_back_to_provider(tmp_path, filename):
    arguments = ["serve", "-c", filename, "--model", "ollama/my-model"]
    if filename == "reef.yaml":
        (tmp_path / "reef.yaml").write_text("invalid: [yaml")
    with pytest.raises(SystemExit) as result:
        main(arguments)
    assert result.value.code == 2
    assert not (tmp_path / ".reef").exists()


@pytest.mark.usefixtures("provider_environment")
def test_provider_startup_can_take_all_inputs_from_declared_environment(monkeypatch, captured_stack):
    monkeypatch.setenv("REEF_UPSTREAM_URL", "http://localhost:8000")
    monkeypatch.setenv("REEF_UPSTREAM_MODEL", "env-model")
    with pytest.raises(SystemExit) as result:
        main(["serve"])
    assert result.value.code == 0
    assert service_settings_from_config(captured_stack["config"]).upstream_model == "env-model"


@pytest.mark.usefixtures("provider_environment")
def test_model_shorthand_accepts_explicit_provider_key(captured_stack):
    with pytest.raises(SystemExit) as result:
        main(["serve", "--model", "openai/demo", "--reef.upstream-api-key", "explicit-key"])
    assert result.value.code == 0
    settings = service_settings_from_config(captured_stack["config"])
    assert settings.upstream_url == "https://api.openai.com"
    assert settings.upstream_api_key == "explicit-key"


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize("field", ["upstream_model", "upstream_api_key"])
def test_model_defaults_preserve_the_last_explicit_alias(captured_stack, field):
    with pytest.raises(SystemExit) as result:
        main(["serve", "--model", "ollama/demo", f"--{field.replace('_', '-')}", "first", f"--{field}", "last"])
    assert result.value.code == 0
    assert captured_stack["config"]["reef"][field] == "last"


@pytest.mark.usefixtures("provider_environment")
def test_provider_temp_config_is_removed_if_stack_preparation_fails(tmp_path, monkeypatch):
    written = []
    original_write = orchestrator._write_override_config

    def write(config):
        path = original_write(config)
        written.append(path)
        return path

    monkeypatch.setattr(orchestrator, "_write_override_config", write)
    (tmp_path / ".reef").write_text("not a directory")
    with pytest.raises(OSError):
        orchestrator._run_orchestrator(None, {"upstream-url": "http://localhost:8000", "upstream-model": "demo"})
    assert len(written) == 1 and not written[0].exists()


@pytest.mark.parametrize("ready", [[], [1], [""], 42, {}])
def test_invalid_readiness_commands_are_rejected(ready):
    with pytest.raises(DeployConfigError, match="ready must be"):
        validate_services({"services": [{"name": "reef", "command": ["python"], "ready": ready}]}, "stack.yaml")


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize(
    "options,match",
    [
        ([], "requires --inference.upstream-url"),
        (["--upstream-url", "http://localhost:8000"], "requires --inference.upstream-url"),
        (["--model", "bare-model"], "requires --inference.upstream-url"),
        (["--model", "ollama/demo", "--upstream-url", "file:///private/provider-secret"], "HTTP\\(S\\)"),
        (["--model", "ollama/demo", "--port", "0"], "between 1 and 65535"),
        (["--model", "ollama/demo", "--port", "nope"], "reef.port"),
        (["--model", "ollama/demo", "--host", ""], "must be non-empty"),
        (["--model", "ollama/demo", "--no-config"], "unknown configuration flag"),
        (["--model", "ollama/demo", "--model-path", "org/model"], "cannot be combined"),
        (
            ["--model", "ollama/demo", "--recipe.implementation", "missing.module:Recipe"],
            "cannot import recipe reference",
        ),
        (["--model", "ollama/demo", "--upstream-modle", "typo"], "unknown configuration flag"),
        (["--model", "ollama/demo", "--services", "[]"], "unknown configuration flag"),
        (["--model", "ollama/demo", "--inference-timeout-s", "0"], "must be positive"),
    ],
)
def test_invalid_provider_inputs_fail_before_downloads_or_processes(tmp_path, monkeypatch, capsys, options, match):
    def must_not_run(*args, **kwargs):
        pytest.fail("invalid configuration must fail before downloads or processes")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", must_not_run)
    monkeypatch.setattr(orchestrator, "_Stack", must_not_run)
    with pytest.raises(SystemExit) as result:
        main(["serve", *options])
    assert result.value.code == 2
    stderr = capsys.readouterr().err
    assert re.search(match, stderr)
    assert "provider-secret" not in stderr and "Traceback" not in stderr
    assert not (tmp_path / ".reef").exists()


@pytest.mark.parametrize("host,probe_host", [("0.0.0.0", "127.0.0.1"), ("::", "[::1]"), ("::1", "[::1]")])
def test_provider_probe_uses_a_reachable_bind_address(host, probe_host):
    config = {
        "reef": {"recipe": "recipe", "host": host, "upstream_url": "http://localhost:8000", "upstream_model": "demo"}
    }
    assemble_provider_services(config)
    services = validate_services(config, "<command line>")
    assert services[0]["ready"][-1] == f"http://{probe_host}:8900/healthz"


def _unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
def test_provider_cli_serves_and_records_feedback_without_yaml_or_gpu_imports(tmp_path):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers["Authorization"], payload))
            body = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hello"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    # Fail any optional training import in both the launcher and HTTP child.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class NoTrainingImports:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in {'torch', 'slime', 'ray', 'sglang', 'megatron'}:\n"
        "            raise RuntimeError('unexpected training import: ' + fullname)\n"
        "sys.meta_path.insert(0, NoTrainingImports())\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("REEF_")}
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(Path(reef.__file__).resolve().parents[1])))
    env["REEF_UPSTREAM_API_KEY"] = "test-upstream-key"
    port = _unused_port()
    base_url = f"http://127.0.0.1:{port}"
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    log_path = tmp_path / "launch.log"
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "reef.cli",
                    "serve",
                    "--inference.upstream-url",
                    f"http://127.0.0.1:{provider.server_port}",
                    "--inference.upstream-model",
                    "demo",
                    "--reef.port",
                    str(port),
                    "--reef.token",
                    "test-token",
                ],
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        with urlopen(f"{base_url}/healthz", timeout=1) as response:
                            assert response.status == 200
                        break
                    except (OSError, URLError):
                        assert time.monotonic() < deadline, log_path.read_text()
                        time.sleep(0.1)
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer test-token",
                    "x-reef-scenario": "demo",
                }
                payload = {"model": "demo", "messages": [{"role": "user", "content": "hello"}], "temperature": 0.3}
                with urlopen(
                    Request(f"{base_url}/v1/chat/completions", json.dumps(payload).encode(), headers), timeout=15
                ) as response:
                    receipt = response.headers["x-reef-agent-record-id"]
                    assert json.load(response)["choices"][0]["message"]["content"] == "hello"
                assert requests == [("/v1/chat/completions", "Bearer test-upstream-key", payload)]
                report = {"score": 1.0, "references": [receipt]}
                with urlopen(
                    Request(f"{base_url}/reef/report", json.dumps(report).encode(), headers), timeout=15
                ) as response:
                    assert json.load(response)["scenario"] == "demo"
            finally:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            assert process.returncode == 0, log_path.read_text()
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
    assert not (tmp_path / "reef.yaml").exists()
    assert "test-upstream-key" not in log_path.read_text()
    with socket.socket() as sock:
        assert sock.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.usefixtures("provider_environment")
@pytest.mark.parametrize("model_flag", ["--inference.upstream-model", "--upstream_model"])
def test_profile_accepts_explicit_provider_fields_without_model_shorthand(monkeypatch, captured_stack, model_flag):
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", Path(__file__).resolve().parents[2])
    with pytest.raises(SystemExit) as result:
        main(
            [
                "serve",
                "--recipe",
                "harness-evolve",
                "--inference.upstream-url",
                "http://127.0.0.1:11434",
                model_flag,
                "gemma4:26b",
            ]
        )
    assert result.value.code == 0
    config = captured_stack["config"]["reef"]
    assert config["upstream_url"] == "http://127.0.0.1:11434"
    assert config["upstream_model"] == "gemma4:26b"
    assert config["recipe"] == "reef.recipe.cordis:CordisRecipe"
    assert config["data"]["training_mode"] == "hybrid"
    assert captured_stack["started"] and captured_stack["stopped"]
