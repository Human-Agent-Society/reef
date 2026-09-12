"""Managed SGLang launch contracts, using a CPU stand-in for lifecycle tests."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest

import reef
from reef.cli import main
from reef.service.deploy import orchestrator
from reef.service.deploy.config import load_config
from reef.service.deploy.process import _command_argv


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("REEF_"):
            monkeypatch.delenv(key)


@pytest.mark.usefixtures("isolated")
def test_model_is_resolved_once_and_shared_with_the_children(tmp_path, monkeypatch):
    captured = {}
    model = tmp_path / "downloaded model"
    model.mkdir()

    def resolve(config):
        captured["original"] = config["reef"]["model_path"]
        config["reef"]["model_path"] = str(model)
        return True

    monkeypatch.setattr(orchestrator, "resolve_model_paths", resolve)

    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            captured["config"] = load_config(config_path)

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    with pytest.raises(SystemExit) as result:
        main(
            [
                "serve",
                "--inference.model-path",
                "org/model",
                "--inference.tensor-parallel-size",
                "2",
                "--inference.options.mem-fraction-static",
                "0.8",
            ]
        )
    assert result.value.code == 0
    config = captured["config"]
    engine, http = config["services"]
    command = _command_argv(config, engine["command"])
    assert config["reef"]["tensor_parallel_size"] == 2 and captured["original"] == "org/model"
    assert command[command.index("--model-path") + 1] == str(model)
    assert "--mem-fraction-static=0.8" in command
    assert command[command.index("--served-model-name") + 1] == "org/model"
    assert http["depends_on"] == ["sglang"]
    assert config["reef"]["upstream_url"] == engine["endpoint"]
    assert config["reef"]["upstream_model"] == "org/model"
    assert config["reef"]["inference_backend"] == "sglang"


@pytest.mark.usefixtures("isolated")
@pytest.mark.parametrize(
    "extra,match",
    [
        (["--tensor-parallel-size", "0"], "must be positive"),
        (["--tensor-parallel-size", "oops"], "tensor_parallel_size"),
        (["--inference-backend", "unknown"], "supports"),
        (["--upstream-url", "http://localhost:8000"], "cannot be combined"),
        (["--upstream-model", "remote"], "cannot be combined"),
        (["--port", "0"], "valid --port"),
    ],
)
def test_local_invalid_settings_fail_before_downloads(monkeypatch, capsys, extra, match):
    def unexpected(*args):
        pytest.fail("invalid settings must not download models")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", unexpected)
    with pytest.raises(SystemExit) as result:
        main(["serve", "--inference.model-path", "org/model", *extra])
    assert result.value.code == 2
    assert match in capsys.readouterr().err


_FAKE_SERVER = """import argparse, json, os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
parser = argparse.ArgumentParser()
for key in ('model-path', 'served-model-name', 'host', 'port', 'tp'):
    parser.add_argument('--' + key, required=True)
args = parser.parse_args()
Path('engine.json').write_text(json.dumps({**vars(args), 'pid': os.getpid()}))
if os.environ.get('FAKE_SGLANG_WAIT'):
    time.sleep(60)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        Path('request.json').write_text(json.dumps(request))
        body = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'local reply'}}]}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
HTTPServer((args.host, int(args.port)), Handler).serve_forever()
"""


def _wait_file(path, process, log):
    deadline = time.monotonic() + 20
    while not path.exists() or not path.read_text():
        assert process.poll() is None, log.read_text()
        assert time.monotonic() < deadline, log.read_text()
        time.sleep(0.05)


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["serve", "interrupt_startup", "reef_failure"])
def test_managed_lifecycle_with_cpu_engine_standin(tmp_path, mode):
    # This package simulates engine HTTP behavior, not GPU inference.
    (tmp_path / "sglang").mkdir()
    (tmp_path / "sglang/__init__.py").write_text("")
    (tmp_path / "sglang/launch_server.py").write_text(_FAKE_SERVER)
    model = tmp_path / "model"
    model.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("REEF_")}
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(Path(reef.__file__).resolve().parents[1])))
    if mode == "interrupt_startup":
        env["FAKE_SGLANG_WAIT"] = "1"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        if mode != "reef_failure":
            reserved.close()
        log_path = tmp_path / "launch.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "reef.cli", "serve", "--inference.model-path", str(model), "--port", str(port)],
                cwd=tmp_path,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_file(tmp_path / "engine.json", process, log_path)
                engine = json.loads((tmp_path / "engine.json").read_text())
                if mode == "serve":
                    deadline = time.monotonic() + 20
                    while True:
                        assert process.poll() is None, log_path.read_text()
                        try:
                            with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1):
                                break
                        except (OSError, URLError):
                            assert time.monotonic() < deadline, log_path.read_text()
                            time.sleep(0.1)
                    payload = {"model": str(model), "messages": [{"role": "user", "content": "hello"}]}
                    headers = {"Content-Type": "application/json", "x-reef-scenario": "local"}
                    with urlopen(
                        Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(payload).encode(), headers),
                        timeout=10,
                    ) as response:
                        assert response.headers["x-reef-agent-record-id"]
                        assert json.load(response)["choices"][0]["message"]["content"] == "local reply"
                    assert json.loads((tmp_path / "request.json").read_text()) == payload
                if mode != "reef_failure":
                    process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=20) == (1 if mode == "reef_failure" else 0), log_path.read_text()
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        with pytest.raises(ProcessLookupError):
            os.kill(engine["pid"], 0)
        with socket.socket() as check:
            assert check.connect_ex(("127.0.0.1", int(engine["port"]))) != 0
    assert not (tmp_path / "reef.yaml").exists()


def test_backend_definition_controls_launch_without_backend_specific_code(monkeypatch):
    from reef.service.deploy.inference import INFERENCE_BACKENDS, InferenceBackend, prepare_inference
    from reef.service.deploy.settings import ServiceSettings

    monkeypatch.setitem(
        INFERENCE_BACKENDS,
        "example",
        InferenceBackend(
            command=(
                "{python}",
                "-m",
                "example.engine",
                "--model",
                "{model_path}",
                "--parallel",
                "{tensor_parallel_size}",
            ),
            health_path="/readyz",
        ),
    )
    config = {"reef": {"model_path": "org/model"}}
    settings = ServiceSettings(
        recipe="recipe",
        model_path="org/model",
        inference_backend="example",
        tensor_parallel_size=2,
        inference_options={"max-tokens": 128},
    )
    service = prepare_inference(config, settings)
    assert service["name"] == "example"
    assert service["command"][1:] == [
        "-m",
        "example.engine",
        "--model",
        "${reef.model_path}",
        "--parallel",
        "2",
        "--max-tokens=128",
    ]
    assert service["ready"][-1] == service["endpoint"] + "/readyz"
    assert config["reef"]["upstream_url"] == service["endpoint"]
