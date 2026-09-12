"""Weight recipes assemble their driver and bridge connection through the shared config path."""

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from reef.cli import main
from reef.service.deploy import orchestrator
from reef.service.deploy.config import interpolate_config, validate_services
from reef.service.deploy.orchestrator import _Stack, resolve_deployment_config
from reef.service.deploy.provider import command_line_config

RECIPE = "recipes.sao.recipe:SAORecipe"


def training_config():
    return {
        "schema-version": 2,
        "recipe": {"implementation": RECIPE, "config": {"batch-size": 1}},
        "inference": {"model-path": "/models/demo"},
        "training": {"options": {"lr": 0.000001, "use-critic": True}},
    }


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith("REEF_") or key in {"RAY_ADDRESS", "SLIME_ARGS_FILE"}:
            monkeypatch.delenv(key)


def test_cli_and_yaml_share_selected_recipe_and_native_option_parsing(tmp_path):
    overrides = {
        "recipe.implementation": RECIPE,
        "inference.model-path": "/models/demo",
        "recipe.config.batch-size": "2",
        "training.options.lr": "0.000002",
        "training.options.use-critic": "true",
        "training.ray-namespace": "custom",
        "training.ray-actor-name": "bridge-custom",
        "training.ready-timeout": "45",
    }
    file_config, _ = resolve_deployment_config(training_config(), overrides, tmp_path / "serve.yaml")
    cli_config, _ = resolve_deployment_config(command_line_config({}), overrides, tmp_path / "cli", standard=True)
    for config in (file_config, cli_config):
        reef = config["reef"]
        assert reef["batch_size"] == 2
        assert reef["training_backend_options"] == {
            "lr": "0.000002",
            "use-critic": True,
            "hf-checkpoint": "${reef.model_path}",
        }
        assert reef["training_backend"] == "slime"
        assert config["execution"] == {"training": "ray", "rollout": "ray"}
        driver, http = validate_services(config, "test")
        assert driver["command"] == [sys.executable, "-m", "reef.service.slime_driver"]
        assert driver["ready_timeout"] == 45
        assert interpolate_config(config, driver["env"]["REEF_RAY_NAMESPACE"]) == "custom"
        assert interpolate_config(config, driver["env"]["REEF_RAY_ACTOR_NAME"]) == "bridge-custom"
        assert http["depends_on"] == [driver["name"]]
        assert driver["executor"] == http["executor"] == "uni"
        assert "inference_url" not in reef
        assert "SGLangChatTrainingInferenceBackend" in reef["inference_backend_factory"]


@pytest.mark.parametrize(
    "override,match",
    [
        ({"inference.model-path": ""}, "requires --inference.model-path"),
        ({"training.backend": "unknown"}, "supports training.backend: slime"),
        ({"training.ready-timeout": "0"}, "must be positive"),
        ({"training.timeout-s": "0"}, "must be positive"),
        ({"training.options.hf-checkpoint": "different/model"}, "must match inference.model-path"),
        ({"training.options.ready-file": "/tmp/marker"}, "managed by Reef"),
        ({"training.options.ready": "/tmp/marker"}, "managed by Reef"),
        ({"inference.upstream-url": "http://localhost:8000"}, "training-owned inference"),
        ({"inference.url": "http://localhost:8000"}, "connection from the bridge"),
        ({"inference.options.tp-size": "2"}, "Slime owns inference workers"),
        ({"inference.tensor-parallel-size": "2"}, "Slime owns inference workers"),
        ({"execution.training.backend": "uni"}, "requires execution.training.backend: ray"),
        ({"recipe.config.batch-szie": "2"}, "unknown configuration flag"),
    ],
)
def test_invalid_training_inputs_fail_before_downloads_and_processes(tmp_path, monkeypatch, capsys, override, match):
    path = tmp_path / "serve.yaml"
    path.write_text(yaml.safe_dump(training_config()))

    def unexpected(*args, **kwargs):
        pytest.fail("invalid training input reached downloads or process startup")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", unexpected)
    monkeypatch.setattr(orchestrator, "_Stack", unexpected)
    argv = ["serve", "-c", str(path)]
    for key, value in override.items():
        argv.extend([f"--{key}", value])
    with pytest.raises(SystemExit) as result:
        main(argv)
    assert result.value.code == 2
    assert match in capsys.readouterr().err


def test_explicit_services_preserve_custom_training_topology(tmp_path):
    raw = training_config()
    raw["training"]["backend"] = "custom"
    raw["services"] = [{"name": "custom-driver", "command": ["custom"]}]
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert config["services"] == raw["services"]
    assert "execution" not in config
    assert "inference_backend_factory" not in config["reef"]


@pytest.mark.parametrize("checkpoint", ["/models/demo", "${inference.model-path}"])
def test_native_checkpoint_tracks_the_resolved_model_path(tmp_path, checkpoint):
    raw = training_config()
    raw["training"]["options"]["hf-checkpoint"] = checkpoint
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    config["reef"]["model_path"] = "/downloaded/snapshot"
    assert (
        interpolate_config(config, config["reef"]["training_backend_options"]["hf-checkpoint"])
        == "/downloaded/snapshot"
    )


def test_matching_home_relative_model_and_checkpoint_are_accepted(tmp_path):
    raw = training_config()
    raw["inference"]["model-path"] = "~/models/demo"
    raw["training"]["options"]["hf-checkpoint"] = "~/models/demo"
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert config["reef"]["training_backend_options"]["hf-checkpoint"] == "${reef.model_path}"


def test_cli_only_training_downloads_once_and_transports_the_resolved_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloads, configs, paths = [], [], []
    from reef.service.deploy import config as config_module

    def download(model):
        downloads.append(model)
        return "/downloaded/snapshot"

    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            configs.append(config_module.load_config(config_path))
            paths.append(config_path)
            assert config_path.stat().st_mode & 0o077 == 0

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(config_module, "resolve_hf_snapshot", download)
    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    with pytest.raises(SystemExit) as result:
        main(
            [
                "serve",
                "--recipe.implementation",
                RECIPE,
                "--inference.model-path",
                "org/model",
                "--recipe.config.batch-size",
                "2",
                "--training.options.use-critic",
                "true",
            ]
        )
    assert result.value.code == 0
    assert downloads == ["org/model"]
    reef = configs[0]["reef"]
    assert reef["model_path"] == "/downloaded/snapshot"
    assert reef["batch_size"] == 2
    assert interpolate_config(configs[0], reef["training_backend_options"]["hf-checkpoint"]) == reef["model_path"]
    assert not paths[0].exists()
    assert not (tmp_path / "reef.yaml").exists()


def test_training_planning_does_not_import_gpu_packages(tmp_path):
    script = """
import sys
class NoTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'slime', 'ray', 'sglang', 'megatron'}:
            raise RuntimeError('unexpected training import: ' + fullname)
sys.meta_path.insert(0, NoTrainingImports())
from reef.service.deploy.orchestrator import resolve_deployment_config
import json
config, _ = resolve_deployment_config(json.loads(sys.argv[1]), None, sys.argv[2])
assert len(config['services']) == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(training_config()), str(tmp_path / "serve.yaml")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("external_cluster,driver_fails", [(False, False), (True, False), (False, True)])
def test_generated_services_share_connection_wait_for_bridge_and_clean_up(
    tmp_path, monkeypatch, external_cluster, driver_fails
):
    # Real local workers and readiness probes; only the GPU workloads and Ray lease are stand-ins.
    released = []

    def acquire(address):
        assert not external_cluster
        return SimpleNamespace(address="127.0.0.1:6379", close=lambda: released.append(True))

    monkeypatch.setattr(orchestrator, "acquire_ray_runtime", acquire)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    raw = training_config()
    raw["service"] = {"port": port, "ready-timeout": 15}
    raw["training"].update({"ray-namespace": "test-namespace", "ray-actor-name": "test-actor"})
    if external_cluster:
        raw["training"]["ray-address"] = "127.0.0.1:6380"
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    driver, http = config["services"]
    driver_script = """
import json, os, time
from pathlib import Path
keys = ['RAY_ADDRESS', 'REEF_RAY_NAMESPACE', 'REEF_RAY_ACTOR_NAME', 'SLIME_ARGS_FILE']
Path('driver-env.json').write_text(json.dumps({key: os.environ[key] for key in keys}))
Path(os.environ['REEF_BRIDGE_READY_FILE']).write_text('reef-slime-bridge-ready')
time.sleep(120)
"""
    http_script = """
import json, os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import yaml
assert Path('stack/slime-driver/bridge.ready').read_text() == 'reef-slime-bridge-ready'
config = yaml.safe_load(Path(os.environ['REEF_CONFIG']).read_text())
Path('http-config.json').write_text(json.dumps(config))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
HTTPServer(('127.0.0.1', config['reef']['port']), Handler).serve_forever()
"""
    driver["command"] = [sys.executable, "-c", "raise SystemExit(7)" if driver_fails else driver_script]
    http["command"] = [sys.executable, "-c", http_script]
    for service in (driver, http):
        service["cwd"] = str(tmp_path)
    run_dir = tmp_path / "stack"
    run_dir.mkdir()
    stack = _Stack(config, validate_services(config, "test.yaml"), run_dir, 15, tmp_path / "serve.yaml")
    try:
        if driver_fails:
            with pytest.raises(RuntimeError, match=r"slime-driver.*exited"):
                stack.start()
            assert not (tmp_path / "http-config.json").exists()
        else:
            stack.start()
            environment = json.loads((tmp_path / "driver-env.json").read_text())
            child = json.loads((tmp_path / "http-config.json").read_text())["reef"]
            assert environment["RAY_ADDRESS"] == child["ray_address"]
            assert environment["REEF_RAY_NAMESPACE"] == child["ray_namespace"] == "test-namespace"
            assert environment["REEF_RAY_ACTOR_NAME"] == child["ray_actor_name"] == "test-actor"
            assert environment["SLIME_ARGS_FILE"] == ""
            assert child["ray_address"] == ("127.0.0.1:6380" if external_cluster else "127.0.0.1:6379")
    finally:
        stack.shutdown(grace=1)
    assert released == ([] if external_cluster else [True])
    for path in run_dir.glob("*.worker.json"):
        for pid in json.loads(path.read_text())["pids"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
