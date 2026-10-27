"""Weight recipes assemble their inference component and HTTP service through the shared config path."""

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from reef.cli import main
from reef.runtime.jobs import ReleaseFormat
from reef.service.deploy import inference as inference_deploy
from reef.service.deploy import orchestrator
from reef.service.deploy.config_utils import DeployConfigError, interpolate_config
from reef.service.deploy.execution import validate_services
from reef.service.deploy.inference import (
    INFERENCE_COMPONENT_SERVICE,
    command_line_config,
    free_loopback_port,
    local_inference_component,
)
from reef.service.deploy.orchestrator import _Stack, resolve_deployment_config
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.service.deploy.serving import serving_runtime_config
from reef.train.slime_backend.launch import SlimeDeployment, global_batch_size, slime_arguments

RECIPE = "recipes.sao.recipe:SAORecipe"
CONTROL_TOKEN_ENV = "REEF_REMOTE_CONTROL_TOKEN"
FULL_WEIGHTS = ReleaseFormat("full")
INFERENCE_TOKEN_ENV = "REEF_REMOTE_INFERENCE_TOKEN"


def training_config():
    return {
        "recipe": {"implementation": RECIPE},
        "pipeline": {"task2job": {"config": {"buffer_size": 4}}},
        "inference": {"model-path": "/models/demo"},
        "training": {"options": {"lr": 0.000001, "use-critic": True, "global-batch-size": 4}},
    }


def slime_runtime_config(config):
    return SlimeDeployment().component_config(asdict(service_config_from_mapping(config)), loss_family="")


def slime_argv(config):
    """The arguments the HTTP service gives Slime's parser for this resolved deployment config."""
    return slime_arguments(slime_runtime_config(config)["settings"], loss_family="", environ={})


def serving_config(config, release_format=FULL_WEIGHTS):
    """The serving runtime config that the HTTP service builds for Slime's releases of ``release_format``."""
    return serving_runtime_config(service_config_from_mapping(config), SlimeDeployment(), release_format)


@pytest.fixture(autouse=True)
def clean_environment():
    # Deployment resolution generates the inference component's tokens into os.environ; drop them afterwards.
    with patch.dict(os.environ):
        for key in [key for key in os.environ if key.startswith("REEF_") or key == "RAY_ADDRESS"]:
            os.environ.pop(key)
        yield


def test_cli_and_yaml_share_selected_recipe_and_native_option_parsing(tmp_path):
    overrides = {
        "recipe.implementation": RECIPE,
        "inference.model-path": "/models/demo",
        "pipeline.task2job.config.buffer_size": "8",
        "training.options.lr": "0.000002",
        "training.options.use-critic": "true",
        "training.options.global-batch-size": "4",
        "training.ray-namespace": "custom",
        "training.ready-timeout": "45",
    }
    file_config, _ = resolve_deployment_config(training_config(), overrides, tmp_path / "serve.yaml")
    cli_config, _ = resolve_deployment_config(command_line_config({}), overrides, tmp_path / "cli")
    for config in (file_config, cli_config):
        reef = config["reef"]
        assert config["pipeline"] == {"task2job": {"config": {"buffer_size": 8}}}
        assert reef["training_backend_options"] == {
            "lr": "0.000002",
            "use-critic": True,
            "global-batch-size": "4",
            "hf-checkpoint": "${reef.model_path}",
        }
        assert reef["training_backend"] == "slime"
        assert reef["ray_namespace"] == "custom"
        assert config["execution"] == {"training": "ray"}
        assert slime_runtime_config(config)["execution"] == {"execution": {"training": "ray"}}
        component, http = validate_services(config, "test")
        assert component["name"] == INFERENCE_COMPONENT_SERVICE
        assert component["command"][1:3] == ["-m", "reef.inference.remote"]
        assert http["command"] == [sys.executable, "-m", "reef.service"]
        assert component["ready_timeout"] == http["ready_timeout"] == 45
        assert http["depends_on"] == [component["name"]]
        assert component["executor"] == http["executor"] == "uni"
        assert "inference_url" not in reef
        assert reef["inference_remote"]["url"] == component["endpoint"]
        assert reef["inference_backend"] == "sglang"


@pytest.mark.parametrize(
    "override,match",
    [
        ({"inference.model-path": ""}, "must be non-empty"),
        ({"training.backend": "unknown"}, "unknown training backend"),
        ({"training.ready-timeout": "0"}, "must be positive"),
        ({"training.timeout-s": "0"}, "must be positive"),
        ({"training.options.hf-checkpoint": "different/model"}, "must match inference.model-path"),
        ({"training.colocate": "true"}, "unknown configuration flag --training.colocate"),
        ({"inference.upstream-url": "http://localhost:8000"}, "managed inference"),
        ({"inference.url": "http://localhost:8000"}, "inference.url is reserved"),
        ({"inference.options.tp-size": "2"}, "managed by Reef"),
        ({"inference.num-gpus": "3", "inference.tensor-parallel-size": "2"}, "must be divisible"),
        ({"inference.num-gpus": "0"}, "must be positive"),
        ({"inference.backend": "none", "training.colocate": "true"}, "unknown configuration flag --training.colocate"),
        ({"inference.backend": "none", "inference.num-gpus": "8"}, "remove inference.num-gpus"),
        (
            {"inference.backend": "none", "inference.tensor-parallel-size": "2", "inference.options.dtype": "bf16"},
            "remove inference.tensor-parallel-size, inference.options",
        ),
        (
            {
                "inference.backend": "none",
                "inference.handler-factory": "reef.inference.sglang.chat.SGLangInferenceHandler",
            },
            "remove inference.handler-factory",
        ),
        ({"inference.backend": "none", "teacher.model-path": "/models/teacher"}, "remove teacher.model-path"),
        ({"inference.backend": "none", "training.options.rollout-num-gpus": "1"}, "configures inference"),
        ({"inference.tensor-parallel-size": "-1"}, "must be positive"),
        ({"inference.num-gpus": "1.5"}, "valid int"),
        ({"training.options.rollout_num_gpus": "2"}, "configures inference"),
        ({"training.options.rollout-num": "2"}, "configures inference"),
        ({"training.options.sglang-context-length": "1024"}, "configures inference"),
        ({"training.options.sglang-config": "engines.yaml"}, "configures inference"),
        ({"training.options.rollout-external": "true"}, "configures inference"),
        ({"inference.options.config": "engines.yaml"}, "managed by Reef"),
        ({"inference.options.pp-size": "2"}, "managed by Reef"),
        ({"inference.options.sglang-context-length": "1024"}, "without the sglang- prefix"),
        ({"execution.training.backend": "uni"}, "requires execution.training.backend: ray"),
        ({"recipe.config.max-stalenes": "2"}, "unknown configuration flag"),
        (
            {"inference.backend": "vllm", "inference.num-gpus": "2", "inference.tensor-parallel-size": "1"},
            "vllm serves one engine",
        ),
        ({"inference.serving": "trainer"}, "needs a training backend that serves its own releases"),
        ({"inference.serving": "elsewhere"}, "engines, trainer, none or module:Factory"),
        ({"inference.backend": "tinker-engine"}, "unknown inference backend 'tinker-engine'"),
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


def test_cli_only_training_resolves_public_config_references(tmp_path):
    config, _ = resolve_deployment_config(
        command_line_config({}),
        {
            "recipe.implementation": RECIPE,
            "inference.model-path": "/models/demo",
            "training.config.checkpoint_dir": "work/checkpoints",
            "training.options.hf-checkpoint": "${inference.model-path}",
            "training.options.save": "${training.config.checkpoint_dir}/megatron",
            "training.options.global-batch-size": "4",
        },
        tmp_path / "cli",
    )
    options = config["reef"]["training_backend_options"]
    assert interpolate_config(config, options["hf-checkpoint"]) == "/models/demo"
    assert interpolate_config(config, options["save"]) == "work/checkpoints/megatron"


def test_cli_only_training_downloads_once_and_transports_the_resolved_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloads, configs, paths = [], [], []
    from reef.service.deploy import config_utils as config_module
    from reef.service.deploy import inference

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

    monkeypatch.setattr(inference, "resolve_hf_snapshot", download)
    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    with pytest.raises(SystemExit) as result:
        main(
            [
                "serve",
                "--recipe.implementation",
                RECIPE,
                "--inference.model-path",
                "org/model",
                "--pipeline.task2job.config.buffer_size",
                "8",
                "--training.options.use-critic",
                "true",
                "--training.options.global-batch-size",
                "4",
            ]
        )
    assert result.value.code == 0
    assert downloads == ["org/model"]
    reef = configs[0]["reef"]
    assert reef["model_path"] == "/downloaded/snapshot"
    assert configs[0]["pipeline"] == {"task2job": {"config": {"buffer_size": 8}}}
    assert interpolate_config(configs[0], reef["training_backend_options"]["hf-checkpoint"]) == reef["model_path"]
    assert not paths[0].exists()
    assert not (tmp_path / "reef.yaml").exists()


@pytest.mark.parametrize("recipe", [RECIPE, "recipes.openclawrl.recipe:OpenClawRLRecipe"])
def test_training_planning_does_not_import_gpu_packages(tmp_path, recipe):
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
assert config['services'][-1]['name'] == 'reef'
"""
    raw = training_config()
    raw["recipe"]["implementation"] = recipe
    if "openclawrl" in recipe:
        raw["pipeline"]["record2record"] = {
            "config": {"prm_url": "http://external:23001", "prm_tokenizer_path": "/models/judge"}
        }
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(raw), str(tmp_path / "serve.yaml")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


INFERENCE_SCRIPT = """
import json, os, socket, sys
from pathlib import Path
argv = sys.argv[1:]
keys = ['RAY_ADDRESS', 'REEF_REMOTE_CONTROL_TOKEN', 'REEF_REMOTE_INFERENCE_TOKEN']
Path('inference.json').write_text(json.dumps({'argv': argv, 'env': {key: os.environ.get(key) for key in keys}}))
listener = socket.create_server(('127.0.0.1', int(argv[argv.index('--port') + 1])))
while True:
    listener.accept()[0].close()
"""

HTTP_SCRIPT = """
import json, os, socket
from pathlib import Path
from urllib.parse import urlsplit
from http.server import HTTPServer, BaseHTTPRequestHandler
import yaml
config = yaml.safe_load(Path(os.environ['REEF_CONFIG']).read_text())
remote = urlsplit(config['reef']['inference_remote']['url'])
socket.create_connection((remote.hostname, remote.port), timeout=5).close()
keys = ['RAY_ADDRESS', 'REEF_REMOTE_CONTROL_TOKEN', 'REEF_REMOTE_INFERENCE_TOKEN']
Path('http.json').write_text(json.dumps({'reef': config['reef'], 'env': {key: os.environ.get(key) for key in keys}}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
HTTPServer(('127.0.0.1', config['reef']['port']), Handler).serve_forever()
"""


@pytest.mark.parametrize("external_cluster,inference_fails", [(False, False), (True, False), (False, True)])
def test_generated_services_share_tokens_and_ray_cluster_and_clean_up(
    tmp_path, monkeypatch, external_cluster, inference_fails
):
    # Real local workers and readiness probes; only the engines and the Ray lease are stand-ins.
    released = []

    def acquire(address):
        assert not external_cluster
        return SimpleNamespace(address="127.0.0.1:6379", close=lambda: released.append(True))

    monkeypatch.setattr(orchestrator, "acquire_ray_runtime", acquire)
    raw = training_config()
    raw["reef"] = {"port": free_loopback_port(0), "ready-timeout": 15}
    raw["training"].update({"ray-namespace": "test-namespace", "ready-timeout": 15})
    if external_cluster:
        raw["training"]["ray-address"] = "127.0.0.1:6380"
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    component, http = config["services"]
    arguments = component["command"][3:]
    component["command"] = [sys.executable, "-c", "raise SystemExit(7)" if inference_fails else INFERENCE_SCRIPT]
    component["command"].extend(arguments)
    http["command"] = [sys.executable, "-c", HTTP_SCRIPT]
    for service in (component, http):
        service["cwd"] = str(tmp_path)
    run_dir = tmp_path / "stack"
    run_dir.mkdir()
    stack = _Stack(config, validate_services(config, "test.yaml"), run_dir, 15, tmp_path / "serve.yaml")
    try:
        if inference_fails:
            with pytest.raises(RuntimeError, match=r"inference.*exited"):
                stack.start()
            assert not (tmp_path / "http.json").exists()
        else:
            stack.start()
            started = json.loads((tmp_path / "inference.json").read_text())
            child = json.loads((tmp_path / "http.json").read_text())
            cluster = "127.0.0.1:6380" if external_cluster else "127.0.0.1:6379"
            assert started["env"]["RAY_ADDRESS"] == child["env"]["RAY_ADDRESS"] == child["reef"]["ray_address"]
            assert child["reef"]["ray_address"] == cluster
            assert child["reef"]["ray_namespace"] == "test-namespace"
            # An external cluster is named to the component; a managed one reaches it through RAY_ADDRESS.
            assert ("--ray-address" in started["argv"]) is external_cluster
            assert started["env"][CONTROL_TOKEN_ENV] == child["env"][CONTROL_TOKEN_ENV]
            assert started["env"][INFERENCE_TOKEN_ENV] == child["env"][INFERENCE_TOKEN_ENV]
            assert started["env"][CONTROL_TOKEN_ENV] != started["env"][INFERENCE_TOKEN_ENV]
    finally:
        stack.shutdown(grace=1)
    assert released == ([] if external_cluster else [True])
    for path in run_dir.glob("*.worker.json"):
        for pid in json.loads(path.read_text())["pids"].values():
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)


def test_local_inference_component_listens_on_loopback_and_records_its_address(tmp_path):
    config = {"reef": {}, "run_dir": str(tmp_path / "run")}
    settings = ServiceConfig(
        recipe=RECIPE, port=8900, inference_backend="vllm", ray_address="10.0.0.1:6379", training_ready_timeout=77
    )
    service = local_inference_component(config, settings)
    port = int(service["command"][service["command"].index("--port") + 1])
    assert service["name"] == INFERENCE_COMPONENT_SERVICE
    assert service["endpoint"] == config["reef"]["inference_remote"]["url"] == f"http://127.0.0.1:{port}"
    assert port != settings.port
    assert service["command"][1:] == [
        "-m",
        "reef.inference.remote",
        "--engine",
        "vllm",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--work-dir",
        str(tmp_path / "run" / INFERENCE_COMPONENT_SERVICE),
        "--ray-address",
        "10.0.0.1:6379",
    ]
    assert service["ready"][-1] == str(port)
    assert service["ready_timeout"] == 77
    assert config["reef"]["inference_remote"]["control_token_env"] == CONTROL_TOKEN_ENV
    assert os.environ[CONTROL_TOKEN_ENV] and os.environ[INFERENCE_TOKEN_ENV]
    assert os.environ[CONTROL_TOKEN_ENV] != os.environ[INFERENCE_TOKEN_ENV]


def test_local_inference_component_keeps_operator_tokens_and_defaults_to_sglang(tmp_path, monkeypatch):
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "control")
    monkeypatch.setenv(INFERENCE_TOKEN_ENV, "inference")
    config = {"reef": {}}
    service = local_inference_component(config, ServiceConfig(recipe=RECIPE))
    command = service["command"]
    assert command[command.index("--engine") + 1] == "sglang"
    assert command[command.index("--work-dir") + 1] == str(Path(".reef/run") / INFERENCE_COMPONENT_SERVICE)
    assert "--ray-address" not in command
    assert (os.environ[CONTROL_TOKEN_ENV], os.environ[INFERENCE_TOKEN_ENV]) == ("control", "inference")


def test_local_inference_component_refuses_one_token_for_control_and_inference(monkeypatch):
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "shared")
    monkeypatch.setenv(INFERENCE_TOKEN_ENV, "shared")
    with pytest.raises(DeployConfigError, match="must differ"):
        local_inference_component({"reef": {}}, ServiceConfig(recipe=RECIPE))


def test_free_loopback_port_skips_the_reserved_port(monkeypatch):
    class Listener:
        ports = iter((8900, 41000))

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", next(self.ports))

    assert free_loopback_port(8900) != 8900
    monkeypatch.setattr(inference_deploy.socket, "socket", Listener)
    assert free_loopback_port(8900) == 41000


@pytest.mark.parametrize(
    "change",
    [
        {"inference": {"model-path": "/models/demo", "remote": {"url": "https://gpu-host.example"}}},
        {"inference": {"model-path": "/models/demo", "backend": "none"}},
    ],
    ids=["remote-component", "no-inference"],
)
def test_no_local_inference_component_without_local_engines(tmp_path, change):
    raw = {**training_config(), **change}
    if "backend" in change["inference"]:
        raw["recipe"]["implementation"] = "reef.recipe.sft.recipe:SFTRecipe"
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert [service["name"] for service in config["services"]] == ["reef"]
    assert not config["services"][0].get("depends_on")


def test_teacher_endpoint_reaches_the_http_service_environment(tmp_path):
    config, _ = resolve_deployment_config(
        training_config(),
        {"teacher.model-path": "/models/teacher", "teacher.num-gpus": "1", "teacher.port": "30100"},
        tmp_path / "serve.yaml",
    )
    component, teacher, http = config["services"]
    assert (component["name"], teacher["name"]) == (INFERENCE_COMPONENT_SERVICE, "distill-teacher")
    assert http["depends_on"] == [INFERENCE_COMPONENT_SERVICE, "distill-teacher"]
    assert http["env"] == {"REEF_DISTILL_TEACHER_URL": "${endpoints.distill-teacher}"}
    assert not component.get("env")


@pytest.mark.parametrize("mode", [{}, {"options": {"megatron-lora-rank": 8}}], ids=["full", "lora"])
def test_inference_cli_over_yaml_reaches_the_serving_engines_and_not_slime(tmp_path, mode):
    raw = training_config()
    raw["inference"].update(
        {
            "num-gpus": 2,
            "tensor-parallel-size": 1,
            "options": {"mem_fraction_static": 0.8, "router-port": 30000, "disable-cuda-graph": True},
        }
    )
    raw["training"]["options"].update(mode.get("options", {}))
    overrides = {
        "inference.num-gpus": "4",
        "inference.tensor-parallel-size": "2",
        "inference.options.mem-fraction-static": "0.6",
        "inference.options.disable-cuda-graph": "false",
    }
    config, _ = resolve_deployment_config(raw, overrides, tmp_path / "serve.yaml")
    equivalent = training_config()
    equivalent["training"]["options"].update(mode.get("options", {}))
    equivalent["inference"].update(
        {
            "num-gpus": 4,
            "tensor-parallel-size": 2,
            "options": {"mem-fraction-static": "0.6", "router-port": 30000, "disable-cuda-graph": False},
        }
    )
    file_only, _ = resolve_deployment_config(equivalent, None, tmp_path / "other.yaml")
    argv = slime_argv(config)
    assert argv == slime_argv(file_only)
    # Slime starts no engine and receives no engine flag.
    assert "--rollout-num-gpus=0" in argv
    assert not [argument for argument in argv if argument.startswith(("--sglang-", "--rollout-num-gpus-per-engine"))]
    assert all(not key.startswith(("rollout-num", "sglang-")) for key in config["reef"]["training_backend_options"])
    assert raw["inference"]["num-gpus"] == 2
    assert not {"--colocate", "--offload-rollout", "--offload-train"} & set(argv)
    serving = serving_config(config)
    # Each resolution starts its own local component, on its own port.
    assert {**serving, "remote": None} == {**serving_config(file_only), "remote": None}
    assert (serving["num_gpus"], serving["tensor_parallel_size"]) == (4, 2)
    assert serving["options"] == {"mem-fraction-static": "0.6", "router-port": 30000, "disable-cuda-graph": False}


@pytest.mark.parametrize("option", ["colocate", "offload-rollout"])
def test_native_placement_flags_are_rejected_in_training_options(tmp_path, option):
    raw = training_config()
    raw["training"]["options"][option] = True
    with pytest.raises(DeployConfigError, match=f"training.options.{option} shares GPUs between training and serving"):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")


@pytest.mark.parametrize("batch_size", [None, 0, "0", "1.5", "two", True])
def test_slime_deployment_refuses_a_global_batch_size_that_is_not_a_positive_integer(tmp_path, batch_size):
    raw = training_config()
    if batch_size is None:
        raw["training"]["options"].pop("global-batch-size")
    else:
        raw["training"]["options"]["global-batch-size"] = batch_size
    with pytest.raises(DeployConfigError, match=r"training\.options\.global-batch-size must be a positive integer"):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")


def test_slime_runtime_config_carries_the_resolved_global_batch_size(tmp_path):
    raw = training_config()
    raw["training"]["config"] = {"global_batch_size": 8}
    # Slime may derive the global batch size from --num-steps-per-rollout; Reef still reads the explicit value.
    raw["training"]["options"].update(
        {"global-batch-size": "${training.config.global_batch_size}", "num-steps-per-rollout": 1}
    )
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    assert global_batch_size(slime_runtime_config(config)["settings"]["training_backend_options"]) == 8
    assert "--global-batch-size=8" in slime_argv(config)


def test_inference_capacity_defaults_to_one_tensor_parallel_engine(tmp_path):
    config, _ = resolve_deployment_config(training_config(), {"inference.tensor-parallel-size": "4"}, tmp_path / "c")
    assert config["reef"]["inference_num_gpus"] == 4
    assert (serving_config(config)["num_gpus"], serving_config(config)["tensor_parallel_size"]) == (4, 4)


def test_slime_arguments_expand_the_model_path_reference(tmp_path):
    # The deploy layer binds hf-checkpoint to ``${reef.model_path}`` so the model
    # resolves once; Slime must receive the resolved path, not the reference.
    config, _ = resolve_deployment_config(training_config(), None, tmp_path / "serve.yaml")

    assert config["reef"]["training_backend_options"]["hf-checkpoint"] == "${reef.model_path}"
    argv = slime_argv(config)
    assert f"--hf-checkpoint={config['reef']['model_path']}" in argv
    assert not any("${" in argument for argument in argv)


def test_vllm_serving_engines_get_their_options_and_slime_gets_no_engine_flags(tmp_path):
    overrides = {"inference.backend": "vllm", "inference.options.max-model-len": "4096"}
    config, _ = resolve_deployment_config(training_config(), overrides, tmp_path / "serve.yaml")
    reef = config["reef"]
    assert reef["inference_backend"] == "vllm"
    assert reef["inference_options"] == {"max-model-len": "4096"}
    component = config["services"][0]
    assert component["command"][component["command"].index("--engine") + 1] == "vllm"
    argv = slime_argv(config)
    assert "--rollout-num-gpus=0" in argv
    assert not any(argument.startswith("--sglang-") for argument in argv)
    serving = serving_config(config)
    assert (serving["engine"], serving["options"]) == ("vllm", {"max-model-len": "4096"})


def test_a_deployment_without_an_inference_engine_trains_on_every_gpu(tmp_path):
    overrides = {
        "recipe.implementation": "reef.recipe.sft.recipe:SFTRecipe",
        "inference.model-path": "/models/demo",
        "inference.backend": "none",
        "training.options.actor-num-gpus-per-node": "8",
        "training.options.save-interval": "42",
    }
    config, _ = resolve_deployment_config(training_config(), overrides, tmp_path / "serve.yaml")
    reef = config["reef"]
    assert reef["inference_backend"] == "none"
    assert not reef.get("inference_num_gpus") and not reef.get("inference_options")
    argv = slime_argv(config)
    assert "--rollout-num-gpus=0" in argv
    assert "--actor-num-gpus-per-node=8" in argv
    assert not any(
        argument.startswith(("--colocate", "--offload", "--sglang-", "--router-", "--rollout-num-gpus-per-engine"))
        for argument in argv
    )
    assert serving_config(config, None)["type"] == "none"


def test_a_deployment_without_an_inference_engine_refuses_a_composite_recipe(tmp_path):
    raw = {
        "recipe": {
            "implementation": "reef.recipe.composite:CompositeRecipe",
            "config": {
                "components": {
                    "weights": {"implementation": "reef.recipe.sft.recipe:SFTRecipe"},
                    "records": {"implementation": "recipe"},
                }
            },
        },
        "inference": {"model-path": "/models/demo", "backend": "none"},
    }
    with pytest.raises(DeployConfigError, match="cannot serve a composite recipe"):
        resolve_deployment_config(raw, None, tmp_path / "serve.yaml")


def test_slime_serves_on_a_remote_inference_component(tmp_path):
    from reef.service.deploy.training import local_model_required

    raw = training_config()
    raw["inference"]["remote"] = {"url": "https://gpu-host.example"}
    config, _ = resolve_deployment_config(raw, None, tmp_path / "serve.yaml")
    # Megatron loads the model here; the component loads it as the file names it.
    assert local_model_required(config)
    assert config["reef"]["inference_remote"]["model_path"] == "/models/demo"
    serving = serving_config(config)
    assert serving["remote"]["url"] == "https://gpu-host.example"
    assert serving["remote"]["inference_token_env"] == INFERENCE_TOKEN_ENV
    assert (serving["engine"], serving["num_gpus"]) == ("sglang", 1)
    lora = training_config()
    lora["inference"]["remote"] = {"url": "https://gpu-host.example"}
    lora["training"]["options"] = {**lora["training"]["options"], "megatron-lora-rank": 8}
    resolve_deployment_config(lora, None, tmp_path / "serve.yaml")
    refused = training_config()
    refused["inference"]["remote"] = {"url": "https://gpu-host.example"}
    refused["training"] = {**raw["training"], "colocate": True}
    with pytest.raises(DeployConfigError, match=r"unknown config fields: training\.colocate"):
        resolve_deployment_config(refused, None, tmp_path / "serve.yaml")
