"""Shipped examples use service defaults and explicit worker placement overrides."""

from pathlib import Path

import pytest
import yaml
from reef_service.config_helpers import deployment_layout

from reef.recipe.config import recipe_config_from_mapping
from reef.runtime.executor.arguments import native_arguments
from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.service.deploy.execution import service_executor_selection, validate_services

ROOT = Path(__file__).resolve().parents[2]
SERVICE_CONFIGS = (
    "recipes/basic/local-sglang.yaml",
    "recipes/basic/external-provider.yaml",
    "recipes/openclawrl/examples/openclawrl/serve.yaml",
    "recipes/sao/examples/sao/serve.yaml",
    "recipes/beta/coral/examples/coral_demo/serve.yaml",
    "recipes/tttd/examples/tttd/serve.yaml",
    "recipes/tttd/examples/guidance_ttt/serve.yaml",
    "tutorials/evolve-your-harness/configs/serve.yaml",
    "tutorials/evolve-your-harness/configs/serve-native.yaml",
    "tutorials/evolve-your-harness/configs/deployment.yaml",
)
EVOLUTION_CONFIGS = (
    "recipes/skillclaw/skillclaw.yaml",
    "tutorials/evolve-your-harness/configs/serve.yaml",
    "tutorials/evolve-your-harness/configs/serve-native.yaml",
    "tutorials/evolve-your-harness/configs/deployment.yaml",
)

TRAINING_CONFIGS = (
    "recipes/sao/examples/sao/serve.yaml",
    "recipes/beta/coral/examples/coral_demo/serve.yaml",
    "recipes/tttd/examples/tttd/serve.yaml",
    "recipes/tttd/examples/guidance_ttt/serve.yaml",
)


@pytest.mark.parametrize("relative", TRAINING_CONFIGS)
def test_training_examples_use_managed_ray_without_reserving_driver_gpus(relative):
    path = ROOT / relative
    config = deployment_layout(yaml.safe_load(path.read_text()))
    services = validate_services(config, relative)
    assert [service["name"] for service in services] == ["slime-driver", "reef"]
    assert "ray_address" not in config["reef"]
    assert not {"cuda_visible_devices", "ray_port", "ray_dashboard_port"} & config["training"].keys()
    for service in services:
        assert "cuda" not in service
        assert "RAY_ADDRESS" not in service.get("env", {})
        assert "CUDA_VISIBLE_DEVICES" not in service.get("env", {})
        assert not service.get("resources")
        assert service_executor_selection(config, service).settings.backend == "uni"
    assert not services[0].get("depends_on")
    assert services[1]["depends_on"] == ["slime-driver"]
    if "coral" not in relative:
        assert "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}" in path.with_name("run.sh").read_text()
    if "tttd" in relative:
        assert config["training"]["num_gpus"] == 2
        assert config["reef"]["training_backend_options"]["actor-num-gpus-per-node"] == str(
            config["training"]["num_gpus"]
        )
        assert config["reef"]["training_backend_options"]["colocate"] is True
    elif "sao" in relative:
        assert config["reef"]["training_backend_options"]["actor-num-gpus-per-node"] == "1"
        assert config["reef"]["training_backend_options"]["rollout-num-gpus"] == "1"


def test_recipe_yamls_no_longer_launch_ray_head_services():
    for pattern in ("*.yaml", "*.yml"):
        for path in (ROOT / "recipes").rglob(pattern):
            if "work" not in path.parts:
                assert "ray start" not in path.read_text(), path


@pytest.mark.parametrize("relative", SERVICE_CONFIGS)
def test_examples_select_service_executors_from_resources_and_slime_workers_on_ray(relative):
    config = deployment_layout(yaml.safe_load((ROOT / relative).read_text()))
    assert "services" not in config.get("execution", {})
    assert "execution" not in config or config["execution"]
    assert role_executor_settings(config, "services").backend == "auto"
    services = validate_services(config, relative)
    for service in services:
        expected = "ray" if service.get("resources") else "uni"
        assert service_executor_selection(config, service).settings.backend == expected
    if any(service["name"] == "slime-driver" for service in services):
        for role in ("training", "rollout"):
            assert config["execution"][role] == "ray"
            assert select_executor(role_executor_settings(config, role), role=role).settings.backend == "ray"


@pytest.mark.parametrize("relative", EVOLUTION_CONFIGS)
def test_cpu_evolution_examples_select_uni_then_mp_without_changing_isolation(relative):
    config = recipe_config_from_mapping(yaml.safe_load((ROOT / relative).read_text()))
    evolution = config["evolution"]
    assert config["execution"]["evolution"]["workers"] == 1
    assert set(config["execution"]["evolution"]) == {"workers"}
    assert not {"episode_workers", "worker_executor", "worker_resources"} & evolution.keys()
    assert evolution.get("executor", "local") == "local"
    for workers, expected in ((1, "uni"), (2, "mp")):
        config["execution"]["evolution"]["workers"] = workers
        settings = role_executor_settings(config, "evolution")
        assert settings.backend == "auto"
        assert select_executor(settings, role="evolution").settings.backend == expected


def test_gepa_allows_backend_selection_without_default_resource_boilerplate():
    config = yaml.safe_load((ROOT / "recipes/gepa/examples/aime/gepa.yaml").read_text())
    execution = config["execution"]["evolution"]
    assert set(execution) == {"backend", "workers"}
    assert execution["backend"] == "${REEF_GEPA_EXECUTOR}"


def test_openclawrl_isolates_external_models_from_reef_training():
    root = ROOT / "recipes/openclawrl/examples/openclawrl"
    config = deployment_layout(yaml.safe_load((root / "serve.yaml").read_text()))
    compose = yaml.safe_load((root / "docker-compose.yaml").read_text())["services"]
    driver, http = validate_services(config, "serve.yaml")
    assert [driver["name"], http["name"]] == ["slime-driver", "reef"]
    assert not driver.get("resources") and not driver.get("depends_on")
    assert http["depends_on"] == ["slime-driver"]
    assert compose["reef"]["environment"]["RAY_ADDRESS"] == "${RAY_ADDRESS:-}"
    pools = {
        name: set(service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"])
        for name, service in compose.items()
    }
    assert pools == {"prm": {"1"}, "user-model": {"2"}, "reef": {"3", "4", "5", "6", "7"}}
    for name in ("prm", "user-model"):
        command = compose[name]["command"]
        assert command[:3] == ["python", "-m", "sglang.launch_server"]
        assert int(command[command.index("--tp") + 1]) == len(pools[name])
        assert "RAY_ADDRESS" not in compose[name].get("environment", {})
    prm = compose["prm"]["command"]
    assert config["reef"]["prm_url"] == f"http://127.0.0.1:{prm[prm.index('--port') + 1]}"
    assert config["reef"]["prm_tokenizer_path"] == prm[prm.index("--model-path") + 1]
    assert compose["reef"]["depends_on"] == {"prm": {"condition": "service_healthy"}}
    flags = dict(
        argument.removeprefix("--").split("=", 1)
        for argument in native_arguments(config["reef"]["training_backend_options"])
        if "=" in argument
    )
    training_gpus = int(flags["actor-num-nodes"]) * int(flags["actor-num-gpus-per-node"])
    rollout_gpus = int(flags["rollout-num-gpus"])
    assert (training_gpus, rollout_gpus, int(flags["num-gpus-per-node"])) == (4, 1, 5)
    assert training_gpus + rollout_gpus == len(pools["reef"])
