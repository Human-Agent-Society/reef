"""Shipped example placement stays explicit without starting any services."""

from pathlib import Path

import pytest
import yaml

from reef.runtime.executor.config import role_executor_settings, select_executor
from reef.service.deploy.config import validate_services
from reef.service.deploy.execution import service_executor_selection

ROOT = Path(__file__).resolve().parents[2]
SERVICE_CONFIGS = (
    "recipes/basic/local-sglang.yaml",
    "recipes/basic/external-provider.yaml",
    "recipes/openclawrl/examples/openclawrl/serve.yaml",
    "recipes/sao/examples/sao/serve.yaml",
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


@pytest.mark.parametrize("relative", SERVICE_CONFIGS)
def test_examples_keep_service_processes_local_and_slime_workers_on_ray(relative):
    config = yaml.safe_load((ROOT / relative).read_text())
    assert config["execution"]["services"] == "local"
    services = validate_services(config, relative)
    for service in services:
        assert service_executor_selection(config, service).settings.backend == "local"
    if any(service["name"] == "slime-driver" for service in services):
        for role in ("training", "rollout"):
            assert config["execution"][role] == "ray"
            assert select_executor(role_executor_settings(config, role), role=role).settings.backend == "ray"


@pytest.mark.parametrize("relative", EVOLUTION_CONFIGS)
def test_cpu_evolution_examples_select_uni_then_mp_without_changing_isolation(relative):
    config = yaml.safe_load((ROOT / relative).read_text())
    evolution = config["evolution"]
    assert config["execution"]["evolution"]["workers"] == 1
    assert config["execution"]["evolution"]["backend"] == "auto"
    assert not {"episode_workers", "worker_executor", "worker_resources"} & evolution.keys()
    assert evolution.get("executor", "local") == "local"
    for workers, expected in ((1, "uni"), (2, "mp")):
        config["execution"]["evolution"]["workers"] = workers
        settings = role_executor_settings(config, "evolution")
        assert select_executor(settings, role="evolution").settings.backend == expected
