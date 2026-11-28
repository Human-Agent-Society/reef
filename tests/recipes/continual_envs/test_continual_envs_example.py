"""The example deployment of the continual-environments recipe: it resolves, and its stages and generator start."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.support.task_service_stubs import generator_state

from recipes.beta.continual_envs import TrafficTaskGenerator
from reef.deploy.orchestrator import resolve_deployment_config
from reef.deploy.service_config import service_config_from_mapping
from reef.recipe.generated_tasks import GeneratedTaskSamples, GeneratedTaskSettings

EXAMPLE = Path(__file__).resolve().parents[3] / "recipes/beta/continual_envs/examples/terminalworld/serve.yaml"


@pytest.mark.unit
def test_the_example_waits_for_whole_windows_of_traffic_and_builds_its_generator(tmp_path: Path, monkeypatch) -> None:
    environ = {
        "REEF_TOKEN": "t",
        "REEF_ACCESS_TOKEN": "rft_1",
        "REEF_CONTINUAL_ENVS_STATE_DIR": str(tmp_path),
        "REEF_CONTINUAL_ENVS_MODEL_PATH": str(tmp_path / "model"),
        "REEF_PLATFORM_URL": "https://console.example",
        "REEF_GENERATION_SPEC": "terminal-failures",
        "REEF_GENERATION_GENERATOR": "tw-ours",
    }
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    resolved, _ = resolve_deployment_config(yaml.safe_load(EXAMPLE.read_text()), None, EXAMPLE)
    stage = service_config_from_mapping(resolved).pipeline_settings["record2sample"]["config"]
    settings = GeneratedTaskSettings.parse(
        {**stage, "generator_url": "http://127.0.0.1:8910"}, plugin_class=GeneratedTaskSamples
    )
    assert settings.source_records_per_round == settings.min_source_records_per_round == 64
    assert settings.source_tag == "source=traffic" and settings.pool_capacity > 0
    # The generation run ends inside the deadline of its generation, so a timeout cancels the run first.
    assert stage["generator"]["config"]["run_timeout_seconds"] < settings.generation_deadline_seconds
    generator = TrafficTaskGenerator.from_config(
        stage["generator"]["config"], scenario="traffic", environ=environ, state=generator_state(tmp_path / "state")
    )
    assert (generator.settings.spec, generator.settings.generator) == ("terminal-failures", "tw-ours")
    assert generator.settings.params == {"per_failure": 2} and generator.settings.max_score == 0.5
