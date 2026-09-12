"""Versioned layout and native backend transport public contracts."""

from __future__ import annotations

import argparse

import pytest
import yaml

from reef.runtime.executor.arguments import native_arguments
from reef.service.deploy import orchestrator
from reef.service.deploy.cli import _apply_overrides, _parse_overrides
from reef.service.deploy.component_config import (
    component_config_arguments,
    normalize_component_config,
    normalize_component_layout,
    translate_layout,
    translate_references,
)
from reef.service.deploy.config_utils import DeployConfigError, load_config
from reef.service.deploy.service_config import normalize_service_config, service_config_from_mapping


def test_public_layout_and_cli_preserve_values_and_opaque_options():
    config = translate_layout(
        {
            "schema-version": 2,
            "reef": {"port": 8000, "token": "001"},
            "inference": {
                "model-path": "org/model",
                "options": {"mem_fraction_static": 0.8, "trust-remote-code": True},
            },
            "training": {"options": {"lr": 1e-6, "optimizer": "adam"}},
            "storage": {"agent-record-dir": ".reef/data"},
        }
    )
    overrides = _parse_overrides(
        [
            "--reef.port",
            "9000",
            "--inference.options.mem-fraction-static",
            "0.6",
            "--inference.options.trust-remote-code",
            "false",
            "--training.options.lr",
            "2e-6",
        ]
    )
    config = normalize_service_config(_apply_overrides(config, overrides))
    settings = service_config_from_mapping(config)
    assert settings.port == 9000 and settings.tokens == ("001",)
    assert settings.model_path == "org/model" and settings.agent_record_dir == ".reef/data"
    assert native_arguments(settings.inference_options) == ["--mem-fraction-static=0.6"]
    assert native_arguments(settings.training_backend_options) == ["--lr=2e-6", "--optimizer=adam"]


@pytest.mark.parametrize(
    "value, message",
    [
        ({"schema-version": 3}, "unsupported"),
        ({"schema-version": True}, "unsupported"),
        ({"schema-version": 2, "service": {}}, "does not accept service/services"),
        ({"schema-version": 2, "services": []}, "does not accept service/services"),
        ({"schema-version": 2, "infernce": {}}, "unknown config sections"),
        ({"schema-version": 2, "inference": {"model-pth": {}}}, "unknown config fields"),
        ({"schema-version": 2, "reef": "oops"}, "must be an object"),
        ({"schema-version": 2, "inference": {"model-path": "a", "model_path": "b"}}, "duplicate"),
    ],
)
def test_invalid_versioned_layout_fails(value, message):
    with pytest.raises(DeployConfigError, match=message):
        translate_layout(value)


def test_component_yaml_spelling_and_public_references():
    config = translate_layout(
        {
            "schema-version": 2,
            "reef": {"port": 9001},
            "execution": {"evolution": {"backend": "uni", "resources": {"cpus-per-worker": 0.5}}},
            "inference": {"options": {"custom": "${reef.port}"}},
        }
    )
    arguments = component_config_arguments(config)
    config = normalize_component_layout(config, arguments)
    config = normalize_component_config(normalize_service_config(config), arguments)
    assert config["execution"]["evolution"]["resources"]["cpus_per_worker"] == 0.5
    assert translate_references(config, arguments)["reef"]["inference_options"]["custom"] == "9001"


def test_native_options_reach_the_backend_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--use-critic", action="store_true")
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--metadata")
    parser.add_argument("--model")
    arguments = native_arguments(
        {"lr": 1e-6, "use_critic": True, "layers": [1, 2], "metadata": {"key": "value"}, "model": "001"}
    )
    parsed = parser.parse_args(arguments)
    assert parsed.lr == 1e-6 and parsed.use_critic and parsed.layers == [1, 2]
    assert parsed.metadata == '{"key": "value"}' and parsed.model == "001"
    with pytest.raises(SystemExit):
        parser.parse_args(native_arguments({"unknown": 1}))


@pytest.mark.parametrize("options", [{"tp_size": 2}, {"model-path": "other"}, {"port": 8000}])
def test_managed_options_cannot_be_overridden(options):
    with pytest.raises(DeployConfigError, match="managed by Reef"):
        native_arguments(options, reserved={"tp-size", "model-path", "port"})


def test_native_option_aliases_and_object_replacement():
    config = {"reef": {"inference_options": {"mem_fraction_static": 0.8, "other": 2}}}
    result = _apply_overrides(config, {"inference.options.mem-fraction-static": "0.6"})
    assert result["reef"]["inference_options"] == {"mem-fraction-static": "0.6", "other": 2}
    replaced = normalize_service_config(_apply_overrides(result, {"inference.options": "{}"}))
    assert replaced["reef"]["inference_options"] == {}
    with pytest.raises(DeployConfigError, match="duplicate"):
        native_arguments({"foo_bar": 1, "foo-bar": 2})


def test_versioned_files_always_pass_normalized_config_to_children(tmp_path, monkeypatch):
    raw = {"schema-version": 2, "inference": {"upstream-url": "http://localhost:8000", "upstream-model": "001"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    captured = {}
    monkeypatch.chdir(tmp_path)

    class Stack:
        exit_code = 0

        def __init__(self, config, services, run_dir, ready_timeout_default, config_path, source_root=None):
            captured["path"] = config_path
            captured["config"] = load_config(config_path)

        def start(self):
            pass

        def block(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(orchestrator, "_Stack", Stack)
    assert orchestrator._run_orchestrator(str(path)) == 0
    assert captured["path"] != path and not captured["path"].exists()
    assert captured["config"]["reef"]["upstream_model"] == "001"
    assert "schema-version" not in captured["config"]
    assert yaml.safe_load(path.read_text()) == raw


def test_automatic_stack_rejects_unused_training_options(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"schema-version": 2, "training": {"options": {"lr": 1e-6}}}))
    with pytest.raises(DeployConfigError, match="select a weight-training recipe"):
        orchestrator._run_orchestrator(str(path))


def test_whole_native_object_then_leaf_override_and_flag_values():
    config = _apply_overrides({}, {"inference.options": '{"max-tokens": 10}', "inference.options.max-tokens": "20"})
    assert native_arguments(config["reef"]["inference_options"]) == ["--max-tokens=20"]
    with pytest.raises(DeployConfigError, match="cannot contain flags"):
        native_arguments({"layers": ["--port=123"]})


def test_public_references_keep_opaque_key_spelling():
    config = {"reef": {"inference_options": {"custom-key": 2}}, "reference": "${inference.options.custom-key}"}
    assert translate_references(config, ())["reference"] == "${reef.inference_options.custom-key}"


def test_shipped_reef_yamls_use_the_public_layout():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    configs = []
    for directory in ("recipes", "tutorials", "reef/service/profiles", "tests/packaging"):
        for path in (root / directory).rglob("*.yaml"):
            if "work" in path.relative_to(root).parts:
                continue
            text = path.read_text()
            if not re.search(r"(?m)^(schema-version|reef|implementation):", text):
                continue
            configs.append(path)
            config = yaml.safe_load(text)
            assert config.get("schema-version") == 2, path
            assert not {"service", "services", "implementation", "data", "evolution"} & config.keys(), path
    assert len(configs) >= 15


def test_declared_recipe_sections_support_leaf_overrides():
    config = {
        "reef": {
            "recipe": "reef.recipe.cordis:CordisRecipe",
            "data": {"training_mode": "auto"},
            "evolution": {"adapter": "pi", "tasks": ["one"], "episode_workers": 2},
        }
    }
    arguments = component_config_arguments(config)
    result = _apply_overrides(
        config,
        {"recipe.config.evolution.episode_workers": "4", "recipe.config.training-mode": "manual"},
        arguments=arguments,
    )
    assert result["reef"]["evolution"] == {"adapter": "pi", "tasks": ["one"], "episode_workers": 4}
    assert result["reef"]["data"]["training_mode"] == "manual"
    assert config["reef"]["evolution"]["episode_workers"] == 2
