"""Recipe/runtime declarations share the public CLI and YAML value contract."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest
import yaml

from reef.core.config import config_arguments, config_option, parse_config_values
from reef.recipe import Recipe, RecipeConfigError, WeightTrainingRecipe, config_field
from reef.recipe.config_fields import resolve_config_field_values
from reef.runtime import InferenceProxyRuntime, RuntimeConfigError, RuntimeRegistry
from reef.runtime.executor.config import executor_settings
from reef.runtime.registry import RuntimeFactory
from reef.service.deploy import orchestrator
from reef.service.deploy.cli import _apply_overrides, _parse_overrides, build_serve_parser
from reef.service.deploy.config_utils import DeployConfigError, interpolate_environment
from reef.service.deploy.deployment_config import component_config_arguments, normalize_component_config
from reef.service.deploy.service_config import normalize_service_config, service_config_from_mapping


@dataclass(frozen=True)
class ExtensionRecipe(WeightTrainingRecipe):
    label: str = config_field("default", env="REEF_EXTENSION_LABEL", help="Label kept verbatim.")
    enabled: bool = config_field(True)
    count: int = config_field(4, env="REEF_EXTENSION_COUNT")
    scale: float = config_field(0.5, allow_nonfinite=False)
    tags: tuple[str, ...] = config_field(())
    options: Mapping[str, Any] = config_field(default_factory=dict)


@dataclass(frozen=True)
class ExtensionServingRecipe(Recipe):
    label: str = config_field("default")


@dataclass(frozen=True)
class ExtensionRuntimeSettings:
    label: str = config_option("default", env="REEF_EXTENSION_LABEL")
    enabled: bool = config_option(True)
    count: int = config_option(4, env="REEF_EXTENSION_COUNT")
    scale: float = config_option(0.5)
    tags: tuple[str, ...] = config_option(())
    options: Mapping[str, Any] = config_option(default_factory=dict)


class ExtensionRuntimeFactory(RuntimeFactory):
    kind = "extension"

    def config_type(self):
        return ExtensionRuntimeSettings

    def __call__(self, config, model_path, recipe_config, environ):
        runtime = InferenceProxyRuntime(model_path=model_path, base_url="http://provider")
        runtime.parsed_config = config
        return runtime


@pytest.fixture
def extension(monkeypatch):
    module = ModuleType("reef_config_extension")
    module.Training = ExtensionRecipe
    module.Serving = ExtensionServingRecipe
    module.runtime = ExtensionRuntimeFactory()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def resolve_deployment(config, argv=()):
    overrides = _parse_overrides(list(argv))
    selected = _apply_overrides(config, overrides)
    arguments = component_config_arguments(selected)
    combined = _apply_overrides(config, overrides, arguments=arguments)
    expanded = interpolate_environment(combined, "stack.yaml")
    return normalize_component_config(normalize_service_config(expanded), arguments)


@pytest.mark.parametrize(
    "field, value, cli",
    [
        ("label", "001", "001"),
        ("label", "true", "true"),
        ("label", "null", "null"),
        ("label", "", ""),
        ("enabled", False, "false"),
        ("count", 0, "0"),
        ("scale", 0.125, "0.125"),
        ("tags", ["001", "true"], '["001", "true"]'),
        ("tags", [], "[]"),
        ("options", {"nested": {"enabled": False}}, '{"nested": {"enabled": false}}'),
        ("options", {}, "{}"),
    ],
)
def test_extension_yaml_cli_and_runtime_values_are_equivalent(extension, field, value, cli):
    base = {"reef": {"recipe": "reef_config_extension:Training"}}
    yaml_config = resolve_deployment({"reef": {**base["reef"], field: value}})
    cli_config = resolve_deployment(base, [f"--{field}={cli}"])
    yaml_values = resolve_config_field_values(ExtensionRecipe, {field: yaml_config["reef"][field]}, {})
    cli_values = resolve_config_field_values(ExtensionRecipe, {field: cli_config["reef"][field]}, {})
    backend = extension.runtime.parse_config({field: value}, {})
    assert yaml_values[field] == cli_values[field] == backend[field]
    assert type(yaml_values[field]) is type(cli_values[field]) is type(backend[field])
    restored = yaml.safe_load(yaml.safe_dump(cli_config))
    child_values = service_config_from_mapping(restored).recipe_settings
    assert resolve_config_field_values(ExtensionRecipe, {field: child_values[field]}, {})[field] == cli_values[field]


def test_cli_overrides_yaml_then_environment_then_defaults(extension, monkeypatch):
    monkeypatch.setenv("REEF_EXTENSION_COUNT", "8")
    config = {"reef": {"recipe": "reef_config_extension:Training", "count": "${MISSING_COUNT:?}"}}
    combined = resolve_deployment(config, ["--count", "2"])
    env = {"REEF_EXTENSION_COUNT": "8"}
    assert resolve_config_field_values(ExtensionRecipe, {"count": combined["reef"]["count"]}, env)["count"] == 2
    assert resolve_config_field_values(ExtensionRecipe, {"count": "6"}, env)["count"] == 6
    assert resolve_config_field_values(ExtensionRecipe, {}, env)["count"] == 8
    assert resolve_config_field_values(ExtensionRecipe, {}, {})["count"] == 4
    assert extension.runtime.parse_config({}, env)["count"] == 8
    assert extension.runtime.parse_config({"count": 0}, env)["count"] == 0


def test_last_alias_wins_and_boolean_can_be_disabled(extension):
    config = resolve_deployment(
        {"reef": {"recipe": "reef_config_extension:Training", "count": 9}},
        ["--count", "1", "--reef.count", "2", "--count", "3", "--no-enabled"],
    )
    assert config["reef"]["count"] == 3
    assert config["reef"]["enabled"] is False


@pytest.mark.parametrize("argv", [["--label"], ["--count"], ["--no-enabled=true"], ["--no-enabled", "true"]])
def test_component_flags_validate_arity_after_discovery(extension, argv):
    with pytest.raises(ValueError, match="value"):
        resolve_deployment({"reef": {"recipe": "reef_config_extension:Training"}}, argv)


@pytest.mark.parametrize("field,value", [("count", True), ("scale", "nan"), ("tags", [2]), ("label", 1)])
def test_recipe_and_runtime_reject_the_same_invalid_values(extension, field, value):
    with pytest.raises(RecipeConfigError, match=field):
        resolve_config_field_values(ExtensionRecipe, {field: value}, {})
    with pytest.raises(RuntimeConfigError, match=field):
        extension.runtime.parse_config({field: value}, {})


def test_recipe_and_runtime_errors_do_not_echo_values(extension):
    secret = "private-invalid-number"
    for resolve in (
        lambda: resolve_config_field_values(ExtensionRecipe, {"count": secret}, {}),
        lambda: extension.runtime.parse_config({"count": secret}, {}),
    ):
        with pytest.raises((RecipeConfigError, RuntimeConfigError)) as caught:
            resolve()
        assert secret not in str(caught.value)


def test_runtime_definition_is_loaded_from_selected_dotted_factory(extension):
    config = resolve_deployment(
        {"reef": {"recipe": "reef_config_extension:Serving", "runtime": {"type": "reef_config_extension:runtime"}}},
        ["--reef.data.label", "001", "--reef.runtime.label", "true", "--no-reef.runtime.enabled"],
    )
    runtime = RuntimeRegistry().build(config["reef"]["runtime"], model_path="model", environ={})
    assert runtime.parsed_config["label"] == "true"
    assert runtime.parsed_config["enabled"] is False
    assert runtime.parsed_config["type"] == "reef_config_extension:runtime"
    recipe = ExtensionServingRecipe.from_environment({}, config=config["reef"], runtime=runtime)
    assert recipe.label == "001"


def test_unknown_declared_recipe_and_runtime_fields_are_rejected(extension):
    with pytest.raises(RecipeConfigError, match=r"reef\.typo"):
        resolve_deployment({"reef": {"recipe": "reef_config_extension:Training", "typo": 4}})
    with pytest.raises(RuntimeConfigError, match=r"unknown config fields.*typo"):
        RuntimeRegistry().build({"type": "reef_config_extension:runtime", "typo": 1}, model_path="model")


def test_executor_aliases_preserve_profile_and_parse_resources(extension):
    config = {
        "executors": {"local": {"backend": "mp", "workers": 2, "options": {}}},
        "execution": {"evolution": "local"},
    }
    combined = resolve_deployment(
        config,
        ["--execution.evolution.workers", "4", "--execution.evolution.resources.cpus-per-worker", "0.5"],
    )
    settings = executor_settings(combined, combined["execution"]["evolution"])
    assert settings.backend == "mp"
    assert settings.workers == 4
    assert settings.resources.cpus_per_worker == 0.5
    assert combined["executors"]["local"]["workers"] == 2


def test_component_help_has_namespaced_fields_without_destination_collisions(extension):
    config = {"reef": {"recipe": "reef_config_extension:Training"}}
    parser = build_serve_parser(config=config)
    help_text = parser.format_help()
    assert "--label" in help_text and "Label kept verbatim" in help_text
    assert "--execution.training.backend" in help_text
    args = vars(parser.parse_args(["--execution.training.backend", "ray", "--execution.rollout.backend", "custom:X"]))
    assert args["execution.training.backend"] == "ray"
    assert args["execution.rollout.backend"] == "custom:X"


def test_component_field_cannot_shadow_public_cli_option(extension):
    @dataclass(frozen=True)
    class ConflictingRecipe(WeightTrainingRecipe):
        port: int = config_field(1)

    extension.Conflicting = ConflictingRecipe
    with pytest.raises(ValueError, match=r"conflicts.*--port"):
        component_config_arguments({"reef": {"recipe": "reef_config_extension:Conflicting"}})


def test_mutable_component_defaults_are_independent():
    arguments = config_arguments(ExtensionRuntimeSettings)
    first = parse_config_values(arguments, {})
    second = parse_config_values(arguments, {})
    first["options"]["changed"] = True
    assert second["options"] == {}


def test_launch_rejects_invalid_extension_before_model_download(extension, tmp_path, monkeypatch):
    path = tmp_path / "stack.yaml"
    path.write_text(yaml.safe_dump({"reef": {"recipe": "reef_config_extension:Training"}, "services": []}))

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid configuration reached model download or startup")

    monkeypatch.setattr(orchestrator, "resolve_model_paths", forbidden)
    monkeypatch.setattr(orchestrator._Stack, "start", forbidden)
    with pytest.raises(DeployConfigError, match="count"):
        orchestrator._run_orchestrator(str(path), _parse_overrides(["--count", "invalid"]))


def test_selected_recipe_help_works_without_optional_gpu_modules(tmp_path):
    package = tmp_path / "light_recipe"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from dataclasses import dataclass\nfrom reef.recipe import Recipe, config_field\n"
        "@dataclass(frozen=True)\nclass LightRecipe(Recipe):\n"
        "    label: str = config_field('default', help='Lightweight label.')\n"
    )
    path = tmp_path / "stack.yaml"
    path.write_text("reef:\n  recipe: light_recipe:LightRecipe\n")
    script = (
        "import sys\nfrom reef.service.deploy.orchestrator import main\n"
        "try:\n    main(['-c', sys.argv[1], '--help'])\n"
        "except SystemExit as e:\n    assert e.code == 0\n"
        "assert not {'torch', 'slime', 'sglang', 'light_unselected'} & sys.modules.keys()\n"
    )
    result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--reef.data.label" in result.stdout and "Lightweight label" in result.stdout


def test_training_recipe_resolves_fields_once_before_connecting(extension, monkeypatch):
    from reef_service.runtime_stubs import StubTrainingRuntime

    from reef.service import assembly
    from reef.service.deploy.service_config import ServiceConfig

    resolved = []
    original = assembly.resolve_config_field_values

    def resolve(*args, **kwargs):
        values = original(*args, **kwargs)
        resolved.append(values)
        return values

    def connect(**kwargs):
        assert len(resolved) == 1
        assert kwargs["max_staleness"] == resolved[0]["max_staleness"]
        return StubTrainingRuntime(max_staleness=kwargs["max_staleness"])

    monkeypatch.setattr(assembly, "resolve_config_field_values", resolve)
    settings = ServiceConfig(
        recipe="reef_config_extension:Training",
        model_path="model",
        ray_address="local",
        recipe_settings={"count": "2", "max_staleness": "3"},
    )
    recipe = assembly._training_recipe(ExtensionRecipe, settings, {}, connect)
    assert recipe.count == 2 and recipe.max_staleness == 3
    assert len(resolved) == 1


@pytest.mark.parametrize("runtime_type", ["ray_training", "executor_training"])
def test_training_adapters_share_typed_fields_without_allocating_resources(runtime_type):
    from reef.runtime.registry import runtime_factory_for

    factory = runtime_factory_for(runtime_type)
    parsed = factory.parse_config({"type": runtime_type, "inference_timeout_s": "5.5", "max_staleness": "2"}, {})
    assert parsed["inference_timeout_s"] == 5.5
    assert parsed["max_staleness"] == 2
    with pytest.raises(RuntimeConfigError, match="max_staleness"):
        factory.parse_config({"max_staleness": True}, {})
    with pytest.raises(RuntimeConfigError, match="inference_timeout_s"):
        factory.parse_config({"inference_timeout_s": "nan"}, {})


def test_optional_extension_null_and_string_null_are_distinct():
    @dataclass(frozen=True)
    class Nullable:
        label: str | None = config_option("default")
        count: int | None = config_option(3)

    arguments = config_arguments(Nullable)
    assert parse_config_values(arguments, {"label": None, "count": None}) == {"label": None, "count": None}
    assert parse_config_values(arguments, {"label": "null", "count": "null"}) == {"label": "null", "count": None}


def test_switching_recipe_uses_only_the_final_selected_schema(extension):
    config = {"reef": {"recipe": "missing_recipe:Missing"}}
    combined = resolve_deployment(config, ["--reef.recipe", "reef_config_extension:Training", "--label", "001"])
    assert combined["reef"]["label"] == "001"
    assert "missing_recipe" not in sys.modules


def test_required_component_field_fails_without_inventing_a_default():
    @dataclass(frozen=True)
    class Required:
        name: str = config_option()

    arguments = config_arguments(Required)
    with pytest.raises(ValueError, match=r"reef\.name is required"):
        parse_config_values(arguments, {})
    assert parse_config_values(arguments, {"name": "001"}) == {"name": "001"}


def test_opaque_python_backend_options_preserve_backend_objects():
    marker = object()
    parsed = executor_settings({}, {"backend": "custom:Executor", "options": {"backend_object": marker}})
    assert parsed.options["backend_object"] is marker


def test_profile_overrides_reach_the_recipe_without_reloading_original_yaml(extension, tmp_path):
    from reef.service.assembly import _serving_recipe

    # This source file is deliberately different from the merged deployment.
    original = tmp_path / "example.yaml"
    original.write_text("implementation: reef_config_extension:Serving\ndata: {label: original}\n")
    combined = resolve_deployment(
        {
            "reef": {"recipe": "example", "upstream_model": "model"},
            "implementation": "reef_config_extension:Serving",
            "data": {"label": "yaml"},
            "model": {"path": "model"},
            "runtime": {"type": "reef_config_extension:runtime"},
        },
        ["--data.label", "001", "--runtime.label", "true", "--no-runtime.enabled"],
    )
    child = service_config_from_mapping(yaml.safe_load(yaml.safe_dump(combined)))
    recipe = _serving_recipe("example", child, {"REEF_RECIPE_CONFIG_DIR": str(tmp_path)}, None)
    assert recipe.label == "001"
    assert recipe.runtime.parsed_config["label"] == "true"
    assert recipe.runtime.parsed_config["enabled"] is False
    assert "original" in original.read_text()


def test_references_expand_inside_declared_recipe_containers(extension):
    combined = resolve_deployment(
        {
            "reef": {
                "recipe": "reef_config_extension:Training",
                "tags": ["${shared.label}"],
                "options": {"nested": {"label": "${shared.label}"}},
            },
            "shared": {"label": "001"},
        }
    )
    assert combined["reef"]["tags"] == ("001",)
    assert combined["reef"]["options"] == {"nested": {"label": "001"}}


def test_profile_can_omit_its_runtime_with_an_empty_object(extension):
    combined = resolve_deployment(
        {
            "reef": {"recipe": "example"},
            "implementation": "reef_config_extension:Serving",
            "runtime": {},
        }
    )
    assert combined["runtime"] == {}


def test_versioned_recipe_layout_uses_selected_schema(extension):
    from reef.service.deploy.deployment_config import (
        normalize_component_layout,
        translate_layout,
        translate_references,
    )

    config = translate_layout(
        {
            "schema-version": 2,
            "recipe": {"implementation": "reef_config_extension:Training", "config": {"label": "001", "count": 2}},
            "inference": {"model-path": "org/model", "options": {"custom": "${recipe.config.count}"}},
        }
    )
    arguments = component_config_arguments(config)
    config = normalize_component_layout(config, arguments)
    config = _apply_overrides(config, {"recipe.config.count": "5"}, arguments=arguments)
    config = normalize_component_config(normalize_service_config(config), arguments)
    assert config["reef"]["count"] == 5 and config["reef"]["label"] == "001"
    assert "data" not in config["reef"]
    assert translate_references(config, arguments)["reef"]["inference_options"]["custom"] == "${reef.count}"
