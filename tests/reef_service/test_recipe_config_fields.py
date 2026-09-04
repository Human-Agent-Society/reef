"""Config fields: one dataclass field is the whole configuration surface.

Covers the field-metadata mechanism (``reef.recipe.config_fields``): key and env
derivation, type-aware casting from the annotation, precedence, and the loud
failures for values nothing would consume.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any

import pytest
import yaml
from reef_service.runtime_stubs import StubTrainingRuntime

from reef.recipe import RecipeConfigError, config_field
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import _parse_float, recipe_config_fields, resolve_config_field_values
from reef.records import RecordStore
from reef.train.evaluation import DefaultCandidateEvaluationPlugin
from reef.train.slime_backend.backend import SlimeTrainingBackend

from ._threshold_processor import ThresholdProcessor


@dataclass(frozen=True)
class ConfiguredRecipe(WeightTrainingRecipe):
    """Every supported config field annotation on one recipe."""

    _: KW_ONLY
    batch_size: int = config_field(4, env="REEF_CONFIGURED_BATCH_SIZE")
    temperature: float = config_field(1.0, env="REEF_CONFIGURED_TEMPERATURE")
    strict: bool = config_field(False, env="REEF_CONFIGURED_STRICT")
    label: str = config_field("default", env="REEF_CONFIGURED_LABEL")
    name: str = "configured"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(step_preparer="sft", loss_family="sft", processor=ThresholdProcessor)


@pytest.mark.unit
def test_config_fields_declare_the_whole_configuration_surface() -> None:
    config_fields = recipe_config_fields(ConfiguredRecipe)

    assert sorted(config_fields) == ["batch_size", "label", "max_staleness", "strict", "temperature"]
    assert config_fields["batch_size"].env == "REEF_CONFIGURED_BATCH_SIZE"
    assert config_fields["batch_size"].default == 4
    assert config_fields["max_staleness"].env == "REEF_MAX_STALENESS"


@pytest.mark.unit
def test_config_field_precedence_is_config_over_environment_over_default() -> None:
    environ = {"REEF_CONFIGURED_BATCH_SIZE": "8"}

    assert resolve_config_field_values(ConfiguredRecipe, {}, {})["batch_size"] == 4
    assert resolve_config_field_values(ConfiguredRecipe, {}, environ)["batch_size"] == 8
    assert resolve_config_field_values(ConfiguredRecipe, {"batch_size": 2}, environ)["batch_size"] == 2
    assert resolve_config_field_values(ConfiguredRecipe, {}, {"REEF_MAX_STALENESS": "3"})["max_staleness"] == 3


@pytest.mark.unit
def test_config_field_casting_is_type_aware_for_every_annotation() -> None:
    values = resolve_config_field_values(
        ConfiguredRecipe,
        {"batch_size": "16", "temperature": "0.25", "strict": "true", "label": "run-a"},
        {},
    )

    assert values == {
        "batch_size": 16,
        "max_staleness": 0,
        "temperature": 0.25,
        "strict": True,
        "label": "run-a",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf"), "nan", "inf", "+inf", "-inf"],
)
def test_parse_float_rejects_non_finite_values(non_finite: Any) -> None:
    with pytest.raises(RecipeConfigError, match=r"temperature must be finite"):
        _parse_float(non_finite, "temperature")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1.0),
        (2.5, 2.5),
        (" 3.14 ", 3.14),
        (" -0.5 ", -0.5),
    ],
)
def test_parse_float_accepts_finite_numbers(value: Any, expected: float) -> None:
    assert _parse_float(value, "temperature") == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "non_finite",
    [float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"],
)
def test_float_config_field_rejects_non_finite_from_config(non_finite: Any) -> None:
    with pytest.raises(RecipeConfigError, match=r"temperature must be finite"):
        resolve_config_field_values(ConfiguredRecipe, {"temperature": non_finite}, {})


@pytest.mark.unit
@pytest.mark.parametrize("yaml_input", ["temperature: .nan", "temperature: .inf", "temperature: -.inf"])
def test_float_config_field_rejects_non_finite_from_yaml(yaml_input: str) -> None:
    yaml_value = yaml.safe_load(yaml_input)
    with pytest.raises(RecipeConfigError, match=r"temperature must be finite"):
        resolve_config_field_values(ConfiguredRecipe, yaml_value, {})


@pytest.mark.unit
@pytest.mark.parametrize("env_value", ["nan", "inf", "+inf", "-inf"])
def test_float_config_field_rejects_non_finite_from_env(env_value: str) -> None:
    environ = {"REEF_CONFIGURED_TEMPERATURE": env_value}
    with pytest.raises(RecipeConfigError, match=r"REEF_CONFIGURED_TEMPERATURE must be finite"):
        resolve_config_field_values(ConfiguredRecipe, {}, environ)


@pytest.mark.unit
def test_float_config_field_survives_service_config_uncast() -> None:
    # The old service layer int()-cast every recipe setting, silently
    # truncating floats; the config field's annotation now owns the cast.
    config = ConfiguredRecipe.service_config({"temperature": 0.25}, model_path="/models/demo")
    assert config["data"]["temperature"] == 0.25

    interpolated = ConfiguredRecipe.service_config({"temperature": "0.25"}, model_path="/models/demo")
    assert interpolated["data"]["temperature"] == 0.25

    recipe = ConfiguredRecipe.from_environment({}, config=interpolated, runtime=StubTrainingRuntime())
    assert recipe.temperature == 0.25
    assert isinstance(recipe.temperature, float)


@pytest.mark.unit
def test_from_environment_resolves_every_config_field_source() -> None:
    recipe = ConfiguredRecipe.from_environment(
        {"REEF_CONFIGURED_TEMPERATURE": "0.5", "REEF_CONFIGURED_STRICT": "1"},
        config={"data": {"batch_size": 2}},
        runtime=StubTrainingRuntime(),
    )

    assert (recipe.batch_size, recipe.temperature, recipe.strict, recipe.label) == (2, 0.5, True, "default")


@pytest.mark.unit
def test_service_config_rejects_settings_no_config_field_consumes() -> None:
    with pytest.raises(RecipeConfigError) as excinfo:
        ConfiguredRecipe.service_config({"group_size": 2, "batch_size": 1}, model_path="/models/demo")

    message = str(excinfo.value)
    assert "ConfiguredRecipe does not consume reef.group_size" in message
    # The error lists what the recipe does consume.
    for known in (
        "reef.batch_size",
        "reef.temperature",
        "reef.strict",
        "reef.label",
        "reef.max_staleness",
        "reef.checkpoint_every_n_versions",
    ):
        assert known in message


@pytest.mark.unit
def test_data_section_rejects_keys_no_config_field_consumes() -> None:
    with pytest.raises(
        RecipeConfigError, match=r"does not consume config key.*'group_size'.*known ConfiguredRecipe config fields"
    ):
        ConfiguredRecipe.from_environment(
            {},
            config={"data": {"group_size": 2}},
            runtime=StubTrainingRuntime(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"batch_size": "eight"}, r"reef\.batch_size must be an integer, got 'eight'"),
        ({"batch_size": 2.5}, r"reef\.batch_size must be an integer, got 2\.5"),
        ({"temperature": "warm"}, r"reef\.temperature must be a number, got 'warm'"),
        ({"strict": "maybe"}, r"reef\.strict must be a boolean \(true/false\), got 'maybe'"),
        ({"label": 3}, r"reef\.label must be a string, got 3"),
        ({"checkpoint_every_n_versions": "x"}, r"reef\.checkpoint_every_n_versions must be an integer"),
    ],
)
def test_service_config_names_the_setting_and_the_expected_type(settings, match) -> None:
    with pytest.raises(RecipeConfigError, match=match):
        ConfiguredRecipe.service_config(settings, model_path="/models/demo")


@pytest.mark.unit
def test_invalid_field_values_surface_as_recipe_config_errors() -> None:
    # __post_init__ range checks (ValueError on direct construction) become
    # RecipeConfigError on the configuration path.
    from recipes.sao import SAORecipe

    with pytest.raises(RecipeConfigError, match="invalid SAORecipe configuration: batch_size must be positive"):
        SAORecipe.from_environment({}, config={"data": {"batch_size": -1}}, runtime=StubTrainingRuntime())
    with pytest.raises(RecipeConfigError, match="invalid SAORecipe configuration"):
        SAORecipe.from_environment({"REEF_SAO_BATCH_SIZE": "0"}, config={}, runtime=StubTrainingRuntime())


@pytest.mark.unit
def test_config_field_annotations_outside_the_supported_scalars_fail_at_declaration() -> None:
    @dataclass(frozen=True)
    class BadRecipe(WeightTrainingRecipe):
        _: KW_ONLY
        sizes: list = config_field(None)  # noqa: RUF009 - the invalid declaration under test

    with pytest.raises(TypeError, match="must be annotated with one of: int, float, bool, str"):
        recipe_config_fields(BadRecipe)


@pytest.mark.unit
def test_default_build_uses_declared_processor_and_config_fields() -> None:
    trainer = ConfiguredRecipe(StubTrainingRuntime(), batch_size=2).build("scenario", RecordStore())

    assert isinstance(trainer.processor, ThresholdProcessor)
    assert isinstance(trainer.training_backend, SlimeTrainingBackend)
    assert isinstance(trainer.candidate_evaluator, DefaultCandidateEvaluationPlugin)
    assert trainer.candidate_evaluator.evaluator is trainer.training_backend
    assert trainer.training_backend.step_preparer == "sft"
    assert trainer.processor.context.config["batch_size"] == 2
    assert "max_staleness" not in trainer.processor.context.config


@pytest.mark.unit
def test_default_build_requires_processor_and_step_preparer_declarations() -> None:
    from reef.records import RecordStore

    @dataclass(frozen=True)
    class NoProcessorRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(step_preparer="sft", loss_family="sft")

    @dataclass(frozen=True)
    class NoPreparerRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(step_preparer="", loss_family="sft", processor=ThresholdProcessor)

    with pytest.raises(TypeError, match=r"declares no processor.*training_spec\(\).*override build"):
        NoProcessorRecipe(StubTrainingRuntime()).build("scenario", RecordStore())
    with pytest.raises(TypeError, match=r"declares no step_preparer.*registered preparer name.*'module:callable'"):
        NoPreparerRecipe(StubTrainingRuntime()).build("scenario", RecordStore())
