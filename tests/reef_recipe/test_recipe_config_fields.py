"""Config fields: one dataclass field is the whole configuration surface.

Covers the field-metadata mechanism (``reef.recipe.config_fields``): key and env
derivation, type-aware casting from the annotation, precedence, and the loud
failures for values nothing would consume.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import pytest
import yaml
from tests.support._threshold_processor import ThresholdBatches, ThresholdTasks
from tests.support.runtime_stubs import StubTrainingRuntime, runtime_bindings

from reef.core.config import config_arguments
from reef.evaluation import BackendAlwaysSelectPlugin
from reef.recipe import RecipeConfigError, config_field
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import resolve_config_field_values
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.pipeline import PipelineConfig, PipelineSpec
from reef.train.runtime_backend import RuntimeCandidateBackend


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
        return WeightTrainingSpec(objective="sft")


@dataclass(frozen=True)
class StagedRecipe(WeightTrainingRecipe):
    """A recipe that declares its stages; their config comes from the pipeline section."""

    _: KW_ONLY
    name: str = "staged"

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(objective="sft")

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(record2task=ThresholdTasks, task2batch=ThresholdBatches)


@pytest.mark.unit
def test_config_fields_declare_the_whole_configuration_surface() -> None:
    config_fields = {argument.name: argument for argument in config_arguments(ConfiguredRecipe)}

    assert sorted(config_fields) == [
        "batch_size",
        "label",
        "max_staleness",
        "strict",
        "temperature",
        "training_mode",
    ]
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
        "training_mode": "auto",
        "temperature": 0.25,
        "strict": True,
        "label": "run-a",
    }


@pytest.mark.unit
def test_float_config_field_parses_negative_infinity_from_yaml_and_env() -> None:
    # An accept-everything score bar is float("-inf"); the operator must be
    # able to spell it explicitly through both config field channels. YAML's native
    # spelling is `-.inf` (parsed as a float by safe_load), and the
    # environment fallback arrives as the string "-inf" — both must reach the
    # recipe as float("-inf") rather than needing a sentinel.
    yaml_value = yaml.safe_load("temperature: -.inf")
    assert yaml_value == {"temperature": float("-inf")}

    assert resolve_config_field_values(ConfiguredRecipe, yaml_value, {})["temperature"] == float("-inf")
    environ = {"REEF_CONFIGURED_TEMPERATURE": "-inf"}
    assert resolve_config_field_values(ConfiguredRecipe, {}, environ)["temperature"] == float("-inf")


@pytest.mark.unit
def test_float_config_field_survives_service_config_uncast() -> None:
    # The old service layer int()-cast every recipe setting, silently
    # truncating floats; the config field's annotation now owns the cast.
    config = ConfiguredRecipe.service_config({"temperature": 0.25}, model_path="/models/demo")
    assert config["data"]["temperature"] == 0.25

    interpolated = ConfiguredRecipe.service_config({"temperature": "0.25"}, model_path="/models/demo")
    assert interpolated["data"]["temperature"] == "0.25"  # Translation leaves parsing to field resolution.

    recipe = ConfiguredRecipe.from_environment({}, config=interpolated, **runtime_bindings(StubTrainingRuntime()))
    assert recipe.temperature == 0.25
    assert isinstance(recipe.temperature, float)


@pytest.mark.unit
def test_from_environment_resolves_every_config_field_source() -> None:
    recipe = ConfiguredRecipe.from_environment(
        {"REEF_CONFIGURED_TEMPERATURE": "0.5", "REEF_CONFIGURED_STRICT": "1"},
        config={"data": {"batch_size": 2}},
        **runtime_bindings(StubTrainingRuntime()),
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
def test_data_section_reads_either_spelling_and_refuses_both() -> None:
    """A deployment file writes group-size where the field is group_size; naming a field twice is a mistake."""
    resolved = resolve_config_field_values(ConfiguredRecipe, {"batch-size": 2, "temperature": 0.5}, {})
    assert resolved["batch_size"] == 2 and resolved["temperature"] == 0.5
    with pytest.raises(RecipeConfigError, match="names the field 'batch_size' twice"):
        resolve_config_field_values(ConfiguredRecipe, {"batch-size": 2, "batch_size": 3}, {})


@pytest.mark.unit
def test_data_section_rejects_keys_no_config_field_consumes() -> None:
    with pytest.raises(
        RecipeConfigError, match=r"does not consume config key.*'group_size'.*known ConfiguredRecipe config fields"
    ):
        ConfiguredRecipe.from_environment(
            {},
            config={"data": {"group_size": 2}},
            **runtime_bindings(StubTrainingRuntime()),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"batch_size": "eight"}, r"reef\.batch_size must be a valid int"),
        ({"batch_size": 2.5}, r"reef\.batch_size must be a valid int"),
        ({"temperature": "warm"}, r"reef\.temperature must be a valid float"),
        ({"strict": "maybe"}, r"reef\.strict must be a valid bool"),
        ({"label": 3}, r"reef\.label must be a string"),
    ],
)
def test_service_config_names_the_setting_and_the_expected_type(settings, match) -> None:
    with pytest.raises(RecipeConfigError, match=match):
        config = ConfiguredRecipe.service_config(settings, model_path="/models/demo")
        resolve_config_field_values(ConfiguredRecipe, config["data"], {})


@pytest.mark.unit
def test_invalid_field_values_surface_as_recipe_config_errors() -> None:
    # __post_init__ range checks (ValueError on direct construction) become
    # RecipeConfigError on the configuration path.
    with pytest.raises(RecipeConfigError, match="invalid ConfiguredRecipe configuration: max_staleness must be"):
        ConfiguredRecipe.from_environment(
            {}, config={"data": {"max_staleness": -1}}, **runtime_bindings(StubTrainingRuntime())
        )


@pytest.mark.unit
def test_a_stage_setting_in_the_recipe_config_is_refused_with_its_pipeline_key() -> None:
    from recipes.sao import SAORecipe
    from reef.pipeline_context import PipelineContext

    with pytest.raises(RecipeConfigError, match="does not consume"):
        SAORecipe.from_environment({}, config={"data": {"batch_size": 4}}, **runtime_bindings(StubTrainingRuntime()))
    # The stage's own range check fails when the pipeline builds.
    recipe = SAORecipe.from_environment(
        {},
        config={"pipeline": {"task2batch": {"config": {"batch_size": -1}}}},
        **runtime_bindings(StubTrainingRuntime()),
    )
    with pytest.raises(RecipeConfigError, match="batch_size must be positive"):
        recipe.build_pipeline(PipelineContext("math"))
    # The removed environment fallback no longer configures the stage.
    recipe = SAORecipe.from_environment(
        {"REEF_SAO_BATCH_SIZE": "4"}, config={}, **runtime_bindings(StubTrainingRuntime())
    )
    assert recipe.build_pipeline(PipelineContext("math")).batch_size == 128


@pytest.mark.unit
def test_config_field_annotations_outside_the_supported_scalars_fail_at_declaration() -> None:
    @dataclass(frozen=True)
    class BadRecipe(WeightTrainingRecipe):
        _: KW_ONLY
        sizes: list = config_field(None)  # noqa: RUF009 - the invalid declaration under test

    with pytest.raises(TypeError, match="unsupported config field type: sizes"):
        config_arguments(BadRecipe)


@pytest.mark.unit
def test_default_build_uses_declared_stages_and_their_pipeline_config() -> None:
    stages = PipelineConfig.from_setting(
        {"task2batch": {"config": {"batch_size": 2}}}, spec=StagedRecipe.pipeline_spec(), environ={}
    )
    trainer = StagedRecipe(
        **runtime_bindings(StubTrainingRuntime(max_staleness=3)), max_staleness=3, pipeline_stages=stages
    ).build("scenario", SQLiteRecordStore())

    assert isinstance(trainer.pipeline.record2task, ThresholdTasks)
    assert isinstance(trainer.pipeline.task2batch, ThresholdBatches)
    assert isinstance(trainer.candidate_backend, RuntimeCandidateBackend)
    assert isinstance(trainer.candidate_evaluator, BackendAlwaysSelectPlugin)
    assert trainer.candidate_evaluator._candidate_backend is trainer.candidate_backend
    assert trainer.candidate_backend.training_method_selector.method.objective == "sft"
    assert trainer.pipeline.batch_size == 2
    # The shared recipe field reaches the stages as the typed context field, not as stage config.
    assert trainer.pipeline.context.max_staleness == 3


@pytest.mark.unit
def test_default_build_requires_processor_and_objective_declarations() -> None:
    from reef.storage.sqlite import SQLiteRecordStore

    @dataclass(frozen=True)
    class NoTask2BatchRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(objective="sft")

    @dataclass(frozen=True)
    class NoObjectiveRecipe(WeightTrainingRecipe):
        @classmethod
        def training_spec(cls) -> WeightTrainingSpec:
            return WeightTrainingSpec(objective="")

        @classmethod
        def pipeline_spec(cls) -> PipelineSpec:
            return PipelineSpec(task2batch=ThresholdBatches)

    with pytest.raises(TypeError, match=r"declares no task2batch.*pipeline_spec\(\).*override build"):
        NoTask2BatchRecipe(**runtime_bindings(StubTrainingRuntime())).build("scenario", SQLiteRecordStore())
    with pytest.raises(TypeError, match=r"declares no objective.*registered objective name.*'module:Objective'"):
        NoObjectiveRecipe(**runtime_bindings(StubTrainingRuntime())).build("scenario", SQLiteRecordStore())


@pytest.mark.unit
def test_opd_config_keeps_objective_options_out_of_recipe_fields() -> None:
    from recipes.opd import OPDRecipe

    values = resolve_config_field_values(OPDRecipe, {}, {})
    assert values["max_staleness"] == 0
    # The stage settings belong to the pipeline section, and the objective options to the backend.
    for name in ("batch_size", "max_teacher_tokens", "teacher_checkpoint", "tokenizer_path"):
        assert name not in {argument.name for argument in config_arguments(OPDRecipe)}
