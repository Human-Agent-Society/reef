"""Config fields: one dataclass field is the whole configuration surface.

Covers the field-metadata mechanism (``reef.recipe.config_fields``): key and env
derivation, type-aware casting from the annotation, precedence, and the loud
failures for values nothing would consume.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import pytest
import yaml
from tests.support._threshold_processor import ThresholdJobData, ThresholdSamples
from tests.support.runtime_stubs import runtime_bindings

from reef.core.config import config_arguments
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.recipe import RecipeConfigError, config_field
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import resolve_config_field_values
from reef.storage.sqlite import SQLiteRecordStore
from reef.train.jobs import FixedJobPlan, JobPlanContext
from reef.train.pipeline import PipelineConfig, PipelineSpec
from reef.train.types import JobData


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
        return PipelineSpec(record2sample=ThresholdSamples, sample2job=ThresholdJobData)


@pytest.mark.unit
def test_config_fields_declare_the_whole_configuration_surface() -> None:
    config_fields = {argument.name: argument for argument in config_arguments(ConfiguredRecipe)}

    assert sorted(config_fields) == [
        "batch_size",
        "job_parameters",
        "label",
        "strict",
        "temperature",
        "training_mode",
    ]
    assert config_fields["batch_size"].env == "REEF_CONFIGURED_BATCH_SIZE"
    assert config_fields["batch_size"].default == 4


@pytest.mark.unit
def test_config_field_precedence_is_config_over_environment_over_default() -> None:
    environ = {"REEF_CONFIGURED_BATCH_SIZE": "8"}

    assert resolve_config_field_values(ConfiguredRecipe, {}, {})["batch_size"] == 4
    assert resolve_config_field_values(ConfiguredRecipe, {}, environ)["batch_size"] == 8
    assert resolve_config_field_values(ConfiguredRecipe, {"batch_size": 2}, environ)["batch_size"] == 2


@pytest.mark.unit
def test_config_field_casting_is_type_aware_for_every_annotation() -> None:
    values = resolve_config_field_values(
        ConfiguredRecipe,
        {"batch_size": "16", "temperature": "0.25", "strict": "true", "label": "run-a"},
        {},
    )

    assert values == {
        "batch_size": 16,
        "job_parameters": None,
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

    recipe = ConfiguredRecipe.from_environment({}, config=interpolated, **runtime_bindings())
    assert recipe.temperature == 0.25
    assert isinstance(recipe.temperature, float)


@pytest.mark.unit
def test_from_environment_resolves_every_config_field_source() -> None:
    recipe = ConfiguredRecipe.from_environment(
        {"REEF_CONFIGURED_TEMPERATURE": "0.5", "REEF_CONFIGURED_STRICT": "1"},
        config={"data": {"batch_size": 2}},
        **runtime_bindings(),
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
            **runtime_bindings(),
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
    with pytest.raises(RecipeConfigError, match="invalid ConfiguredRecipe configuration: training_mode must be"):
        ConfiguredRecipe.from_environment({}, config={"data": {"training_mode": "never"}}, **runtime_bindings())


@pytest.mark.unit
def test_the_recipe_refuses_max_staleness_which_the_record_stage_owns() -> None:
    with pytest.raises(RecipeConfigError, match=r"does not consume config key.*'max_staleness'"):
        ConfiguredRecipe.from_environment({}, config={"data": {"max-staleness": 2}}, **runtime_bindings())


@pytest.mark.unit
def test_a_stage_setting_in_the_recipe_config_is_refused_with_its_pipeline_key() -> None:
    from recipes.sao import SAORecipe
    from reef.pipeline_context import PipelineContext

    with pytest.raises(RecipeConfigError, match="does not consume"):
        SAORecipe.from_environment({}, config={"data": {"batch_size": 4}}, **runtime_bindings())
    # The stage's own range check fails when the pipeline builds.
    recipe = SAORecipe.from_environment(
        {},
        config={"pipeline": {"sample2job": {"config": {"buffer_size": -1}}}},
        **runtime_bindings(),
    )
    with pytest.raises(RecipeConfigError, match="buffer_size must be None or a positive int"):
        recipe.build_pipeline(PipelineContext("math"))


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
        {"sample2job": {"config": {"job_samples": 2, "buffer_size": 8}}}, spec=StagedRecipe.pipeline_spec(), environ={}
    )
    trainer = StagedRecipe(**runtime_bindings(), pipeline_stages=stages).build("scenario", SQLiteRecordStore())

    assert isinstance(trainer.pipeline.record2sample, ThresholdSamples)
    assert isinstance(trainer.pipeline.sample2job, ThresholdJobData)
    assert trainer.job_loop is not None
    assert isinstance(trainer.job_loop.evaluator, AlwaysSelectCheckpointPlugin)
    plan = trainer.job_loop.plan
    assert isinstance(plan, FixedJobPlan)
    request = plan.next_job(JobPlanContext(JobData("scenario:1"), 0, None))
    assert request.parameters["method"]["objective"] == "sft"
    # The job size and the buffer size come from the stage config.
    assert (trainer.job_samples, trainer.pipeline.buffer_size) == (2, 8)


@pytest.mark.unit
def test_the_default_job_plan_sends_the_configured_job_parameters_with_every_job() -> None:
    config = {"job_parameters": {"scheduling": {"packing_length": 4096}, "save_interval": 50}}
    job_parameters = resolve_config_field_values(StagedRecipe, config, {})["job_parameters"]
    recipe = StagedRecipe(**runtime_bindings(), job_parameters=job_parameters)

    request = recipe.job_plan().next_job(JobPlanContext(JobData("scenario:1"), 0, None))

    assert request.parameters == {
        "scheduling": {"packing_length": 4096},
        "save_interval": 50,
        "method": {"objective": "sft", "learning_rate_schedule": None},
    }
    with pytest.raises(RecipeConfigError, match="job_parameters cannot set method"):
        StagedRecipe(**runtime_bindings(), job_parameters={"method": {"objective": "other"}})


@pytest.mark.unit
def test_a_weight_recipe_without_job_samples_fails_its_build() -> None:
    with pytest.raises(RecipeConfigError, match=r"set pipeline\.sample2job\.config\.job_samples"):
        StagedRecipe(**runtime_bindings()).build("scenario", SQLiteRecordStore())


@pytest.mark.unit
def test_default_build_requires_processor_and_objective_declarations() -> None:
    from reef.storage.sqlite import SQLiteRecordStore

    @dataclass(frozen=True)
    class NoSample2JobRecipe(WeightTrainingRecipe):
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
            return PipelineSpec(sample2job=ThresholdJobData)

    with pytest.raises(TypeError, match=r"declares no sample2job.*pipeline_spec\(\).*override build"):
        NoSample2JobRecipe(**runtime_bindings()).build("scenario", SQLiteRecordStore())
    with pytest.raises(TypeError, match=r"declares no objective.*registered objective name.*'module:Objective'"):
        NoObjectiveRecipe(**runtime_bindings()).build("scenario", SQLiteRecordStore())


@pytest.mark.unit
def test_opd_config_keeps_objective_options_out_of_recipe_fields() -> None:
    from recipes.opd import OPDRecipe

    # The stage settings belong to the pipeline section, and the objective options to the backend.
    for name in ("batch_size", "max_staleness", "max_teacher_tokens", "teacher_checkpoint", "tokenizer_path"):
        assert name not in {argument.name for argument in config_arguments(OPDRecipe)}
