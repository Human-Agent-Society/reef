"""CoralRecipe: deployment wiring for the CORAL example.

Requires the reef checkout (recipe base + cookbook registrations); skipped
otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.support.runtime_stubs import runtime_bindings

reef_recipe = pytest.importorskip("reef.recipe.base", reason="requires a reef checkout")

from reef.pipeline_context import PipelineContext
from reef.recipe.errors import RecipeConfigError
from reef.train.algos.registry import resolve_objective


def _recipe_cls():
    from recipes.beta.coral.recipe import CoralRecipe

    return CoralRecipe


EXAMPLE = Path(__file__).resolve().parents[3] / "recipes" / "beta" / "coral" / "examples" / "coral_demo" / "serve.yaml"


def coral_recipe(config):
    return _recipe_cls().from_environment({}, config=config, **runtime_bindings())


def test_training_spec_binds_processor_and_grouped_machinery():
    recipe_cls = _recipe_cls()
    spec = recipe_cls.training_spec()
    stages = recipe_cls.pipeline_spec()
    from recipes.beta.coral.processor import CoralJobData, CoralSamples

    assert (stages.record2sample, stages.sample2job) == (CoralSamples, CoralJobData)
    assert spec.objective == "tttd"
    assert spec.loss_family == "tttd"


def test_importing_the_recipe_registers_the_reused_objective():
    _recipe_cls()
    assert resolve_objective("tttd").loss_family == "tttd"


def test_example_pipeline_section_builds_the_coral_stages():
    from recipes.beta.coral.processor import CoralGrouping, CoralJobData, CoralSamples

    deployment = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    pipeline = coral_recipe({"pipeline": deployment["pipeline"]}).build_pipeline(PipelineContext("coral-demo"))

    assert isinstance(pipeline.record2sample, CoralSamples)
    assert isinstance(pipeline.record2sample.grouping, CoralGrouping)
    assert pipeline.record2sample.grouping.group_size == 2
    assert pipeline.record2sample.assembly.settings.accept_multi_turn_policy_samples
    assert isinstance(pipeline.sample2job, CoralJobData)


def test_default_group_size_is_four():
    pipeline = coral_recipe({}).build_pipeline(PipelineContext("coral-demo"))

    assert pipeline.record2sample.grouping.group_size == 4


@pytest.mark.parametrize("key", ["group_size", "group-size"])
def test_group_size_in_the_recipe_config_is_refused(key):
    with pytest.raises(RecipeConfigError, match="does not consume"):
        coral_recipe({"data": {key: 2}})


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"record2sample": {"config": {"group_size": 1}}}, "at least two"),
        ({"sample2job": {"config": {"group_size": 2}}}, "CoralJobData: unknown config fields: group_size"),
        ({"record2sample": {"config": {"max_staleness": -1}}}, "max_staleness must be a non-negative int"),
    ],
)
def test_bad_stage_config_is_refused_when_the_pipeline_builds(pipeline, message):
    recipe = coral_recipe({"pipeline": pipeline})
    with pytest.raises(RecipeConfigError, match=message):
        recipe.build_pipeline(PipelineContext("coral-demo"))
