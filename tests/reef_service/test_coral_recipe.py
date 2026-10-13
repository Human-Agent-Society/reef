"""CoralRecipe: deployment wiring for the CORAL example.

Requires the reef checkout (recipe base + cookbook registrations); skipped
otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

reef_recipe = pytest.importorskip("reef.recipe.base", reason="requires a reef checkout")

from reef.pipeline_context import PipelineContext
from reef.recipe.errors import RecipeConfigError
from reef.train.algos.registry import resolve_objective


def _recipe_cls():
    from recipes.beta.coral.recipe import CoralRecipe

    return CoralRecipe


EXAMPLE = Path(__file__).resolve().parents[2] / "recipes" / "beta" / "coral" / "examples" / "coral_demo" / "serve.yaml"


def coral_recipe(config):
    return _recipe_cls().from_environment({}, config=config, **runtime_bindings(StubTrainingRuntime()))


def test_training_spec_binds_processor_and_grouped_machinery():
    recipe_cls = _recipe_cls()
    spec = recipe_cls.training_spec()
    stages = recipe_cls.pipeline_spec()
    from recipes.beta.coral.processor import CoralBatches, CoralTasks
    from recipes.beta.coral.recipe import CoralRecord2Record

    assert (stages.record2record, stages.record2task, stages.task2batch) == (
        CoralRecord2Record,
        CoralTasks,
        CoralBatches,
    )
    assert spec.objective == "tttd"
    assert spec.loss_family == "tttd"


def test_importing_the_recipe_registers_the_reused_objective():
    _recipe_cls()
    assert resolve_objective("tttd").loss_family == "tttd"


def test_example_pipeline_section_builds_the_coral_stages():
    from recipes.beta.coral.processor import CoralBatches, CoralTasks
    from recipes.beta.coral.recipe import CoralGrouping, CoralRecord2Record

    deployment = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    pipeline = coral_recipe({"pipeline": deployment["pipeline"]}).build_pipeline(PipelineContext("coral-demo"))

    assert isinstance(pipeline.record2record, CoralRecord2Record)
    assert isinstance(pipeline.record2record.grouping, CoralGrouping)
    assert pipeline.record2record.grouping.group_size == 2
    assert isinstance(pipeline.record2task, CoralTasks)
    assert pipeline.record2task.assembly.settings.accept_multi_turn_policy_samples
    assert isinstance(pipeline.task2batch, CoralBatches)
    assert pipeline.batch_size == 1


def test_default_group_size_is_four():
    pipeline = coral_recipe({}).build_pipeline(PipelineContext("coral-demo"))

    assert pipeline.record2record.grouping.group_size == 4


@pytest.mark.parametrize("key", ["group_size", "group-size"])
def test_group_size_in_the_recipe_config_is_refused(key):
    with pytest.raises(RecipeConfigError, match="does not consume"):
        coral_recipe({"data": {key: 2}})


@pytest.mark.parametrize(
    ("pipeline", "message"),
    [
        ({"record2record": {"config": {"group_size": 1}}}, "at least two"),
        ({"task2batch": {"config": {"group_size": 2}}}, "CoralBatches takes no config"),
        ({"record2task": {"config": {"group_size": 2}}}, "group_size"),
    ],
)
def test_bad_stage_config_is_refused_when_the_pipeline_builds(pipeline, message):
    recipe = coral_recipe({"pipeline": pipeline})
    with pytest.raises(RecipeConfigError, match=message):
        recipe.build_pipeline(PipelineContext("coral-demo"))
