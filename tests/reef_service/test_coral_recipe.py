"""CoralRecipe: deployment wiring for the CORAL example.

Requires the reef checkout (recipe base + cookbook registrations); skipped
otherwise.
"""

from __future__ import annotations

import pytest

reef_recipe = pytest.importorskip("reef.recipe.base", reason="requires a reef checkout")

from reef.train.algos.registry import resolve_objective


def _recipe_cls():
    from recipes.beta.coral.recipe import CoralRecipe

    return CoralRecipe


def test_training_spec_binds_processor_and_grouped_machinery():
    recipe_cls = _recipe_cls()
    spec = recipe_cls.training_spec()
    from recipes.beta.coral.processor import CoralProcessor

    assert spec.processor is CoralProcessor
    assert spec.objective == "tttd"
    assert spec.loss_family == "tttd"


def test_importing_the_recipe_registers_the_reused_objective():
    _recipe_cls()
    assert resolve_objective("tttd").loss_family == "tttd"


def test_group_size_floor():
    recipe_cls = _recipe_cls()
    from reef.inference.http import InferenceProxyRuntime

    runtime = InferenceProxyRuntime(model_path="demo-model", base_url="http://localhost:8000")
    with pytest.raises(ValueError, match="at least two"):
        recipe_cls(training_runtime=None, runtime=runtime, group_size=1)
