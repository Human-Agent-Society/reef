"""First-class pre-publication candidate evaluation plugins."""

from __future__ import annotations

import pytest
from tests.support._candidate_evaluation_plugin import CheckpointEvaluator
from tests.support.runtime_stubs import StubInferenceRuntime, runtime_bindings

from recipes.sao import SAORecipe
from reef.evaluation import CandidateEvaluationConfig, build_candidate_evaluation
from reef.recipe import RecipeConfigError
from reef.storage.sqlite import SQLiteRecordStore

PLUGIN = "tests.support._candidate_evaluation_plugin"


def evaluation_config(*, score: float = 0.25, threshold: float = 0.8) -> dict:
    return {
        "module": f"{PLUGIN}:CheckpointEvaluator",
        "config": {
            "score": score,
            "threshold": threshold,
            "token_env": "EVALUATION_TOKEN",
        },
    }


@pytest.mark.unit
def test_plugin_class_builds_scenario_local_candidate_evaluator() -> None:
    config = CandidateEvaluationConfig.from_dict(
        evaluation_config(),
        environ={"EVALUATION_TOKEN": "secret"},
    )

    evaluator = build_candidate_evaluation(config, runtime=StubInferenceRuntime(), scenario="math")

    assert evaluator.scenario == "math"
    assert evaluator.token == "secret"
    assert evaluator.threshold == pytest.approx(0.8)


@pytest.mark.unit
def test_recipe_carries_evaluation_config_into_each_trainer() -> None:
    config = SAORecipe.service_config({}, model_path="/models/student")
    config["evaluation"] = evaluation_config()
    config["pipeline"] = {"sample2job": {"config": {"job_samples": 1}}}
    recipe = SAORecipe.from_environment(
        {"EVALUATION_TOKEN": "secret"},
        config=config,
        **runtime_bindings(),
    )

    trainer = recipe.build("math", SQLiteRecordStore())

    evaluator = trainer.job_loop.evaluator
    assert isinstance(evaluator, CheckpointEvaluator)
    assert evaluator.scenario == "math"
    assert evaluator.token == "secret"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, r"evaluation\.module"),
        ({"module": "not-dotted"}, r"must be 'package\.module:ClassName'"),
        ({"module": f"{PLUGIN}:missing"}, r"cannot import evaluation"),
        ({"module": f"{PLUGIN}:plain_function"}, r"must name a class that inherits ConfiguredEvaluationPlugin"),
        ({"module": f"{PLUGIN}:EvaluatorOnly"}, r"must name a class that inherits ConfiguredEvaluationPlugin"),
        ({"module": f"{PLUGIN}:DuckPlugin"}, r"must name a class that inherits ConfiguredEvaluationPlugin"),
        ({"module": f"{PLUGIN}:IncompletePlugin"}, r"names an abstract class"),
        ({"module": f"{PLUGIN}:CheckpointEvaluator", "extra": True}, r"unknown key.*extra"),
    ],
)
def test_invalid_evaluation_plugins_fail_before_training(config, match) -> None:
    recipe_config = SAORecipe.service_config({}, model_path="/models/student")
    recipe_config["evaluation"] = config
    with pytest.raises(RecipeConfigError, match=match):
        SAORecipe.from_environment({}, config=recipe_config, **runtime_bindings())
