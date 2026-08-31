"""First-class pre-publication candidate evaluation plugins."""

from __future__ import annotations

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime

from recipes.sao import SAORecipe
from reef.recipe import RecipeConfigError
from reef.records import RecordStore
from reef.train.evaluation import CandidateEvaluationConfig, CandidateEvaluationConfigError, build_candidate_evaluation

PLUGIN = "reef_service._candidate_evaluation_plugin"


def evaluation_config(*, score: float = 0.25, threshold: float = 0.8) -> dict:
    return {
        "module": f"{PLUGIN}:build_evaluator",
        "config": {
            "score": score,
            "threshold": threshold,
            "token_env": "EVALUATION_TOKEN",
        },
    }


@pytest.mark.unit
def test_dotted_factory_builds_scenario_local_candidate_evaluator() -> None:
    runtime = StubTrainingRuntime()
    config = CandidateEvaluationConfig.from_dict(
        evaluation_config(),
        environ={"EVALUATION_TOKEN": "secret"},
    )

    evaluator = build_candidate_evaluation(config, runtime=runtime, scenario="math")

    assert evaluator.scenario == "math"
    assert evaluator.token == "secret"
    assert evaluator.threshold == pytest.approx(0.8)


@pytest.mark.unit
def test_recipe_carries_evaluation_config_into_each_trainer() -> None:
    runtime = StubTrainingRuntime()
    config = SAORecipe.service_config(
        {"batch_size": 1},
        model_path="/models/student",
    )
    config["evaluation"] = evaluation_config()
    recipe = SAORecipe.from_environment(
        {"EVALUATION_TOKEN": "secret"},
        config=config,
        runtime=runtime,
    )

    trainer = recipe.build("math", RecordStore())

    assert trainer.candidate_evaluator is not None
    assert trainer.candidate_evaluator.scenario == "math"
    assert trainer.candidate_evaluator.token == "secret"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (f"{PLUGIN}:build_invalid", r"does not provide evaluate\(candidate\)"),
        (f"{PLUGIN}:build_evaluator_only", r"does not provide decide\(candidate, evaluation\)"),
    ],
)
def test_factory_must_return_the_declared_evaluator_contract(factory, match) -> None:
    config = CandidateEvaluationConfig(
        module=factory,
        config={"score": 1.0, "threshold": 0.8},
    )

    with pytest.raises(CandidateEvaluationConfigError, match=match):
        build_candidate_evaluation(config, runtime=StubTrainingRuntime(), scenario="math")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({}, r"evaluation\.module"),
        ({"module": "not-dotted"}, r"must be 'package\.module:factory_name'"),
        ({"module": f"{PLUGIN}:missing"}, r"cannot import candidate evaluation plugin factory"),
        ({"module": f"{PLUGIN}:build_evaluator", "extra": True}, r"unknown key.*'extra'"),
    ],
)
def test_invalid_evaluation_plugins_fail_before_training(config, match) -> None:
    runtime = StubTrainingRuntime()
    recipe_config = SAORecipe.service_config({}, model_path="/models/student")
    recipe_config["evaluation"] = config
    with pytest.raises(RecipeConfigError, match=match):
        SAORecipe.from_environment({}, config=recipe_config, runtime=runtime)
