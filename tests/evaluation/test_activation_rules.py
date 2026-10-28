"""The activation rule decides whether an artifact that training makes serves; the ``activation`` section names it."""

from __future__ import annotations

import pytest
from tests.support._activation_rule_plugin import ComponentRule
from tests.support.runtime_stubs import runtime_bindings

from recipes.sao import SAORecipe
from reef.core.activation import MadeArtifact
from reef.core.evaluation import PolicyEvaluation, PolicyHarness, PolicyModel, RolloutScore, RolloutScores
from reef.evaluation import AlwaysActivate, EvaluatedActivation, NeverActivate
from reef.evaluation.activation import activation_reference, build_activation_rule
from reef.recipe import RecipeConfigError

PLUGIN = "tests.support._activation_rule_plugin"


def evaluation(*values: float) -> PolicyEvaluation:
    scores = RolloutScores((("t", tuple(RolloutScore.scored(value) for value in values)),))
    return PolicyEvaluation(PolicyHarness("terminus-2"), PolicyModel("m"), scores)


@pytest.mark.unit
def test_the_built_in_rules_activate_always_never_or_after_an_evaluation_that_passes() -> None:
    weights = MadeArtifact("weights")
    assert build_activation_rule(None).activates(weights), "always is the default"
    assert AlwaysActivate().activates(weights) and not NeverActivate().activates(weights)
    reviewed = AlwaysActivate(frozenset({"skill"}))
    assert not reviewed.activates(MadeArtifact("harness", frozenset({"skill", "prompt"})))
    assert reviewed.activates(MadeArtifact("harness", frozenset({"prompt"})))
    assert not AlwaysActivate().activates(MadeArtifact("harness", frozenset({"native_loop"}))), "always reviewed"
    gated = EvaluatedActivation(0.5)
    assert not gated.activates(weights)
    assert gated.activates_evaluated("a1", evaluation(1.0, 0.0)) and not gated.activates_evaluated(
        "a1", evaluation(0.0)
    )
    failed = PolicyEvaluation(
        PolicyHarness("terminus-2"), PolicyModel("m"), RolloutScores((("t", (RolloutScore.failed(),)),))
    )
    assert not gated.activates_evaluated("a1", failed), "an evaluation with no mean score activates nothing"
    assert not AlwaysActivate().activates_evaluated("a1", evaluation(1.0))


@pytest.mark.unit
def test_an_activation_section_names_a_built_in_rule_or_a_plugin_class() -> None:
    assert isinstance(build_activation_rule(activation_reference("never")), NeverActivate)
    rule = build_activation_rule(activation_reference({"module": "always", "config": {"review_kinds": ["skill"]}}))
    assert isinstance(rule, AlwaysActivate) and rule.review_kinds == frozenset({"skill"})
    gated = build_activation_rule(activation_reference({"module": "evaluated", "config": {"min_mean_score": 0.8}}))
    assert isinstance(gated, EvaluatedActivation) and gated.min_mean_score == 0.8
    custom = build_activation_rule(
        activation_reference({"module": f"{PLUGIN}:ComponentRule", "config": {"component": "weights"}})
    )
    assert isinstance(custom, ComponentRule) and custom.activates(MadeArtifact("weights"))


@pytest.mark.unit
def test_the_recipe_carries_its_activation_rule_into_each_scenario() -> None:
    config = SAORecipe.service_config({}, model_path="/models/student")
    config["activation"] = {"module": "never"}
    config["pipeline"] = {"sample2job": {"config": {"job_samples": 1}}}
    recipe = SAORecipe.from_environment({}, config=config, **runtime_bindings())
    assert isinstance(recipe.activation_rule(), NeverActivate)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"module": "not-dotted"}, r"must be 'package\.module:ClassName'"),
        ({"module": f"{PLUGIN}:missing"}, r"cannot import activation"),
        ({"module": f"{PLUGIN}:plain_function"}, r"must name a class that inherits ActivationRule"),
        ({"module": f"{PLUGIN}:EvaluatorOnly"}, r"must name a class that inherits ActivationRule"),
        ({"module": f"{PLUGIN}:DuckPlugin"}, r"must name a class that inherits ActivationRule"),
        ({"module": f"{PLUGIN}:IncompleteRule"}, r"names an abstract class"),
        ({"module": "always", "extra": True}, r"unknown key.*extra"),
        ({"module": "always", "config": {"review_kinds": "skill"}}, r"review_kinds must be a list"),
        ({"module": "never", "config": {"x": 1}}, r"takes no config"),
        ({"module": "evaluated", "config": {}}, r"exactly min_mean_score"),
        ({"module": "evaluated", "config": {"min_mean_score": "high"}}, r"finite number"),
    ],
)
def test_an_invalid_activation_section_fails_the_deployment(config, match) -> None:
    recipe_config = SAORecipe.service_config({}, model_path="/models/student")
    recipe_config["activation"] = config
    with pytest.raises(RecipeConfigError, match=match):
        SAORecipe.from_environment({}, config=recipe_config, **runtime_bindings())
