"""SPADE's Slime objective: task jobs on the slime family that center by SPADE's rule and carry delayed samples."""

from __future__ import annotations

import pytest

import recipes.beta.spade  # noqa: F401  (registers the spade objective)
from recipes.beta.spade.slime import SpadeObjective
from reef.train.algos.group_advantages import GroupAdvantageRule, group_advantages
from reef.train.slime_backend.launch import SlimeDeployment
from reef.train.slime_backend.objectives import resolve_slime_objective


@pytest.mark.unit
def test_the_spade_objective_trains_task_jobs_on_the_slime_family_with_spades_rule() -> None:
    objective = resolve_slime_objective("spade")
    assert isinstance(objective, SpadeObjective)
    assert (objective.loss_family, objective.carries_samples) == ("slime", True)
    assert objective.advantage_rule == GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)
    plays = group_advantages([1, 0, 0, 1, 1, 0, 0, 0], objective.advantage_rule)
    assert (plays[0], plays[1]) == (pytest.approx(1.290994, abs=1e-6), pytest.approx(-0.774597, abs=1e-6))


@pytest.mark.unit
def test_a_deployment_whose_task_objective_is_spade_starts_on_the_slime_family() -> None:
    settings = {
        "recipe": "spade",
        "ray_address": "ray://head:10001",
        "ray_namespace": "reef",
        "execution_config": {},
        "train_timeout_s": 600.0,
        "model_path": "/models/m",
        "training_backend_options": {},
        "teacher_model_path": None,
        "teacher_timeout": 60.0,
    }
    config = SlimeDeployment().component_config(settings, default_parameters={"task": {"objective": "spade"}})
    assert config["loss_family"] == "slime"
