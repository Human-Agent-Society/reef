"""A Slime objective that carries delayed samples, and Slime's options with which its task jobs run.

``CARRYING_OBJECTIVE`` is the dotted ``module:Objective`` reference that the resolver
(``reef.train.slime_backend.objectives.resolve_objective``) accepts, so that the Slime suites share one class under
the objective's name.
"""

from __future__ import annotations

from collections.abc import Mapping

from reef.core.job_data import JobData
from reef.train.slime_backend.group_advantages import GroupAdvantageRule
from reef.train.slime_backend.objectives import SlimeObjective
from reef.train.slime_backend.signals import TrainingSignal

#: SPADE's rule: each group centered on its mean and divided by its population standard deviation.
SPADE = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)


class CarryingObjective(SlimeObjective):
    """A method on the slime family that centers by SPADE's rule and carries delayed samples, as SPADE's Designer
    objective does."""

    name = "test-carrying"
    loss_family = "slime"
    advantage_rule = SPADE
    carries_samples = True

    def prepare(self, data: JobData, state: Mapping[str, object]) -> TrainingSignal:
        raise AssertionError("a task job prepares nothing")


CARRYING_OBJECTIVE = "tests.support.carrying_objective:CarryingObjective"

#: Slime's parsed options that a task job of ``CarryingObjective`` reads, with values that work: groups of 2 in
#: rollouts of 4 prompts, and an optimizer step of 7 samples.
CARRYING_OPTIONS: Mapping[str, object] = {
    "advantage_estimator": "grpo",
    "n_samples_per_prompt": 2,
    "rollout_batch_size": 4,
    "global_batch_size": 7,
    "use_tis": True,
    "normalize_advantages": False,
    "custom_advantage_function_path": None,
    "use_dynamic_batch_size": True,
    "custom_tis_function_path": None,
    "use_rollout_logprobs": False,
    "rollout_temperature": 0.6,
    "rollout_top_p": 1.0,
    "rollout_top_k": -1,
    "use_score_centering": False,
    "score_centering_top_k": 2,
}
