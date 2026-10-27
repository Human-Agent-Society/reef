"""Contract for method-owned preparation and backend loss selection."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from recipes.sao.objective import SaoObjective
from recipes.tttd.objective import TttdObjective
from reef.recipe import WeightTrainingSpec
from reef.runtime.interfaces import TrainingMethod
from reef.train.algos import TrainingObjective, TrainingSignal
from reef.train.algos.registry import (
    ObjectiveRegistry,
    job_training,
    register_objective,
    resolve_objective,
    unregister_objective,
)
from reef.train.algos.schedule import StepScheduling
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.tinker_backend.losses import resolve_tinker_loss
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.types import JobData


class ExampleObjective(TrainingObjective):
    name = "test-example-objective"
    loss_family = "sft"

    def prepare(self, batch: JobData, state: Mapping[str, object]) -> TrainingSignal:
        return TrainingSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


OBJECTIVE_INSTANCE = ExampleObjective()


def plain_preparation(batch, state):
    return TrainingSignal("skip", state)


def test_objective_resolution_accepts_names_classes_and_instances():
    registry = ObjectiveRegistry()
    objective = registry.resolve(f"{__name__}:ExampleObjective")
    assert registry.resolve(objective.name) is objective
    assert registry.resolve(f"{__name__}:ExampleObjective") is objective
    assert registry.register(objective) is objective
    assert registry.names == (objective.name,)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(ExampleObjective())
    registry.unregister(objective.name)
    assert registry.resolve(f"{__name__}:OBJECTIVE_INSTANCE") is OBJECTIVE_INSTANCE


@pytest.mark.parametrize("reference", ["plain_preparation", "MissingObjective", "TrainingSignal"])
def test_dotted_objective_rejects_functions_missing_attributes_and_unrelated_classes(reference):
    with pytest.raises(TypeError, match="TrainingObjective class or instance"):
        ObjectiveRegistry().resolve(f"{__name__}:{reference}")


def test_objective_resolution_reports_unknown_and_malformed_references():
    registry = ObjectiveRegistry()
    with pytest.raises(ValueError, match=r"unknown objective.*available objectives"):
        registry.resolve("missing")
    with pytest.raises(ValueError, match=r"must be 'package\.module:Objective'"):
        registry.resolve(":ExampleObjective")
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("missing")


def test_objective_registration_validates_the_method_contract():
    registry = ObjectiveRegistry()
    with pytest.raises(TypeError, match="inherit TrainingObjective"):
        registry.register(plain_preparation)
    objective = ExampleObjective()
    objective.name = ""
    with pytest.raises(ValueError, match="non-empty name"):
        registry.register(objective)
    objective.name = ExampleObjective.name
    objective.loss_family = ""
    with pytest.raises(ValueError, match="non-empty loss_family"):
        registry.register(objective)
    objective.loss_family = ExampleObjective.loss_family
    objective.supports_multiple_epochs = "yes"
    with pytest.raises(ValueError, match="supports_multiple_epochs as a bool"):
        registry.register(objective)


def test_recipe_has_one_objective_binding_and_derives_its_loss_family():
    @register_objective
    class CustomObjective(ExampleObjective):
        name = "test-recipe-objective"
        loss_family = "tttd"

    try:
        spec = WeightTrainingSpec(objective=CustomObjective.name)
        assert spec.loss_family == "tttd"
        assert "scheduling" not in WeightTrainingSpec.__dataclass_fields__
        assert isinstance(resolve_objective(CustomObjective.name), CustomObjective)
        with pytest.raises(TypeError, match="loss_family"):
            WeightTrainingSpec(objective=CustomObjective.name, loss_family="sft")
    finally:
        unregister_objective(CustomObjective.name)


def test_job_training_reads_the_method_and_the_schedule_the_objective_accepts():
    # The schedule is a job parameter, not part of the objective's signal.
    assert "scheduling" not in TrainingSignal.__dataclass_fields__
    method = TrainingMethod("sao").to_dict()
    assert job_training({"method": method}) == (TrainingMethod("sao"), StepScheduling())
    assert job_training({"method": method, "scheduling": {"epochs": 2}}) == (
        TrainingMethod("sao"),
        StepScheduling(epochs=2),
    )

    # An objective only says what its loss tolerates: SAO's clipped ratio
    # accepts a second pass, TTTD's unclipped importance sampling does not.
    assert SaoObjective.supports_multiple_epochs is True
    assert TttdObjective.supports_multiple_epochs is False
    with pytest.raises(ValueError, match="'tttd' does not support 2 epochs per job"):
        job_training({"method": TrainingMethod("tttd").to_dict(), "scheduling": {"epochs": 2}})
    TttdObjective().validate_scheduling(StepScheduling(shuffle=True))


@pytest.mark.parametrize("parameters", [{}, {"method": "tttd"}, {"scheduling": {}}])
def test_job_training_refuses_a_job_without_a_method_object(parameters):
    with pytest.raises(ValueError, match="needs the 'method' parameter as an object"):
        job_training(parameters)


def test_objectives_keep_each_sample_its_own_rollout_by_default():
    assert TrainingObjective.rollout_unit == "sample"
    assert SaoObjective.rollout_unit == TttdObjective.rollout_unit == "sample"


def test_both_backends_preserve_skip_and_proposed_state_without_resolving_a_loss():
    objective = ExampleObjective()
    objective.loss_family = "no-backend-needed-for-skip"
    register_objective(objective)
    try:
        state = {"steps": 7}
        batch = JobData("empty", ())
        slime = prepare_slime_job(batch, TrainingMethod(objective.name), state, StepScheduling(), step_size=1)
        tinker = prepare_tinker_job(
            batch, TrainingMethod(objective.name), state, StepScheduling(), batch_size=1, kl_coef=0.0
        )
        assert slime.action == "skip" and slime.payload is None
        assert tinker.loss is None and tinker.steps == ()
        assert slime.next_algorithm_state == tinker.next_algorithm_state == {"steps": 7, "skipped": True}
        assert state == {"steps": 7}
        assert slime.metrics == tinker.metrics == {"reason": "no signal"}
    finally:
        unregister_objective(objective.name)


def test_group_advantages_are_shared_across_backends_before_optimizer_partitioning():
    # One sample per optimizer step: every group is split across steps, yet
    # its advantages come from the whole group because prepare ran first.
    one_sample_steps = StepScheduling()
    rewards = (0.0, 2.0, 3.0, 1.0)
    batch = JobData(
        "two-groups",
        tuple(
            replace(policy_trajectory(str(index), (10, 11), (1,), (-0.1,), reward), group_id=str(index // 2))
            for index, reward in enumerate(rewards)
        ),
    )
    state = {"steps": 4}
    expected = tuple(
        advantage
        for group in (rewards[:2], rewards[2:])
        for advantage in TttdObjective.adaptive_entropic_advantages(list(group))[0]
    )
    slime = prepare_slime_job(batch, TrainingMethod("tttd"), state, one_sample_steps, step_size=1)
    tinker = prepare_tinker_job(batch, TrainingMethod("tttd"), state, one_sample_steps, batch_size=1, kl_coef=0.0)
    assert slime.payload["loss"] == "tttd" and tinker.loss is resolve_tinker_loss("tttd")
    assert slime.payload["advantages"] == pytest.approx(expected)
    assert [rows[0].advantage for rows in tinker.steps] == pytest.approx(expected)
    assert slime.payload["external_step_sizes"] == [1, 1, 1, 1]
    assert len(tinker.steps) == 4
    assert slime.next_algorithm_state == tinker.next_algorithm_state == {"steps": 5}
    assert state == {"steps": 4}
    assert slime.metrics["adaptive_betas"] == tinker.metrics["adaptive_betas"]


def test_method_objectives_resolve_without_importing_training_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from recipes.tttd import TTTDRecipe\n"
            "from recipes.sao import SAORecipe\n"
            "from recipes.openclawrl import OpenClawRLRecipe\n"
            "from reef.train.algos.registry import resolve_objective\n"
            "from reef.train.algos.schedule import StepScheduling\n"
            "for recipe in (TTTDRecipe, SAORecipe, OpenClawRLRecipe):\n"
            "    spec = recipe.training_spec()\n"
            "    objective = resolve_objective(spec.objective)\n"
            "    assert objective.loss_family == spec.loss_family\n"
            "    objective.validate_scheduling(StepScheduling())\n"
            "assert not {'torch', 'ray', 'slime', 'tinker'} & sys.modules.keys()\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
