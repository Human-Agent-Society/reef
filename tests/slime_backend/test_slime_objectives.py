"""Contract for Slime's objectives: their registration, their resolution and the loss family that each selects."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from recipes.sao.slime import SaoObjective
from recipes.tttd.objective import adaptive_entropic_advantages
from recipes.tttd.slime import TttdObjective
from reef.train.slime_backend.loss_families import loss_family_reference, resolve_loss_family
from reef.train.slime_backend.objectives import (
    OBJECTIVES,
    SlimeObjective,
    register_objective,
    resolve_objective,
    unregister_objective,
)
from reef.train.slime_backend.parameters import SlimeJobParameters
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.slime_backend.schedule import StepScheduling
from reef.train.slime_backend.signals import TrainingSignal
from reef.train.types import JobData


class ExampleObjective(SlimeObjective):
    name = "test-example-objective"
    loss_family = "sft"

    def prepare(self, batch: JobData, state: Mapping[str, object]) -> TrainingSignal:
        return TrainingSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


OBJECTIVE_INSTANCE = ExampleObjective()


class MisnamedObjective(ExampleObjective):
    name = "test-another-name"


def plain_preparation(batch, state):
    return TrainingSignal("skip", state)


@pytest.fixture
def no_objectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.train.slime_backend.objectives.OBJECTIVES", {})


def test_objective_resolution_accepts_names_classes_and_instances(no_objectives):
    objective = resolve_objective(f"{__name__}:ExampleObjective")
    assert resolve_objective(objective.name) is objective
    assert resolve_objective(f"{__name__}:ExampleObjective") is objective
    register_objective(objective.name, objective)
    with pytest.raises(ValueError, match="already registered"):
        register_objective(objective.name, ExampleObjective())
    unregister_objective(objective.name)
    assert resolve_objective(f"{__name__}:OBJECTIVE_INSTANCE") is OBJECTIVE_INSTANCE


def test_a_registered_reference_is_imported_at_its_first_resolution(no_objectives):
    from reef.train.slime_backend import objectives

    reference = f"{__name__}:ExampleObjective"
    register_objective(ExampleObjective.name, reference)
    register_objective(ExampleObjective.name, reference)
    assert {ExampleObjective.name: reference} == objectives.OBJECTIVES
    with pytest.raises(ValueError, match="already registered"):
        register_objective(ExampleObjective.name, f"{__name__}:OBJECTIVE_INSTANCE")

    objective = resolve_objective(ExampleObjective.name)
    assert isinstance(objective, ExampleObjective)
    assert resolve_objective(ExampleObjective.name) is objective
    # A job may also name the registered objective by its reference.
    assert resolve_objective(reference) is objective

    # A reference whose objective declares another name stays a reference, and every resolution refuses it.
    register_objective("test-misnamed", f"{__name__}:MisnamedObjective")
    for _ in range(2):
        with pytest.raises(ValueError, match="cannot register under the name 'test-misnamed'"):
            resolve_objective("test-misnamed")


@pytest.mark.parametrize("reference", ["plain_preparation", "MissingObjective", "TrainingSignal"])
def test_dotted_objective_rejects_functions_missing_attributes_and_unrelated_classes(reference):
    with pytest.raises(ValueError, match="SlimeObjective class or instance"):
        resolve_objective(f"{__name__}:{reference}")


def test_objective_resolution_reports_unknown_and_malformed_references():
    with pytest.raises(ValueError, match=r"unknown Slime objective.*available objectives"):
        resolve_objective("missing")
    with pytest.raises(ValueError, match=r"must be 'package\.module:Objective'"):
        resolve_objective(":ExampleObjective")
    with pytest.raises(ValueError, match=r"must be 'package\.module:Objective'"):
        register_objective("malformed", "no_colon")
    with pytest.raises(ValueError, match="cannot import objective"):
        resolve_objective("no_such_package.module:Objective")
    with pytest.raises(KeyError, match="not registered"):
        unregister_objective("missing")


def test_objective_registration_validates_the_method_contract():
    with pytest.raises(TypeError, match="inherit SlimeObjective"):
        register_objective("plain", plain_preparation)
    objective = ExampleObjective()
    with pytest.raises(ValueError, match="cannot register under the name 'other'"):
        register_objective("other", objective)
    objective.name = ""
    with pytest.raises(ValueError, match="non-empty name"):
        register_objective("", objective)
    objective.name = ExampleObjective.name
    objective.loss_family = ""
    with pytest.raises(ValueError, match="non-empty loss_family"):
        register_objective(objective.name, objective)
    objective.loss_family = ExampleObjective.loss_family
    objective.supports_multiple_epochs = "yes"
    with pytest.raises(ValueError, match="supports_multiple_epochs as a bool"):
        register_objective(objective.name, objective)
    assert ExampleObjective.name not in OBJECTIVES


def test_slime_owns_its_built_in_objectives():
    assert resolve_objective("sft").loss_family == "sft"
    assert isinstance(resolve_objective("sft"), SlimeObjective)
    assert resolve_objective("slime_dataset").loss_family == "slime"
    with pytest.raises(ValueError, match="objective 'task_group'"):
        resolve_objective("task_group")


def test_a_method_registers_its_loss_family_with_its_objective():
    # The cookbook's public half registers the objective only. Its module defines the family, and names the
    # reference that a worker process imports.
    objective = resolve_objective("tttd")
    assert resolve_loss_family(objective.loss_family).loss_family == "tttd"
    assert loss_family_reference("tttd") == "recipes.tttd.slime:TttdAlgorithm"


def test_a_job_checks_its_schedule_against_its_objective():
    assert SlimeJobParameters.from_parameters({"objective": "sao", "scheduling": {"epochs": 2}}).scheduling == (
        StepScheduling(epochs=2)
    )

    # An objective only says what its loss tolerates: SAO's clipped ratio
    # accepts a second pass, TTTD's unclipped importance sampling does not.
    assert SaoObjective.supports_multiple_epochs is True
    assert TttdObjective.supports_multiple_epochs is False
    with pytest.raises(ValueError, match="'tttd' does not support 2 epochs per job"):
        SlimeJobParameters.from_parameters({"objective": "tttd", "scheduling": {"epochs": 2}})
    TttdObjective().validate_scheduling(StepScheduling(shuffle=True))


@pytest.mark.parametrize("parameters", [{}, {"objective": ""}, {"objective": 3}, {"scheduling": {}}])
def test_a_job_without_an_objective_is_refused(parameters):
    with pytest.raises(ValueError, match="needs the objective parameter"):
        SlimeJobParameters.from_parameters(parameters)


def test_objectives_keep_each_sample_its_own_rollout_by_default():
    assert SlimeObjective.rollout_unit == "sample"
    assert SaoObjective.rollout_unit == TttdObjective.rollout_unit == "sample"


def test_a_skipped_job_keeps_the_proposed_state_without_resolving_a_loss():
    objective = ExampleObjective()
    objective.loss_family = "no-backend-needed-for-skip"
    register_objective(objective.name, objective)
    try:
        state = {"steps": 7}
        prepared = prepare_slime_job(JobData("empty", ()), objective.name, state, StepScheduling(), step_size=1)
        assert prepared.action == "skip" and prepared.payload is None
        assert prepared.next_algorithm_state == {"steps": 7, "skipped": True}
        assert state == {"steps": 7}
        assert prepared.metrics == {"reason": "no signal"}
    finally:
        unregister_objective(objective.name)


def test_group_advantages_are_computed_before_optimizer_partitioning():
    # One sample per optimizer step: every group is split across steps, yet
    # its advantages come from the whole group because prepare ran first.
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
        advantage for group in (rewards[:2], rewards[2:]) for advantage in adaptive_entropic_advantages(list(group))[0]
    )
    prepared = prepare_slime_job(batch, "tttd", state, StepScheduling(), step_size=1)
    assert prepared.payload["loss"] == "tttd"
    assert prepared.payload["advantages"] == pytest.approx(expected)
    assert prepared.payload["external_step_sizes"] == [1, 1, 1, 1]
    assert prepared.next_algorithm_state == {"steps": 5}
    assert state == {"steps": 4}
    assert len(prepared.metrics["adaptive_betas"]) == 2


def test_method_objectives_resolve_without_importing_training_backends():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from recipes.tttd import TTTDRecipe\n"
            "from recipes.sao import SAORecipe\n"
            "from recipes.openclawrl import OpenClawRLRecipe\n"
            "assert not {'recipes.tttd.slime', 'recipes.tttd.tinker', 'recipes.sao.slime'} & sys.modules.keys()\n"
            "from reef.train.slime_backend.loss_families import loss_family_reference, resolve_loss_family\n"
            "from reef.train.slime_backend.objectives import resolve_objective\n"
            "from reef.train.slime_backend.schedule import StepScheduling\n"
            "for recipe in (TTTDRecipe, SAORecipe, OpenClawRLRecipe):\n"
            "    objective = resolve_objective(recipe.default_parameters()['trajectory']['objective'])\n"
            "    objective.validate_scheduling(StepScheduling())\n"
            "    family = resolve_loss_family(objective.loss_family)\n"
            "    assert loss_family_reference(family.loss_family).endswith(':' + type(family).__name__)\n"
            "assert not {'torch', 'ray', 'slime', 'tinker'} & sys.modules.keys()\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
