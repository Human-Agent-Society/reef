"""Contract for method-owned preparation and backend loss selection."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import pytest
from tests.support._trajectories import policy_trajectory

from recipes.sao.slime import SaoObjective
from recipes.tttd.objective import adaptive_entropic_advantages
from recipes.tttd.slime import TttdObjective
from reef.train.algos.schedule import StepScheduling
from reef.train.algos.signals import TrainingSignal
from reef.train.slime_backend.objectives import (
    SlimeObjective,
    add_slime_objective,
    register_slime_objective,
    resolve_slime_objective,
    unregister_slime_objective,
)
from reef.train.slime_backend.parameters import SlimeJobParameters
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_job
from reef.train.tinker_backend.losses import resolve_tinker_loss
from reef.train.tinker_backend.objectives import (
    TinkerObjective,
    register_tinker_objective,
    resolve_tinker_objective,
    unregister_tinker_objective,
)
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.types import JobData


class ExampleObjective(SlimeObjective):
    name = "test-example-objective"
    loss_family = "sft"

    def prepare(self, batch: JobData, state: Mapping[str, object]) -> TrainingSignal:
        return TrainingSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


OBJECTIVE_INSTANCE = ExampleObjective()


def plain_preparation(batch, state):
    return TrainingSignal("skip", state)


class TinkerExampleObjective(TinkerObjective):
    name = "test-example-objective"
    loss_family = "no-backend-needed-for-skip"

    def prepare(self, batch: JobData, state: Mapping[str, object]) -> TrainingSignal:
        return TrainingSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


@pytest.fixture
def no_slime_objectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.train.slime_backend.objectives.SLIME_OBJECTIVES", {})


def test_objective_resolution_accepts_names_classes_and_instances(no_slime_objectives):
    objective = resolve_slime_objective(f"{__name__}:ExampleObjective")
    assert resolve_slime_objective(objective.name) is objective
    assert resolve_slime_objective(f"{__name__}:ExampleObjective") is objective
    assert add_slime_objective(objective) is objective
    with pytest.raises(ValueError, match="already registered"):
        add_slime_objective(ExampleObjective())
    unregister_slime_objective(objective.name)
    assert resolve_slime_objective(f"{__name__}:OBJECTIVE_INSTANCE") is OBJECTIVE_INSTANCE


@pytest.mark.parametrize("reference", ["plain_preparation", "MissingObjective", "TrainingSignal"])
def test_dotted_objective_rejects_functions_missing_attributes_and_unrelated_classes(reference):
    with pytest.raises(ValueError, match="SlimeObjective class or instance"):
        resolve_slime_objective(f"{__name__}:{reference}")
    with pytest.raises(ValueError, match="TinkerObjective class or instance"):
        resolve_tinker_objective(f"{__name__}:{reference}")


def test_objective_resolution_reports_unknown_and_malformed_references():
    with pytest.raises(ValueError, match=r"unknown Slime objective.*available objectives"):
        resolve_slime_objective("missing")
    with pytest.raises(ValueError, match=r"unknown Tinker objective.*available objectives"):
        resolve_tinker_objective("missing")
    with pytest.raises(ValueError, match=r"must be 'package\.module:Objective'"):
        resolve_slime_objective(":ExampleObjective")
    with pytest.raises(ValueError, match="cannot import objective"):
        resolve_slime_objective("no_such_package.module:Objective")
    with pytest.raises(KeyError, match="not registered"):
        unregister_slime_objective("missing")


def test_objective_registration_validates_the_method_contract():
    with pytest.raises(TypeError, match="inherit SlimeObjective"):
        add_slime_objective(plain_preparation)
    objective = ExampleObjective()
    objective.name = ""
    with pytest.raises(ValueError, match="non-empty name"):
        add_slime_objective(objective)
    objective.name = ExampleObjective.name
    objective.loss_family = ""
    with pytest.raises(ValueError, match="non-empty loss_family"):
        add_slime_objective(objective)
    objective.loss_family = ExampleObjective.loss_family
    objective.supports_multiple_epochs = "yes"
    with pytest.raises(ValueError, match="supports_multiple_epochs as a bool"):
        add_slime_objective(objective)


def test_each_backend_owns_its_built_in_objectives():
    assert resolve_slime_objective("sft").loss_family == resolve_tinker_objective("sft").loss_family == "sft"
    assert isinstance(resolve_slime_objective("sft"), SlimeObjective)
    assert isinstance(resolve_tinker_objective("sft"), TinkerObjective)
    assert resolve_slime_objective("slime_dataset").loss_family == "slime"
    for resolve in (resolve_slime_objective, resolve_tinker_objective):
        with pytest.raises(ValueError, match="objective 'task_group'"):
            resolve("task_group")
    with pytest.raises(ValueError, match="unknown Tinker objective 'slime_dataset'"):
        resolve_tinker_objective("slime_dataset")


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


def test_both_backends_preserve_skip_and_proposed_state_without_resolving_a_loss():
    objective = ExampleObjective()
    objective.loss_family = "no-backend-needed-for-skip"
    register_slime_objective(objective)
    register_tinker_objective(TinkerExampleObjective())
    try:
        state = {"steps": 7}
        batch = JobData("empty", ())
        slime = prepare_slime_job(batch, objective.name, state, StepScheduling(), step_size=1)
        tinker = prepare_tinker_job(
            batch,
            objective.name,
            state,
            StepScheduling(),
            batch_size=1,
            kl_coef=0.0,
            score_centering=None,
        )
        assert slime.action == "skip" and slime.payload is None
        assert tinker.loss is None and tinker.steps == ()
        assert slime.next_algorithm_state == tinker.next_algorithm_state == {"steps": 7, "skipped": True}
        assert state == {"steps": 7}
        assert slime.metrics == tinker.metrics == {"reason": "no signal"}
    finally:
        unregister_slime_objective(objective.name)
        unregister_tinker_objective(objective.name)


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
        advantage for group in (rewards[:2], rewards[2:]) for advantage in adaptive_entropic_advantages(list(group))[0]
    )
    slime = prepare_slime_job(batch, "tttd", state, one_sample_steps, step_size=1)
    tinker = prepare_tinker_job(
        batch, "tttd", state, one_sample_steps, batch_size=1, kl_coef=0.0, score_centering=None
    )
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
            "from reef.train.algos.schedule import StepScheduling\n"
            "from reef.train.slime_backend.objectives import resolve_slime_objective\n"
            "for recipe in (TTTDRecipe, SAORecipe, OpenClawRLRecipe):\n"
            "    objective = resolve_slime_objective(recipe.default_parameters()['trajectory']['objective'])\n"
            "    objective.validate_scheduling(StepScheduling())\n"
            "assert not {'torch', 'ray', 'slime', 'tinker'} & sys.modules.keys()\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
