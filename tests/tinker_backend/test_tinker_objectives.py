"""Contract for Tinker's objectives and loss families: their registration and their resolution."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import pytest
from tests.support.trajectories import policy_trajectory

from recipes.tttd.objective import adaptive_entropic_advantages
from recipes.tttd.tinker import TttdTinkerLoss, TttdTinkerObjective
from reef.core.plugins import PluginConfigError
from reef.train.tinker_backend.loss_families import (
    LOSS_FAMILIES,
    register_loss_family,
    resolve_loss_family,
    unregister_loss_family,
)
from reef.train.tinker_backend.losses import CrossEntropyLoss, ImportanceSamplingLoss, TinkerLoss
from reef.train.tinker_backend.objectives import (
    OBJECTIVES,
    TinkerObjective,
    register_objective,
    resolve_objective,
    unregister_objective,
)
from reef.train.tinker_backend.preparation import prepare_tinker_job
from reef.train.tinker_backend.schedule import StepScheduling
from reef.train.tinker_backend.signals import TrainingSignal
from reef.train.types import JobData


class ExampleObjective(TinkerObjective):
    name = "test-example-objective"
    loss_family = "no-backend-needed-for-skip"

    def prepare(self, batch: JobData, state: Mapping[str, object]) -> TrainingSignal:
        return TrainingSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


class MisnamedObjective(ExampleObjective):
    name = "test-another-name"


def plain_preparation(batch, state):
    return TrainingSignal("skip", state)


@pytest.fixture
def no_objectives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("reef.train.tinker_backend.objectives.OBJECTIVES", {})


def test_objective_resolution_accepts_names_and_class_references(no_objectives):
    objective = resolve_objective(f"{__name__}:ExampleObjective")
    assert resolve_objective(objective.name) is objective
    assert resolve_objective(f"{__name__}:ExampleObjective") is objective
    register_objective(objective.name, objective)
    with pytest.raises(ValueError, match="already registered"):
        register_objective(objective.name, ExampleObjective())
    unregister_objective(objective.name)
    assert resolve_objective(f"{__name__}:ExampleObjective") is not objective


def test_a_registered_reference_is_imported_at_its_first_resolution(no_objectives):
    from reef.train.tinker_backend import objectives

    reference = f"{__name__}:ExampleObjective"
    register_objective(ExampleObjective.name, reference)
    register_objective(ExampleObjective.name, reference)
    assert {ExampleObjective.name: reference} == objectives.OBJECTIVES
    with pytest.raises(ValueError, match="already registered"):
        register_objective(ExampleObjective.name, f"{__name__}:MisnamedObjective")

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


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("plain_preparation", "must name a class that inherits TinkerObjective"),
        ("TrainingSignal", "must name a class that inherits TinkerObjective"),
        ("MissingObjective", "cannot import Tinker objective"),
        ("TinkerObjective", "names an abstract class"),
    ],
)
def test_dotted_objective_rejects_functions_missing_attributes_and_other_classes(reference, message):
    with pytest.raises(PluginConfigError, match=message):
        resolve_objective(f"{__name__}:{reference}")


def test_objective_resolution_reports_unknown_and_malformed_references():
    with pytest.raises(ValueError, match=r"unknown Tinker objective.*available objectives"):
        resolve_objective("missing")
    with pytest.raises(PluginConfigError, match=r"must be 'package\.module:ClassName'"):
        resolve_objective(":ExampleObjective")
    with pytest.raises(ValueError, match=r"must be 'package\.module:Objective'"):
        register_objective("malformed", "no_colon")
    with pytest.raises(PluginConfigError, match="cannot import Tinker objective"):
        resolve_objective("no_such_package.module:Objective")
    with pytest.raises(KeyError, match="not registered"):
        unregister_objective("missing")


def test_objective_registration_validates_the_method_contract():
    with pytest.raises(TypeError, match="inherit TinkerObjective"):
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


def test_tinker_owns_its_built_in_objectives_and_loss_families():
    assert resolve_objective("sft").loss_family == "sft"
    assert isinstance(resolve_objective("sft"), TinkerObjective)
    assert isinstance(resolve_loss_family("sft"), CrossEntropyLoss)
    assert isinstance(resolve_loss_family("importance_sampling"), ImportanceSamplingLoss)
    with pytest.raises(ValueError, match="objective 'task_group'"):
        resolve_objective("task_group")
    with pytest.raises(ValueError, match="unknown Tinker objective 'slime_dataset'"):
        resolve_objective("slime_dataset")
    with pytest.raises(ValueError, match=r"unsupported Tinker loss family 'slime'.*register_loss_family\(\)"):
        resolve_loss_family("slime")


def test_a_loss_family_registers_under_its_name():
    class ToyLoss(TinkerLoss):
        loss_family = "test-toy-loss"

        def inputs(self, rows, base_logprobs, *, kl_coef):
            return []

    assert register_loss_family(ToyLoss) is ToyLoss
    try:
        toy = resolve_loss_family("test-toy-loss")
        assert isinstance(toy, ToyLoss)
        assert register_loss_family(toy) is toy
        with pytest.raises(ValueError, match="'test-toy-loss' is already registered"):
            register_loss_family(ToyLoss())
    finally:
        unregister_loss_family("test-toy-loss")
    assert "test-toy-loss" not in LOSS_FAMILIES
    with pytest.raises(KeyError, match="not registered"):
        unregister_loss_family("test-toy-loss")

    class NamelessLoss(ToyLoss):
        loss_family = ""

    with pytest.raises(ValueError, match="non-empty loss_family"):
        register_loss_family(NamelessLoss)
    with pytest.raises(TypeError, match="registers a TinkerLoss"):
        register_loss_family(plain_preparation)
    with pytest.raises(TypeError, match="registers a TinkerLoss"):
        register_loss_family(ExampleObjective)


def test_a_method_registers_its_loss_family_with_its_objective():
    # The cookbook's public half registers the objective only. Its module defines the family.
    objective = resolve_objective("tttd")
    assert isinstance(objective, TttdTinkerObjective)
    assert isinstance(resolve_loss_family(objective.loss_family), TttdTinkerLoss)


def test_a_skipped_job_keeps_the_proposed_state_without_resolving_a_loss():
    objective = ExampleObjective()
    register_objective(objective.name, objective)
    try:
        state = {"steps": 7}
        steps = prepare_tinker_job(
            JobData("empty", ()),
            objective.name,
            state,
            StepScheduling(),
            batch_size=1,
            kl_coef=0.0,
            score_centering=None,
        )
        assert steps.loss is None and steps.steps == ()
        assert steps.next_algorithm_state == {"steps": 7, "skipped": True}
        assert state == {"steps": 7}
        assert steps.metrics == {"reason": "no signal"}
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
    steps = prepare_tinker_job(batch, "tttd", state, StepScheduling(), batch_size=1, kl_coef=0.0, score_centering=None)
    assert steps.loss is resolve_loss_family("tttd")
    assert [rows[0].advantage for rows in steps.steps] == pytest.approx(expected)
    assert len(steps.steps) == 4
    assert steps.next_algorithm_state == {"steps": 5}
    assert state == {"steps": 4}
    assert len(steps.metrics["adaptive_betas"]) == 2


def test_a_method_objective_resolves_without_importing_the_tinker_sdk_or_slime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "from recipes.tttd import TTTDRecipe\n"
            "assert not {'recipes.tttd.slime', 'recipes.tttd.tinker'} & sys.modules.keys()\n"
            "from reef.train.tinker_backend.loss_families import resolve_loss_family\n"
            "from reef.train.tinker_backend.objectives import resolve_objective\n"
            "from reef.train.tinker_backend.schedule import StepScheduling\n"
            "objective = resolve_objective(TTTDRecipe.default_parameters()['trajectory']['objective'])\n"
            "objective.validate_scheduling(StepScheduling())\n"
            "assert type(resolve_loss_family(objective.loss_family)).__name__ == 'TttdTinkerLoss'\n"
            "assert 'recipes.tttd.slime' not in sys.modules\n"
            "assert not {'torch', 'ray', 'slime', 'tinker'} & sys.modules.keys()\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
