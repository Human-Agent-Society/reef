"""Contract for method-owned preparation and backend loss selection."""

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import pytest
from reef_service._trajectories import policy_trajectory

from recipes.sao.objective import SaoObjective
from recipes.tttd.objective import TttdObjective
from reef.recipe import WeightTrainingSpec
from reef.train.algos import StepScheduling, StepSignal, TrainingObjective
from reef.train.algos.registry import ObjectiveRegistry, register_objective, resolve_objective, unregister_objective
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.tinker_backend.preparation import prepare_tinker_step
from reef.train.types import TrainingBatch


class ExampleObjective(TrainingObjective):
    name = "test-example-objective"
    loss_family = "sft"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, object]) -> StepSignal:
        return StepSignal("skip", {**state, "skipped": True}, metrics={"reason": "no signal"})


OBJECTIVE_INSTANCE = ExampleObjective()


def plain_preparation(batch, state):
    return StepSignal("skip", state)


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


@pytest.mark.parametrize("reference", ["plain_preparation", "MissingObjective", "StepSignal"])
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
        assert spec.scheduling == StepScheduling()
        assert isinstance(resolve_objective(CustomObjective.name), CustomObjective)
        with pytest.raises(TypeError, match="loss_family"):
            WeightTrainingSpec(objective=CustomObjective.name, loss_family="sft")
    finally:
        unregister_objective(CustomObjective.name)


def test_recipe_owns_the_step_schedule_and_the_objective_rejects_unsupported_epochs():
    # The schedule is recipe configuration, not part of the objective's signal.
    assert "scheduling" not in StepSignal.__dataclass_fields__
    two_passes = StepScheduling(unit="sample", batch_size="actual", epochs=2)
    assert WeightTrainingSpec(objective="sao", scheduling=two_passes).scheduling is two_passes

    # An objective only says what its loss tolerates: SAO's clipped ratio
    # accepts a second pass, TTTD's unclipped importance sampling does not.
    assert SaoObjective.supports_multiple_epochs is True
    SaoObjective().validate_scheduling(two_passes)
    assert TttdObjective.supports_multiple_epochs is False
    with pytest.raises(ValueError, match=r"'tttd' does not support StepScheduling\(epochs=2\)"):
        TttdObjective().validate_scheduling(two_passes)
    TttdObjective().validate_scheduling(StepScheduling(unit="sample", batch_size=2, shuffle=True))

    # Both backends check the schedule before touching the batch.
    batch = TrainingBatch("two-groups", ())
    with pytest.raises(ValueError, match="does not support StepScheduling"):
        prepare_slime_step(batch, "tttd", {"steps": 0}, two_passes)
    with pytest.raises(ValueError, match="does not support StepScheduling"):
        prepare_tinker_step(batch, "tttd", {"steps": 0}, two_passes, batch_size=1)


def test_both_backends_preserve_skip_and_proposed_state_without_resolving_a_loss():
    objective = ExampleObjective()
    objective.loss_family = "no-backend-needed-for-skip"
    register_objective(objective)
    try:
        state = {"steps": 7}
        batch = TrainingBatch("empty", ())
        slime = prepare_slime_step(batch, objective.name, state, StepScheduling())
        tinker = prepare_tinker_step(batch, objective.name, state, StepScheduling(), batch_size=1)
        assert slime == tinker
        assert slime.action == "skip" and slime.payload is None
        assert slime.next_algorithm_state == {"steps": 7, "skipped": True}
        assert state == {"steps": 7}
        assert slime.metrics == {"reason": "no signal"}
    finally:
        unregister_objective(objective.name)


def test_group_advantages_are_shared_across_backends_before_optimizer_partitioning():
    # One sample per optimizer step: every group is split across steps, yet
    # its advantages come from the whole group because prepare ran first.
    one_sample_steps = StepScheduling(unit="sample", batch_size=1)
    rewards = (0.0, 2.0, 3.0, 1.0)
    batch = TrainingBatch(
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
    slime = prepare_slime_step(batch, "tttd", state, one_sample_steps)
    tinker = prepare_tinker_step(batch, "tttd", state, one_sample_steps, batch_size=1)
    assert slime.payload["loss"] == tinker.payload["loss"] == "tttd"
    assert slime.payload["advantages"] == pytest.approx(expected)
    assert [rows[0]["advantage"] for rows in tinker.payload["batches"]] == pytest.approx(expected)
    assert slime.payload["external_step_sizes"] == [1, 1, 1, 1]
    assert len(tinker.payload["batches"]) == 4
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
            "for recipe in (TTTDRecipe, SAORecipe, OpenClawRLRecipe):\n"
            "    spec = recipe.training_spec()\n"
            "    objective = resolve_objective(spec.objective)\n"
            "    assert objective.loss_family == spec.loss_family\n"
            "    objective.validate_scheduling(spec.scheduling)\n"
            "assert not {'torch', 'ray', 'slime', 'tinker'} & sys.modules.keys()\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
