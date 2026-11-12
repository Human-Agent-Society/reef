"""AutoEnvScaling's Tinker objective: mean-centered advantages within each task's rollouts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from tests.support.trajectories import policy_trajectory

from recipes.beta.autoenvscaling.tinker import AutoEnvScalingTinkerObjective
from reef.core.job_data import JobData
from reef.core.trajectories import TrajectorySample
from reef.train.tinker_backend.objectives import resolve_objective

EXAMPLES = Path(__file__).resolve().parents[3] / "recipes/beta/autoenvscaling/examples"
OBJECTIVE = "recipes.beta.autoenvscaling.tinker:AutoEnvScalingTinkerObjective"


def sample(task: str, index: int, reward: float) -> TrajectorySample:
    """One rollout's sample in the task group ``task``, as a task job's rollout makes it."""
    trajectory = policy_trajectory(f"{task}-{index}", (1, 2, 3), (1, 1), (-0.1, -0.2), reward, artifact_id="r1")
    return dataclasses.replace(trajectory, group_id=task)


@pytest.mark.unit
def test_a_task_job_names_the_objective_by_its_dotted_reference_on_tinker() -> None:
    objective = resolve_objective(OBJECTIVE)
    assert isinstance(objective, AutoEnvScalingTinkerObjective) and objective.loss_family == "importance_sampling"


@pytest.mark.unit
def test_the_objective_centers_rewards_within_each_task_group() -> None:
    first, second = "harbor-00000-000", "harbor-00000-001"
    data = JobData("d", (sample(first, 0, 1.0), sample(first, 1, 0.0), sample(second, 0, 0.5), sample(second, 1, 0.5)))
    signal = resolve_objective(OBJECTIVE).prepare(data, {})
    assert signal.action == "train"
    assert signal.advantages == (0.5, -0.5, 0.0, 0.0)
    assert signal.metrics["constant_groups"] == 1


@pytest.mark.unit
def test_a_group_whose_rewards_differ_a_little_gets_small_advantages() -> None:
    # Partial credit: a reward of 0.9 against 1.0 is not scaled up to a full success against a full failure.
    task = "harbor-00000-000"
    data = JobData("d", (sample(task, 0, 1.0), sample(task, 1, 0.9)))
    assert resolve_objective(OBJECTIVE).prepare(data, {}).advantages == pytest.approx((0.05, -0.05))


@pytest.mark.unit
def test_a_job_whose_groups_are_all_constant_is_skipped_and_keeps_the_step_count() -> None:
    tasks = ("harbor-00000-000", "harbor-00000-001")
    data = JobData("d", tuple(sample(task, index, 1.0) for task in tasks for index in range(2)))
    signal = resolve_objective(OBJECTIVE).prepare(data, {"steps": 3})
    assert signal.action == "skip" and signal.next_algorithm_state == {"steps": 3}


@pytest.mark.unit
@pytest.mark.parametrize("example", ["recorded_sessions/serve-tinker.yaml", "tmax_seeds/serve-tinker.yaml"])
def test_the_tinker_examples_name_this_objective(example: str) -> None:
    reference = yaml.safe_load((EXAMPLES / example).read_text())["recipe"]["config"]["job_parameters"]["objective"]
    assert isinstance(resolve_objective(reference), AutoEnvScalingTinkerObjective)
