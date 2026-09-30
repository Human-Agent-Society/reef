"""CPU contracts for per-job training methods on the Slime bridge: loss-family switches and LR schedules."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import ray
from reef_service._trajectories import policy_trajectory
from reef_service.slime_coordinator import build_slime_coordinator

from reef.core.training_method import LearningRateSchedule, LearningRateScheduleState, TrainingMethod
from reef.core.trajectories import trajectory_reward
from reef.runtime.recovery import marker_path, read_marker
from reef.runtime.scheduler import TrainingCoordinator
from reef.train.algos import StepScheduling, StepSignal, TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult
from reef.train.slime_backend.loss_families import register_loss_family, unregister_loss_family
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import TRAINING_METHOD_KEY
from reef.train.slime_backend.reef_adapters.training_job.storage import LEARNING_RATE_SCHEDULES_FILENAME
from reef.train.types import TrainingBatch, trajectories

from .test_slime_bridge import _DurableGroup, _FakeRolloutManager

#: A policy objective over the suite's ``pg`` family, named by dotted reference like a cookbook method's.
REWARD_OBJECTIVE = "reef_service.test_slime_training_methods:RewardObjective"
WARMUP = LearningRateSchedule("warmup", 1e-4, warmup_steps=4)
# One optimizer step per sample: two samples make a two-step job.
ONE_SAMPLE_STEPS = StepScheduling(unit="sample", batch_size=1)


class RewardObjective(TrainingObjective):
    """Every sample's reward as its advantage, over Slime's policy loss."""

    name = "test-reward-pg"
    loss_family = "pg"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        advantages = tuple(trajectory_reward(item) for item in trajectories(batch))
        return StepSignal("train", {"steps": next_steps(state)}, {}, advantages)


class OptionsFamily(SlimeAlgorithm):
    """A family whose flags only a startup family could have parsed."""

    loss_family = "test-needs-options"
    loss_type = "custom_loss"
    requires_driver_options = True

    def validate_specific_args(self, args, source):
        pass


class CriticWarmupFamily(SlimeAlgorithm):
    """The suite's supervised family, trained as a critic-only warmup step: the actor never steps."""

    loss_family = "sft"
    loss_type = "sft_loss"
    advantages = "forbidden"

    def validate_specific_args(self, args, source):
        pass

    def train(self, rollout_id, rollout_data_refs, *, actor_group, critic_group, resolve):
        return TrainResult([], actor_trained=False)


@pytest.fixture(autouse=True)
def _local_ray_get(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda value, **kwargs: value)


def worker_args(**overrides: object) -> SlimeArguments:
    """The arguments supervised (``sft``) workers started with, as far as the bridge reads them."""
    values: dict[str, object] = {
        "loss_family": "sft",
        "loss_type": "sft_loss",
        "advantage_estimator": "grpo",
        "compute_advantages_and_returns": False,
        "use_rollout_logprobs": True,
        "custom_rollout_data_keys": ("producing_runtime_load_spans", "producing_runtime_load_ids"),
        "reef_rollout_tensor_dtypes": {},
        "reef_external_batch_keys": (),
        "reef_rollout_log_skip_keys": (),
        "score_centering": False,
        "global_batch_size": 1,
        "num_steps_per_rollout": 1,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
    }
    values.update(overrides)
    return SlimeArguments(**values)


class Stack:
    """A coordinator over a Slime bridge whose workers started with the supervised family."""

    def __init__(self, tmp_path: Path, *, start_rollout_id: int = 0, **bridge: Any) -> None:
        self.template = str(tmp_path / "checkpoint-{rollout_id}")
        self.group = _DurableGroup(self.template)
        self.manager = _FakeRolloutManager(["packed"])
        self.coordinator: TrainingCoordinator = build_slime_coordinator(
            self.group,
            self.manager,
            batch_processor=self.manager,
            save_hf_template=self.template,
            start_rollout_id=start_rollout_id,
            **{"loss_family": "sft", "args": worker_args(), **bridge},
        )

    @property
    def bridge(self):
        return self.coordinator._training

    def payload(self, method: TrainingMethod, *, step: int) -> dict[str, Any]:
        batch = TrainingBatch(
            f"batch-{step}",
            tuple(
                policy_trajectory(f"s{step}-{index}", (1, 2, 3), (1, 1), (-0.2, -0.1), 1.0, "v1") for index in range(2)
            ),
        )
        prepared = self.coordinator.prepare_training_step(batch, method, {}, ONE_SAMPLE_STEPS)
        assert prepared.payload is not None
        return {**prepared.payload, "scenario_step": step, "expected_runtime_load_id": "v1"}

    def run(self, method: TrainingMethod, *, step: int):
        result = self.coordinator.execute_training_job(self.payload(method, step=step))
        assert result.outcome == "checkpoint"
        published = self.coordinator.update_serving_weights(result.training_job_id)
        self.coordinator.acknowledge_training_commit(result.training_job_id)
        return published

    def activation(self, job: int) -> Mapping[str, Any] | None:
        """What the job's data told the actor workers to switch to, if anything."""
        return self.manager.calls[job].get(TRAINING_METHOD_KEY)


@pytest.mark.unit
def test_a_single_method_job_leaves_the_workers_as_they_started(tmp_path) -> None:
    stack = Stack(tmp_path)
    stack.run(TrainingMethod("sft"), step=0)
    stack.run(TrainingMethod("sft"), step=1)
    assert [stack.activation(job) for job in (0, 1)] == [None, None]
    assert len(stack.group.train_calls) == 2
    assert not (tmp_path / LEARNING_RATE_SCHEDULES_FILENAME).exists()


@pytest.mark.unit
def test_a_job_of_another_family_switches_the_actor_workers_and_back(tmp_path) -> None:
    stack = Stack(tmp_path)
    stack.run(TrainingMethod("sft"), step=0)
    stack.run(TrainingMethod(REWARD_OBJECTIVE), step=1)
    stack.run(TrainingMethod(REWARD_OBJECTIVE), step=2)
    stack.run(TrainingMethod("sft"), step=3)

    policy = stack.activation(1)
    assert policy is not None
    assert policy["learning_rate_schedule"] is None
    assert policy["loss_family_args"]["loss_family"] == "pg"
    assert policy["loss_family_args"]["loss_type"] == "policy_loss"
    # The workers already run the policy family for the next job.
    assert stack.activation(2) is None
    supervised = stack.activation(3)
    assert supervised is not None
    assert supervised["loss_family_args"]["loss_family"] == "sft"
    assert supervised["loss_family_args"]["loss_type"] == "sft_loss"
    assert set(supervised["loss_family_args"]) == set(policy["loss_family_args"])
    # Every job trained with its own family: the policy jobs shipped advantages, the supervised ones none.
    assert ["advantages" in data for data in stack.manager.calls] == [False, True, True, False]
    assert len(stack.group.train_calls) == 4


@pytest.mark.unit
def test_a_method_the_workers_cannot_train_is_refused_before_a_job_exists(tmp_path) -> None:
    register_loss_family(OptionsFamily())
    try:

        stack = Stack(tmp_path)
        with pytest.raises(RuntimeError, match="requires driver options"):
            stack.bridge.loss_algorithm("test-needs-options")
        # A family that configures the critic started with the workers or not at all.
        with pytest.raises(RuntimeError, match="configures the critic"):
            stack.bridge.loss_algorithm("sao")
        # Without the workers' arguments a bridge trains its startup family only.
        without_arguments = Stack(tmp_path / "bare", args=None)
        with pytest.raises(RuntimeError, match="trains loss family 'sft' only"):
            without_arguments.payload(TrainingMethod(REWARD_OBJECTIVE), step=0)
        assert without_arguments.group.train_calls == []
        assert read_marker(marker_path(without_arguments.template)) is None
    finally:
        unregister_loss_family("test-needs-options")


@pytest.mark.unit
def test_a_run_distils_with_one_distillation_family(tmp_path) -> None:
    stack = Stack(tmp_path)
    stack.bridge.loss_algorithm("sdft")
    with pytest.raises(RuntimeError, match="second distillation family"):
        stack.bridge.loss_algorithm("sdpo")


@pytest.mark.unit
def test_a_schedule_advances_by_optimizer_steps_and_continues_after_a_restart(tmp_path) -> None:
    stack = Stack(tmp_path)
    first = stack.run(TrainingMethod("sft", WARMUP), step=0)
    activation = stack.activation(0)
    assert activation is not None
    assert activation["learning_rate_schedule"] == LearningRateScheduleState(WARMUP, 0).to_dict()
    # Two samples, one optimizer step each: the job's last step ran at the rate of step 1.
    assert first.metrics["learning_rate"] == pytest.approx(2.5e-5)
    assert first.metrics["learning_rate_schedule"] == {"name": "warmup", "completed_steps": 2}
    stack.run(TrainingMethod("sft", WARMUP), step=1)
    second = stack.activation(1)
    assert second is not None
    assert second["learning_rate_schedule"]["completed_steps"] == 2
    # The progress is written with the checkpoint, beside the job marker.
    record = json.loads((tmp_path / LEARNING_RATE_SCHEDULES_FILENAME).read_text())
    assert record == {"schedules": {"": {"rollout_id": 1, **LearningRateScheduleState(WARMUP, 4).to_dict()}}}

    # Workers that loaded the second checkpoint continue the warmup where it stopped.
    restarted = Stack(tmp_path, start_rollout_id=2)
    restarted.run(TrainingMethod("sft", WARMUP), step=2)
    continued = restarted.activation(0)
    assert continued is not None
    assert continued["learning_rate_schedule"]["completed_steps"] == 4
    # A job that selects no schedule keeps the active one.
    restarted.run(TrainingMethod("sft"), step=3)
    kept = restarted.activation(1)
    assert kept is not None
    assert kept["learning_rate_schedule"]["completed_steps"] == 6


@pytest.mark.unit
def test_a_job_dropped_before_training_leaves_the_schedule_where_it_was(tmp_path) -> None:
    stack = Stack(tmp_path)
    stale = {**stack.payload(TrainingMethod("sft", WARMUP), step=0), "expected_runtime_load_id": "old"}
    assert stack.coordinator.execute_training_job(stale).outcome == "stale"
    assert stack.group.train_calls == []
    # The retried batch starts the warmup at step 0, as the dropped attempt would have.
    stack.run(TrainingMethod("sft", WARMUP), step=0)
    retried = stack.activation(0)
    assert retried is not None
    assert retried["learning_rate_schedule"]["completed_steps"] == 0


@pytest.mark.unit
def test_progress_newer_than_the_loaded_checkpoint_refuses_to_start(tmp_path) -> None:
    stack = Stack(tmp_path)
    stack.run(TrainingMethod("sft", WARMUP), step=0)
    stack.run(TrainingMethod("sft", WARMUP), step=1)
    # Workers that loaded the first checkpoint would repeat a step the record already counts.
    with pytest.raises(RuntimeError, match="after the checkpoint the workers loaded"):
        Stack(tmp_path, start_rollout_id=1)


@pytest.mark.unit
def test_a_critic_only_step_leaves_the_actor_schedule_where_it_was(tmp_path) -> None:
    stack = Stack(tmp_path, loss_family=None, loss_runtime=CriticWarmupFamily())
    result = stack.run(TrainingMethod("sft", WARMUP), step=0)
    assert "learning_rate" not in (result.metrics or {})
    assert not (tmp_path / LEARNING_RATE_SCHEDULES_FILENAME).exists()
    # The actor never fetched the job's data, so the next job carries the activation again.
    stack.run(TrainingMethod("sft", WARMUP), step=1)
    assert stack.activation(1) == stack.activation(0)


@pytest.mark.unit
def test_a_job_family_brings_its_loss_type_and_advantage_routing_and_keeps_the_startup_options() -> None:
    from reef.train.slime_backend.loss_families import resolve_loss_family
    from reef.train.slime_backend.reef_adapters.slime_arguments import loss_family_job_args

    # Workers that started with a pg-primitive family: Slime's estimator was routed to cispo.
    startup = worker_args(advantage_estimator="cispo", reef_configured_advantage_estimator="grpo", kl_coef=0.1)
    policy = loss_family_job_args(startup, resolve_loss_family("pg"))
    assert (policy.loss_family, policy.loss_type, policy.advantage_estimator) == ("pg", "policy_loss", "grpo")
    assert policy.compute_advantages_and_returns is False
    # TTTD keeps Slime's pre-train advantage pass for its frozen-base KL, and validates the startup --kl-coef.
    tttd = loss_family_job_args(startup, resolve_loss_family("tttd"))
    assert (tttd.loss_type, tttd.compute_advantages_and_returns) == ("custom_loss", True)
    assert tttd.loss_family_ref == "recipes.tttd.slime:TttdAlgorithm"
    with pytest.raises(RuntimeError, match="positive finite --kl-coef"):
        loss_family_job_args(worker_args(kl_coef=0.0), resolve_loss_family("tttd"))
    # The startup arguments are the workers' own and stay as they are.
    assert (startup.loss_family, startup.loss_type, startup.advantage_estimator) == ("sft", "sft_loss", "cispo")
