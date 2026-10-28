"""CPU contract tests for the Slime training component: one Slime rollout per job, on the component's own workers.

The actor and critic groups are fakes that record what they train and write
the checkpoint directories Slime's writers write, so no GPU, Ray worker or
Megatron import takes part. The component reads the module-level Ray import
of its batch processor, so the suite needs Ray installed.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from tests.support._trajectories import policy_trajectory
from tests.support.slime_component import FakeTrainGroup, RecordingBatchProcessor, checkpoint_storage, finished

from reef.core.artifact_ref import RuntimeLoadSpan
from reef.core.job_data import TaskSample
from reef.core.trajectories import trajectory_reward
from reef.runtime.durable_files import write_json
from reef.runtime.executor import ExecutorFuture, base
from reef.runtime.interfaces import LearningRateSchedule, LearningRateScheduleState, TrainingMethod
from reef.runtime.jobs import JobStatus, ReleaseFormat, TrainingJob
from reef.runtime.serving import ADAPTER_DIR
from reef.train.algos import TrainingObjective, TrainingSignal
from reef.train.algos.helpers import next_steps
from reef.train.slime_backend import component as slime_component
from reef.train.slime_backend.algorithm import SlimeAlgorithm, TrainResult
from reef.train.slime_backend.component import (
    ROLLOUT_FILE,
    START_STATE,
    SlimeRound,
    SlimeSettings,
    SlimeStart,
    SlimeTrainingComponent,
    WorkerTrainingSettings,
    create_slime_component,
)
from reef.train.slime_backend.loss_families import register_loss_family, resolve_loss_family, unregister_loss_family
from reef.train.slime_backend.reef_adapters import startup
from reef.train.slime_backend.reef_adapters.arguments import SlimeArguments
from reef.train.slime_backend.reef_adapters.batches import WORKER_SWITCH_KEY, TrainingBatchProcessor
from reef.train.slime_backend.reef_adapters.preflight import prepare_checkpoint_storage, validate_slime_args
from reef.train.slime_backend.reef_adapters.training_job.storage import (
    CheckpointStorage,
    RetentionConfig,
    _allocated_bytes,
    retention_options,
)
from reef.train.types import JobData, trajectories

#: A policy objective over the suite's ``pg`` family, named by dotted reference like a cookbook method's.
REWARD_OBJECTIVE = "tests.slime_backend.test_slime_component:RewardObjective"
#: The objective of the critic-only family below, by dotted reference.
CRITIC_WARMUP_OBJECTIVE = "tests.slime_backend.test_slime_component:CriticWarmupObjective"
WARMUP = LearningRateSchedule("warmup", 1e-4, warmup_steps=4)


class RewardObjective(TrainingObjective):
    """Every sample's reward as its advantage, over Slime's policy loss."""

    name = "test-reward-pg"
    loss_family = "pg"

    def prepare(self, batch: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        advantages = tuple(trajectory_reward(item) for item in trajectories(batch))
        return TrainingSignal("train", {"steps": next_steps(state)}, {}, advantages)


class SkipObjective(TrainingObjective):
    """Skips every job's data; the data of a job whose ID starts with ``count`` adds one to ``skips``."""

    name = "test-skip"
    loss_family = "sft"

    def prepare(self, batch: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        if batch.data_id.startswith("count"):
            return TrainingSignal("skip", {"skips": int(state.get("skips", 0)) + 1}, {"skipped": 1})
        return TrainingSignal("skip", dict(state), {"skipped": 1})


#: The skipping objective, by dotted reference.
SKIP_OBJECTIVE = "tests.slime_backend.test_slime_component:SkipObjective"


class OptionsFamily(SlimeAlgorithm):
    """A family whose flags only a startup family could have parsed."""

    loss_family = "test-needs-options"
    loss_type = "custom_loss"
    requires_driver_options = True

    def validate_specific_args(self, args, source):
        pass


class CriticWarmupFamily(SlimeAlgorithm):
    """A supervised family trained as a critic-only warmup step: the actor never steps."""

    loss_family = "test-critic-warmup"
    loss_type = "sft_loss"
    advantages = "forbidden"

    def validate_specific_args(self, args, source):
        pass

    def train(self, rollout_id, rollout_data_refs, *, actor_group, critic_group, resolve):
        return TrainResult([], actor_trained=False)


class CriticWarmupObjective(TrainingObjective):
    """The objective that trains ``CriticWarmupFamily``."""

    name = "test-critic-warmup"
    loss_family = "test-critic-warmup"

    def prepare(self, batch: JobData, state: Mapping[str, Any]) -> TrainingSignal:
        return TrainingSignal("train", {"steps": next_steps(state)}, {})


def worker_args(tmp_path: Path, **overrides: object) -> SlimeArguments:
    """The arguments supervised (``sft``) workers started with, as far as the component reads them."""
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
        "save_hf": str(tmp_path / "hf" / "{rollout_id}"),
        "start_rollout_id": 0,
        "critic_save": None,
        "critic_save_interval": 1,
        "critic_steps_per_actor": None,
        "num_critic_only_steps": 0,
        "decoupled_lr": None,
    }
    values.update(overrides)
    return SlimeArguments(**values)


def batch_for(step: int, *, rewards: tuple[float, ...] = (1.0, 1.0)) -> JobData:
    return JobData(
        f"batch-{step}",
        tuple(
            policy_trajectory(f"s{step}-{index}", (1, 2, 3), (1, 1), (-0.2, -0.1), reward, f"inc:{index}")
            for index, reward in enumerate(rewards)
        ),
    )


def parameters(method: TrainingMethod) -> dict[str, Any]:
    """A job's parameters with the default step schedule: one optimizer step per sample, two steps a batch."""
    return {"method": method.to_dict()}


class Stack:
    """A Slime component over fake groups whose workers started with the supervised family.

    ``start`` follows the component's checkpoints the way Reef's job loop
    does: each job starts from the last checkpoint the previous job saved.
    """

    def __init__(
        self,
        tmp_path: Path,
        *,
        lora: bool = False,
        critic: bool = False,
        storage: CheckpointStorage | None = None,
        loss_family: str = "sft",
        rank_metrics: list[dict[str, Any]] | None = None,
        train_timeout_s: float = 600.0,
        **overrides: object,
    ) -> None:
        self.args = worker_args(tmp_path, **overrides)
        self.root = Path(self.args.save_hf.format(rollout_id=0)).parent.parent
        if storage is None:
            storage = checkpoint_storage(self.root, critic_root=self.root / "megatron-critic" if critic else None)
        self.storage = storage
        self.group = FakeTrainGroup(
            self.args.save_hf, lora=lora, megatron_root=self.storage.megatron_root, rank_metrics=rank_metrics
        )
        self.critic = FakeTrainGroup(None, megatron_root=self.storage.critic_root) if critic else None
        self.processor = RecordingBatchProcessor()
        self.component = SlimeTrainingComponent(
            self.args,
            self.group,
            self.critic,
            batch_processor=self.processor,
            storage=self.storage,
            lora=lora,
            loss_family=loss_family,
            train_timeout_s=train_timeout_s,
        )
        start_rollout_id = int(self.args.start_rollout_id or 0)
        self.start: str | None = (
            None if start_rollout_id == 0 else str(self.root / "reef-starts" / f"rollout_{start_rollout_id - 1}")
        )

    def run(self, method: TrainingMethod, *, step: int, batch: JobData | None = None) -> JobStatus:
        job_id = f"job-{step}"
        data = batch_for(step) if batch is None else batch
        self.component.submit(TrainingJob(job_id, "slime", self.start, data, parameters(method)))
        status = finished(self.component, job_id)
        if status.checkpoints:
            self.start = status.checkpoints[-1].path
        return status

    def worker_switch(self, job: int) -> Mapping[str, Any] | None:
        """What the job's data told the actor workers to switch to, if anything."""
        return self.processor.calls[job].get(WORKER_SWITCH_KEY)


@pytest.fixture
def stacks():
    """Build stacks and close their components after the test."""
    built: list[Stack] = []

    def build(tmp_path: Path, **options: Any) -> Stack:
        stack = Stack(tmp_path, **options)
        built.append(stack)
        return stack

    yield build
    for stack in built:
        stack.component.close()


# -- One job, one rollout, one checkpoint


@pytest.mark.unit
def test_a_job_trains_one_rollout_and_saves_it_as_the_start_of_the_next_job(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    assert stack.component.name == "slime"

    first = stack.run(TrainingMethod("sft"), step=0)

    assert (first.state, first.metrics["rollout_id"]) == ("finished", 0)
    (checkpoint,) = first.checkpoints
    start = tmp_path / "reef-starts" / "rollout_0"
    # The checkpoint, after the job's two optimizer steps, is the job's start directory; Reef publishes the
    # rollout's Hugging Face export.
    assert (checkpoint.step, checkpoint.path, checkpoint.release_path) == (2, str(start), str(tmp_path / "hf" / "0"))
    assert checkpoint.metrics["train/loss"] == 0.5
    assert SlimeStart.read(start) == SlimeStart(0, "sft", {"steps": 1}, None)
    assert json.loads((start / START_STATE).read_text())["rollout_id"] == 0
    assert stack.group.train_calls == [(0, ["packed"])]
    assert stack.group.saved == [(0, True)]
    assert stack.processor.calls[0]["rollout_ids"] == [0, 1]
    # Submitting the same job again does nothing.
    stack.component.submit(TrainingJob("job-0", "slime", None, batch_for(0), parameters(TrainingMethod("sft"))))
    assert stack.component.status("job-0") == first

    second = stack.run(TrainingMethod("sft"), step=1)

    assert second.checkpoints[0].path == str(tmp_path / "reef-starts" / "rollout_1")
    # The objective's state continues from the start the job named.
    assert SlimeStart.read(Path(second.checkpoints[0].path)).algorithm_state == {"steps": 2}
    assert [rollout_id for rollout_id, _ in stack.group.train_calls] == [0, 1]


@pytest.mark.unit
def test_a_save_interval_trains_each_run_of_steps_as_its_own_rollout_and_checkpoint(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    job = TrainingJob("job-0", "slime", None, batch_for(0), {**parameters(TrainingMethod("sft")), "save_interval": 1})
    stack.component.submit(job)
    status = finished(stack.component, "job-0")

    starts = tmp_path / "reef-starts"
    assert [(checkpoint.step, checkpoint.path) for checkpoint in status.checkpoints] == [
        (1, str(starts / "rollout_0")),
        (2, str(starts / "rollout_1")),
    ]
    assert status.metrics["rollout_id"] == 1
    # Each rollout trains one step of one sample, and the workers save after each.
    assert [rollout_id for rollout_id, _ in stack.group.train_calls] == [0, 1]
    assert [call["external_step_sizes"] for call in stack.processor.calls] == [[1], [1]]
    assert stack.group.saved == [(0, True), (1, True)]
    # Only the last checkpoint has trained all the data that the objective's next state is for;
    # the earlier one keeps the state of the job's start.
    assert SlimeStart.read(starts / "rollout_0") == SlimeStart(0, None, {}, None)
    assert SlimeStart.read(starts / "rollout_1") == SlimeStart(1, "sft", {"steps": 1}, None)
    assert stack.component.loaded_rollout_id == 1


@pytest.mark.unit
def test_each_rollout_of_a_job_records_the_schedule_progress_up_to_it(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    method = TrainingMethod("sft", WARMUP)
    stack.component.submit(
        TrainingJob("job-0", "slime", None, batch_for(0), {**parameters(method), "save_interval": 1})
    )
    status = finished(stack.component, "job-0")

    progress = [SlimeStart.read(Path(checkpoint.path)).schedule_state for checkpoint in status.checkpoints]
    assert progress == [LearningRateScheduleState(WARMUP, 1), LearningRateScheduleState(WARMUP, 2)]


@pytest.mark.unit
def test_a_restarted_job_continues_after_its_last_saved_rollout(tmp_path, stacks) -> None:
    jobs = tmp_path / "reef-jobs" / "job-0"
    job_parameters = {**parameters(TrainingMethod("sft")), "save_interval": 1}
    write_json(jobs / "job.json", {"start": None, "data": batch_for(0).to_dict(), "parameters": job_parameters})
    # The job saved and recorded its first rollout before the restart; the restarted workers loaded it.
    first = tmp_path / "reef-starts" / "rollout_0"
    SlimeStart(0, None, {}, None).write(first)
    (tmp_path / "hf" / "0").mkdir(parents=True)
    (tmp_path / "megatron" / "iter_0000000").mkdir(parents=True)
    storage = checkpoint_storage(tmp_path)
    storage.complete("job-0", 0, reward=None)
    write_json(jobs / ROLLOUT_FILE, {"rollout_id": 0, "step": 1})
    write_json(
        jobs / "checkpoints.json",
        {"checkpoints": [{"step": 1, "path": str(first), "release_path": str(tmp_path / "hf" / "0"), "metrics": {}}]},
    )

    stack = stacks(tmp_path, start_rollout_id=1, storage=storage)
    status = finished(stack.component, "job-0")

    assert [checkpoint.step for checkpoint in status.checkpoints] == [1, 2]
    assert [rollout_id for rollout_id, _ in stack.group.train_calls] == [1]
    assert status.checkpoints[1].path == str(tmp_path / "reef-starts" / "rollout_1")
    assert SlimeStart.read(Path(status.checkpoints[1].path)) == SlimeStart(1, "sft", {"steps": 1}, None)


@pytest.mark.unit
def test_a_packed_job_trains_global_batch_size_packs_per_step(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    scheduling = {"packing_length": 2 * len(trajectories(batch_for(0))[0].training["tokens"])}
    job = TrainingJob(
        "job-0", "slime", None, batch_for(0), {**parameters(TrainingMethod("sft")), "scheduling": scheduling}
    )
    stack.component.submit(job)
    status = finished(stack.component, "job-0")

    # The two samples fill one pack, so --global-batch-size 1 trains them in one step.
    assert status.state == "finished" and status.checkpoints[0].step == 1
    assert stack.processor.calls[0]["external_step_sizes"] == [2]


@pytest.mark.unit
def test_a_job_starts_only_from_the_weights_the_workers_hold(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    with pytest.raises(ValueError, match="cannot load a published release"):
        stack.component.import_start(tmp_path)
    job = TrainingJob("elsewhere", "slime", str(tmp_path / "other"), batch_for(0), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="other is not a Slime start"):
        stack.component.submit(job)
    other = tmp_path / "rollout_5"
    SlimeStart(5, None, {}, None).write(other)
    job = TrainingJob("elsewhere", "slime", str(other), batch_for(0), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="hold the startup weights; a job cannot start from rollout 5"):
        stack.component.submit(job)
    stack.run(TrainingMethod("sft"), step=0)
    # After a job, the workers hold its rollout: a job from any other start is refused.
    job = TrainingJob("other", "slime", str(other), batch_for(1), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="hold the weights of rollout 0"):
        stack.component.submit(job)
    # A start without a rollout names the startup weights, which the workers no longer hold.
    job = TrainingJob("again", "slime", None, batch_for(1), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="hold the weights of rollout 0; a job cannot start from the startup"):
        stack.component.submit(job)
    with pytest.raises(KeyError):
        stack.component.status("other")
    assert len(stack.group.train_calls) == 1


@pytest.mark.unit
def test_restarted_workers_continue_from_the_checkpoint_they_loaded_and_refuse_the_startup_weights(
    tmp_path, stacks
) -> None:
    # The workers restarted at rollout 3, so they loaded the checkpoint of rollout 2.
    stack = stacks(tmp_path, start_rollout_id=3)
    assert stack.component.loaded_rollout_id == 2
    other = tmp_path / "rollout_5"
    SlimeStart(5, None, {}, None).write(other)
    for start in (str(other), None):
        job = TrainingJob("other", "slime", start, batch_for(0), parameters(TrainingMethod("sft")))
        with pytest.raises(ValueError, match="hold the weights of rollout 2"):
            stack.component.submit(job)

    assert stack.start == str(tmp_path / "reef-starts" / "rollout_2")
    SlimeStart(2, None, {}, None).write(Path(stack.start))
    status = stack.run(TrainingMethod("sft"), step=3)

    assert status.state == "finished" and status.metrics["rollout_id"] == 3
    assert status.checkpoints[0].path == str(tmp_path / "reef-starts" / "rollout_3")
    assert stack.component.loaded_rollout_id == 3


@pytest.mark.unit
def test_a_queued_job_whose_start_the_workers_left_fails_when_it_runs(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    # Both jobs named the startup weights when they were submitted; the first one moved the workers on.
    stack.run(TrainingMethod("sft"), step=0)
    late = TrainingJob("late", "slime", None, batch_for(1), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="hold the weights of rollout 0; a job cannot start from the startup"):
        stack.component.run(late, tmp_path / "late")
    assert len(stack.group.train_calls) == 1


@pytest.mark.unit
def test_a_fresh_start_resets_the_workers_training_state_once_and_keeps_the_weights(tmp_path, stacks) -> None:
    stack = stacks(tmp_path, critic=True)
    first = stack.run(TrainingMethod("sft"), step=0)

    fresh = stack.component.fresh_start(first.checkpoints[0].path)

    assert fresh == str(tmp_path / "reef-starts" / "fresh_rollout_0")
    assert SlimeStart.read(Path(fresh)) == SlimeStart(0, None, {}, None, fresh=True)
    stack.start = fresh
    second = stack.run(TrainingMethod("sft"), step=1)
    # The workers reset before the job trained, and the objective's state started empty.
    assert (stack.group.events, stack.critic.events) == (["reset"], ["reset"])
    assert second.metrics["rollout_id"] == 1
    assert SlimeStart.read(Path(second.checkpoints[0].path)) == SlimeStart(1, "sft", {"steps": 1}, None)
    # The next job continues from the checkpoint and does not reset again.
    stack.run(TrainingMethod("sft"), step=2)
    assert stack.group.events == ["reset"]
    # The same start gives the same fresh start; a job from it is refused once the workers moved on.
    assert stack.component.fresh_start(first.checkpoints[0].path) == fresh
    job = TrainingJob("late", "slime", fresh, batch_for(3), parameters(TrainingMethod("sft")))
    with pytest.raises(ValueError, match="hold the weights of rollout 2"):
        stack.component.submit(job)
    assert stack.component.fresh_start(None) == str(tmp_path / "reef-starts" / "fresh_startup")


@pytest.mark.unit
def test_a_job_of_another_objective_starts_with_an_empty_objective_state(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    stack.run(TrainingMethod("sft"), step=0)
    register_loss_family(CriticWarmupFamily())
    try:
        second = stack.run(TrainingMethod(CRITIC_WARMUP_OBJECTIVE), step=1)
    finally:
        unregister_loss_family("test-critic-warmup")
    state = SlimeStart.read(Path(second.checkpoints[0].path))
    assert (state.objective, state.algorithm_state) == (CRITIC_WARMUP_OBJECTIVE, {"steps": 1})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("job_parameters", "message"),
    [
        (
            {**parameters(TrainingMethod("sft")), "learning_rate": 1e-5},
            r"accept no parameters \['learning_rate'\]",
        ),
        ({"scheduling": {}}, "'method' parameter as an object"),
        ({"method": "sft"}, "'method' parameter as an object"),
        ({**parameters(TrainingMethod("sft")), "scheduling": None}, "scheduling parameter must be an object"),
        ({**parameters(TrainingMethod(REWARD_OBJECTIVE)), "scheduling": {"epochs": 2}}, "does not support 2 epochs"),
        ({**parameters(TrainingMethod("sft")), "save_interval": 0}, "save_interval must be a positive int"),
        ({**parameters(TrainingMethod("sft")), "save_interval": True}, "save_interval must be a positive int"),
    ],
)
def test_a_job_with_parameters_slime_does_not_take_is_refused_and_records_nothing(
    tmp_path, stacks, job_parameters, message
) -> None:
    stack = stacks(tmp_path)
    with pytest.raises(ValueError, match=message):
        stack.component.submit(TrainingJob("job-0", "slime", None, batch_for(0), job_parameters))
    with pytest.raises(KeyError):
        stack.component.status("job-0")
    assert stack.group.train_calls == []


@pytest.mark.unit
def test_a_job_of_tasks_is_refused_because_slime_rolls_no_task_out(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    tasks = JobData("tasks", (TaskSample(tmp_path / "task"),))
    with pytest.raises(ValueError, match="does not roll tasks out"):
        stack.component.submit(TrainingJob("job-0", "slime", None, tasks, parameters(TrainingMethod("sft"))))
    with pytest.raises(KeyError):
        stack.component.status("job-0")
    assert stack.group.train_calls == []


@pytest.mark.unit
def test_a_lora_release_holds_the_adapter_where_serving_loads_it(tmp_path, stacks) -> None:
    stack = stacks(tmp_path, lora=True)

    status = stack.run(TrainingMethod("sft"), step=0)

    (checkpoint,) = status.checkpoints
    release = Path(checkpoint.path) / "release"
    assert checkpoint.release_path == str(release)
    assert (release / ADAPTER_DIR / "adapter_config.json").is_file()
    assert (release / ADAPTER_DIR / "adapter_model.safetensors").read_text() == "adapter-0"
    assert {path.name for path in release.iterdir()} == {ADAPTER_DIR}


@pytest.mark.unit
def test_the_release_format_names_full_weights_or_the_lora_adapter_in_the_engines_module_names(
    tmp_path, stacks
) -> None:
    assert stacks(tmp_path / "full").component.release_format == ReleaseFormat("full")
    lora = stacks(tmp_path / "lora", lora=True, megatron_lora_rank=8, megatron_lora_target_modules=None)
    assert lora.component.release_format == ReleaseFormat(
        "adapter", 8, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    )
    attention = stacks(
        tmp_path / "attention", lora=True, megatron_lora_rank=4, megatron_lora_target_modules=["linear_qkv"]
    )
    assert attention.component.release_format == ReleaseFormat("adapter", 4, ("q_proj", "k_proj", "v_proj"))


@pytest.mark.unit
def test_a_symlinked_checkpoint_fails_the_job(tmp_path, stacks, monkeypatch) -> None:
    stack = stacks(tmp_path)
    target = tmp_path / "checkpoint-target"
    target.mkdir()

    def save_model(rollout_id, force_sync=False):
        Path(stack.args.save_hf.format(rollout_id=rollout_id)).parent.mkdir(parents=True, exist_ok=True)
        Path(stack.args.save_hf.format(rollout_id=rollout_id)).symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(stack.group, "save_model", save_model)

    status = stack.run(TrainingMethod("sft"), step=0)

    assert status.state == "failed" and "missing or unsafe" in status.error
    assert status.checkpoints == ()


@pytest.mark.unit
@pytest.mark.parametrize("progress", ["before_save", "after_save", "recorded"])
def test_a_restarted_component_resumes_its_unfinished_job_without_training_a_saved_rollout(
    tmp_path, stacks, progress
) -> None:
    jobs = tmp_path / "reef-jobs" / "job-0"
    data = batch_for(0)
    write_json(
        jobs / "job.json", {"start": None, "data": data.to_dict(), "parameters": parameters(TrainingMethod("sft"))}
    )
    # The job started training rollout 0 before the restart.
    write_json(jobs / ROLLOUT_FILE, {"rollout_id": 0, "step": 2})
    start_rollout_id = 0
    if progress in ("after_save", "recorded"):
        # Megatron saved the rollout, which the restarted workers loaded again.
        (tmp_path / "hf" / "0").mkdir(parents=True)
        (tmp_path / "megatron" / "iter_0000000").mkdir(parents=True)
        start_rollout_id = 1
    saved_start = tmp_path / "reef-starts" / "rollout_0"
    if progress == "recorded":
        SlimeStart(0, "sft", {"steps": 1}, None).write(saved_start)
        write_json(
            jobs / "checkpoints.json",
            {
                "checkpoints": [
                    {"step": 2, "path": str(saved_start), "release_path": str(tmp_path / "hf" / "0"), "metrics": {}}
                ]
            },
        )

    stack = stacks(tmp_path, start_rollout_id=start_rollout_id)
    status = finished(stack.component, "job-0")

    assert status.state == "finished"
    (checkpoint,) = status.checkpoints
    if progress == "before_save":
        # Nothing was saved: the job trains its rollout again.
        assert stack.group.train_calls == [(0, ["packed"])]
        assert checkpoint.path == str(tmp_path / "reef-starts" / "rollout_0")
    elif progress == "after_save":
        assert stack.group.train_calls == []
        assert (status.metrics["rollout_id"], status.metrics["resumed"]) == (0, 1)
        assert checkpoint.path == str(tmp_path / "reef-starts" / "rollout_0")
        assert SlimeStart.read(Path(checkpoint.path)).algorithm_state == {"steps": 1}
        # The storage records the rollout, so the next admission does not see unowned assets.
        stored = json.loads(stack.storage._record_path(0).read_text())
        assert (stored["job_id"], stored["reward"]) == ("job-0", 1.0)
    else:
        assert stack.group.train_calls == []
        assert checkpoint.path == str(saved_start)


@pytest.mark.unit
def test_a_skipped_job_saves_no_checkpoint_and_the_next_job_continues_the_start(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    for step, data_id in ((0, "plain"), (1, "count-1")):
        status = stack.run(TrainingMethod(SKIP_OBJECTIVE), step=step, batch=JobData(data_id, batch_for(step).samples))
        # A skip trains nothing and saves nothing, whatever state the objective proposed.
        assert (status.state, status.checkpoints, status.metrics) == ("finished", (), {"skipped": 1})
    assert (stack.group.train_calls, stack.group.saved) == ([], [])


# -- Loss families and learning-rate schedules


@pytest.mark.unit
def test_a_single_method_job_leaves_the_workers_as_they_started(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    stack.run(TrainingMethod("sft"), step=0)
    stack.run(TrainingMethod("sft"), step=1)
    assert [stack.worker_switch(job) for job in (0, 1)] == [None, None]
    assert len(stack.group.train_calls) == 2


@pytest.mark.unit
def test_a_job_of_another_family_switches_the_actor_workers_and_back(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    for step, method in enumerate(("sft", REWARD_OBJECTIVE, REWARD_OBJECTIVE, "sft")):
        assert stack.run(TrainingMethod(method), step=step).state == "finished"

    policy = stack.worker_switch(1)
    assert policy is not None
    assert policy["learning_rate_schedule_state"] is None
    assert policy["loss_family_args"]["loss_family"] == "pg"
    assert policy["loss_family_args"]["loss_type"] == "policy_loss"
    # The workers already run the policy family for the next job.
    assert stack.worker_switch(2) is None
    supervised = stack.worker_switch(3)
    assert supervised is not None
    assert supervised["loss_family_args"]["loss_family"] == "sft"
    assert supervised["loss_family_args"]["loss_type"] == "sft_loss"
    assert set(supervised["loss_family_args"]) == set(policy["loss_family_args"])
    # Every job trained with its own family: the policy jobs shipped advantages, the supervised ones none.
    assert ["advantages" in data for data in stack.processor.calls] == [False, True, True, False]
    assert len(stack.group.train_calls) == 4


@pytest.mark.unit
def test_a_family_the_workers_cannot_train_is_refused(tmp_path, stacks) -> None:
    register_loss_family(OptionsFamily())
    try:
        stack = stacks(tmp_path)
        with pytest.raises(RuntimeError, match="requires driver options"):
            stack.component.algorithm_for_loss_family("test-needs-options")
        # A family that configures the critic started with the workers or not at all.
        with pytest.raises(RuntimeError, match="configures the critic"):
            stack.component.algorithm_for_loss_family("sao")
        # A worker keeps one teacher, built from the first distillation family's settings.
        stack.component.algorithm_for_loss_family("sdft")
        with pytest.raises(RuntimeError, match="second distillation family"):
            stack.component.algorithm_for_loss_family("sdpo")
    finally:
        unregister_loss_family("test-needs-options")


@pytest.mark.unit
def test_a_schedule_advances_by_optimizer_steps_and_continues_after_a_restart(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    first = stack.run(TrainingMethod("sft", WARMUP), step=0)
    worker_switch = stack.worker_switch(0)
    assert worker_switch is not None
    assert worker_switch["learning_rate_schedule_state"] == LearningRateScheduleState(WARMUP, 0).to_dict()
    # Two samples, one optimizer step each: the job's last step ran at the rate of step 1.
    assert first.checkpoints[0].metrics["learning_rate"] == pytest.approx(2.5e-5)
    assert first.checkpoints[0].metrics["learning_rate_schedule"] == {"name": "warmup", "completed_steps": 2}
    second = stack.run(TrainingMethod("sft", WARMUP), step=1)
    assert stack.worker_switch(1)["learning_rate_schedule_state"]["completed_steps"] == 2
    # The progress is saved with the rollout's start.
    assert SlimeStart.read(Path(second.checkpoints[0].path)).schedule_state == LearningRateScheduleState(WARMUP, 4)
    stack.component.close()

    # Workers that loaded the second checkpoint continue the warmup where it stopped.
    restarted = stacks(tmp_path, start_rollout_id=2)
    assert restarted.start == second.checkpoints[0].path
    restarted.run(TrainingMethod("sft", WARMUP), step=2)
    assert restarted.worker_switch(0)["learning_rate_schedule_state"]["completed_steps"] == 4
    # A job that selects no schedule keeps the active one.
    restarted.run(TrainingMethod("sft"), step=3)
    assert restarted.worker_switch(1)["learning_rate_schedule_state"]["completed_steps"] == 6


@pytest.mark.unit
def test_a_schedule_with_decoupled_lr_fails_the_job_before_training(tmp_path, stacks) -> None:
    stack = stacks(tmp_path, decoupled_lr=1e-5)
    status = stack.run(TrainingMethod("sft", WARMUP), step=0)
    assert status.state == "failed" and "decoupled-lr" in status.error
    assert stack.group.train_calls == []
    assert status.checkpoints == ()


@pytest.mark.unit
def test_a_critic_only_step_leaves_the_actor_schedule_where_it_was(tmp_path, stacks) -> None:
    register_loss_family(CriticWarmupFamily())
    try:
        stack = stacks(tmp_path, loss_family="test-critic-warmup")
        first = stack.run(TrainingMethod(CRITIC_WARMUP_OBJECTIVE, WARMUP), step=0)
        assert "learning_rate" not in first.checkpoints[0].metrics
        assert SlimeStart.read(Path(first.checkpoints[0].path)).schedule_state == LearningRateScheduleState(WARMUP, 0)
        # The actor never fetched the job's data, so the next job carries the worker switch again.
        stack.run(TrainingMethod(CRITIC_WARMUP_OBJECTIVE, WARMUP), step=1)
        assert stack.worker_switch(1) == stack.worker_switch(0)
    finally:
        unregister_loss_family("test-critic-warmup")


@pytest.mark.unit
def test_a_job_family_brings_its_loss_type_and_advantage_routing_and_keeps_the_startup_options(tmp_path) -> None:
    from reef.train.slime_backend.reef_adapters.slime_arguments import loss_family_job_args

    # Workers that started with a pg-primitive family: Slime's estimator was routed to cispo.
    startup = worker_args(
        tmp_path, advantage_estimator="cispo", reef_configured_advantage_estimator="grpo", kl_coef=0.1
    )
    policy = loss_family_job_args(startup, resolve_loss_family("pg"))
    assert (policy.loss_family, policy.loss_type, policy.advantage_estimator) == ("pg", "policy_loss", "grpo")
    assert policy.compute_advantages_and_returns is False
    # TTTD keeps Slime's pre-train advantage pass for its frozen-base KL, and validates the startup --kl-coef.
    tttd = loss_family_job_args(startup, resolve_loss_family("tttd"))
    assert (tttd.loss_type, tttd.compute_advantages_and_returns) == ("custom_loss", True)
    assert tttd.loss_family_ref == "recipes.tttd.slime:TttdAlgorithm"
    with pytest.raises(RuntimeError, match="positive finite --kl-coef"):
        loss_family_job_args(worker_args(tmp_path, kl_coef=0.0), resolve_loss_family("tttd"))
    # The startup arguments are the workers' own and stay as they are.
    assert (startup.loss_family, startup.loss_type, startup.advantage_estimator) == ("sft", "sft_loss", "cispo")


# -- Worker results, metrics and the rows' versions


@pytest.mark.unit
def test_worker_results_wait_for_the_configured_timeout_and_metrics_come_from_every_rank(
    tmp_path, stacks, monkeypatch
) -> None:
    waits = []

    class TrainingResult(ExecutorFuture):
        def result(self, timeout=None):
            waits.append(timeout)
            return {"loss": 0.1}

    # Slime logs a train step on the last pipeline stage, not on rank 0.
    stack = stacks(
        tmp_path,
        train_timeout_s=43200.0,
        rank_metrics=[
            {"train/step": 0, "perf/step_time": 1.5},
            {"train/step": 0, "train/loss": 1.25, "train/grad_norm": 2.0},
        ],
    )
    stack.group.train_result = [TrainingResult()]
    monkeypatch.setattr(base, "monotonic", lambda: 100.0)

    metrics = stack.run(TrainingMethod("sft"), step=0).checkpoints[0].metrics

    assert waits == [43200.0]
    assert (metrics["loss"], metrics["train/loss"], metrics["train/grad_norm"]) == (0.1, 1.25, 2.0)
    assert metrics["perf/step_time"] == 1.5


@pytest.mark.unit
def test_the_workers_receive_each_rows_producing_versions(tmp_path, stacks) -> None:
    stack = stacks(tmp_path)
    mixed = policy_trajectory(
        "mixed",
        (10, 20, 21, 22),
        (1, 1, 1),
        (-0.1, -0.2, -0.3),
        1.0,
        None,
        runtime_load_spans=(RuntimeLoadSpan(0, 1, "inc:6"), RuntimeLoadSpan(1, 3, "inc:7")),
    )
    batch = JobData("mixed", (*batch_for(0).samples, mixed))
    stack.run(TrainingMethod("sft"), step=0, batch=batch)
    (data,) = stack.processor.calls
    # Training metadata, not a check: a sample's version, or the versions of its token spans.
    assert data["producing_runtime_load_ids"] == ["inc:0", "inc:1", None]
    assert data["producing_runtime_load_spans"] == [
        [],
        [],
        [{"start": 0, "end": 1, "runtime_load_id": "inc:6"}, {"start": 1, "end": 3, "runtime_load_id": "inc:7"}],
    ]


# -- Checkpoint storage


@pytest.mark.unit
def test_checkpoint_storage_records_each_rollout_and_refuses_a_job_it_cannot_hold(tmp_path, stacks) -> None:
    root = tmp_path / "checkpoints"
    megatron_root = root / "megatron"
    source_hf, source_megatron = tmp_path / "source-hf", tmp_path / "source-megatron"
    source_hf.mkdir()
    (source_hf / "model.safetensors").write_text("weights-0")
    (source_megatron / "iter_0000000").mkdir(parents=True)
    (source_megatron / "iter_0000000" / "state").write_text("megatron")
    (source_megatron / "latest_checkpointed_iteration.txt").write_text("0")
    hf_bytes = _allocated_bytes(source_hf)
    pair_bytes = max(hf_bytes + _allocated_bytes(source_megatron / "iter_0000000"), 8 * hf_bytes)
    template = str(root / "hf" / "{rollout_id}")
    storage = CheckpointStorage(
        RetentionConfig(max_storage_bytes=pair_bytes, min_free_space_bytes=0),
        hf_template=template,
        megatron_root=megatron_root,
        source_hf=source_hf,
        source_megatron=source_megatron,
    )
    stack = stacks(tmp_path, save_hf=template, storage=storage)

    first = stack.run(TrainingMethod("sft"), step=0, batch=batch_for(0, rewards=(1.0, 0.0)))

    assert first.state == "finished"
    stored = json.loads(storage._record_path(0).read_text())
    assert (stored["rollout_id"], stored["job_id"], stored["reward"]) == (0, "job-0", pytest.approx(0.5))
    # The next job's start is protected, so the cap leaves no room for another pair.
    blocked = stack.run(TrainingMethod("sft"), step=1)
    assert blocked.state == "failed" and "checkpoint storage is full" in blocked.error
    assert len(stack.group.train_calls) == len(stack.group.saved) == 1


@pytest.mark.unit
def test_storage_preflight_resolves_the_paths_and_refuses_a_storage_it_cannot_fill(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    args = SlimeArguments(
        save_hf="~/checkpoints/hf/{rollout_id}",
        save="~/checkpoints/megatron",
        use_critic=True,
        critic_save=None,
        hf_checkpoint=None,
        load=None,
        megatron_lora_rank=0,
        critic_save_interval=1,
        no_save_optim=False,
    )
    monkeypatch.setattr(CheckpointStorage, "validate_capacity", lambda self, **kwargs: {"blocked": False})
    storage = prepare_checkpoint_storage(args, RetentionConfig())
    assert args.save_hf == str(tmp_path / "checkpoints/hf/{rollout_id}") == storage.hf_template
    # The critic's weights and optimizer survive restarts beside the actor's.
    assert args.critic_save == str(tmp_path / "checkpoints/megatron-critic")
    monkeypatch.setattr(
        CheckpointStorage,
        "validate_capacity",
        lambda self, **kwargs: {"blocked": True, "reasons": ["the filesystem free-space floor"]},
    )
    with pytest.raises(RuntimeError, match=r"storage preflight.*free-space floor"):
        prepare_checkpoint_storage(args, RetentionConfig())


# -- Closing


@pytest.mark.unit
def test_a_critic_group_needs_the_storages_critic_root(tmp_path) -> None:
    for critic, critic_root in ((FakeTrainGroup(None), None), (None, tmp_path / "megatron-critic")):
        with pytest.raises(ValueError, match="critic root: give both or neither"):
            SlimeTrainingComponent(
                worker_args(tmp_path),
                FakeTrainGroup(str(tmp_path / "hf" / "{rollout_id}")),
                critic,
                batch_processor=RecordingBatchProcessor(),
                storage=checkpoint_storage(tmp_path, critic_root=critic_root),
                lora=False,
                loss_family="sft",
            )


@pytest.mark.unit
def test_closing_releases_the_critic_then_the_actor_then_the_resources(tmp_path) -> None:
    events: list[str] = []

    class Resources:
        def close(self):
            events.append("resources")

    actor = FakeTrainGroup(str(tmp_path / "hf" / "{rollout_id}"))
    critic = FakeTrainGroup(None, release_error=RuntimeError("critic unavailable"))
    actor.events = critic.events = events
    component = SlimeTrainingComponent(
        worker_args(tmp_path),
        actor,
        critic,
        batch_processor=RecordingBatchProcessor(),
        storage=checkpoint_storage(tmp_path, critic_root=tmp_path / "megatron-critic"),
        lora=False,
        loss_family="sft",
        resources=Resources(),
    )
    with pytest.raises(RuntimeError, match="critic unavailable"):
        component.close()
    component.close()
    assert events == ["release", "release", "resources"]
    with pytest.raises(RuntimeError, match="closed"):
        component.submit(TrainingJob("late", "slime", None, batch_for(0), parameters(TrainingMethod("sft"))))


# -- Building the component


@pytest.fixture
def build_calls(tmp_path, monkeypatch):
    """Replace Slime's parser, the Ray resources and the workers; record what ``create_slime_component`` gives them."""
    calls: dict[str, Any] = {"events": []}
    args = worker_args(
        tmp_path,
        num_rollout=1,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        offload_rollout=False,
        reef_executor_backend="ray",
        custom_megatron_init_path=None,
        custom_critic_args_hook_path=None,
        megatron_lora_rank=0,
        use_critic=True,
        use_wandb=False,
        wandb_key=None,
        megatron_to_hf_mode="raw",
        no_save_optim=False,
        release_train=False,
    )

    def parse(arguments):
        calls["arguments"] = list(arguments)
        return args

    class Resources:
        def __init__(self, received, **options):
            assert received is args
            calls["resources"] = options
            self.placement_groups = {"actor": "training-bundles", "critic": "training-bundles"}

        def start(self):
            calls["events"].append("start")

        def close(self):
            calls["events"].append("close")

    worker_class = type("ReefMegatronTrainRayActor", (), {})

    def create_groups(received, placement_groups, *, actor_cls, train_timeout_s):
        assert received is args and actor_cls is worker_class
        calls["timeout"] = train_timeout_s
        if calls.get("group_error"):
            raise RuntimeError("workers failed to start")
        calls["groups"] = (FakeTrainGroup(args.save_hf), FakeTrainGroup(None))
        return calls["groups"]

    def storage(received, retention):
        calls["retention"] = retention
        return checkpoint_storage(tmp_path, critic_root=tmp_path / "megatron-critic")

    from reef.train.slime_backend.reef_adapters import slime_arguments

    monkeypatch.setattr(startup, "parse_slime_args", parse)
    monkeypatch.setattr(
        startup,
        "configure_executors",
        lambda received, execution, arguments: calls.update(execution=execution),
    )
    monkeypatch.setattr(startup, "slime_executor_class", lambda backend: None)
    monkeypatch.setattr(slime_component, "SlimeDeploymentResources", Resources)
    monkeypatch.setattr(slime_component, "create_train_groups", create_groups)
    monkeypatch.setattr(slime_component, "prepare_checkpoint_storage", storage)
    monkeypatch.setattr(slime_arguments, "configure_reef_loss_args", lambda received: None)
    # The worker class loads torch, Megatron and Slime's Ray actor; the fake groups never start it.
    worker_module = types.ModuleType("reef.train.slime_backend.reef_adapters.megatron.train_actor")
    worker_module.ReefMegatronTrainRayActor = worker_class  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, worker_module.__name__, worker_module)
    monkeypatch.setenv("PYTHONPATH", "/repo")
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    calls["args"] = args
    return calls


def settings(loss_family: str = "sft") -> SlimeSettings:
    return SlimeSettings(
        arguments=("--reef-checkpoint-policy=best_reward", "--reef-checkpoint-max-storage=100GiB", "--lr=1e-06"),
        loss_family=loss_family,
        recipe="recipes.test:Recipe",
        execution={"execution": {"training": "ray"}},
        ray_address="ray://head:10001",
        namespace="test",
        train_timeout_s=43200.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"loss_family": ""}, "loss_family must be a non-empty string"),
        ({"recipe": ""}, "recipe must be a non-empty string"),
        ({"ray_address": ""}, "ray_address must be a non-empty string"),
        ({"execution": {"serving": {}}}, "only the execution and executors sections"),
        ({"train_timeout_s": 0.0}, "train_timeout_s must be positive"),
    ],
)
def test_slime_settings_refuse_values_the_component_cannot_start_from(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        SlimeSettings(**{**vars(settings()), **overrides})


@pytest.mark.unit
def test_a_slime_start_and_the_worker_settings_check_their_values() -> None:
    for rollout_id in (-1, True):
        with pytest.raises(ValueError, match="rollout_id must be a non-negative int"):
            SlimeStart(rollout_id, None, {}, None)
    with pytest.raises(TypeError, match="algorithm_state must be a mapping"):
        SlimeStart(0, None, [], None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="objective of a non-empty state, and only of one"):
        SlimeStart(0, None, {"steps": 1}, None)
    with pytest.raises(ValueError, match="objective of a non-empty state, and only of one"):
        SlimeStart(0, "sft", {}, None)
    with pytest.raises(ValueError, match="a fresh SlimeStart has no objective state and no schedule progress"):
        SlimeStart(0, None, {}, LearningRateScheduleState(WARMUP, 1), fresh=True)
    with pytest.raises(ValueError, match="a fresh SlimeStart has no objective state and no schedule progress"):
        SlimeStart(0, "sft", {"steps": 1}, None, fresh=True)
    with pytest.raises(ValueError, match="algorithm_state must encode as JSON"):
        SlimeStart(0, "sft", {"steps": float("nan")}, None)
    with pytest.raises(TypeError, match="schedule_state must be a LearningRateScheduleState"):
        SlimeStart(0, None, {}, {"completed_steps": 1})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fresh must be a bool"):
        SlimeStart(0, None, {}, None, fresh=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="loss_family must be a non-empty string"):
        WorkerTrainingSettings("", None)
    with pytest.raises(TypeError, match="schedule_state must be a LearningRateScheduleState"):
        WorkerTrainingSettings("sft", {"completed_steps": 1})  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"rollout_id": -1}, ValueError, "rollout_id must be a non-negative int"),
        ({"step": True}, ValueError, "step must be a positive int"),
        ({"step": 0}, ValueError, "step must be a positive int"),
        ({"next_schedule": LearningRateScheduleState(WARMUP, 1)}, ValueError, "before and after the rollout"),
        ({"fresh": None}, TypeError, "fresh must be a bool"),
        ({"objective": None, "algorithm_state": {"steps": 1}}, ValueError, "objective of a non-empty state"),
    ],
)
def test_a_slime_rollout_checks_its_values(values, error, message) -> None:
    rollout = {
        "rollout_id": 0,
        "step": 1,
        "schedule_state": None,
        "next_schedule": None,
        "fresh": False,
        "objective": "sft",
        "algorithm_state": {"steps": 1},
    }
    SlimeRound(**rollout)
    with pytest.raises(error, match=message):
        SlimeRound(**{**rollout, **values})


@pytest.mark.unit
def test_the_batch_processor_registers_a_familys_keys_and_refuses_another_dtype() -> None:
    args = SlimeArguments(custom_rollout_data_keys=("a",), reef_rollout_tensor_dtypes={"a": "int"})
    processor = TrainingBatchProcessor(args, {"dp_size": 2})
    processor.register_keys(("b", "a"), {"b": "float32", "a": "int"})
    assert (processor.rollout_data_keys, processor.tensor_dtypes) == (("a", "b"), {"a": "int", "b": "float32"})
    with pytest.raises(ValueError, match="rollout key 'b' is declared as long"):
        processor.register_keys(("c",), {"c": "int", "b": "long"})
    # A refused family changes nothing, and the startup arguments stay as the workers started with them.
    assert (processor.rollout_data_keys, processor.tensor_dtypes) == (("a", "b"), {"a": "int", "b": "float32"})
    assert (args.custom_rollout_data_keys, args.reef_rollout_tensor_dtypes) == (("a",), {"a": "int"})
    with pytest.raises(ValueError, match="positive dp_size"):
        TrainingBatchProcessor(args, {"dp_size": 0})


@pytest.mark.unit
def test_the_component_starts_its_workers_on_the_training_gpus_without_serving_engines(build_calls) -> None:
    component = create_slime_component(settings())
    try:
        # Reef's retention options never reach Slime's parser.
        assert build_calls["arguments"] == ["--lr=1e-06"]
        assert build_calls["retention"] == RetentionConfig(policy="best_reward", max_storage_bytes=100 * 1024**3)
        assert build_calls["resources"] == {
            "ray_address": "ray://head:10001",
            "namespace": "test",
            "runtime_env": {
                "env_vars": {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0", "PYTHONPATH": "/repo"}
            },
        }
        # The deployment's execution sections select the training executor.
        assert build_calls["execution"] == {"execution": {"training": "ray"}}
        assert build_calls["timeout"] == 43200.0
        assert component.args is build_calls["args"]
        assert component.train_timeout_s == 43200.0
        assert build_calls["events"] == ["start"]
    finally:
        component.close()
    assert build_calls["events"] == ["start", "close"]


@pytest.mark.unit
def test_a_failed_worker_start_closes_the_resources(build_calls) -> None:
    build_calls["group_error"] = True
    with pytest.raises(RuntimeError, match="workers failed to start"):
        create_slime_component(settings())
    assert build_calls["events"] == ["start", "close"]


@pytest.mark.unit
def test_a_recipe_with_an_unknown_loss_family_is_refused_before_slime_parses(build_calls) -> None:
    with pytest.raises(RuntimeError, match="declares unsupported loss family 'grpo'"):
        create_slime_component(settings("grpo"))
    assert "arguments" not in build_calls


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"debug_train_only": True}, "debug-train-only"),
        ({"debug_rollout_only": True}, "debug-rollout-only"),
        ({"colocate": True}, "remove --colocate"),
        ({"offload_rollout": True}, "remove --offload-rollout"),
        ({"num_rollout": 0}, "positive --num-rollout"),
        ({"save_hf": "/checkpoints/hf"}, "rollout_id"),
        ({"compute_advantages_and_returns": True}, "disable-compute-advantages-and-returns"),
        ({"release_train": True}, "remove --release-train"),
    ],
)
def test_slime_arguments_the_training_component_cannot_run_are_refused(overrides, message) -> None:
    values = {
        "num_rollout": 1,
        "save_hf": "/checkpoints/hf/{rollout_id}",
        "compute_advantages_and_returns": False,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "offload_rollout": False,
        "release_train": False,
    }
    validate_slime_args(SlimeArguments(**values), resolve_loss_family("sft"))
    with pytest.raises(ValueError, match=message):
        validate_slime_args(SlimeArguments(**{**values, **overrides}), resolve_loss_family("sft"))


@pytest.mark.unit
def test_the_workers_runtime_env_carries_the_cuda_defaults_and_this_processs_pythonpath() -> None:
    defaults = {"CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": "0"}
    assert startup.job_runtime_env({}) == {"env_vars": defaults}
    assert startup.job_runtime_env({"PYTHONPATH": "   ", "OTHER": "x"}) == {"env_vars": defaults}
    # The operator's own values win over the defaults.
    overridden = {"CUDA_DEVICE_MAX_CONNECTIONS": "8", "NCCL_NVLS_ENABLE": "1"}
    assert startup.job_runtime_env(overridden) == {"env_vars": overridden}
    # Cookbook loss families resolve inside the workers, which need the service's PYTHONPATH.
    assert startup.job_runtime_env({"PYTHONPATH": "/repo:/opt/sglang/python"}) == {
        "env_vars": {**defaults, "PYTHONPATH": "/repo:/opt/sglang/python"}
    }


@pytest.mark.unit
def test_retention_options_are_parsed_with_units() -> None:
    retention, remaining = retention_options(["--reef-checkpoint-min-free-space=10GB", "--loss-type", "sft_loss"])
    assert retention == RetentionConfig(min_free_space_bytes=10 * 1000**3)
    assert remaining == ["--loss-type", "sft_loss"]
    for size in ("100", "0GB"):
        with pytest.raises(SystemExit):
            retention_options([f"--reef-checkpoint-max-storage={size}"])


@pytest.mark.unit
def test_slime_rejects_its_legacy_raw_wandb_flags() -> None:
    startup.validate_tracking_args(SimpleNamespace(use_wandb=False, wandb_key=None))
    with pytest.raises(RuntimeError, match=r"observability\.wandb"):
        startup.validate_tracking_args(SimpleNamespace(use_wandb=True, wandb_key=None))
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        startup.validate_tracking_args(SimpleNamespace(use_wandb=False, wandb_key="secret"))


@pytest.mark.unit
def test_the_component_stamps_a_dotted_family_reference_for_the_workers() -> None:
    dotted = SimpleNamespace(loss_family="toy")
    startup.stamp_loss_family_reference(dotted, "toy_pkg.family:ToyAlgorithm")
    assert dotted.loss_family_ref == "toy_pkg.family:ToyAlgorithm"

    cookbook = SimpleNamespace(loss_family="sao")
    startup.stamp_loss_family_reference(cookbook, "sao")
    assert cookbook.loss_family_ref == "recipes.sao.slime:SaoAlgorithm"

    unreferenced = SimpleNamespace(loss_family="pg")
    startup.stamp_loss_family_reference(unreferenced, "pg")
    assert vars(unreferenced) == {"loss_family": "pg"}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ref_load", "expected"), [("/checkpoints/initial", "/checkpoints/initial"), (None, "/models/hf")]
)
def test_bridge_mode_first_start_falls_back_to_initial_checkpoint(tmp_path: Path, ref_load, expected) -> None:
    args = SimpleNamespace(
        megatron_to_hf_mode="bridge",
        load=str(tmp_path / "empty-resume-directory"),
        ref_load=ref_load,
        hf_checkpoint="/models/hf",
        start_rollout_id=9,
    )
    startup.use_megatron_bridge_first_start_load(args)
    assert (args.load, args.start_rollout_id) == (expected, 0)


@pytest.mark.unit
def test_bridge_mode_restart_keeps_resumable_checkpoint(tmp_path: Path) -> None:
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "latest_checkpointed_iteration.txt").write_text("3")
    args = SimpleNamespace(
        megatron_to_hf_mode="bridge",
        load=str(resume),
        ref_load="/checkpoints/initial",
        hf_checkpoint="/models/hf",
        start_rollout_id=4,
    )
    startup.use_megatron_bridge_first_start_load(args)
    assert (args.load, args.start_rollout_id) == (str(resume), 4)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("no_save_optim", "same_dir", "tracked", "expected"),
    [
        (True, True, True, True),
        # A first start has nothing of its own to resume.
        (True, True, False, False),
        # A checkpoint from elsewhere may well carry its optimizer.
        (True, False, True, False),
        (False, True, True, False),
    ],
)
def test_a_restart_without_a_saved_optimizer_does_not_load_one(
    tmp_path: Path, no_save_optim: bool, same_dir: bool, tracked: bool, expected: bool
) -> None:
    save = tmp_path / "megatron"
    load = save if same_dir else tmp_path / "other"
    load.mkdir(parents=True)
    if tracked:
        (load / "latest_checkpointed_iteration.txt").write_text("23")
    args = SimpleNamespace(no_save_optim=no_save_optim, no_load_optim=False, load=str(load), save=str(save))
    startup.apply_unsaved_optimizer_resume(args)
    assert args.no_load_optim is expected


# -- The loss family's checks of the workers' arguments, which the component runs before the workers start


@pytest.mark.unit
@pytest.mark.parametrize(
    ("loss_family", "loss_type", "use_rollout_logprobs"), [("sft", "sft_loss", False), ("pg", "policy_loss", True)]
)
def test_matching_reef_and_slime_objectives_are_accepted(loss_family, loss_type, use_rollout_logprobs) -> None:
    resolve_loss_family(loss_family).validate_backend_args(
        SimpleNamespace(loss_type=loss_type, use_rollout_logprobs=use_rollout_logprobs)
    )


@pytest.mark.parametrize(
    ("loss_family", "loss_type", "use_rollout_logprobs", "message"),
    [
        ("sft", "policy_loss", False, "sft_loss"),
        ("pg", "sft_loss", True, "policy_loss"),
        ("pg", "policy_loss", False, "use-rollout-logprobs"),
    ],
)
def test_mismatched_reef_and_slime_objectives_are_refused(
    loss_family, loss_type, use_rollout_logprobs, message
) -> None:
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family(loss_family).validate_backend_args(
            SimpleNamespace(loss_type=loss_type, use_rollout_logprobs=use_rollout_logprobs)
        )


@pytest.mark.unit
def test_the_tttd_custom_objective_is_accepted() -> None:
    resolve_loss_family("tttd").validate_backend_args(
        SimpleNamespace(
            loss_type="custom_loss", use_rollout_logprobs=True, compute_advantages_and_returns=True, kl_coef=0.1
        ),
    )


def sao_backend_args(**overrides) -> SimpleNamespace:
    values = {
        "loss_type": "policy_loss",
        "use_rollout_logprobs": True,
        "custom_advantage_function_path": "recipes.sao.slime.objective.sao_advantages",
        "custom_pg_loss_function_path": "recipes.sao.slime.objective.sao_loss",
        "use_critic": True,
        "eps_clip": 0.3,
        "eps_clip_high": 5.0,
        "critic_steps_per_actor": 2,
        "num_critic_only_steps": 0,
        "sao_length_adaptive_lambda": True,
        "sao_lambda_alpha": 1.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eps_clip", 1.0, "eps-clip in"),
        ("eps_clip_high", -0.1, "eps-clip-high"),
        ("critic_steps_per_actor", 0, "critic-steps-per-actor"),
        ("num_critic_only_steps", -1, "num-critic-only-steps"),
        ("sao_lambda_alpha", 0.0, "sao-lambda-alpha"),
        # SAO opts in to its value model with the explicit --use-critic flag.
        ("use_critic", False, "requires a value model"),
    ],
)
def test_the_sao_backend_contract_is_checked(field, value, message) -> None:
    resolve_loss_family("sao").validate_backend_args(sao_backend_args())
    with pytest.raises(RuntimeError, match=message):
        resolve_loss_family("sao").validate_backend_args(sao_backend_args(**{field: value}))


@pytest.mark.unit
def test_sao_options_are_stripped_from_slimes_arguments_and_projected_onto_args() -> None:
    # The --sao-* family is owned by the sao plugin: the component strips the
    # flags before Slime's parser sees them and projects the values back onto
    # args so the Megatron-side advantage function can observe them.
    from recipes.sao.slime import SaoSettings

    spec = resolve_loss_family("sao")
    settings_value, remaining = spec.parse_specific_options(
        ["--sao-length-adaptive-lambda", "--sao-lambda-alpha", "2.0", "--loss-type", "policy_loss"]
    )
    assert settings_value == SaoSettings(length_adaptive_lambda=True, lambda_alpha=2.0)
    assert remaining == ["--loss-type", "policy_loss"]

    args = SimpleNamespace()
    spec.apply_driver_options(args, settings_value)
    assert (args.sao_length_adaptive_lambda, args.sao_lambda_alpha) == (True, 2.0)

    defaults = SimpleNamespace()
    spec.apply_driver_options(defaults, None)
    assert (defaults.sao_length_adaptive_lambda, defaults.sao_lambda_alpha) == (False, 1.5)
