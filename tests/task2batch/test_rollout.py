"""The pipeline's rollout loop: saved tasks roll out as batches, and the fresh groups of their episodes train."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.batches import TaskItem, TrainDataItem, TrainingBatch, TrajectoryItem
from reef.pipeline_context import PipelineContext
from reef.record2task.assembly import SampleAssembly, SampleAssemblySettings
from reef.task2batch import RolloutResult, RolloutTask2Batch, TaskPool
from reef.train.pipeline import TrainingPipeline

pytestmark = pytest.mark.unit

#: The trainer's batch size of samples. A rollout batch trains on its fresh groups, whatever this is.
SAMPLE_COUNT = 4


def sample(group: str, runtime_load_id: str = "v1") -> TrajectoryItem:
    """One rollout episode's sample, assembled from an in-memory model call that the given version answered."""
    training = {"tokens": [1, 2, 3], "loss_mask": [1], "rollout_log_probs": [-0.5], "runtime_load_id": runtime_load_id}
    call = AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload={"messages": [], "response": {"training": training}}
    )
    assembly = SampleAssembly(SampleAssemblySettings(accept_multi_turn_policy_samples=True))
    return assembly.assemble((call,), 1.0, source_id=f"episode-{group}").with_metadata(group=group)


class FakeRollout(RolloutTask2Batch):
    """Records each submitted batch; ``finish`` hands back each task use's samples on the next look."""

    def __init__(self, context: PipelineContext, tasks_per_step: int) -> None:
        super().__init__(context, None)
        self.tasks_per_step = tasks_per_step
        self.task_pool = TaskPool()
        self.submitted: list[tuple[str, tuple[str, ...]]] = []
        self.finished: list[RolloutResult] = []
        self.trained: list[tuple[str, tuple[str, ...]]] = []
        self.ended: list[bool] = []
        self.refuses = False

    def make_batch(self, tasks: tuple[TrainDataItem, ...], batch_id: str) -> TrainingBatch:
        return TrainingBatch(batch_id, tasks)

    def submit(self, batch: str, tasks: tuple[TaskItem, ...]) -> tuple[str, ...]:
        if self.refuses:
            raise RuntimeError("the rollout worker is closed")
        task_uses = tuple(f"{batch}/{task.task_path.name}" for task in tasks)
        self.submitted.append((batch, task_uses))
        return task_uses

    def finish(self, samples: dict[str, tuple[TrajectoryItem, ...]]) -> None:
        batch, task_uses = self.submitted[-1]
        self.finished.append(RolloutResult(batch, {task_uses[int(index)]: group for index, group in samples.items()}))

    def take_finished(self) -> tuple[RolloutResult, ...]:
        finished, self.finished = tuple(self.finished), []
        return finished

    def running(self) -> bool:
        return False

    def rollout_trained(self, batch: str, task_uses: tuple[str, ...]) -> None:
        self.trained.append((batch, task_uses))

    def rollout_ended(self, batch: str, *, trained: bool) -> None:
        self.ended.append(trained)


def rollout_pipeline(tasks: int = 2, tasks_per_step: int = 2) -> tuple[TrainingPipeline, FakeRollout]:
    context = PipelineContext("s")
    rollout = FakeRollout(context, tasks_per_step)
    for index in range(tasks):
        rollout.task_pool.add(f"task-{index}", TaskItem(Path(f"task-{index}")), group_key=0)
    return TrainingPipeline(context, task2batch=rollout), rollout


def test_a_rollout_batch_trains_on_its_groups_and_marks_its_tasks_once_the_commit_is_durable() -> None:
    pipeline, rollout = rollout_pipeline()
    assert not pipeline.ready(SAMPLE_COUNT) and len(rollout.task_pool) == 0
    ((batch, task_uses),) = rollout.submitted
    assert pipeline.background_jobs_running(), "the episodes still run"
    assert not pipeline.ready(SAMPLE_COUNT), "no episode ended yet"
    rollout.finish({"0": (sample("a"), sample("a")), "1": (sample("b"), sample("b"))})
    assert pipeline.ready(SAMPLE_COUNT)
    built = pipeline.build_batch(SAMPLE_COUNT)
    assert len(built.items) == 4 and pipeline.build_batch(SAMPLE_COUNT) is built
    consumed = pipeline.acknowledge(built.batch_id)
    assert consumed == {f"rollout:{task_use}" for task_use in task_uses} and rollout.ended == [True]
    assert rollout.trained == [], "the commit is not durable yet"
    pipeline.release_records(pipeline.releasable_record_ids())
    assert rollout.trained == [(batch, task_uses)] and pipeline.rollout is None
    assert rollout.task_pool.is_idle, "trained tasks leave the pool"


def test_a_stale_group_does_not_train_and_its_task_goes_back_to_the_pool() -> None:
    pipeline, rollout = rollout_pipeline()
    for version in ("v1", "v2", "v3"):
        pipeline.observe_serving_runtime_load_id(version)
    pipeline.ready(SAMPLE_COUNT)
    rollout.finish({"0": (sample("a", "v1"), sample("a", "v1")), "1": (sample("b", "v3"), sample("b", "v3"))})
    assert pipeline.ready(SAMPLE_COUNT)
    built = pipeline.build_batch(SAMPLE_COUNT)
    assert [item.metadata["group"] for item in built.items] == ["b", "b"], "the group two versions behind is stale"
    pipeline.acknowledge(built.batch_id)
    assert [task.task_id for task in rollout.task_pool.waiting.values()] == ["task-0"]


def test_a_rollout_without_a_group_ends_and_its_tasks_stay_out_until_a_restart() -> None:
    pipeline, rollout = rollout_pipeline(tasks=3)
    pipeline.ready(SAMPLE_COUNT)
    rollout.finish({})
    assert not pipeline.ready(SAMPLE_COUNT), "no group can train, so the next rollout starts"
    assert rollout.ended == [False] and len(rollout.submitted) == 2
    assert [task.task_id for task in rollout.task_pool.taken.values()] == ["task-2"], "only the new batch holds one"


def test_a_dropped_batch_puts_its_tasks_back_and_a_failed_submit_keeps_them() -> None:
    pipeline, rollout = rollout_pipeline()
    pipeline.ready(SAMPLE_COUNT)
    rollout.finish({"0": (sample("a"), sample("a")), "1": (sample("b"), sample("b"))})
    pipeline.ready(SAMPLE_COUNT)
    assert pipeline.acknowledge(pipeline.build_batch(SAMPLE_COUNT).batch_id, trained=False) == frozenset()
    assert rollout.ended == [False] and len(rollout.task_pool) == 2, "the backend dropped the batch as stale"
    rollout.refuses = True
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.ready(SAMPLE_COUNT)
    assert len(rollout.task_pool) == 2 and pipeline.rollout is None


def test_a_pipeline_that_rolls_tasks_out_takes_no_uploaded_trajectories() -> None:
    context = PipelineContext("s")
    rollout = FakeRollout(context, 1)
    rollout.accepts_trajectories = True
    with pytest.raises(TypeError, match="cannot take uploaded trajectories"):
        TrainingPipeline(context, task2batch=rollout)
