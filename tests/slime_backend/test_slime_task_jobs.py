"""Slime's task jobs: Harbor rollouts as the generate step of Slime's own loop, on the component's rollout group."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("ray", reason="requires the optional Ray runtime")

from tests.slime_backend.test_slime_component import Stack, batch_for, parameters
from tests.support.job_data import delayed_sample, job_data_ref
from tests.support.slime_component import finished

from reef.core.job_data import JobData, TaskSample
from reef.execution.executor.placement import GpuBundles
from reef.storage.durable_files import read_json
from reef.train.algos.group_advantages import GroupAdvantageRule
from reef.train.component import TrainingJob
from reef.train.slime_backend import rollout_group as rollout_group_module
from reef.train.slime_backend.dataset import DATASET_FILE
from reef.train.slime_backend.harbor_rollouts import (
    DELAYED_ROWS_FILE,
    TASK_ROLLOUTS_FILE,
    TaskJobSource,
    read_delayed_rows,
)
from reef.train.slime_backend.objectives import SlimeObjective, register_slime_objective, unregister_slime_objective
from reef.train.slime_backend.rollout_group import SlimeRolloutGroup

OBJECTIVE = "slime_dataset"
FAMILY = "reef.train.slime_backend.native:SlimeNativeAlgorithm"
HARBOR = {"environment": "modal", "concurrency": 4}


class TaskRolloutGroup(SlimeRolloutGroup):
    """The serving session's engines: each rollout writes the results that the rollout manager would write.

    ``outcomes`` holds one ``(group_index, assembled)`` list per rollout; the default is a group of two samples.
    """

    def __init__(self, events: list[str], root: Path, outcomes: list[list[tuple[int, bool]]] | None = None) -> None:
        super().__init__(SimpleNamespace(manager=None), colocated=True, router_url="http://router:3000")
        self.events = events
        self.root = root
        self.outcomes = outcomes or []
        # The data source's record when each rollout generated.
        self.records: list[dict[str, Any]] = []

    def sync(self, actor_group) -> None:
        self.events.append("sync")

    def release(self) -> None:
        self.events.append("release")

    def generate(self, rollout_id: int) -> str:
        self.events.append(f"generate:{rollout_id}")
        record = read_json(self.root / DATASET_FILE)
        self.records.append(record)
        source = TaskJobSource.from_record(record)
        outcome = self.outcomes.pop(0) if self.outcomes else [(0, True), (0, True)]
        with open(Path(source.job_directory) / TASK_ROLLOUTS_FILE, "a", encoding="utf-8") as results:
            for group_index, assembled in outcome:
                result = {
                    "token": source.token,
                    "task": source.tasks[0][0],
                    "group_index": group_index,
                    "reward": 1.0 if assembled else None,
                    "assembled": assembled,
                    "error": "" if assembled else "the sandbox did not start",
                }
                results.write(json.dumps(result) + "\n")
        return f"rollout-data-{rollout_id}"

    def close(self) -> None:
        self.events.append("close")


def task(root: Path, name: str) -> TaskSample:
    path = root / name
    path.mkdir(parents=True)
    return TaskSample(path, metadata={"task": {"name": name}})


@pytest.fixture
def task_stack(tmp_path, monkeypatch):
    """A Slime component on the slime family, with a fake rollout group."""
    from tests.slime_backend import test_slime_component

    original = test_slime_component.worker_args

    def worker_args(tmp_path, **overrides):
        return original(
            tmp_path,
            **{
                "loss_family": "slime",
                "loss_type": "policy_loss",
                "compute_advantages_and_returns": True,
                "rollout_batch_size": 2,
                **overrides,
            },
        )

    monkeypatch.setattr(test_slime_component, "worker_args", worker_args)
    events: list[str] = []
    group = TaskRolloutGroup(events, tmp_path)
    stack = Stack(
        tmp_path,
        loss_family=FAMILY,
        rollout_group=group,
        rank_metrics=[{"rollout/raw_reward": 0.25}],
        hf_checkpoint=str(tmp_path / "model"),
    )
    group.root = stack.root
    yield stack, events, group
    stack.component.close()


def task_job(stack: Stack, root: Path, *, count: int = 3, save_interval: int | None = None) -> TrainingJob:
    tasks = tuple(task(root / "tasks", f"task-{index}") for index in range(count))
    job_parameters: dict[str, Any] = {"objective": OBJECTIVE, "harbor": HARBOR}
    if save_interval is not None:
        job_parameters["save_interval"] = save_interval
    return TrainingJob(
        "job-0", "slime", stack.start, job_data_ref(JobData("tasks", tasks)), job_parameters, scenario="s"
    )


@pytest.mark.unit
def test_a_task_job_runs_slimes_loop_over_its_tasks_and_reports_their_rewards(tmp_path, task_stack) -> None:
    stack, events, group = task_stack
    job = task_job(stack, tmp_path, count=3)
    stack.component.submit(job)
    status = finished(stack.component, "job-0")
    assert status.state == "finished", status.error
    # Three tasks in batches of two: two rollouts of Slime's own sequence, so the last wraps to the first task.
    assert events == [event for rollout in range(2) for event in ("sync", f"generate:{rollout}", "release")]
    assert stack.group.train_calls == [(rollout, f"rollout-data-{rollout}") for rollout in range(2)]
    assert status.metrics["task_rollouts"] == 2
    assert [Path(checkpoint.path).name for checkpoint in status.checkpoints] == ["rollout_1"]
    # The rollout manager reads how to run Harbor from the record, which holds no training method.
    record = group.records[0]
    assert [name for name, _ in record["tasks"]] == ["task-0", "task-1", "task-2"]
    assert (record["skip_rollouts"], record["harbor"]["environment"]) == (0, "modal")
    assert status.metrics["rollout_rewards"] == {"task-0": [1.0] * 4, "task-1": [], "task-2": []}
    assert (status.metrics["rollout_samples"], status.metrics["removed_rollouts"]) == (4, 0)


@pytest.mark.unit
def test_a_group_left_with_two_samples_trains_with_the_others_removed(tmp_path, task_stack) -> None:
    stack, _, group = task_stack
    group.outcomes = [[(0, True), (0, True), (1, False), (1, True)]]
    stack.component.submit(task_job(stack, tmp_path, count=2))
    status = finished(stack.component, "job-0")
    assert status.state == "finished", status.error
    assert (status.metrics["rollout_samples"], status.metrics["removed_rollouts"]) == (3, 1)


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome",
    [
        [(0, False), (0, False)],
        # A group of one sample has advantage 0 with Slime's group baseline.
        [(0, True), (0, False), (1, True), (1, False)],
    ],
)
def test_a_rollout_with_no_group_to_train_fails_the_job_before_the_workers_train(
    tmp_path, task_stack, outcome
) -> None:
    stack, events, group = task_stack
    group.outcomes = [outcome]
    stack.component.submit(task_job(stack, tmp_path, count=2))
    status = finished(stack.component, "job-0")
    assert status.state == "failed" and "the sandbox did not start" in status.error
    assert stack.group.train_calls == [] and not status.checkpoints
    # The engines released their memory first.
    assert events == ["sync", "generate:0", "release"]


@pytest.mark.unit
def test_a_resumed_task_job_skips_the_rollouts_it_saved(tmp_path, task_stack, monkeypatch) -> None:
    stack, _, group = task_stack
    job = task_job(stack, tmp_path, count=4, save_interval=1)
    directory = stack.component.job_directory("job-0")
    directory.mkdir(parents=True)
    # The first driver stops after its first checkpoint, one of the job's two rollouts.
    monkeypatch.setattr(stack.component, "stopping", lambda directory: len(stack.group.saved) == 1)
    stack.component.run(job, directory)
    assert [checkpoint.step for checkpoint in stack.component.saved(directory)] == [1]
    first_token = group.records[0]["token"]
    monkeypatch.setattr(stack.component, "stopping", lambda directory: False)
    group.records.clear()
    stack.group.train_calls.clear()
    metrics = stack.component.run(job, directory)
    # The data source skips the saved rollout's tasks, under the new driver's token.
    assert [record["skip_rollouts"] for record in group.records] == [1]
    assert group.records[0]["token"] != first_token
    assert [rollout for rollout, _ in stack.group.train_calls] == [1]
    assert [checkpoint.step for checkpoint in stack.component.saved(directory)] == [1, 2]
    assert metrics["rollout_samples"] == 4, "both drivers' rollouts report"


@pytest.mark.unit
def test_a_task_job_is_checked_at_submit(tmp_path, task_stack) -> None:
    stack, events, _ = task_stack
    add = task(tmp_path / "a", "add")
    method = {"objective": OBJECTIVE}
    for items, job_parameters, message in (
        ((add,), method, "needs the harbor parameter"),
        ((add, *batch_for(0).samples), {**method, "harbor": HARBOR}, "both trajectories and tasks"),
        ((add, task(tmp_path / "b", "add")), {**method, "harbor": HARBOR}, "distinct names"),
        ((add,), {**method, "harbor": {"rollouts_per_task": 2}}, r"takes no \['rollouts_per_task'\]"),
        ((add,), {"objective": "sft", "harbor": HARBOR}, "needs the 'slime' loss family, not 'sft'"),
        ((add,), {**method, "harbor": HARBOR, "scheduling": {"shuffle": True}}, "remove learning_rate_schedule"),
        (
            (add,),
            {
                **method,
                "harbor": HARBOR,
                "learning_rate_schedule": {"name": "warmup", "peak_learning_rate": 1e-5, "warmup_steps": 2},
            },
            "remove learning_rate_schedule",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            stack.component.submit(
                TrainingJob(
                    "j",
                    "slime",
                    stack.component.base_start,
                    job_data_ref(JobData("d", items)),
                    job_parameters,
                    scenario="s",
                )
            )
    assert events == []


@pytest.mark.unit
def test_slimes_options_that_cannot_work_refuse_a_task_job_at_submit(tmp_path, task_stack) -> None:
    stack, events, _ = task_stack
    stack.args.group_rm = True
    with pytest.raises(ValueError, match="--group-rm"):
        stack.component.submit(task_job(stack, tmp_path))
    assert events == []


class ImportanceSamplingObjective(SlimeObjective):
    """A method that a deployment provides, on the importance_sampling family."""

    name = "test-importance-sampling"
    loss_family = "importance_sampling"

    def prepare(self, data, state):
        raise AssertionError("a checked job prepares nothing")


@pytest.mark.unit
def test_a_job_names_an_objective_whose_loss_family_slime_trains(tmp_path) -> None:
    stack = Stack(tmp_path)
    try:
        # A provided method's family, importance_sampling, is Slime's too. It trains trajectories.
        stack.component.check_job(
            TrainingJob(
                "j",
                "slime",
                stack.component.base_start,
                job_data_ref(batch_for(0)),
                parameters(f"{__name__}:ImportanceSamplingObjective"),
                scenario="s",
            )
        )

        class Elsewhere(SlimeObjective):
            name = "elsewhere"
            loss_family = "only-on-another-backend"

            def prepare(self, data, state):
                raise AssertionError("a refused job prepares nothing")

        register_slime_objective(Elsewhere())
        try:
            with pytest.raises(ValueError, match="no loss family 'only-on-another-backend'"):
                stack.component.submit(
                    TrainingJob(
                        "j",
                        "slime",
                        stack.component.base_start,
                        job_data_ref(batch_for(0)),
                        parameters("elsewhere"),
                        scenario="s",
                    )
                )
        finally:
            unregister_slime_objective("elsewhere")
    finally:
        stack.component.close()


class FakeManager:
    """The rollout manager's Ray handle: each remote call records its name."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __getattr__(self, name: str) -> Any:
        manager = self

        class Method:
            def remote(self, *args: Any) -> str:
                manager.events.append(name)
                return name

        return Method()


class ActorGroup:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def update_weights(self) -> None:
        self.events.append("update_weights")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("colocated", "synced", "released"),
    [
        # Colocated engines take back their weights' memory, then their KV cache, around Slime's weight transfer.
        (True, ["onload_weights", "update_weights", "onload_kv"], ["offload"]),
        # Engines on their own GPUs keep their memory.
        (False, ["update_weights"], []),
    ],
)
def test_the_rollout_group_holds_gpu_memory_only_between_sync_and_release(monkeypatch, colocated, synced, released):
    events: list[str] = []
    monkeypatch.setattr(rollout_group_module.ray, "get", lambda value: value)
    group = SlimeRolloutGroup(
        SimpleNamespace(manager=FakeManager(events)), colocated=colocated, router_url="http://router:3000"
    )
    group.sync(ActorGroup(events))
    assert events == synced
    events.clear()
    group.release()
    assert events == released


@pytest.mark.unit
def test_closing_the_rollout_group_detaches_the_driver_and_keeps_the_session() -> None:
    closes: list[bool] = []
    startup = SimpleNamespace(
        manager=None,
        placements={"actor": ("pg", [0, 1], [0, 1]), "critic": None, "rollout": ("pg", [0, 1], [0, 1])},
        close=lambda *, failed: closes.append(failed),
    )
    group = SlimeRolloutGroup(startup, colocated=True, router_url="http://router:3000")
    # The workers train on the session's GPUs.
    assert group.placements["actor"] == GpuBundles("pg", [0, 1], [0, 1]) and group.placements["critic"] is None
    group.close()
    # Slime's failure cleanup detaches the driver; its success cleanup would dispose the session.
    assert closes == [True]


class CarryingObjective(SlimeObjective):
    """A method on the slime family that centers by SPADE's rule and carries delayed samples."""

    name = "test-carrying"
    loss_family = "slime"
    rollout_advantages = GroupAdvantageRule(divide_by_std=True, std="population", epsilon=1e-8)
    carries_samples = True

    def prepare(self, data, state):
        raise AssertionError("a task job prepares nothing")


CARRYING = f"{__name__}:CarryingObjective"
#: Slime's options that a task job that carries samples needs.
CARRYING_OPTIONS = {
    "rollout_batch_size": 4,
    "global_batch_size": 7,
    "use_tis": True,
    "normalize_advantages": False,
    "custom_advantage_function_path": None,
    "use_dynamic_batch_size": True,
    "use_score_centering": False,
    "custom_tis_function_path": None,
    "use_rollout_logprobs": False,
    "rollout_temperature": 0.6,
    "rollout_top_k": -1,
}


@pytest.mark.unit
def test_a_task_job_that_carries_samples_leads_its_first_rollout_with_them(tmp_path, task_stack) -> None:
    stack, _, group = task_stack
    for name, value in CARRYING_OPTIONS.items():
        setattr(stack.args, name, value)
    tasks = tuple(task(tmp_path / "tasks", f"task-{index}") for index in range(3))
    data = JobData("tasks", tasks, delayed=(delayed_sample(2.0), delayed_sample(-2.0), delayed_sample(0.5)))
    job = TrainingJob(
        "job-0", "slime", stack.start, job_data_ref(data), {"objective": CARRYING, "harbor": HARBOR}, scenario="s"
    )
    stack.component.submit(job)
    status = finished(stack.component, "job-0")
    assert status.state == "finished", status.error

    # Three delayed samples in groups of two take two prompts: with three tasks, two rollouts of four prompts.
    assert status.metrics["task_rollouts"] == 2 and status.metrics["delayed_samples"] == 3
    record = group.records[0]
    assert (record["carries_samples"], record["delayed_count"]) == (True, 3)
    assert record["rollout_advantages"] == {"divide_by_std": True, "std": "population", "epsilon": 1e-8}
    rows = read_delayed_rows(Path(record["job_directory"]) / DELAYED_ROWS_FILE)
    assert [advantage for _, advantage in rows] == [2.0, -2.0, 0.5]
    # Slime's raw reward counts the delayed rows, so retention ranks the checkpoint by the task rollouts' rewards.
    stored = json.loads(stack.storage._record_path(1).read_text())
    assert stored["reward"] == 1.0


@pytest.mark.unit
def test_a_resumed_carrying_job_keeps_the_delayed_samples_that_its_first_driver_chose(
    tmp_path, task_stack, monkeypatch
) -> None:
    stack, _, group = task_stack
    for name, value in CARRYING_OPTIONS.items():
        setattr(stack.args, name, value)
    tasks = tuple(task(tmp_path / "tasks", f"task-{index}") for index in range(4))
    data = JobData("tasks", tasks, delayed=tuple(delayed_sample(float(index)) for index in range(6)))
    parameters = {"objective": CARRYING, "harbor": HARBOR, "save_interval": 1}
    job = TrainingJob("job-0", "slime", stack.start, job_data_ref(data), parameters, scenario="s")
    directory = stack.component.job_directory("job-0")
    directory.mkdir(parents=True)
    # Six samples in groups of two lead the first rollout of four prompts with one task; the first driver stops
    # after that rollout.
    monkeypatch.setattr(stack.component, "stopping", lambda directory: len(stack.group.saved) == 1)
    stack.component.run(job, directory)
    assert group.records[0]["delayed_count"] == 6
    # The job resumes with a step that would keep only three samples; it keeps the six, so the task offsets hold.
    stack.args.global_batch_size = 3
    monkeypatch.setattr(stack.component, "stopping", lambda directory: False)
    group.records.clear()
    stack.component.run(job, directory)
    assert [(record["skip_rollouts"], record["delayed_count"]) for record in group.records] == [(1, 6)]


@pytest.mark.unit
def test_a_carrying_job_without_tis_is_refused_and_another_objective_leaves_the_samples_out(
    tmp_path, task_stack
) -> None:
    stack, events, _ = task_stack
    for name, value in {**CARRYING_OPTIONS, "use_tis": False}.items():
        setattr(stack.args, name, value)
    data = JobData("tasks", (task(tmp_path / "tasks", "add"),), delayed=(delayed_sample(),))

    def job(job_id: str, objective: str) -> TrainingJob:
        parameters = {"objective": objective, "harbor": HARBOR}
        return TrainingJob(job_id, "slime", stack.start, job_data_ref(data), parameters, scenario="s")

    with pytest.raises(ValueError, match="without --use-tis"):
        stack.component.submit(job("refused", CARRYING))
    assert events == []
    # An objective that carries no samples trains the tasks and leaves the delayed samples out, so the stage, which
    # builds the same job data after a refusal, cannot stop the scenario's training.
    stack.component.submit(job("trained", OBJECTIVE))
    status = finished(stack.component, "trained")
    assert status.state == "finished", status.error
    assert status.metrics["delayed_left_out"] == {"objective": 1}
