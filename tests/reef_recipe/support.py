"""Shared pieces of the generated-task tests: a scripted generator, an inline worker, and a pipeline that commits each
job's submission, with the helpers that build and train it."""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Self

from tests.support.samples import open_sample_store
from tests.support.task_service_stubs import StandInGenerator, task_for

from reef.artifact import SnapshotRef
from reef.core.job_data import JobOutcome
from reef.core.tasks import HarborTask
from reef.execution.jobs import BackgroundJobRunner, BackgroundJobWorker, Failed, SupportsReceipt
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskJobData
from reef.recipe.generated_tasks.generator import (
    GeneratedTaskResult,
    GeneratorState,
    TaskGenerator,
    TaskValidationResult,
)
from reef.recipe.generated_tasks.processor import GeneratedTaskSamples, GeneratedTaskSettings
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.sample2job import Sample2JobSettings
from reef.storage.commits import CommitRecord
from reef.storage.samples import SampleStore
from reef.storage.scenario import ScenarioStore
from reef.train.pipeline import TrainingPipeline

DESCRIPTION = "shell tasks with hidden state under /var and /etc"


class ScriptedGenerator(TaskGenerator):
    """A task generator that returns scripted tasks, refuses scripted names, and keeps what it was sent."""

    built: list[dict[str, object]] = []

    def __init__(
        self,
        rounds: list[tuple[HarborTask, ...]] | None = None,
        *,
        invalid: Mapping[str, str] | None = None,
        failing: str | None = None,
        failing_results: bool = False,
    ) -> None:
        self.rounds = list(rounds) if rounds is not None else []
        self.invalid = dict(invalid or {})
        self.failing = failing
        self.failing_results = failing_results
        self.requests: list[TaskGenerationRequest] = []
        self.validated: list[str] = []
        self.results: list[tuple[GeneratedTaskResult, ...]] = []
        self.result_generations: list[int] = []

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str], state: GeneratorState
    ) -> Self:
        cls.built.append({"config": dict(config), "scenario": scenario, "environ": dict(environ), "state": state})
        return cls()

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        self.requests.append(request)
        if not self.rounds:
            return ()
        return self.rounds.pop(0)

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        self.validated.append(task.name)
        if task.name == self.failing:
            raise RuntimeError("the checks could not run")
        error = self.invalid.get(task.name)
        return TaskValidationResult((error,) if error else ())

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        self.results.append(results)
        self.result_generations.append(generation)
        if self.failing_results:
            raise RuntimeError("the generator lost its state directory")


def generated(round_number: int, *indices: int) -> tuple[HarborTask, ...]:
    """Tasks of one round."""
    return tuple(
        task_for(f"harbor-{round_number:05d}-{index:03d}", f"designer-{index}", 9000 + 10 * round_number + index)
        for index in indices
    )


class Unbound(BackgroundJobRunner):
    """The runner of a test worker that runs rounds itself. The real worker thread never calls it."""

    async def run(self, job: SupportsReceipt) -> SupportsReceipt:
        raise RuntimeError("no processor bound")


class InlineWorker(BackgroundJobWorker):
    """Runs each round the moment it is submitted, on this thread, so a test reads the outcome right away."""

    def __init__(self) -> None:
        super().__init__(Unbound(), thread_name="test-rounds", concurrency=1)
        self.generation: BackgroundJobRunner | None = None
        self.submitted: list[SupportsReceipt] = []
        self.outcomes: list[SupportsReceipt] = []

    def submit(self, job: SupportsReceipt) -> bool:
        self.submitted.append(job)
        if self.generation is None:
            return False
        try:
            self.outcomes.append(asyncio.run(self.generation.run(job)))
        except Exception:
            self.outcomes.append(Failed(job.receipt))
        return True

    def poll(self) -> list[SupportsReceipt]:
        outcomes, self.outcomes = self.outcomes, []
        return outcomes

    def close(self) -> None:
        self.generation = None


class CommittingPipeline(TrainingPipeline):
    """A pipeline whose ``acknowledge`` also commits what the job data consumed, as the trainer's commit of a
    job's submission does: the commit counts the stored samples trained. Each look also makes the samples first,
    as the samples service does before the trainer reads the sample store."""

    def __init__(self, context: PipelineContext, session: ScenarioStore, **stages: Any) -> None:
        super().__init__(context, **stages)
        self.session = session

    def advance(self) -> None:
        self.produce()
        super().advance()

    def acknowledge(self, data_id: str) -> frozenset[str]:
        consumed = super().acknowledge(data_id)
        commit_number = len(self.session.history()) + 1
        commit = CommitRecord(
            scenario="tasks",
            commit_number=commit_number,
            snapshot=SnapshotRef(f"snapshot:{commit_number}", "base"),
            checkpoint=True,
            algorithm_state={},
            high_water_sequence=commit_number,
            high_water_offset=commit_number,
            consumed_ids=consumed,
            recorded_at=1000.0 + commit_number,
        )
        self.session.append_commit(expected_commit_number=commit_number - 1, commit=commit)
        return consumed


def stage_settings(config: Mapping[str, object]) -> tuple[GeneratedTaskSettings, Sample2JobSettings]:
    """Split one flat config, as a test writes it, into the settings of the two stages."""
    sample_names = {field.name for field in dataclasses.fields(GeneratedTaskSettings)}
    batch_names = {field.name for field in dataclasses.fields(Sample2JobSettings)}
    unknown = sorted(set(config) - sample_names - batch_names)
    if unknown:
        raise ValueError(f"unknown stage settings: {', '.join(unknown)}")
    return (
        GeneratedTaskSettings(**{key: value for key, value in config.items() if key in sample_names}),
        Sample2JobSettings(**{key: value for key, value in config.items() if key in batch_names}),
    )


def job_data_stage(settings: Sample2JobSettings, context: PipelineContext) -> GeneratedTaskJobData:
    return GeneratedTaskJobData(context, buffer_size=settings.buffer_size, job_samples=settings.job_samples)


def scenario_session(tmp_path: Path, scenario: str = "tasks") -> tuple[ScenarioStore, SampleStore]:
    """A scenario session in a SQLite storage under ``tmp_path``, and its sample store in a data store."""
    return open_sample_store(tmp_path / "records", scenario)


def rounds(
    tmp_path: Path,
    task_generator: ScriptedGenerator,
    service: StandInGenerator | None = None,
    *,
    worker: InlineWorker | None = None,
    **config: object,
) -> tuple[TrainingPipeline, StandInGenerator]:
    """A generated-task pipeline; ``worker`` runs the generation and delivery jobs, inline by default."""
    service = service if service is not None else StandInGenerator(tmp_path / "tasks")
    worker = worker if worker is not None else InlineWorker()
    session, samples = scenario_session(tmp_path)
    context = PipelineContext(
        "tasks",
        task_storage=session.tasks,
        samples=samples.for_trainer(""),
        training_mode=str(config.pop("training_mode", "auto")),
    )
    sample_settings, batch_settings = stage_settings(
        {"buffer_size": 2, "rounds": 3, "description": DESCRIPTION, **config}
    )
    generation = GeneratedTaskSamples(
        context, sample_settings, task_generator=task_generator, generator_service=service, worker=worker
    )
    built = CommittingPipeline(
        context,
        session,
        record2sample=generation,
        sample2job=job_data_stage(batch_settings, context),
    )
    worker.generation = generation
    return built, service


def handed_out_names(p: TrainingPipeline) -> list[str]:
    """The names of the tasks that the next job's data takes: in ``manual``, the oldest training request's."""
    if p.training_mode == "auto":
        entries = p.auto_entries()
    else:
        request = next(iter(p.training_requests.values()), None)
        entries = () if request is None else p.request_entries(request)
    return [sample.metadata["task"]["name"] for _, sample in entries]


def job_size(p: TrainingPipeline) -> int:
    """The tasks that one task job takes: the ``job_samples`` of the pipeline's ``GeneratedTaskJobData``."""
    size = p.sample2job.job_samples
    assert size is not None
    return size


def finished_job(job_id: str, rewards: Mapping[str, Sequence[float]]) -> JobOutcome:
    """The end of the finished task job ``job_id`` whose rollouts of each task earned ``rewards[name]``."""
    rollout_rewards = {name: list(values) for name, values in rewards.items()}
    return JobOutcome(job_id, "tinker", "finished", {"rollout_rewards": rollout_rewards, "rollout_samples": 4})


def submit_job(p: TrainingPipeline) -> list[str]:
    """Take the handed-out task job as the trainer does, commit its submission, and return its task names. The
    submission's commit counts the tasks trained."""
    data = p.build_job_data(job_size(p))
    p.acknowledge(data.data_id)
    return [item.metadata["task"]["name"] for item in data.samples]


def train_job(
    p: TrainingPipeline,
    rewards: Mapping[str, Sequence[float]] | None = None,
    *,
    error: str | None = None,
    job_id: str = "job",
) -> list[str]:
    """Train the handed-out task job as the trainer and the training component do, and return its task names.

    The trainer commits the job's submission (``submit_job``), then the job's end. Each task's rollouts earn
    ``rewards[name]``, 1.0 and 0.0 by default. A job with ``error`` failed and reports no metrics.
    """
    names = submit_job(p)
    if error is None:
        p.job_ended(finished_job(job_id, {name: (rewards or {}).get(name, (1.0, 0.0)) for name in names}))
    else:
        p.job_ended(JobOutcome(job_id, "tinker", "failed", error=error))
    return names
