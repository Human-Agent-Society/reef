"""The training status that ``GET /reef/status`` reports, and when a local training worker may sleep.

The status tests bind a scenario double to the dispatcher's registry, so each
one checks one status field. The wait tests pin when a worker looks again
without a new record: a running training job saves checkpoints, and a
background job, such as a judgment, can make a batch ready. A scenario
created with no record starts its worker when its pipeline starts such a job.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from typing import Literal

import pytest
from tests.support.policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent, runtime_bindings
from tests.support.samples import made_task
from tests.support.threshold_processor import ThresholdJobData
from tests.support.trainers import OneTrainerBuilder

from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core.job_data import JobOutcome
from reef.dispatcher import Dispatcher
from reef.dispatcher.errors import TrainingErrors, local_error_source
from reef.dispatcher.local_workers import POLL_SECONDS, LocalBackendWorker, LocalTrainingWorkers
from reef.pipeline_context import PipelineContext
from reef.recipe import CompositeRecipe, Recipe
from reef.record2sample import NoSamples
from reef.storage.samples import stage_item_id
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.jobs import JobLoop, JobLoopState, RunningJob
from reef.train.pipeline import TrainingPipeline
from reef.train.trainer import Trainer
from reef.train.types import Sample

pytestmark = pytest.mark.unit


def scenario_double(
    *,
    background_jobs_running: bool = False,
    job_running: bool = False,
    job_data_ready: bool = False,
    pipeline_status: dict | None = None,
    runtime=None,
    last_training_commit: dict | None = None,
    job_loop=None,
    state: dict | None = None,
) -> SimpleNamespace:
    """A scenario with one trainer; it carries what the status, the local workers and ``Dispatcher.close`` read."""
    pipeline = SimpleNamespace(
        background_jobs_running=lambda: background_jobs_running,
        training_mode="auto",
        supported_training_modes=frozenset({"auto"}),
    )
    trainer = SimpleNamespace(
        pipeline=pipeline,
        trains=True,
        job_running=job_running,
        needs_polling=lambda: job_running or job_data_ready or background_jobs_running,
        job_loop=job_loop,
        state={} if state is None else state,
        job_data_ready=lambda: job_data_ready,
        pipeline_status=lambda: dict(pipeline_status or {}),
    )
    return SimpleNamespace(
        trainer=trainer,
        trainers=(SimpleNamespace(component=None, trainer=trainer),),
        trainer_for=lambda component: trainer,
        runtime=runtime,
        commit_status={"commit_number": 0, "last_training_commit": last_training_commit},
        artifact_status=lambda: {"weights": {"active": "base", "training_start": "base"}},
        served_artifact_id=lambda: "base",
        close=lambda: None,
    )


def registry_double(scenario: SimpleNamespace, experiment_reports=None) -> SimpleNamespace:
    """A registry of ``scenario`` alone; ``experiment_reports`` is what ``Dispatcher.close`` closes."""
    return SimpleNamespace(
        training_status_scenario_names=("s",),
        experiment_reports=experiment_reports,
        get_optional=lambda name: scenario if name == "s" else None,
        loaded_scenarios=lambda: (scenario,),
        preload_errors={},
    )


def bound_dispatcher(scenario: SimpleNamespace) -> Dispatcher:
    root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
    initial = root / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=root / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    dispatcher.registry = registry_double(scenario, dispatcher.registry.experiment_reports)
    dispatcher.local_workers.registry = dispatcher.registry
    dispatcher.training_status.registry = dispatcher.registry
    return dispatcher


def test_status_exposes_the_ready_state() -> None:
    dispatcher = bound_dispatcher(scenario_double(job_data_ready=True))

    assert dispatcher.build_training_status()["scenarios"]["s"]["job_data_ready"] is True


def test_status_exposes_the_last_training_commit() -> None:
    committed = {
        "commit_number": 3,
        "recorded_at": 1_756_400_000.0,
        "metrics": {"skipped": "no proposal", "selection": {"reason": "candidate lost"}},
    }
    dispatcher = bound_dispatcher(scenario_double(last_training_commit=committed))

    assert dispatcher.build_training_status()["scenarios"]["s"]["last_training_commit"] == committed


def test_status_names_the_served_artifact_and_the_inference_admission() -> None:
    class Runtime:
        open = False
        per_scenario_adapters = False

        @property
        def inference_admission_status(self):
            return {"open": self.open, "active": 0}

    runtime = Runtime()
    scenario = scenario_double(runtime=runtime)
    dispatcher = bound_dispatcher(scenario)

    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["served_artifact_id"] == "base"
    assert status["inference_admission"] == {"open": False, "active": 0}

    scenario.served_artifact_id = lambda: "trained"
    runtime.open = True
    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["served_artifact_id"] == "trained"
    assert status["inference_admission"] == {"open": True, "active": 0}


def test_status_exposes_terminal_processor_outcomes() -> None:
    outcome = {"failed_steps": [{"step": 1, "reason": "mixed_release_ids", "release_ids": ["old", "new"]}]}
    dispatcher = bound_dispatcher(scenario_double(pipeline_status=outcome))

    assert dispatcher.build_training_status()["scenarios"]["s"]["processor"] == outcome


def test_status_reports_the_running_job_and_the_finished_jobs_of_a_job_trainer() -> None:
    finished = JobOutcome("job-2", "stub", "finished")
    jobs = (
        RunningJob("job-3", "stub:base", checkpoints_handled=1),
        RunningJob("job-4", "stub:other", train_from="a1"),
    )
    running = JobLoopState("stub:base", jobs, finished_jobs=2, last_job=finished)
    loop = JobLoop(StubTrainingComponent(), scenario="s")
    dispatcher = bound_dispatcher(scenario_double(job_loop=loop, state=running.to_dict()))

    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["training_jobs"] == [{**job.to_dict(), "backend": "stub"} for job in jobs]
    assert status["training_jobs"][1] == {
        "job_id": "job-4",
        "start": "stub:other",
        "checkpoints_handled": 0,
        "train_from": "a1",
        "backend": "stub",
    }
    assert status["finished_jobs"] == 2

    dispatcher.registry.get_optional("s").trainer.state = JobLoopState(
        "stub:base", finished_jobs=3, last_job=finished
    ).to_dict()
    status = dispatcher.build_training_status()["scenarios"]["s"]
    assert status["training_jobs"] == []
    assert status["finished_jobs"] == 3


class ResidencyRuntime(StubInferenceRuntime):
    """A runtime that reports ``residency`` as its adapter residency, or raises ``failure`` when it is asked."""

    def __init__(self, residency: Mapping[str, int] | None = None, failure: Exception | None = None) -> None:
        super().__init__()
        self.residency = residency
        self.failure = failure

    def adapter_residency_status(self) -> Mapping[str, int] | None:
        if self.failure is not None:
            raise self.failure
        return self.residency


def serving_status(recipe: Recipe, tmp_path: Path) -> Mapping[str, object]:
    """The ``serving`` block of the status that a dispatcher of ``recipe`` builds."""
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
        coordination=memory_coordination(),
    )
    try:
        return dispatcher.build_training_status()["serving"]
    finally:
        dispatcher.close()


def test_status_reports_the_adapter_residency_that_the_runtime_of_the_weights_reports(tmp_path: Path) -> None:
    residency = {"resident": 1, "capacity": 2}
    weights = TestPolicyRecipe(**runtime_bindings(runtime=ResidencyRuntime(residency)))

    assert serving_status(weights, tmp_path) == {"test_policy": {"adapters": residency}}
    # A composite holds the block under the name of its weight component.
    composite = CompositeRecipe(components={"records": Recipe(), "policy": weights})
    assert serving_status(composite, tmp_path) == {"composite": {"policy": {"adapters": residency}}}
    # A runtime that reports no residency, and a recipe that trains no weights, add nothing.
    assert serving_status(TestPolicyRecipe(**runtime_bindings()), tmp_path) == {}
    assert serving_status(Recipe(runtime=ResidencyRuntime(residency)), tmp_path) == {}
    assert serving_status(CompositeRecipe(components={"records": Recipe(), "other": Recipe()}), tmp_path) == {}
    # A runtime that cannot answer leaves the rest of the status readable.
    failing = TestPolicyRecipe(**runtime_bindings(runtime=ResidencyRuntime(failure=RuntimeError("engines are down"))))
    assert serving_status(failing, tmp_path) == {"test_policy": {"error": "RuntimeError: engines are down"}}


def test_status_of_a_trainer_without_a_job_loop_has_no_job_fields() -> None:
    status = bound_dispatcher(scenario_double()).build_training_status()["scenarios"]["s"]

    assert "training_jobs" not in status
    assert "finished_jobs" not in status


def test_status_reports_build_failure_instead_of_raising() -> None:
    dispatcher = bound_dispatcher(scenario_double())

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher.registry.get_optional("s").trainer.job_data_ready = fail

    status = dispatcher.build_training_status()

    assert status["scenarios"] == {}
    assert status["error"] == "s: RuntimeError: processor status is unavailable"

    dispatcher.registry.get_optional("s").trainer.job_data_ready = lambda: False
    recovered = dispatcher.build_training_status()
    assert recovered["error"] is None
    assert "s" in recovered["scenarios"]


def test_status_reports_training_and_build_failures() -> None:
    dispatcher = bound_dispatcher(scenario_double())
    dispatcher.errors.record("s", "RuntimeError: backend failed", local_error_source(None))

    def fail() -> bool:
        raise RuntimeError("processor status is unavailable")

    dispatcher.registry.get_optional("s").trainer.job_data_ready = fail

    assert dispatcher.build_training_status()["error"] == (
        "s: RuntimeError: backend failed\ns: RuntimeError: processor status is unavailable"
    )


class RecordingEvent(Event):
    """An event whose ``wait`` returns at once and records the timeout it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.timeouts: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.timeouts.append(timeout)
        return True


def worker_wait_timeout(scenario: SimpleNamespace) -> float | None:
    """The timeout that the worker of ``scenario``'s trainer waits with."""
    workers = LocalTrainingWorkers(registry_double(scenario), TrainingErrors(), held=False)
    ready = RecordingEvent()
    workers.workers[("s", None)] = LocalBackendWorker(ready, Thread(target=print))
    assert workers.wait("s", None, ready) is True
    return ready.timeouts[-1]


def test_an_idle_worker_sleeps_until_the_next_record() -> None:
    assert worker_wait_timeout(scenario_double()) is None


def test_a_worker_with_a_running_job_polls_for_its_checkpoints() -> None:
    assert worker_wait_timeout(scenario_double(job_running=True)) == POLL_SECONDS


def test_a_worker_with_running_background_jobs_polls_on_a_bounded_interval() -> None:
    # A judgment that lands makes a batch ready without a new record waking the worker:
    # sleeping without a timeout stalls training on a quiet workload.
    timeout = worker_wait_timeout(scenario_double(background_jobs_running=True))

    assert timeout is not None and timeout > 0


class BackgroundStep(NoSamples):
    """Run one step from no record, as the rounds of generated tasks do, and make the step's one task.

    The first look starts the step. The step runs until a look after ``finished`` is set; that look ends it, and
    the stage then makes the step's one task from no record set.
    """

    def __init__(self, finished: Event) -> None:
        super().__init__()
        self.finished = finished
        self.step: Literal["not started", "running", "ended"] = "not started"
        self.made = False

    def advance(self, now: float) -> None:
        super().advance(now)
        if self.step == "not started":
            self.step = "running"
        elif self.step == "running" and self.finished.is_set():
            self.step = "ended"

    def background_jobs_running(self) -> bool:
        return self.step == "running"

    def made_samples(self) -> tuple[tuple[str, Sample], ...]:
        if self.made or self.step != "ended":
            return ()
        self.made = True
        return (("generated-task", made_task("generated-task")),)


@dataclass(frozen=True, kw_only=True)
class BackgroundStepRecipe(TestPolicyRecipe):
    """A weights recipe that trains the task of a ``BackgroundStep``; ``step_finished`` ends the step."""

    step_finished: Event = field(default_factory=Event, compare=False)

    def background_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        return TrainingPipeline(
            context,
            record2sample=BackgroundStep(self.step_finished),
            sample2job=ThresholdJobData(context, None, 1),
        )


class BackgroundStepTrainers(OneTrainerBuilder):
    """Builds the weight trainer of ``recipe``, which trains the task of a ``BackgroundStep``."""

    def __init__(self, recipe: BackgroundStepRecipe) -> None:
        self.recipe = recipe

    def build_trainer(
        self,
        spec,
        scenario,
        records,
        *,
        algorithm_state,
        experiment_logger,
        task_storage,
        artifacts,
        samples,
        root,
    ):
        return Trainer.build(
            scenario,
            records,
            pipeline_factory=self.recipe.background_pipeline,
            job_loop=JobLoop(spec.job_component, scenario=scenario),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            samples=samples,
            root=root,
        )


def background_step_dispatcher(tmp_path: Path) -> tuple[Dispatcher, BackgroundStepRecipe, StubTrainingComponent]:
    """A dispatcher of ``BackgroundStepRecipe`` with no scenario yet, and the component that runs its jobs."""
    initial = tmp_path / "initial"
    initial.mkdir()
    component = StubTrainingComponent(tmp_path / "checkpoints")
    recipe = BackgroundStepRecipe(**runtime_bindings(component))
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "agent-record"),
        coordination=memory_coordination(),
        trainer_builder=BackgroundStepTrainers(recipe),
    )
    return dispatcher, recipe, component


def assert_trains_the_generated_task(component: StubTrainingComponent) -> None:
    """Wait up to ten seconds for the one job, which trains the step's task: it passes the jobs queue, whose
    workers poll, so a loaded host takes longer than an idle one."""
    deadline = time.monotonic() + 10
    while not component.submitted and time.monotonic() < deadline:
        time.sleep(0.01)
    [job] = component.submitted
    assert [sample.task_path.name for sample in job.read_data().samples] == [
        stage_item_id("generated", "generated-task")
    ]


def test_a_scenario_created_after_start_trains_the_job_data_its_pipeline_makes_without_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What POST /reef/scenarios does for a recipe that needs no record, such as generated tasks: no record or
    # upload ever wakes the worker, so the worker must start with the scenario and poll while the step runs.
    monkeypatch.setattr("reef.dispatcher.local_workers.POLL_SECONDS", 0.01)
    dispatcher, recipe, component = background_step_dispatcher(tmp_path)
    try:
        dispatcher.get_or_create_scenario("generated", allow_implicit_creation=True)
        recipe.step_finished.set()
        assert_trains_the_generated_task(component)
    finally:
        dispatcher.close()


def test_a_status_read_leaves_a_finished_step_to_the_worker_which_trains_its_job_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A status read between the worker's last step and its wait does not advance the pipeline: the finished
    # step stays running until the worker's next cycle collects it, so the worker keeps polling and trains it.
    monkeypatch.setattr("reef.dispatcher.local_workers.POLL_SECONDS", 0.01)
    wait = LocalTrainingWorkers.wait
    waits = 0
    read_ready: list[bool] = []

    def wait_after_a_status_read(
        self: LocalTrainingWorkers, scenario: str, component: str | None, ready: Event
    ) -> bool:
        nonlocal waits
        waits += 1
        if waits == 2:
            # The worker ran its first step, which found the step running; no record wakes it again.
            ready.clear()
            recipe.step_finished.set()
            read_ready.append(dispatcher.build_training_status()["scenarios"][scenario]["job_data_ready"])
        return wait(self, scenario, component, ready)

    monkeypatch.setattr(LocalTrainingWorkers, "wait", wait_after_a_status_read)
    dispatcher, recipe, training_component = background_step_dispatcher(tmp_path)
    try:
        dispatcher.get_or_create_scenario("generated", allow_implicit_creation=True)
        assert_trains_the_generated_task(training_component)
        assert read_ready == [False]
    finally:
        dispatcher.close()
