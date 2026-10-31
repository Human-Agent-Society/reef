"""Shared setup for the tests of a scenario with one trainer per release component."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.artifact import ArtifactRef, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.activation import ActivationRule
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import AlwaysActivate
from reef.execution.jobs import SupportsReceipt
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.runtime.jobs import JobStatus, SavedCheckpoint, TrainingJob
from reef.scenario import Scenario
from reef.storage.samples import TrainerSamples
from reef.storage.sql_coordination import open_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.surface.weights import WeightLoader
from reef.train import ComponentTrainer, Trainer
from reef.train.jobs import JobLoop
from reef.train.pipeline import TrainingPipeline

WEIGHTS = "weights"
HARNESS = "harness"
from reef.execution.workers import SERVICE_COMPONENTS


class ComponentJobs(StubTrainingComponent):
    """The training component of one release component: its jobs run in ``submit``, on the trainer's own thread.

    Each job runs ``step``, which saves one checkpoint holding ``<component>.txt`` = ``<component> step <n>`` for the
    ``n``-th job, with the metrics ``{prepared}``. ``prepared`` counts the steps that started, a step that raised
    included; a step that raises records no job, so ``submit`` raises as a refused job does and the trainer's cycle
    fails. ``reject_next`` ends the next job without a checkpoint, with ``selected: False``. ``hold_next`` makes the
    next checkpoint change the ``HELD`` kind, which the test recipe's activation rule does not activate. ``started``
    is the text of the last start that ``import_start`` took. A subclass blocks or fails a step in ``step``.
    """

    def __init__(self, component: str, artifact_dir: Path) -> None:
        super().__init__(artifact_dir, name=component, shares_scenarios=True, release_format=None)
        self.component = component
        self.artifact_dir = artifact_dir
        self.prepared = 0
        self.reject_next = False
        self.hold_next = False
        self.started: str | None = None

    def unblock(self) -> None:
        """Let go of any step the test holds, so the dispatcher can close."""

    def import_start(self, release: Path) -> str:
        self.started = (release / f"{self.component}.txt").read_text(encoding="utf-8")
        return super().import_start(release)

    def step(self, job: TrainingJob) -> JobStatus:
        """The job's end: a checkpoint of the component's file, or none when the step is rejected."""
        number = len(self.submitted) + 1
        if self.reject_next:
            # A rejected step publishes nothing: the job ends without a checkpoint.
            self.reject_next = False
            return JobStatus(job.job_id, "finished", metrics={"prepared": self.prepared, "selected": False})
        path = self.artifact_dir / self.component / uuid.uuid4().hex
        path.mkdir(parents=True)
        (path / f"{self.component}.txt").write_text(f"{self.component} step {number}", encoding="utf-8")
        checkpoint = SavedCheckpoint(
            0,
            str(path),
            str(path),
            {"prepared": self.prepared},
            frozenset({HELD}) if self.hold_next else frozenset(),
        )
        self.hold_next = False
        return JobStatus(job.job_id, "finished", (checkpoint,))

    def submit(self, job: TrainingJob) -> None:
        if job.job_id in self.statuses:
            return
        self.prepared += 1
        status = self.step(job)
        self.submitted.append(job)
        self.statuses[job.job_id] = status


def finish_job(scenario: Scenario, component: str, checkpoint_metrics: Mapping[str, Any] | None = None) -> None:
    """Commit the checkpoint and the end of ``component``'s running job, when its trainer's component is a
    ``ComponentJobs``, whose jobs end in ``submit``: the commits that follow the job's submission. ``checkpoint_metrics``
    join the checkpoint's commit. A job of another component runs in the training service, so it is left alone."""
    trainer = scenario.trainer_for(component)
    if not isinstance(trainer.job_loop_required().component, ComponentJobs):
        return
    while trainer.job_running:
        result = scenario.prepare_training_step(component)
        assert result is not None, "a job that ended in submit has its checkpoint and its end to commit"
        if checkpoint_metrics is not None and result.metrics["training_job"]["event"] == "checkpoint":
            result = trainer.add_commit_metrics(result, checkpoint_metrics)
        scenario.commit(result, component=component)


class WeightsJobComponent(StubTrainingComponent):
    """The stub training component whose ``n``-th job saves the weights tree ``weights.txt`` = ``weights job <n>``.

    The checkpoint also holds the stub's ``model.txt`` = ``w<n>``, the version
    that ``StubInferenceRuntime`` serves it under.
    """

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        checkpoints = super().train(job)
        (Path(checkpoints[-1].path) / f"{WEIGHTS}.txt").write_text(
            f"weights job {len(self.submitted)}", encoding="utf-8"
        )
        return checkpoints


@dataclass(frozen=True)
class Judgment(SupportsReceipt):
    receipt: str


class JudgedAtOnce:
    """A judge worker whose every judgment is ready at the next poll."""

    def __init__(self) -> None:
        self.done: list[Judgment] = []

    def submit(self, job: Judgment) -> bool:
        self.done.append(job)
        return True

    def poll(self) -> list[Judgment]:
        done, self.done = self.done, []
        return done

    def close(self) -> None:
        return None


class AutoOnlyThresholdJobData(ThresholdJobData):
    """The same batches, without ``manual``: the stage takes no instruction."""

    supported_training_modes = frozenset({"auto"})


#: The node kind of a held artifact: the test recipe's activation rule does not activate it.
HELD = "held"

#: How many artifacts behind the served one a row may be and still train. A test appends rows before the jobs that
#: it waits for move the served artifact, so its rows stay trainable until it ends.
MAX_STALENESS = 100


@dataclass(frozen=True)
class _TwoTrainerRecipe(Recipe):
    """A scenario serving weights and a harness tree, each evolved by its own training component."""

    backends: Mapping[str, ComponentJobs]
    auto_only_components: frozenset[str] = frozenset()

    def activation_rule(self) -> ActivationRule:
        return AlwaysActivate(frozenset({HELD}))

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                # A training component's weights load through the runtime, so train-from imports them as a start.
                WEIGHTS: (
                    ComponentSurface() if self.training_component is None else ComponentSurface(loader=WeightLoader())
                ),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )

    def pipeline_for(self, component: str, context: PipelineContext) -> TrainingPipeline:
        if component in self.auto_only_components:
            return threshold_pipeline(
                context, sample2job=AutoOnlyThresholdJobData(context, None, None), max_staleness=MAX_STALENESS
            )
        return threshold_pipeline(context, max_staleness=MAX_STALENESS)

    def build_trainers(
        self,
        scenario,
        records,
        *,
        surface,
        algorithm_states,
        experiment_logger=None,
        task_storage=None,
        releases=None,
        samples=None,
    ):
        def factory_for(component: str):
            return lambda context: self.pipeline_for(component, context)

        # Each trainer reads the samples as the component that its commits name; the only trainer names none.
        alone = len(self.backends) + (self.training_component is not None) == 1

        def samples_for(component: str) -> TrainerSamples | None:
            return None if samples is None else TrainerSamples(samples, "" if alone else component)

        trainers = tuple(
            ComponentTrainer(
                component,
                Trainer.build(
                    scenario,
                    records,
                    pipeline_factory=factory_for(component),
                    job_loop=JobLoop(backend, scenario=scenario),
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    default_job_samples=1,
                    samples=samples_for(component),
                ),
            )
            for component, backend in self.backends.items()
        )
        training = self.training_component
        if training is None:
            return trainers
        # The weights trainer submits jobs to the training component; it comes first, as the local weights trainer does.
        weights = Trainer.build(
            scenario,
            records,
            pipeline_factory=factory_for(WEIGHTS),
            job_loop=JobLoop(training, scenario=scenario),
            algorithm_state=algorithm_states.get(WEIGHTS),
            experiment_logger=experiment_logger,
            default_job_samples=1,
            samples=samples_for(WEIGHTS),
        )
        return (ComponentTrainer(WEIGHTS, weights), *trainers)


class _RecordingTracker(ExperimentTracker):
    """Keeps every training event the dispatcher records."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def bind_scenario(self, **kwargs):
        return NullExperimentLogger()

    def correlation_metrics(self, context):
        return {}

    def record(self, event):
        self.events.append(event)

    def record_train_from(self, event):
        pass

    def close(self):
        pass


def _records(step: int, artifact_id: str, scenario: str = "agent") -> tuple[AgentRecord, AgentRecord]:
    """One inference, answered by the artifact ``artifact_id``, and the report on it; rows of scenarios other than
    ``agent`` carry the scenario's name."""
    prefix = "" if scenario == "agent" else f"{scenario}-"
    inference = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2], "artifact_id": artifact_id},
        agent_record_id=f"{prefix}i{step}",
    )
    report = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [f"{prefix}i{step}"]},
        agent_record_id=f"{prefix}r{step}",
        references=(f"{prefix}i{step}",),
    )
    return inference, report


def _append_records(scenario: Scenario, *steps: int) -> None:
    """Append each step's rows, answered by the artifact that ``scenario`` serves now, to the store directly, so the
    test, not a worker, owns when a trainer prepares."""
    for step in steps:
        for record in _records(step, scenario.serving_artifact_id(), scenario.name):
            scenario.records.append(record)


def _scenario(dispatcher: Dispatcher, *steps: int, name: str = "agent") -> Scenario:
    """The scenario, created or loaded on first use, with each step's rows appended.

    Loading a scenario wakes its workers. The call that loads it stops them
    again before the rows land, so the test, not a worker, decides when a
    trainer prepares. An accepted record, ``local_workers.start`` or
    ``preload_scenarios`` starts them again.
    """
    loads = dispatcher.registry.get_optional(name) is None
    scenario = dispatcher.get_or_create_scenario(name)
    assert scenario is not None
    if loads:
        stop_workers(dispatcher, name)
    _append_records(scenario, *steps)
    return scenario


def stop_workers(dispatcher: Dispatcher, scenario: str) -> None:
    """Stop the workers of ``scenario`` and wait for them to end; a wake starts new ones."""
    workers = dispatcher.local_workers
    with workers.lock:
        running = [worker for key, worker in workers.workers.items() if key[0] == scenario]
    workers.remove(scenario)
    for worker in running:
        worker.thread.join(10)


def _seeded_repository(tmp_path: Path, components: tuple[str, ...] = (WEIGHTS, HARNESS)):
    """A repository whose base seeds one file per component, unless the test wrote its own base first."""
    initial = tmp_path / "initial"
    if not initial.exists():
        for component in components:
            (initial / component).mkdir(parents=True)
            (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def component_files(scenario: Scenario, ref: ArtifactRef) -> dict[str, str]:
    """The text of each component's ``<component>.txt`` in the release ``ref``."""
    head = scenario.repository.materialize(ref)
    return {
        component: (TextFileTree().read_files(head.component(component)) or {})[f"{component}.txt"]
        for component in (WEIGHTS, HARNESS)
    }


def _local_pair(tmp_path: Path) -> dict[str, ComponentJobs]:
    return {component: ComponentJobs(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}


def _dispatcher(
    tmp_path: Path,
    *,
    backends: dict[str, ComponentJobs] | None = None,
    recipe: Recipe | None = None,
    auto_only_components: frozenset[str] = frozenset(),
    training: StubTrainingComponent | None = None,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    experiment_tracker: ExperimentTracker | None = None,
    hold_local_cycles: bool = False,
    components: frozenset[str] = frozenset(SERVICE_COMPONENTS),
) -> tuple[Dispatcher, dict[str, ComponentJobs]]:
    """A dispatcher over the two trainer recipe of ``backends``, or over ``recipe`` when one is given. ``components``
    are the services that it runs; without ``samples``, only the test makes samples (``make_samples``)."""
    repository = _seeded_repository(tmp_path) if backend_factory is None else backend_factory
    if recipe is not None:
        backends = {} if backends is None else backends
    elif training is None:
        backends = _local_pair(tmp_path) if backends is None else backends
        recipe = _TwoTrainerRecipe(backends=backends, auto_only_components=auto_only_components)
    else:
        # A training component binds the scenario: a process trains one scenario on it. It trains the weights; the
        # harness keeps its local backend.
        backends = {HARNESS: ComponentJobs(HARNESS, tmp_path / "candidates")} if backends is None else backends
        recipe = _TwoTrainerRecipe(
            backends=backends,
            auto_only_components=auto_only_components,
            runtime=StubInferenceRuntime(),
            training_component=training,
        )
    records = tmp_path / "records" if records_dir is None else records_dir
    dispatcher = Dispatcher(
        recipe,
        repository,
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(records),
        experiment_tracker=experiment_tracker,
        hold_local_cycles=hold_local_cycles,
        # The work queues live beside the records, so a restarted dispatcher finds the running job, as in a deployment.
        coordination=open_coordination("sqlite", agent_record_dir=str(records), record_database_url=None),
        components=components,
    )
    return dispatcher, backends
