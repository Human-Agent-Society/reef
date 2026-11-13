"""Shared setup for the tests of a scenario with one trainer per snapshot component."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tests.support.polling import became_true
from tests.support.runtime_stubs import StubFacts, StubInferenceRuntime, StubTrainingComponent, job_sized_stages
from tests.support.samples import data_storage
from tests.support.threshold_processor import ThresholdJobData, ThresholdSamples

from reef.artifact import DefaultStart, InMemoryRepositoryBackend, SnapshotRef
from reef.artifact.components import ComponentSpec, ScenarioComponents, TextFileTree
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation.activation import activation_reference
from reef.evaluation.harness import EvaluationSettings
from reef.execution.jobs import SupportsReceipt
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.recipe import CompositeRecipe, Recipe
from reef.recipe.base import WeightTrainingRecipe
from reef.recipe.harness import HarnessEvolutionRecipe
from reef.record2sample import Record2Sample
from reef.sample2job import Sample2Job
from reef.scenario import Scenario
from reef.scenario.live_objects import LiveObjects
from reef.storage.sql_coordination import open_coordination
from reef.train.component import JobStatus, SavedCheckpoint, TrainingJob
from reef.train.harness_backend.settings import EvolutionSettings
from reef.train.job_queue import CHECKS_QUEUE, JobQueueComponent
from reef.train.pipeline import PipelineSettings, PipelineSpec

WEIGHTS = "weights"
HARNESS = "harness"
from reef.execution.workers import SERVICE_COMPONENTS
from reef.train.types import TrainingCommit


class ComponentJobs(StubTrainingComponent):
    """The training component of one snapshot component, named as the component. A dispatcher's training worker
    takes each job from the jobs queue and gives it to ``submit``, which runs it in that thread.

    Each job runs ``step``, which saves one checkpoint holding ``<component>.txt`` = ``<component> step <n>`` for the
    ``n``-th job, with the metrics ``{prepared}``. ``prepared`` counts the steps that started, a step that raised
    included; a step that raises fails the job. ``reject_next`` ends the next job without a checkpoint, with
    ``selected: False``. ``hold_next`` adds the next checkpoint's snapshot path to ``held_paths``: such a checkpoint
    changes the ``HELD`` part (``changed_parts``), which the test recipe's activation rule does not activate.
    ``started`` is the text of the last start that ``import_start`` took. A new scenario starts from
    ``<component>.txt`` = ``<component> seed``, or from no file when ``seeded`` is False. A subclass blocks or
    fails a step in ``step``.
    """

    def __init__(self, component: str, artifact_dir: Path, *, seeded: bool = True) -> None:
        super().__init__(artifact_dir, name=component, shares_scenarios=True, artifact_format=None)
        self.component = component
        self.artifact_dir = artifact_dir
        self.seeded = seeded
        self.prepared = 0
        self.reject_next = False
        self.hold_next = False
        self.held_paths: set[str] = set()
        self.started: str | None = None

    def changed_parts(self, start: str, checkpoint: SavedCheckpoint) -> frozenset[str]:
        return frozenset({HELD}) if checkpoint.artifact_path in self.held_paths else frozenset()

    def unblock(self) -> None:
        """Let go of any step the test holds, so the dispatcher can close."""

    def stub_default_start(self) -> DefaultStart:
        return seeded_start(self.component) if self.seeded else DefaultStart()

    def import_start(self, artifact_dir: Path) -> str:
        self.started = (artifact_dir / f"{self.component}.txt").read_text(encoding="utf-8")
        return super().import_start(artifact_dir)

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
        checkpoint = SavedCheckpoint(0, str(path), str(path), {"prepared": self.prepared})
        if self.hold_next:
            self.held_paths.add(str(path))
        self.hold_next = False
        return JobStatus(job.job_id, "finished", (checkpoint,))

    def submit(self, job: TrainingJob) -> None:
        if job.job_id in self.statuses:
            return
        self.prepared += 1
        try:
            status = self.step(job)
        except Exception as exc:
            status = JobStatus(job.job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        self.submitted.append(job)
        self.statuses[job.job_id] = status


def checked_commit(
    scenario: Scenario, component: str | None = None, *, seconds: float = 10.0
) -> TrainingCommit | None:
    """``scenario.next_commit(component)``, called again while the trainer's reserved job waits for the check of the
    training service, as the trainer's worker polls. ``AssertionError`` when no answer comes within ``seconds``."""
    deadline = time.monotonic() + seconds
    while True:
        result = scenario.next_commit(component)
        if result is not None or not scenario.trainer_for(component).job_check_waits:
            return result
        assert time.monotonic() < deadline, "the training service did not check the job"
        time.sleep(0.01)


def wait_for_job_check(scenario: Scenario, component: str | None = None, *, seconds: float = 10.0) -> None:
    """Ask the training service to check the next job of ``component``'s trainer, and wait for its answer: the
    trainer's next cycle then submits the job. ``AssertionError`` when no answer comes within ``seconds``."""
    job_loop = scenario.trainer_for(component).job_loop_required()
    queued = job_loop.component
    assert isinstance(queued, JobQueueComponent)
    assert scenario.next_commit(component) is None and job_loop.waiting_job is not None
    item_id = f"job:{job_loop.waiting_job.job_id}"

    def answered() -> bool:
        item = queued.queue.read(CHECKS_QUEUE, item_id)
        return item is not None and item.final

    assert became_true(answered, timeout_seconds=seconds), "the training service did not check the job"


def finish_job(
    scenario: Scenario,
    component: str | None = None,
    checkpoint_metrics: Mapping[str, Any] | None = None,
    *,
    seconds: float = 10.0,
) -> None:
    """Commit the checkpoint and the end of ``component``'s running job, when the job's component is a
    ``ComponentJobs``, whose jobs end as soon as the training worker submits them: the commits that follow the
    job's submission, as the trainer's worker makes them once the training worker reported them in the jobs queue.
    Then, in a scenario with a runtime, wait until the selected artifact serves, because the next automatic job waits for it.
    ``checkpoint_metrics`` join the checkpoint's commit. A job of another component ends when its test says so, so
    it is left alone. ``AssertionError`` when the training worker reports nothing within ``seconds``."""
    trainer = scenario.trainer_for(component)
    queued = trainer.job_loop_required().component
    assert isinstance(queued, JobQueueComponent) and isinstance(queued.facts, StubFacts)
    if not isinstance(queued.facts.component, ComponentJobs):
        return
    while trainer.job_running:
        deadline = time.monotonic() + seconds
        while (result := scenario.next_commit(component)) is None:
            assert time.monotonic() < deadline, "the training worker did not report the running job"
            time.sleep(0.01)
        if checkpoint_metrics is not None and result.metrics["training_job"]["event"] == "checkpoint":
            result = replace(result, metrics={**result.metrics, **checkpoint_metrics})
            trainer.pending_commit = replace(trainer.pending_commit, result=result)
        scenario.commit(result, component=component)
    if scenario.runtime is not None:
        assert became_true(lambda: scenario.served_artifact_id() == scenario.serving_artifact_id())


class WeightsJobComponent(StubTrainingComponent):
    """The stub training component whose ``n``-th job saves the weights tree ``weights.txt`` = ``weights job <n>``.

    The checkpoint also holds the stub's ``model.txt`` = ``w<n>``, the version
    that ``StubInferenceRuntime`` serves it under. A new scenario starts from ``weights.txt`` = ``weights seed``.
    """

    def stub_default_start(self) -> DefaultStart:
        return seeded_start(WEIGHTS)

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
    """The same batches, without ``manual``: the stage takes no training request."""

    supported_training_modes = frozenset({"auto"})


#: The node kind of a held artifact: the test recipe's activation rule does not activate it.
HELD = "held"

#: How many artifacts behind the served one a row may be and still train. A test appends rows before the jobs that
#: it waits for move the served artifact, so its rows stay trainable until it ends.
MAX_STALENESS = 100


def component_stages(
    record2sample: type[Record2Sample] = ThresholdSamples,
    sample2job: type[Sample2Job] = ThresholdJobData,
    *,
    record2sample_config: Mapping[str, Any] | None = None,
) -> PipelineSettings:
    """The stages of one trainer of the two trainer recipe: ``record2sample`` with ``record2sample_config``, by
    default the threshold stage with the staleness limit ``MAX_STALENESS``, and ``sample2job`` with jobs of one
    sample."""
    config = {"max_staleness": MAX_STALENESS} if record2sample_config is None else dict(record2sample_config)
    return job_sized_stages(PipelineSpec(record2sample, sample2job), {"record2sample": {"config": config}})


@dataclass(frozen=True, kw_only=True)
class WeightsRecipe(WeightTrainingRecipe):
    """The weight recipe of the two trainer recipe: its trainer submits each job to the deployment's training
    component. Its stages are ``pipeline_stages``, by default ``component_stages()``."""

    name: str = "test_weights"

    def __post_init__(self) -> None:
        if self.pipeline_stages is None:
            object.__setattr__(self, "pipeline_stages", component_stages())
        super().__post_init__()

    @classmethod
    def pipeline_spec(cls) -> PipelineSpec:
        return PipelineSpec(ThresholdSamples, ThresholdJobData)


@dataclass(frozen=True)
class TextHarnessRecipe(HarnessEvolutionRecipe):
    """A harness recipe whose component is a tree of text files, with no adapter layout: its trainer submits each
    job to the deployment's harness component. It takes any report, and a new scenario starts from what the
    harness component's facts write, as a weight recipe's does. Its stages are ``pipeline_stages``, by default
    ``component_stages()``."""

    start: DefaultStart = field(default_factory=DefaultStart, kw_only=True)
    name: str = field(default="test_harness", kw_only=True)

    def __post_init__(self) -> None:
        if self.pipeline_stages is None:
            object.__setattr__(self, "pipeline_stages", component_stages())
        super().__post_init__()

    @property
    def report_type(self) -> None:  # type: ignore[override]
        return None

    def build_components(self, scenario: str) -> ScenarioComponents:
        return ScenarioComponents({HARNESS: ComponentSpec(files=TextFileTree())})

    def default_start(self) -> DefaultStart:
        return self.start


def harness_recipe(
    stages: PipelineSettings | None = None, *, seeded: bool = True, training_mode: str = "auto"
) -> TextHarnessRecipe:
    """The harness recipe of the two trainer recipe, whose trainer runs ``stages``. A new scenario starts from
    ``harness.txt`` = ``harness seed``, or from no file when ``seeded`` is False."""
    return TextHarnessRecipe(
        evolution=EvolutionSettings(),
        evaluation=EvaluationSettings(policy="none"),
        pipeline_stages=stages,
        start=seeded_start(HARNESS) if seeded else DefaultStart(),
        training_mode=training_mode,
    )


def two_trainer_recipe(
    *,
    weights: Recipe | None = None,
    harness: Recipe | None = None,
    auto_only_components: frozenset[str] = frozenset(),
    training_mode: str = "auto",
) -> CompositeRecipe:
    """A composite recipe of a scenario that serves weights and a harness tree, each evolved by its own trainer:
    ``weights`` (a ``WeightsRecipe`` by default) on the deployment's training component, and ``harness`` (a
    ``harness_recipe()`` by default) on its harness component. The trainer of each of ``auto_only_components``
    takes no training request. The activation rule does not activate a checkpoint that changes the ``HELD`` part."""
    auto_only = component_stages(sample2job=AutoOnlyThresholdJobData)
    if weights is None:
        weights = WeightsRecipe(
            pipeline_stages=auto_only if WEIGHTS in auto_only_components else None, training_mode=training_mode
        )
    if harness is None:
        harness = harness_recipe(auto_only if HARNESS in auto_only_components else None, training_mode=training_mode)
    return CompositeRecipe(
        components={WEIGHTS: weights, HARNESS: harness},
        activation=activation_reference({"module": "always", "config": {"review_kinds": [HELD]}}),
        training_mode=training_mode,
    )


def two_trainer_live(backends: Mapping[str, StubTrainingComponent]) -> LiveObjects:
    """The live objects of a deployment of the two trainer recipe: ``backends[WEIGHTS]`` as the training component
    and ``backends[HARNESS]`` as the harness component, each with its facts (``StubFacts``), and a stub runtime
    that serves the weights. A component that ``backends`` does not name is left out."""
    training, harness = backends.get(WEIGHTS), backends.get(HARNESS)
    return LiveObjects(
        runtime=StubInferenceRuntime(),
        training_component=training,
        training_facts=None if training is None else StubFacts(training),
        harness_component=harness,
        harness_facts=None if harness is None else StubFacts(harness),
    )


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


def seeded_start(component: str) -> DefaultStart:
    """The default start of one component: ``<component>.txt`` = ``<component> seed``. A composite recipe keeps the
    file under the component's directory."""
    return DefaultStart(files={f"{component}.txt": f"{component} seed"})


def _repository(tmp_path: Path):
    """An in-memory repository under the test's directory."""
    return InMemoryRepositoryBackend.factory(directory=tmp_path / "repository")


def component_files(scenario: Scenario, ref: SnapshotRef) -> dict[str, str]:
    """The text of each component's ``<component>.txt`` in the snapshot ``ref``."""
    head = scenario.repository.materialize(ref)
    return {
        component: (TextFileTree().read_files(head.component(component)) or {})[f"{component}.txt"]
        for component in (WEIGHTS, HARNESS)
    }


def _local_pair(tmp_path: Path) -> dict[str, StubTrainingComponent]:
    return {component: ComponentJobs(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}


def _dispatcher(
    tmp_path: Path,
    *,
    backends: Mapping[str, StubTrainingComponent] | None = None,
    recipe: Recipe | None = None,
    auto_only_components: frozenset[str] = frozenset(),
    training: StubTrainingComponent | None = None,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    experiment_tracker: ExperimentTracker | None = None,
    hold_local_cycles: bool = False,
    components: frozenset[str] = frozenset(SERVICE_COMPONENTS),
    live: LiveObjects | None = None,
) -> tuple[Dispatcher, dict[str, Any]]:
    """A dispatcher over the two trainer recipe, or over ``recipe`` when one is given, and the training components
    of its deployment by scenario component.

    ``backends`` gives the training component of ``WEIGHTS`` and the harness component of ``HARNESS``; each one
    that it leaves out is a ``ComponentJobs``. ``training`` is another way to give the component of ``WEIGHTS``.
    ``live`` replaces the live objects that ``backends`` give (``two_trainer_live``). The dispatcher's own training
    worker runs the jobs of both components from the jobs queue. ``components`` are the services that it runs;
    without ``samples``, only the test makes samples (``make_samples``)."""
    repository = _repository(tmp_path) if backend_factory is None else backend_factory
    built = {
        WEIGHTS: ComponentJobs(WEIGHTS, tmp_path / "candidates") if training is None else training,
        HARNESS: ComponentJobs(HARNESS, tmp_path / "candidates"),
        **({} if backends is None else backends),
    }
    if recipe is None:
        recipe = two_trainer_recipe(auto_only_components=auto_only_components)
    live = two_trainer_live(built) if live is None else live
    records = tmp_path / "records" if records_dir is None else records_dir
    storage = data_storage(records)
    dispatcher = Dispatcher(
        recipe,
        repository,
        live=replace(live, data_store=storage.data),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=storage,
        record_storage=storage.inner,
        experiment_tracker=experiment_tracker,
        hold_local_cycles=hold_local_cycles,
        # The work queues live beside the records, so a restarted dispatcher finds the running job, as in a deployment.
        coordination=open_coordination("sqlite", agent_record_dir=str(records), record_database_url=None),
        components=components,
    )
    return dispatcher, built


def component_id(scenario: Scenario, revision: str, component: str) -> str:
    """Read a component identity from a test's stored snapshot or composed view."""
    selection = scenario.repository.selections.get(revision)
    if selection is not None:
        return selection.components[component]
    ref = scenario.repository.backend.resolve_snapshot(revision)
    return scenario.repository.resolve(ref).component(component).artifact_id
