"""Shared setup for the tests of a scenario with one trainer per release component."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.support._threshold_processor import ThresholdJobData, threshold_pipeline
from tests.support.runtime_stubs import StubInferenceRuntime, StubTrainingComponent

from reef.artifact import Artifact, ArtifactRef, InMemoryRepositoryBackend
from reef.background_jobs import SupportsReceipt
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation import EvaluationResult, UpdateCandidate
from reef.evaluation.evaluators import AlwaysSelectCheckpointPlugin
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.scenario import Scenario
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.surface.base import ServingRuntime
from reef.surface.weights import WeightLoader
from reef.train import CandidateBackend, ComponentTrainer, PreparedStep, Trainer, TrainStepResult
from reef.train.jobs import FixedJobPlan, JobLoop, JobRequest
from reef.train.pipeline import TrainingPipeline

WEIGHTS = "weights"
HARNESS = "harness"


class _ComponentBackend(CandidateBackend):
    """A local candidate cycle publishing one file for its component per step."""

    def __init__(self, component: str, artifact_dir: Path) -> None:
        self.component = component
        self.artifact_dir = artifact_dir
        self.prepared = 0
        self.evaluated = 0
        self.reject_next = False
        self.hold_next = False
        self.result: TrainStepResult | None = None

    def unblock(self) -> None:
        """Let go of any step the test holds, so the dispatcher can close."""

    def initial_state(self) -> Mapping[str, Any]:
        return {"steps": 0}

    def prepare_step(self, batch, state, scenario_step):
        step = int(state["steps"]) + 1
        self.prepared += 1
        if self.reject_next:
            # A rejected step publishes nothing: the row it leaves carries the head's own release.
            self.reject_next = False
            self.result = TrainStepResult({"steps": step}, metrics={"prepared": self.prepared, "selected": False})
            return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state={"steps": step})
        path = self.artifact_dir / self.component / uuid.uuid4().hex
        path.mkdir(parents=True)
        (path / f"{self.component}.txt").write_text(f"{self.component} step {step}", encoding="utf-8")
        self.result = TrainStepResult(
            {"steps": step}, metrics={"prepared": self.prepared}, artifact=Artifact.local(path), pending=self.hold_next
        )
        self.hold_next = False
        return PreparedStep.with_candidate(UpdateCandidate(batch.data_id), state={"steps": step})

    def evaluate(self, candidate):
        self.evaluated += 1
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass


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


class UnversionedWeightLoader(WeightLoader):
    """Load weights through the runtime, but name no serving version: the test rows name none, so none is stale."""

    def served_runtime_load_id(self, artifact: Artifact, runtime: ServingRuntime | None) -> str | None:
        return None


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


class HybridThresholdJobData(ThresholdJobData):
    """The same batches, taking instructions too."""

    supported_training_modes = frozenset({"auto", "manual", "hybrid"})


@dataclass(frozen=True)
class _TwoTrainerRecipe(Recipe):
    """A scenario serving weights and a harness tree, each evolved by its own local backend."""

    backends: Mapping[str, CandidateBackend]
    hybrid_components: frozenset[str] = frozenset()

    def build_surface(self, scenario: str) -> Surface:
        return Surface(
            components={
                # A training component's weights load through the runtime, so a rollback moves the next job's start.
                WEIGHTS: (
                    ComponentSurface()
                    if self.training_component is None
                    else ComponentSurface(loader=UnversionedWeightLoader())
                ),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )

    def pipeline_for(self, component: str, context: PipelineContext) -> TrainingPipeline:
        if component in self.hybrid_components:
            return threshold_pipeline(context, task2job=HybridThresholdJobData(context, None, None))
        return threshold_pipeline(context)

    def build_trainers(
        self, scenario, records, *, surface, algorithm_states, experiment_logger=None, task_storage=None
    ):
        def factory_for(component: str):
            return lambda context: self.pipeline_for(component, context)

        trainers = tuple(
            ComponentTrainer(
                component,
                Trainer.build(
                    scenario,
                    records,
                    pipeline_factory=factory_for(component),
                    candidate_backend=backend,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    default_job_samples=1,
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
            job_loop=JobLoop(
                training,
                FixedJobPlan(JobRequest()),
                AlwaysSelectCheckpointPlugin(),
                scenario=scenario,
            ),
            algorithm_state=algorithm_states.get(WEIGHTS),
            experiment_logger=experiment_logger,
            default_job_samples=1,
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

    def record_rollback(self, event):
        pass

    def close(self):
        pass


def _records(step: int, scenario: str = "agent") -> tuple[AgentRecord, AgentRecord]:
    """One inference and the report on it; rows of scenarios other than ``agent`` carry the scenario's name."""
    prefix = "" if scenario == "agent" else f"{scenario}-"
    inference = AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
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
    """Append each step's rows to the store directly, so the test, not a worker, owns when a trainer prepares."""
    for step in steps:
        for record in _records(step, scenario.name):
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


def _local_pair(tmp_path: Path) -> dict[str, _ComponentBackend]:
    return {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}


def _dispatcher(
    tmp_path: Path,
    *,
    backends: dict[str, _ComponentBackend] | None = None,
    recipe: Recipe | None = None,
    hybrid_components: frozenset[str] = frozenset(),
    training: StubTrainingComponent | None = None,
    records_dir: Path | None = None,
    backend_factory: Any = None,
    experiment_tracker: ExperimentTracker | None = None,
    hold_local_cycles: bool = False,
) -> tuple[Dispatcher, dict[str, _ComponentBackend]]:
    """A dispatcher over the two trainer recipe of ``backends``, or over ``recipe`` when one is given."""
    repository = _seeded_repository(tmp_path) if backend_factory is None else backend_factory
    if recipe is not None:
        backends = {} if backends is None else backends
    elif training is None:
        backends = _local_pair(tmp_path) if backends is None else backends
        recipe = _TwoTrainerRecipe(backends=backends, hybrid_components=hybrid_components)
    else:
        # A training component binds the scenario: a process trains one scenario on it. It trains the weights; the
        # harness keeps its local backend.
        backends = {HARNESS: _ComponentBackend(HARNESS, tmp_path / "candidates")} if backends is None else backends
        recipe = _TwoTrainerRecipe(
            backends=backends,
            hybrid_components=hybrid_components,
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
    )
    return dispatcher, backends
