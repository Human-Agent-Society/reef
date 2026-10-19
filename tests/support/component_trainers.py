"""Shared scenario setup for experiment reporting and commit tests."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.support._threshold_processor import ThresholdBatches, threshold_pipeline
from tests.support.runtime_stubs import StubTrainingRuntime

from reef.artifact import Artifact, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.evaluation import EvaluationResult, UpdateCandidate
from reef.observability import ExperimentTracker, NullExperimentLogger
from reef.pipeline_context import PipelineContext
from reef.recipe import Recipe
from reef.scenario import Scenario
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.surface import ComponentSurface, Surface, TextFileTree
from reef.train import CandidateBackend, ComponentTrainer, PreparedStep, Trainer, TrainStepResult
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
            return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})
        path = self.artifact_dir / self.component / uuid.uuid4().hex
        path.mkdir(parents=True)
        (path / f"{self.component}.txt").write_text(f"{self.component} step {step}", encoding="utf-8")
        self.result = TrainStepResult(
            {"steps": step}, metrics={"prepared": self.prepared}, artifact=Artifact.local(path), pending=self.hold_next
        )
        self.hold_next = False
        return PreparedStep.with_candidate(UpdateCandidate(batch.batch_id), state={"steps": step})

    def evaluate(self, candidate):
        self.evaluated += 1
        return EvaluationResult("test", "1", {})

    def settle_step(self, prepared, decision):
        assert self.result is not None
        return self.result

    def abort_step(self, prepared):
        pass


class HybridThresholdBatches(ThresholdBatches):
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
                WEIGHTS: ComponentSurface(),
                HARNESS: ComponentSurface(files=TextFileTree()),
            }
        )

    def pipeline_for(self, component: str, context: PipelineContext) -> TrainingPipeline:
        if component in self.hybrid_components:
            return threshold_pipeline(context, task2batch=HybridThresholdBatches(context, 1))
        return threshold_pipeline(context)

    def build_trainers(self, scenario, records, *, surface, algorithm_states, experiment_logger=None):
        def factory_for(component: str):
            return lambda context: self.pipeline_for(component, context)

        return tuple(
            ComponentTrainer(
                component,
                Trainer.build(
                    scenario,
                    records,
                    pipeline_factory=factory_for(component),
                    candidate_backend=backend,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                ),
            )
            for component, backend in self.backends.items()
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
    """The scenario, created on first use, with each step's rows appended."""
    scenario = dispatcher.get_or_create_scenario(name)
    assert scenario is not None
    _append_records(scenario, *steps)
    return scenario


def _seeded_repository(tmp_path: Path, components: tuple[str, ...] = (WEIGHTS, HARNESS)):
    """A repository whose base seeds one file per component, unless the test wrote its own base first."""
    initial = tmp_path / "initial"
    if not initial.exists():
        for component in components:
            (initial / component).mkdir(parents=True)
            (initial / component / f"{component}.txt").write_text(f"{component} seed", encoding="utf-8")
    return InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")


def _local_pair(tmp_path: Path) -> dict[str, _ComponentBackend]:
    return {component: _ComponentBackend(component, tmp_path / "candidates") for component in (WEIGHTS, HARNESS)}


def _dispatcher(
    tmp_path: Path,
    *,
    backends: dict[str, _ComponentBackend] | None = None,
    recipe: Recipe | None = None,
    hybrid_components: frozenset[str] = frozenset(),
    training: StubTrainingRuntime | None = None,
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
        # A training runtime binds the scenario to the training thread, which drives the dispatched job.
        backends = _local_pair(tmp_path) if backends is None else backends
        recipe = _TwoTrainerRecipe(
            backends=backends,
            hybrid_components=hybrid_components,
            runtime=training.inference,
            training_runtime=training,
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
