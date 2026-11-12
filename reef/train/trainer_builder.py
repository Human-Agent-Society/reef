"""Build the trainers of one scenario from a recipe's values and the live objects of their holders.

A recipe is configuration: it builds no trainer (RFC #1390). The scenario factory reads a ``TrainerSpec`` from the
recipe, or one for each component of a composite recipe, and gives it to its ``TrainerBuilder`` with the objects of
the scenario's storage session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.components import ScenarioComponents
from reef.core.components import RECORDS_COMPONENT
from reef.core.reports import ReportBase
from reef.observability import ExperimentLogger
from reef.storage.data import DataStore
from reef.storage.records import RecordStore
from reef.storage.samples import SampleKind, SampleStore, TrainerSamples
from reef.storage.tasks import TaskStorage, TaskStore
from reef.train.backend import CommittedArtifacts, ComponentArtifacts
from reef.train.component import ScenarioRoot, TrainingComponent, component_root
from reef.train.jobs import JobLoop
from reef.train.pipeline import PipelineSettings
from reef.train.trainer import ComponentTrainer, Trainer


@dataclass(frozen=True, kw_only=True)
class TrainerSpec:
    """What one trainer is built from, apart from its scenario.

    Values from the recipe: ``name`` is the recipe's name, for error messages. ``stages`` is the pipeline's stage
    classes and their configs. ``report_type`` is the report contract, or None for any report. ``training_mode`` is
    ``auto`` or ``manual``. ``job_parameters`` is the parameters of each job by the kind of its samples; None gives
    every kind empty parameters.

    Live objects from their holder: ``job_component`` runs the trainer's jobs; None is a trainer that only reads
    records. ``data_store`` keeps the data directory of each job; None keeps them in a temporary directory
    (``JobLoop``).

    Purpose: the scenario factory reads one from a recipe (``reef.scenario.factory.read_trainer_specs``), and
    ``build_trainer`` builds the trainer.
    """

    name: str
    stages: PipelineSettings
    report_type: type[ReportBase] | None = None
    training_mode: str = "auto"
    job_parameters: Mapping[SampleKind, Mapping[str, Any]] | None = None
    job_component: TrainingComponent | None = None
    data_store: DataStore | None = None

    def __post_init__(self) -> None:
        if self.training_mode not in ("auto", "manual"):
            raise ValueError("training_mode must be 'auto' or 'manual'")


@dataclass(frozen=True)
class ComponentTrainerSpecs:
    """The trainer specs of a composite recipe.

    Values: ``specs`` holds the spec of each component's trainer under the component's name, in the order that the
    trainers are built. ``report_type`` is the report contract that the scenario's ingress checks, or None for any
    report.

    Purpose: the scenario factory reads one from a composite recipe, and ``SpecTrainerBuilder`` builds one trainer
    for each component.
    """

    specs: Mapping[str, TrainerSpec]
    report_type: type[ReportBase] | None = None


class ComponentTaskStorage(TaskStorage):
    """Opens the task stores of one component under ``<component>/<name>``, so components never share a store.

    The scenario's imported tasks are not a task store: every component reads them from the scenario's sample
    store.
    """

    def __init__(self, storage: TaskStorage, component: str) -> None:
        self.storage = storage
        self.component = component

    def open(self, name: str) -> TaskStore:
        return self.storage.open(f"{self.component}/{name}")


def build_trainer(
    spec: TrainerSpec,
    scenario: str,
    records: RecordStore,
    *,
    algorithm_state: Mapping[str, Any] | None = None,
    experiment_logger: ExperimentLogger | None = None,
    task_storage: TaskStorage | None = None,
    artifacts: ComponentArtifacts | None = None,
    samples: TrainerSamples | None = None,
    root: ScenarioRoot | None = None,
) -> Trainer:
    """Build the trainer of ``spec`` for the scenario ``scenario``.

    With ``spec.job_component``, the trainer submits jobs through a ``JobLoop`` on it, whose scenario component is
    the component of ``artifacts``, and a trainer with no ``algorithm_state`` starts from ``root``. Without it, the
    trainer only reads records. ``records``, ``task_storage`` and ``samples`` belong to the scenario's storage
    session (``Trainer.build``).

    Raise ``PluginConfigError`` when a stage refuses its config, or when a trainer that trains has no
    ``job_samples``.
    """
    if spec.job_component is None:
        job_loop = None
    else:
        job_loop = JobLoop(
            spec.job_component,
            scenario=scenario,
            scenario_component=None if artifacts is None else artifacts.component,
            data_store=spec.data_store,
        )
    return Trainer.build(
        scenario,
        records,
        pipeline_factory=spec.stages.build,
        job_loop=job_loop,
        root=root,
        job_parameters=spec.job_parameters,
        algorithm_state=algorithm_state,
        report_type=spec.report_type,
        experiment_logger=experiment_logger,
        training_mode=spec.training_mode,
        task_storage=task_storage,
        samples=samples,
    )


class TrainerBuilder(ABC):
    """Builds the trainers of a scenario.

    Purpose: ``ScenarioFactory`` holds one builder and calls it once for each scenario that it recovers, with the
    specs that it reads from the deployment's recipe.
    """

    @abstractmethod
    def build_trainers(
        self,
        specs: TrainerSpec | ComponentTrainerSpecs,
        scenario: str,
        records: RecordStore,
        *,
        components: ScenarioComponents,
        algorithm_states: Mapping[str, Mapping[str, Any] | None],
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: CommittedArtifacts | None = None,
        samples: SampleStore | None = None,
        root: ScenarioRoot | None = None,
    ) -> tuple[ComponentTrainer, ...]:
        """Build every trainer of the scenario ``scenario``, each bound to the component that it evolves.

        ``specs`` is the recipe's one spec, or the specs of a composite recipe. ``components`` is what the scenario
        serves. ``algorithm_states`` holds each trainer's recovered state under its component name; a trainer with
        none is new. ``records`` is the scenario's records, ``task_storage`` opens the generated task stores of its
        session, and ``samples`` is the samples that it trains on. ``artifacts`` reads the scenario's committed
        snapshots, and ``root`` is its root snapshot, where a new trainer starts. The scenario runs the trainers as
        separate workers that take turns, one step at a time.
        """


class SpecTrainerBuilder(TrainerBuilder):
    """Builds each trainer from its spec with ``build_trainer``.

    - ``ComponentTrainerSpecs`` give one trainer for each component, in the order of ``specs``. Each trainer gets its
      own view of the scenario's objects: the task stores under ``<component>/``, the committed artifacts and the
      samples of the component, and the component's directory of ``root``. Each trainer releases the reports that
      ``ComponentTrainerSpecs.report_type`` admits and its own contract refuses.
    - One ``TrainerSpec`` gives one trainer. It is bound to the scenario's only component, or to ``records`` when
      the scenario serves none, and it reads the samples as the trainer ``""``.

    Purpose: the scenarios of a dispatcher use it unless the dispatcher is given another builder.
    """

    def build_trainers(
        self,
        specs: TrainerSpec | ComponentTrainerSpecs,
        scenario: str,
        records: RecordStore,
        *,
        components: ScenarioComponents,
        algorithm_states: Mapping[str, Mapping[str, Any] | None],
        experiment_logger: ExperimentLogger | None = None,
        task_storage: TaskStorage | None = None,
        artifacts: CommittedArtifacts | None = None,
        samples: SampleStore | None = None,
        root: ScenarioRoot | None = None,
    ) -> tuple[ComponentTrainer, ...]:
        """Build the trainers of ``specs`` (``TrainerBuilder.build_trainers``).

        Raise ``ValueError`` when one ``TrainerSpec`` is given for a scenario that serves several components.
        """
        if isinstance(specs, ComponentTrainerSpecs):
            trainers = []
            for component, spec in specs.specs.items():
                trainer = build_trainer(
                    spec,
                    scenario,
                    records,
                    algorithm_state=algorithm_states.get(component),
                    experiment_logger=experiment_logger,
                    task_storage=None if task_storage is None else ComponentTaskStorage(task_storage, component),
                    artifacts=None if artifacts is None else ComponentArtifacts(artifacts, component),
                    samples=None if samples is None else samples.for_trainer(component),
                    root=None if root is None else component_root(root, component),
                )
                # Ingress admits what any component accepts; each trainer then releases the reports shaped for another.
                trainer.admit_reports_of(specs.report_type)
                trainers.append(ComponentTrainer(component, trainer))
            return tuple(trainers)
        if len(components.names) > 1:
            raise ValueError(
                f"recipe {specs.name!r} serves components {list(components.names)}: only a composite recipe binds "
                "one trainer per component"
            )
        component = components.names[0] if components.names else RECORDS_COMPONENT
        trainer = build_trainer(
            specs,
            scenario,
            records,
            algorithm_state=algorithm_states.get(component),
            experiment_logger=experiment_logger,
            task_storage=task_storage,
            artifacts=None if artifacts is None else ComponentArtifacts(artifacts, component),
            samples=None if samples is None else samples.for_trainer(""),
            root=root,
        )
        return (ComponentTrainer(component, trainer),)
