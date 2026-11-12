"""Trainers for tests: the trainer of a recipe, and builders for trainers that no recipe gives."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.artifact.components import ScenarioComponents
from reef.core.components import RECORDS_COMPONENT
from reef.observability import ExperimentLogger
from reef.recipe import Recipe
from reef.scenario.factory import read_trainer_spec
from reef.storage.records import RecordStore
from reef.storage.samples import SampleStore, TrainerSamples
from reef.storage.tasks import TaskStorage
from reef.train import ComponentTrainer, Trainer
from reef.train.backend import CommittedArtifacts, ComponentArtifacts
from reef.train.component import ScenarioRoot
from reef.train.trainer_builder import ComponentTrainerSpecs, TrainerBuilder, TrainerSpec, build_trainer


def recipe_trainer(
    recipe: Recipe,
    scenario: str,
    records: RecordStore,
    *,
    algorithm_state: Mapping[str, Any] | None = None,
    task_storage: TaskStorage | None = None,
    artifacts: ComponentArtifacts | None = None,
    samples: TrainerSamples | None = None,
    root: ScenarioRoot | None = None,
) -> Trainer:
    """The trainer that the scenario factory builds for ``recipe``, which is not a composite."""
    return build_trainer(
        read_trainer_spec(recipe),
        scenario,
        records,
        algorithm_state=algorithm_state,
        task_storage=task_storage,
        artifacts=artifacts,
        samples=samples,
        root=root,
    )


class OneTrainerBuilder(TrainerBuilder):
    """Binds the trainer that ``build_trainer`` returns to the scenario's only component, or to ``records`` when the
    scenario serves none. A test subclasses it for a trainer that no recipe gives, such as one whose jobs run on
    the trainer's own thread and not through the jobs queue."""

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
        assert isinstance(specs, TrainerSpec)
        component = components.names[0] if components.names else RECORDS_COMPONENT
        trainer = self.build_trainer(
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

    @abstractmethod
    def build_trainer(
        self,
        spec: TrainerSpec,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, Any] | None,
        experiment_logger: ExperimentLogger | None,
        task_storage: TaskStorage | None,
        artifacts: ComponentArtifacts | None,
        samples: TrainerSamples | None,
        root: ScenarioRoot | None,
    ) -> Trainer:
        """The scenario's only trainer. ``spec`` is the spec of the dispatcher's recipe."""
