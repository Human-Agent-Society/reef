"""Trainers for tests: the trainer of a recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.recipe import Recipe
from reef.scenario.factory import read_trainer_spec
from reef.scenario.live_objects import LiveObjects
from reef.storage.records import RecordStore
from reef.storage.samples import TrainerSamples
from reef.storage.tasks import TaskStorage
from reef.train import Trainer
from reef.train.backend import ComponentArtifacts
from reef.train.component import ScenarioRoot
from reef.train.trainer_builder import build_trainer


def recipe_trainer(
    recipe: Recipe,
    live: LiveObjects,
    scenario: str,
    records: RecordStore,
    *,
    algorithm_state: Mapping[str, Any] | None = None,
    task_storage: TaskStorage | None = None,
    artifacts: ComponentArtifacts | None = None,
    samples: TrainerSamples | None = None,
    root: ScenarioRoot | None = None,
) -> Trainer:
    """The trainer that the scenario factory builds for ``recipe``, which is not a composite, on the objects of
    ``live``."""
    return build_trainer(
        read_trainer_spec(recipe, live),
        scenario,
        records,
        algorithm_state=algorithm_state,
        task_storage=task_storage,
        artifacts=artifacts,
        samples=samples,
        root=root,
    )
