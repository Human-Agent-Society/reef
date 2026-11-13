"""Trainers for tests: the trainer of a recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tests.support.stored_checkpoints import store_checkpoints

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
    ``live``. Its job component runs the jobs on the trainer's own thread, so it stores its own checkpoints
    (``store_checkpoints``)."""
    spec = read_trainer_spec(recipe, live)
    if spec.job_component is not None:
        store_checkpoints(spec.job_component, scenario, root)
    return build_trainer(
        spec,
        scenario,
        records,
        algorithm_state=algorithm_state,
        task_storage=task_storage,
        artifacts=artifacts,
        samples=samples,
        root=root,
    )
