"""The training pipeline of a scenario opens the generated task stores of its scenario session."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from tests.support._threshold_processor import threshold_pipeline

from reef.artifact import InMemoryRepositoryBackend
from reef.dispatcher import Dispatcher
from reef.pipeline_context import PipelineContext
from reef.recipe import CompositeRecipe, Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import SplitSettings
from reef.surface import Surface
from reef.train.pipeline import TrainingPipeline

SPLIT = SplitSettings(eval_fraction=0.0, seed=0)


@dataclass(frozen=True)
class ContextRecipe(Recipe):
    """Builds the default trainer and keeps each pipeline context that it gets."""

    contexts: list[PipelineContext] = field(default_factory=list, compare=False)

    def build_pipeline(self, context: PipelineContext) -> TrainingPipeline:
        self.contexts.append(context)
        return threshold_pipeline(context)


def test_a_scenario_pipeline_opens_the_task_stores_of_its_session(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("initial", encoding="utf-8")
    recipe = ContextRecipe()
    dispatcher = Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("math")
        task_storage = recipe.contexts[-1].task_storage
        assert task_storage is not None
        task_storage.open("tasks", split=SPLIT).begin_generation(generation=0, source_record_ids=("record-1",))
        assert scenario.store.tasks.open("tasks", split=SPLIT).generation(0) is not None
    finally:
        dispatcher.close()


def test_each_composite_component_opens_its_own_task_stores() -> None:
    first, second = ContextRecipe(), ContextRecipe()
    with closing(SQLiteScenarioStorage()) as storage, closing(storage.open("math")) as session:
        trainers = CompositeRecipe(components={"first": first, "second": second}).build_trainers(
            "math", session.records, surface=Surface(), algorithm_states={}, task_storage=session.tasks
        )
        for bound in trainers:
            bound.trainer.close()
        first_storage, second_storage = first.contexts[-1].task_storage, second.contexts[-1].task_storage
        assert first_storage is not None and second_storage is not None
        first_storage.open("tasks", split=SPLIT).begin_generation(generation=0, source_record_ids=("record-1",))

        assert second_storage.open("tasks", split=SPLIT).generations() == ()
        assert session.tasks.open("first/tasks", split=SPLIT).generation(0) is not None
