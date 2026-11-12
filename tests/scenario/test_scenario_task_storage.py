"""The training pipeline of a scenario opens the generated task stores of its scenario session."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

from tests.support.samples import data_storage, sample_store

from reef.artifact import InMemoryRepositoryBackend
from reef.artifact.components import ScenarioComponents
from reef.dispatcher import Dispatcher
from reef.recipe import CompositeRecipe, Recipe
from reef.scenario.factory import read_trainer_specs
from reef.scenario.live_objects import LiveObjects
from reef.storage.sql_coordination import memory_coordination
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.trainer_builder import SpecTrainerBuilder


def test_a_scenario_pipeline_opens_the_task_stores_of_its_session(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "model.txt").write_text("initial", encoding="utf-8")
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(directory=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
        coordination=memory_coordination(),
    )
    try:
        scenario = dispatcher.get_or_create_scenario("math")
        task_storage = scenario.trainer.pipeline.context.task_storage
        assert task_storage is not None
        task_storage.open("tasks").begin_generation(generation=0, source_record_ids=("record-1",))
        assert scenario.store.tasks.open("tasks").generation(0) is not None
    finally:
        dispatcher.close()


def test_each_composite_component_opens_its_own_task_stores_and_shares_the_sample_store() -> None:
    with closing(data_storage()) as storage, closing(storage.open("math")) as session:
        samples = sample_store(storage, "math", session)
        trainers = SpecTrainerBuilder().build_trainers(
            read_trainer_specs(CompositeRecipe(components={"first": Recipe(), "second": Recipe()}), LiveObjects()),
            "math",
            session.records,
            components=ScenarioComponents(),
            algorithm_states={},
            task_storage=session.tasks,
            samples=samples,
        )
        for component_trainer in trainers:
            component_trainer.trainer.close()
        first_storage, second_storage = (
            component_trainer.trainer.pipeline.context.task_storage for component_trainer in trainers
        )
        assert first_storage is not None and second_storage is not None
        first_storage.open("tasks").begin_generation(generation=0, source_record_ids=("record-1",))

        assert second_storage.open("tasks").generations() == ()
        assert session.tasks.open("first/tasks").generation(0) is not None
        # The sample store, with its imported tasks, is one store that every component reads as itself.
        first_samples, second_samples = (
            component_trainer.trainer.pipeline.context.samples for component_trainer in trainers
        )
        assert first_samples is not None and second_samples is not None
        assert first_samples == samples.for_trainer("first") and second_samples == samples.for_trainer("second")
