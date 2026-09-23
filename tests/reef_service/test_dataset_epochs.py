"""Fixed datasets replay across commits without duplicating stored records."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.reports import ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import DataProcessor, PreparedStep, ProcessorContext, Trainer, TrainingBatch
from reef.train.processors.reported import GroupDecision, ReportedFeedbackProcessor
from reef.train.types import TaskItem

from .test_reef_trainer_contracts import ExampleBackend


class InferenceDatasetProcessor(DataProcessor):
    required_request_types = frozenset({RequestType.INFERENCE, RequestType.TRAIN})
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})

    def ingest(self, item):
        if item.request_type is RequestType.INFERENCE:
            self.add_samples(
                item.agent_record_id,
                (TaskItem(Path(item.agent_record_id), source_agent_record_ids=(item.agent_record_id,)),),
            )
        else:
            super().ingest(item)


class ReportDatasetProcessor(ReportedFeedbackProcessor):
    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT, RequestType.TRAIN})
    supported_training_modes = frozenset({"auto", "manual", "hybrid"})

    def make_sample(self, context):
        return TaskItem(Path(context.report.agent_record_id))

    def make_batch(self, items, batch_number):
        return TrainingBatch(f"batch:{batch_number}", items)


class DatasetBackend(ExampleBackend):
    def __init__(self, scenario, batches):
        super().__init__(scenario, [])
        self.batches = batches

    def prepare_step(self, batch, state, scenario_step):
        if batch.request is not None and batch.request.text == "fail":
            raise RuntimeError("instruction failed")
        self.batches.append(batch)
        return PreparedStep.skipped(state={"steps": state.get("steps", 0) + 1})


class GroupedDatasetProcessor(ReportDatasetProcessor):
    def grouping(self, context):
        return context.report.payload["metadata"]["group"], None

    def decide_group(self, key, items):
        return GroupDecision.READY if len(items) == 2 else GroupDecision.INCOMPLETE


@dataclass(frozen=True, kw_only=True)
class DatasetRecipe(Recipe):
    processor_type: type[DataProcessor] = InferenceDatasetProcessor
    batches: list[TrainingBatch] = field(default_factory=list)
    dataset_epochs: int = 2

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: self.processor_type(
                context.with_config({"batch_size": 2, "dataset_epochs": self.dataset_epochs})
            ),
            candidate_backend=DatasetBackend(scenario, self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            report_type=ScoredRolloutReport,
            training_mode=self.training_mode,
        )


def record(record_id, request_type=RequestType.INFERENCE, payload=None):
    return AgentRecord.create(
        scenario="s", request_type=request_type, payload=payload or {}, agent_record_id=record_id
    )


@pytest.mark.parametrize("processor_type", [InferenceDatasetProcessor, ReportDatasetProcessor])
@pytest.mark.parametrize("restart_after", [0, 1, 2, 3, 4])
def test_two_passes_tail_failure_and_durable_restart(tmp_path, processor_type, restart_after):
    initial = tmp_path / "initial"
    initial.mkdir()
    repositories = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    recipe = DatasetRecipe(processor_type=processor_type, training_mode="hybrid")

    def open_dispatcher():
        return Dispatcher(recipe, repositories, scenario_storage=SQLiteScenarioStorage(tmp_path / "records"))

    dispatcher = open_dispatcher()
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        for name in ("a", "b", "c"):
            scenario.records.append(record(name))
            if processor_type is ReportDatasetProcessor:
                scenario.records.append(record(f"r-{name}", RequestType.REPORT, {"score": 1, "references": [name]}))
        # An idle input stream, even with a full batch, is not dataset completion.
        assert scenario.prepare_training_step() is None
        assert recipe.batches == []
        scenario.records.append(record("fail", RequestType.TRAIN, {"text": "fail", "session": "", "release_id": ""}))
        scenario.records.append(record("end", RequestType.REPORT, {"metadata": {"dataset_end": True}}))
        with pytest.raises(RuntimeError, match="instruction failed"):
            scenario.prepare_training_step()
        assert scenario.trainer.fail_pending_instruction("instruction failed")
        skipped = scenario.prepare_training_step()
        scenario.commit(skipped)
        assert scenario.store.history()[0].consumed_ids == {"fail"}
        assert scenario.store.history()[0].metrics["dataset_unit_ids"] == []
        assert recipe.batches == []

        for step in range(5):
            if step == restart_after:
                dispatcher.close()
                dispatcher = open_dispatcher()
                scenario = dispatcher.get_or_create_scenario("s")
            if step == 4:
                break
            result = scenario.prepare_training_step()
            assert result is not None
            prepared = scenario.trainer.prepare_commit(result)
            assert scenario.trainer.prepare_commit(result) is prepared
            scenario.commit(result)
            if step < 2:
                assert scenario.records.get("s", "a") is not None
                assert scenario.records.get("s", "end") is not None
            elif step == 2:
                assert scenario.records.get("s", "a") is None
                assert scenario.records.get("s", "c") is not None

        prefix = "r-" if processor_type is ReportDatasetProcessor else ""
        assert [[str(item.task_path) for item in batch.items] for batch in recipe.batches] == [
            [f"{prefix}a", f"{prefix}b"],
            [f"{prefix}c"],
            [f"{prefix}a", f"{prefix}b"],
            [f"{prefix}c"],
        ]
        assert len({batch.batch_id for batch in recipe.batches}) == 4
        assert scenario.prepare_training_step() is None
        assert scenario.records.count("s") == 0
        assert not scenario.records.append_result(
            record("end", RequestType.REPORT, {"metadata": {"dataset_end": True}})
        ).inserted
        assert [commit.metrics["dataset_epoch"] for commit in scenario.store.history()] == [1, 1, 1, 2, 2]
        assert [row["metrics"]["dataset_epoch"] for row in scenario.releases() if "metrics" in row] == [2, 2, 1, 1, 1]
    finally:
        dispatcher.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_end_signal_is_durable_idempotent_and_requires_opt_in(tmp_path, enabled):
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        DatasetRecipe() if enabled else Recipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )

    async def run():
        async with TestClient(TestServer(create_app(dispatcher))) as client:
            for _ in range(2):
                response = await client.post(
                    "/reef/report",
                    headers={"x-reef-scenario": "s"},
                    json={"agent_record_id": "end", "metadata": {"dataset_end": True}},
                )
                if not enabled:
                    assert response.status == 400
                    assert "dataset_epochs" in await response.text()
                    return
                assert response.status == 200, await response.text()
                assert (await response.json())["request_type"] == "report"
            for invalid in (
                {"metadata": {"dataset_end": "true"}},
                {"metadata": {"dataset_end": True}, "score": 1},
                {"metadata": {"dataset_end": True}, "references": ["a"]},
            ):
                response = await client.post("/reef/report", headers={"x-reef-scenario": "s"}, json=invalid)
                assert response.status == 400

    try:
        asyncio.run(run())
        assert dispatcher.get_or_create_scenario("s").records.count("s") == int(enabled)
    finally:
        dispatcher.close()


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5, "2"])
def test_dataset_epochs_requires_a_positive_integer(epochs):
    with pytest.raises(ValueError, match="positive integer"):
        DataProcessor(ProcessorContext("s", {"dataset_epochs": epochs}))


def test_final_pass_restart_preserves_shared_sources_and_excludes_later_data(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    repositories = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    recipe = DatasetRecipe(processor_type=ReportDatasetProcessor)
    dispatcher = Dispatcher(recipe, repositories, scenario_storage=SQLiteScenarioStorage(tmp_path / "records"))
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        scenario.records.append(record("shared"))
        for name in ("a", "b", "c"):
            scenario.records.append(record(name, RequestType.REPORT, {"score": 1, "references": ["shared"]}))
        scenario.records.append(record("end", RequestType.REPORT, {"metadata": {"dataset_end": True}}))
        scenario.records.append(record("later"))
        for _ in range(3):
            scenario.commit(scenario.prepare_training_step())
        assert scenario.records.get("s", "a") is None
        assert scenario.records.get("s", "shared") is not None
        dispatcher.close()
        dispatcher = Dispatcher(recipe, repositories, scenario_storage=SQLiteScenarioStorage(tmp_path / "records"))
        scenario = dispatcher.get_or_create_scenario("s")
        scenario.commit(scenario.prepare_training_step())
        assert [str(item.task_path) for item in recipe.batches[-1].items] == ["c"]
        assert scenario.records.get("s", "shared") is None
        assert scenario.records.get("s", "later") is not None
        assert scenario.prepare_training_step() is None
    finally:
        dispatcher.close()


@pytest.mark.parametrize("epochs", [1, 3])
def test_released_reservation_retries_same_batch_and_pass(epochs):
    processor = InferenceDatasetProcessor(ProcessorContext("s", {"dataset_epochs": epochs, "batch_size": 2}))
    for name in ("a", "b", "c"):
        processor.ingest_record(record(name))
    processor.ingest_record(record("end", RequestType.REPORT, {"metadata": {"dataset_end": True}}))
    for epoch in range(1, epochs + 1):
        for names in (("a", "b"), ("c",)):
            batch = processor.build_batch()
            assert processor.build_batch() is batch
            processor.release_batch(batch.batch_id)
            retried = processor.build_batch()
            assert retried == batch
            assert processor.dataset_epoch == epoch
            assert tuple(str(item.task_path) for item in retried.items) == names
            processor.acknowledge(retried.batch_id)
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == {"a", "b", "c", "end"}


def test_groups_repeat_in_arrival_order_and_must_be_complete():
    processor = GroupedDatasetProcessor(ProcessorContext("s", {"dataset_epochs": 2, "batch_size": 1}))
    end = record("end", RequestType.REPORT, {"metadata": {"dataset_end": True}})
    processor.ingest_record(record("source"))
    for name in ("a1", "b1", "b2", "a2"):
        processor.ingest_record(
            record(name, RequestType.REPORT, {"score": 1, "references": ["source"], "metadata": {"group": name[0]}})
        )
        if name == "a1":
            with pytest.raises(ValueError, match="incomplete report groups"):
                processor.ingest_record(end)
    processor.ingest_record(end)
    for epoch in (1, 2):
        for names in (("a1", "a2"), ("b1", "b2")):
            batch = processor.build_batch()
            assert processor.dataset_epoch == epoch
            assert tuple(str(item.task_path) for item in batch.items) == names
            processor.acknowledge(batch.batch_id)
    assert not processor.ready()
