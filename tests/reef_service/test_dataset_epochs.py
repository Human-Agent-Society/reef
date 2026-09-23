"""Storage snapshots replay across commits with only a batch resident in memory."""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.recipe import Recipe
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train import DatasetProcessor, PreparedStep, ProcessorContext, Trainer, TrainingBatch
from reef.train.types import TaskItem

from .test_reef_trainer_contracts import ExampleBackend


class InferenceDatasetProcessor(DatasetProcessor):
    def make_sample(self, record):
        return TaskItem(Path(record.agent_record_id), metadata=record.payload)


class DatasetBackend(ExampleBackend):
    def __init__(self, scenario, batches):
        super().__init__(scenario, [])
        self.batches = batches

    def prepare_step(self, batch, state, scenario_step):
        if batch.request is not None and batch.request.text == "fail":
            raise RuntimeError("instruction failed")
        self.batches.append(batch)
        return PreparedStep.skipped(state={"steps": state.get("steps", 0) + 1})


@dataclass(frozen=True, kw_only=True)
class DatasetRecipe(Recipe):
    batches: list[TrainingBatch] = field(default_factory=list)
    dataset_epochs: int = 2

    def build(self, scenario, records, *, algorithm_state=None, experiment_logger=None):
        return Trainer.build(
            scenario,
            records,
            processor_factory=lambda context: InferenceDatasetProcessor(
                context.with_config({"batch_size": 2, "dataset_epochs": self.dataset_epochs})
            ),
            candidate_backend=DatasetBackend(scenario, self.batches),
            algorithm_state=algorithm_state,
            experiment_logger=experiment_logger,
            training_mode=self.training_mode,
        )


def record(record_id, request_type=RequestType.INFERENCE, payload=None, scenario="s"):
    return AgentRecord.create(
        scenario=scenario, request_type=request_type, payload=payload or {}, agent_record_id=record_id
    )


@pytest.mark.parametrize("restart_after", [0, 1, 2, 3, 4])
def test_two_passes_tail_failure_and_durable_restart(tmp_path, restart_after):
    initial = tmp_path / "initial"
    initial.mkdir()
    repositories = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository")
    recipe = DatasetRecipe(training_mode="hybrid")

    def open_dispatcher():
        return Dispatcher(recipe, repositories, scenario_storage=SQLiteScenarioStorage(tmp_path / "records"))

    dispatcher = open_dispatcher()
    try:
        scenario = dispatcher.get_or_create_scenario("s")
        scenario.records.append(record("fail", RequestType.TRAIN, {"text": "fail", "session": "", "release_id": ""}))
        for name in ("a", "b", "c"):
            scenario.records.append(record(name))
        # No completion signal: the first read takes the current storage tail.
        with pytest.raises(RuntimeError, match="instruction failed"):
            scenario.prepare_training_step()
        assert scenario.trainer.fail_pending_instruction("instruction failed")
        skipped = scenario.prepare_training_step()
        scenario.commit(skipped)
        assert scenario.store.history()[0].consumed_ids == {"fail"}
        assert scenario.store.history()[0].metrics["dataset_state"]["cursor"] == 0
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
            elif step == 2:
                assert scenario.records.get("s", "a") is None
                assert scenario.records.get("s", "c") is not None
            if step == 0:
                # Arrivals during training wait for a subsequent snapshot.
                scenario.records.append(record("later"))

        assert [[str(item.task_path) for item in batch.items] for batch in recipe.batches] == [
            ["a", "b"],
            ["c"],
            ["a", "b"],
            ["c"],
        ]
        assert len({batch.batch_id for batch in recipe.batches}) == 4
        assert [commit.metrics["dataset_epoch"] for commit in scenario.store.history()] == [1, 1, 1, 2, 2]
        for _ in range(2):
            result = scenario.prepare_training_step()
            assert result is not None
            scenario.commit(result)
        assert [[str(item.task_path) for item in batch.items] for batch in recipe.batches[-2:]] == [
            ["later"],
            ["later"],
        ]
        assert scenario.prepare_training_step() is None
        assert scenario.records.count("s") == 0
        assert not scenario.records.append_result(record("a")).inserted
        incompatible = InferenceDatasetProcessor(ProcessorContext("s", config={"dataset_epochs": 3}))
        with pytest.raises(ValueError, match="must match the committed"):
            incompatible.restore_consumption(scenario.store.history())
        legacy_commit = replace(scenario.store.history()[-1], metrics={})
        with pytest.raises(ValueError, match="missing dataset consumption state"):
            incompatible.restore_consumption((legacy_commit,))
    finally:
        dispatcher.close()


@pytest.mark.parametrize("epochs", [0, -1, True, "2", 1.5])
def test_invalid_dataset_epochs(epochs):
    with pytest.raises(ValueError, match="dataset_epochs"):
        InferenceDatasetProcessor(ProcessorContext("s", config={"dataset_epochs": epochs}))


def test_byte_budget_retry_and_trailing_non_training_records(tmp_path):
    records = SQLiteRecordStore(tmp_path / "records.sqlite")
    processor = InferenceDatasetProcessor(
        ProcessorContext("s", config={"dataset_epochs": 2, "batch_size": 5, "dataset_batch_bytes": 80})
    )
    try:
        for name in ("a", "b", "c"):
            records.append(record(name, payload={"text": "x" * 50}))
        records.append(record("ignored", RequestType.REPORT))
        watermark, offset = 0, 0
        seen = []
        for _ in range(6):
            watermark, offset = processor.consume(records, after_sequence=watermark, offset=offset)
            batch = processor.build_batch()
            processor.release_batch(batch.batch_id)
            assert processor.build_batch() == batch
            assert len(batch.items) == 1
            seen.append(str(batch.items[0].task_path))
            processor.acknowledge(batch.batch_id)
            compacted = processor.retention_decision().releasable_agent_record_ids
            records.compact("s", compacted)
            processor.compaction_applied(compacted)
        assert seen == ["a", "b", "c", "a", "b", "c"]
        processor.consume(records, after_sequence=watermark, offset=offset)
        assert not processor.ready()
        assert processor.samples == []
        assert records.get("s", "ignored") is not None
    finally:
        records.close()


def test_large_dataset_reads_only_current_batch(tmp_path):
    records = SQLiteRecordStore(tmp_path / "records.sqlite")
    processor = InferenceDatasetProcessor(ProcessorContext("s", config={"batch_size": 2, "dataset_epochs": 2}))
    try:
        # 16 MiB on disk. The sample retains the payload so a full-cache
        # implementation cannot pass by returning a tiny TaskItem instead.
        for index in range(256):
            records.append(record(str(index), payload={"text": "x" * 65536}))
        records.append(record("other", payload={"text": "other scenario"}, scenario="other"))
        assert records.latest_sequence("s") == 256
        tracemalloc.start()
        try:
            watermark, offset = processor.consume(records, after_sequence=0, offset=0)
            batch = processor.build_batch()
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert watermark == offset == 2
        assert len(batch.items) == len(processor.samples) == 2
        assert peak_bytes < 4 * 1024**2
        assert records.count("s") == 256
        assert processor.consumption_metrics()["dataset_state"]["cursor"] == 0
    finally:
        records.close()


def test_oversized_record_makes_progress_and_empty_store_can_receive_later(tmp_path):
    records = SQLiteRecordStore(tmp_path / "records.sqlite")
    processor = InferenceDatasetProcessor(
        ProcessorContext("s", config={"dataset_batch_bytes": 10, "dataset_epochs": 1})
    )
    try:
        assert processor.consume(records, after_sequence=0, offset=0) == (0, 0)
        assert not processor.ready()
        records.append(record("large", payload={"text": "x" * 1024}))
        processor.consume(records, after_sequence=0, offset=0)
        batch = processor.build_batch()
        assert len(batch.items) == 1
        with pytest.raises(ValueError, match="before consumption"):
            processor.set_training_mode("hybrid")
        with pytest.raises(ValueError, match="requires a commit"):
            processor.dropped(batch.batch_id)
        assert processor.acknowledge(batch.batch_id) == {"large"}
    finally:
        records.close()
