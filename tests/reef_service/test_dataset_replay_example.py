"""The tutorial composes existing storage, processors and durable commits."""

import importlib
from contextlib import closing
from pathlib import Path

import pytest

from reef.artifact import InMemoryRepositoryBackend
from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.storage.sqlite import SQLiteRecordStore, SQLiteScenarioStorage
from reef.train.types import ProcessorContext


@pytest.fixture
def example(monkeypatch):
    # Load only the tutorial, so the installed-wheel CI still imports Reef
    # from its wheel rather than accidentally adding the source root.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tutorials"))
    return importlib.import_module("dataset_replay.demo")


def test_two_passes_failed_instruction_restart_and_continual_records(tmp_path, monkeypatch, example):
    initial = tmp_path / "initial"
    initial.mkdir()
    repository = InMemoryRepositoryBackend.factory(initial, root=tmp_path / "artifacts")
    storage_path = tmp_path / "records"
    recipe = example.ReplayRecipe(dataset_last_sequence=3)

    first = Dispatcher(recipe, repository, scenario_storage=SQLiteScenarioStorage(storage_path))
    with closing(first):
        scenario = first.get_or_create_scenario("demo")
        scenario.records.append_many([example.example_record(record_id) for record_id in ("a", "b", "c")])
        scenario.records.append(
            AgentRecord.create(
                scenario="demo",
                agent_record_id="failed-instruction",
                request_type=RequestType.TRAIN,
                payload={"text": "fail this attempt", "session": "demo", "release_id": "base"},
            )
        )
        scenario.set_training_mode("manual")

        def _fail_step(self, batch, state, scenario_step):
            raise RuntimeError("injected instruction failure")

        with monkeypatch.context() as patch:
            patch.setattr(example.DemoBackend, "prepare_step", _fail_step)
            with pytest.raises(RuntimeError, match="injected instruction failure"):
                scenario.prepare_training_step()
        assert scenario.trainer.fail_pending_instruction("injected instruction failure")
        skipped = scenario.prepare_training_step()
        assert skipped is not None
        scenario.commit(skipped)
        assert scenario.store.history()[0].consumed_ids == frozenset({"failed-instruction"})
        assert scenario.store.history()[0].algorithm_state["replay"]["after_sequence"] == 0

        scenario.set_training_mode("auto")
        result = scenario.prepare_training_step()
        assert result is not None and result.metrics["record_ids"] == ["a", "b"]
        scenario.commit(result)
        scenario.records.append(example.example_record("live"))
        uncommitted = scenario.prepare_training_step()
        assert uncommitted is not None and uncommitted.metrics["record_ids"] == ["c"]
        # Even acknowledgement before a failed publication must not move the
        # durable cursor. Simulate a crash before the commit record is written.
        scenario.trainer.prepare_commit(uncommitted)

    restarted = Dispatcher(recipe, repository, scenario_storage=SQLiteScenarioStorage(storage_path))
    with closing(restarted):
        scenario = restarted.get_or_create_scenario("demo")
        for expected in (["c"], ["a", "b"], ["c"], ["live"]):
            result = scenario.prepare_training_step()
            assert result is not None and result.metrics["record_ids"] == expected
            scenario.commit(result)
        assert scenario.prepare_training_step() is None
        history = scenario.store.history()
        trained = [row for row in history if row.metrics.get("phase")]
        assert [row.metrics["dataset_epoch"] for row in trained] == [1, 1, 2, 2, None]
        assert [row.metrics["record_ids"] for row in trained] == [["a", "b"], ["c"], ["a", "b"], ["c"], ["live"]]
        assert len({row.artifact_ref.release_id for row in trained}) == 5
        assert scenario.records.count("demo", request_type=RequestType.INFERENCE) == 4
        assert scenario.trainer.algorithm_state_dict()["replay"]["after_sequence"] == 5

    with closing(Dispatcher(recipe, repository, scenario_storage=SQLiteScenarioStorage(storage_path))) as final:
        scenario = final.get_or_create_scenario("demo")
        assert scenario.prepare_training_step() is None
        scenario.records.append(example.example_record("later"))
        result = scenario.prepare_training_step()
        assert result is not None and result.metrics["record_ids"] == ["later"]
        scenario.commit(result)
        assert result.metrics["phase"] == "stream"


def test_replay_pages_and_buffers_only_one_batch(monkeypatch, example):
    with closing(SQLiteRecordStore()) as records:
        records.append_many([example.example_record(str(number)) for number in range(31)])
        reads = []
        replay_page = records.replay_page

        def _read_page(scenario, *, after_sequence=0, limit=256):
            reads.append(limit)
            return replay_page(scenario, after_sequence=after_sequence, limit=limit)

        monkeypatch.setattr(records, "replay_page", _read_page)
        processor = example.ReplayProcessor(
            ProcessorContext("demo", {"batch_size": 3}), records=records, progress=example.ReplayProgress(30, 2)
        )
        seen = []
        while processor.ready():
            batch = processor.build_batch()
            assert processor.build_batch() is batch
            assert len(batch.items) <= 3
            seen.append((batch.phase, batch.epoch, [item.source_agent_record_ids[0] for item in batch.items]))
            processor.acknowledge(batch.batch_id)
        assert max(reads) == 3
        assert [record_id for phase, _, ids in seen if phase == "dataset" for record_id in ids] == [
            str(number) for _ in range(2) for number in range(30)
        ]
        assert seen[-1] == ("stream", 3, ["30"])
        assert processor.status()["buffered_records"] == 0
        assert records.count("demo") == 31


def test_empty_dataset_and_capacity_eviction_do_not_block_live_records(tmp_path, caplog, example):
    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("demo")) as store:
        store.records.append_many([example.example_record(record_id) for record_id in ("a", "b", "c")])
        processor = example.ReplayProcessor(
            ProcessorContext("demo", {"batch_size": 2}), records=store.records, progress=example.ReplayProgress(3, 2)
        )
        assert processor.ready()
        batch = processor.build_batch()
        processor.acknowledge(batch.batch_id)
        assert storage.prune(days=7, max_bytes=1) > 0
        assert store.records.loss("demo").record_count == 3
        assert caplog.records
        assert not processor.ready()
        store.records.append(example.example_record("live"))
        assert processor.ready()
        batch = processor.build_batch()
        assert batch.phase == "stream"
        assert [item.source_agent_record_ids for item in batch.items] == [("live",)]


@pytest.mark.parametrize(("epoch", "sequence"), [(2, 1), (3, 4)])
def test_replay_progress_round_trips(epoch, sequence, example):
    progress = example.ReplayProgress(3, 2, epoch=epoch, after_sequence=sequence)
    assert example.ReplayProgress.from_dict(progress.to_dict()) == progress


def test_restart_rejects_changing_the_committed_dataset(example):
    with closing(SQLiteRecordStore()) as records, pytest.raises(ValueError, match="cannot change"):
        example.ReplayRecipe(dataset_last_sequence=4).build(
            "demo", records, algorithm_state={"replay": example.ReplayProgress(3, 2).to_dict()}
        )
