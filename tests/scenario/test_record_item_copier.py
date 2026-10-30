"""The record item copier (RFC #1028): a scenario's records follow into the data store, also after a restart."""

from __future__ import annotations

import time
from contextlib import closing
from pathlib import Path

from tests.support.trajectory_uploads import UploadRecipe, dispatcher_for

from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.storage import record_items
from reef.storage.sqlite import SQLiteScenarioStorage


def report(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload={"score": 1.0}, agent_record_id=record_id
    )


def record_item_ids(dispatcher: Dispatcher, count: int) -> list[str]:
    """The IDs of the record items of owner ``s``, after ``count`` of them exist."""
    deadline = time.monotonic() + 10
    while len(items := dispatcher.data_store.query("s", kind="record")) < count:
        if time.monotonic() > deadline:
            raise AssertionError(f"{len(items)} of {count} record items exist")
        time.sleep(0.02)
    return [item.item_id for item in items]


def test_each_accepted_report_becomes_a_record_item(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    try:
        dispatcher.get_or_create_scenario("s")
        dispatcher.accept_record(report("r1"))
        dispatcher.accept_record(report("r2"))
        assert record_item_ids(dispatcher, 2) == ["s/r1", "s/r2"]
        (item,) = dispatcher.data_store.query("s", kind="record", limit=1)
        assert item.scenarios == {"s": "train"}
        deadline = time.monotonic() + 10
        while dispatcher.stored_settings("s").record_items_sequence < 2:
            assert time.monotonic() < deadline, "the copier did not store its sequence"
            time.sleep(0.02)
    finally:
        dispatcher.close()


def test_the_first_load_moves_the_old_records_in_pages(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(record_items, "RECORD_PAGE_SIZE", 2)
    with closing(SQLiteScenarioStorage(tmp_path / "records")) as storage, closing(storage.open("s")) as session:
        for index in range(5):
            session.records.append(report(f"r{index}"))
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    try:
        dispatcher.get_or_create_scenario("s")
        assert record_item_ids(dispatcher, 5) == [f"s/r{index}" for index in range(5)]
    finally:
        dispatcher.close()


def test_another_scenario_of_the_owner_gets_a_copy_when_an_item_names_it(tmp_path: Path) -> None:
    dispatcher = dispatcher_for(tmp_path, UploadRecipe())
    try:
        for scenario in ("s", "t"):
            dispatcher.get_or_create_scenario(scenario)
        dispatcher.set_scenario_owner("t", "s")
        dispatcher.accept_record(report("r1"))
        record_item_ids(dispatcher, 1)
        dispatcher.data_store.update("s", ["s/r1"], {"scenarios": {"t": "train"}})
        dispatcher.sync_items("s", ["t"])
        with dispatcher.locked_scenario("t") as current:
            assert current.store.records.get("t", "s/r1").payload == {"score": 1.0}
        # Archiving t releases the item; the record stays an item of its origin.
        dispatcher.delete_scenario("t")
        (item,) = dispatcher.data_store.query("s", kind="record")
        assert item.scenarios == {"s": "train"}
    finally:
        dispatcher.close()
