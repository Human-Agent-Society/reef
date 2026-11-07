"""A scenario's records are record items in the data store (RFC #1028, RFC #1219)."""

from __future__ import annotations

import time
from pathlib import Path

from tests.support.trajectory_uploads import UploadRecipe, dispatcher_for

from reef.core.records_types import AgentRecord, RequestType
from reef.dispatcher import Dispatcher


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
    finally:
        dispatcher.close()


def test_another_scenario_of_the_owner_trains_on_an_item_that_names_it_and_stores_no_copy(tmp_path: Path) -> None:
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
            trainer = current.trainers[0].trainer
            assert trainer.holds_untrained_records()
            assert trainer.produce_samples().records_read == 1
            assert trainer.produce_samples().records_read == 0, "a process reads each record once"
            assert current.store.records.get("t", "s/r1") is None, "t stores no copy"
            assert current.used_record("s/r1").payload == {"score": 1.0}

        # A new process reads the item again: no commit consumed it.
        dispatcher.registry.reload("t")
        with dispatcher.locked_scenario("t") as current:
            assert current.trainers[0].trainer.produce_samples().records_read == 1

        # Without the use, the read stops.
        dispatcher.data_store.update("s", ["s/r1"], {"scenarios": {"t": None}})
        with dispatcher.locked_scenario("t") as current:
            assert current.used_record("s/r1") is None
        dispatcher.data_store.update("s", ["s/r1"], {"scenarios": {"t": "train"}})

        # Archiving t releases the item; the record stays an item of its origin.
        dispatcher.delete_scenario("t")
        (item,) = dispatcher.data_store.query("s", kind="record")
        assert item.scenarios == {"s": "train"}

        # Deleting the origin releases its record items too, so a new scenario with its name reads none of them.
        dispatcher.delete_scenario("s")
        (item,) = dispatcher.data_store.query("s", kind="record")
        assert item.scenarios == {}
        dispatcher.get_or_create_scenario("s")
        assert dispatcher.data_store.query("s", scenario="s") == ()
    finally:
        dispatcher.close()
