"""The training samples of a scenario (RFC #1140): the items of a data store that it uses, and its training state."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, inspect
from tests.support.samples import made_task
from tests.support.task_service_stubs import task_for

from reef.artifact import SnapshotRef
from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.tasks import write_harbor_task
from reef.storage.commits import CommitRecord
from reef.storage.data import DataStore, MemoryDataStore, NewItem
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.samples import SampleStore, consumed_sample_id, stage_item_id
from reef.storage.scenario import ScenarioStore
from reef.storage.sql_data import PostgresDataStore, SQLiteDataStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.tasks import LocalTaskFileStore


def trajectory(text: str, **fields) -> TrajectorySample:
    return TrajectorySample(
        {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": text}]},
        **fields,
    )


def upload(item_id: str, text: str = "", **scenarios: str) -> NewItem:
    return NewItem(item_id, trajectory(text or item_id).trajectory, {"kind": "trajectory", "scenarios": scenarios})


@dataclass
class Scenario:
    """The session of the scenario ``math`` and its samples: the items of ``acme`` in ``data`` that it uses.
    ``root`` holds the task directories to import."""

    session: ScenarioStore
    data: DataStore
    samples: SampleStore
    root: Path

    def task(self, name: str, split: str, port: int = 9000) -> None:
        directory = write_harbor_task(task_for(name, "record-1", port), self.root / "tasks" / str(port))
        self.data.create("acme", [NewItem(name, directory, {"kind": "task", "scenarios": {"math": split}})])

    def ids(self, trainer: str | None = None, **page) -> list[str]:
        samples = self.samples if trainer is None else self.samples.for_trainer(trainer)
        return [stored.sample_id for stored in samples.page(**page)]


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def scenario(request, tmp_path):
    task_files = LocalTaskFileStore(tmp_path / "task-files")
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        storage = PostgresScenarioStorage(url, tmp_path, schema=schema, task_files=task_files)
        data = PostgresDataStore(url, task_files, schema=schema)
    elif request.param == "sqlite":
        storage = SQLiteScenarioStorage(tmp_path, task_files=task_files)
        data = SQLiteDataStore(tmp_path / "data", task_files)
    else:
        storage = SQLiteScenarioStorage(task_files=task_files)
        data = MemoryDataStore(task_files)
    with closing(storage), closing(data), closing(storage.open("math")) as session:
        yield Scenario(session, data, SampleStore(data, "acme", "math", session.training_state), tmp_path)


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path / "files", schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path / "files")


def test_a_scenario_reads_the_items_that_it_uses_in_use_order(scenario):
    data = scenario.data
    data.create("acme", [upload("t1", math="train"), upload("t2"), upload("t3", other="train")])
    scenario.task("task-a", "eval")
    assert scenario.ids() == ["t1", "task-a"]
    first, task = scenario.samples.page()
    assert (first.sample, first.split, first.component, first.group) == (trajectory("t1"), "train", None, None)
    assert (task.kind, task.split) == ("task", "eval")

    # An older item that joins later reads after the others; a split change is a new use, which reads again.
    data.update("acme", ["t2"], {"scenarios": {"math": "train"}})
    assert scenario.ids() == ["t1", "task-a", "t2"]
    data.update("acme", ["task-a"], {"scenarios": {"math": "train"}})
    assert scenario.ids() == ["t1", "t2", "task-a"]
    assert scenario.ids(after_sequence=first.sequence, limit=1) == ["t2"]
    assert scenario.ids(kind="trajectory") == ["t1", "t2"]
    assert scenario.ids(after_sequence=scenario.samples.page()[-1].sequence) == []


def test_a_listing_shows_the_samples_without_reading_their_content(scenario, monkeypatch):
    scenario.data.create("acme", [upload("t1", math="train")])
    scenario.task("task-a", "eval")
    stored = scenario.samples.page()
    read = scenario.data.scenario_items

    def without_content(*args, **options):
        assert options["content"] is False
        return read(*args, **options)

    monkeypatch.setattr(scenario.data, "scenario_items", without_content)
    listed = scenario.samples.listing()
    assert [(row.sequence, row.sample_id, row.kind, row.split, row.component, row.group) for row in listed] == [
        (row.sequence, row.sample_id, row.kind, row.split, row.component, row.group) for row in stored
    ]
    assert scenario.samples.listing(after_sequence=listed[-1].sequence) == ()


def test_get_reads_the_samples_of_the_ids_that_the_scenario_uses_in_item_order(scenario):
    scenario.data.create("acme", [upload("t1", math="train"), upload("t2"), upload("t3", math="train")])
    assert [stored.sample_id for stored in scenario.samples.get(("t3", "t1", "t2", "missing"))] == ["t1", "t3"]


def test_stage_samples_become_items_of_their_trainer_and_consume_their_records(scenario):
    samples = scenario.samples
    scenario.data.create("acme", [upload("upload", math="train")])
    made = trajectory("a1", group_id="g", source_agent_record_ids=("r1",))
    task = made_task("x", source_agent_record_ids=("r2",), metadata={"task": {"name": "x"}})
    grid = (("a:r1:0", made), ("generated-tasks/x", task))
    assert samples.for_trainer("a").add_record_samples("a:r1", ("r1", "i1", "r2"), grid) == 2
    assert samples.for_trainer("a").add_record_samples("a:r1", ("r1", "i1", "r2"), grid) == 0
    samples.for_trainer("b").add_record_samples(None, ("r3",), (("b:r3:0", trajectory("b")),))

    item_id = stage_item_id("math", "a:r1:0")
    assert (item_id, stage_item_id("math", "generated-tasks/x")) == ("math.a:r1:0", "math.generated-tasks.x")
    metadata = scenario.data.read("acme", item_id).metadata
    assert {key: metadata[key] for key in ("kind", "scenarios", "origin", "group", "group_id")} == {
        "kind": "trajectory",
        "scenarios": {"math": "train"},
        "origin": {"scenario": "math", "component": "a", "records": ["r1"]},
        "group": "a:r1",
        "group_id": "g",
    }
    assert scenario.ids(trainer="a") == ["upload", item_id, "math.generated-tasks.x"]
    assert scenario.ids(trainer="b") == ["upload", "math.b:r3:0"]
    assert len(scenario.ids()) == 4
    _, stored, stored_task = samples.for_trainer("a").page()
    assert (stored.sample, stored.component, stored.group) == (made, "a", "a:r1")
    assert isinstance(stored_task.sample, TaskSample) and stored_task.component == "a"
    assert stored_task.sample.metadata == {"task": {"name": "x"}}
    assert stored_task.sample.source_agent_record_ids == ("r2",)
    assert (stored_task.sample.task_path / "instruction.md").read_text() == "Solve x.\n"
    (other,) = samples.for_trainer("b").page()[1:]
    assert (other.component, other.group, other.sample.source_agent_record_ids) == ("b", None, ())

    # An item that exists stays as it is.
    assert samples.for_trainer("a").add_record_samples(None, ("r4",), (("a:r1:0", trajectory("other")),)) == 0
    assert samples.get((item_id,))[0].sample == made
    state = samples.state
    assert state.consumed_record_ids("a") == {"r1", "i1", "r2", "r4"}
    assert state.consumed_record_ids("b") == {"r3"} and state.consumed_record_ids("") == frozenset()


def training_commit(step: int, *consumed: str, component: str | None = None) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        snapshot=SnapshotRef(f"release:{step}", "base"),
        checkpoint=True,
        algorithm_state={"step": step},
        high_water_sequence=step,
        high_water_offset=step,
        consumed_ids=frozenset(consumed),
        recorded_at=1000.0 + step,
        component=component,
    )


def train_counts(samples: SampleStore) -> dict[str, dict[str, int]]:
    return {stored.sample_id: dict(stored.train_counts) for stored in samples.page()}


def test_a_commit_counts_each_sample_it_consumed_once_for_its_trainer(scenario):
    session = scenario.session
    scenario.data.create("acme", [upload("t1", math="train"), upload("t2", math="train")])
    session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1"), "record-1"))
    assert train_counts(scenario.samples) == {"t1": {"": 1}, "t2": {}}
    # A retry of the stored commit counts nothing again; a later commit counts again.
    session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1"), "record-1"))
    session.commit_step(expected_step=1, commit=training_commit(2, consumed_sample_id("t1"), consumed_sample_id("t2")))
    assert train_counts(scenario.samples) == {"t1": {"": 2}, "t2": {"": 1}}
    # The commit of one component of several counts for that component only.
    session.commit_step(expected_step=2, commit=training_commit(3, consumed_sample_id("t2"), component="b"))
    _, second = scenario.samples.page()
    assert second.train_counts == {"": 1, "b": 1}
    assert (second.train_count(""), second.train_count("b"), second.train_count("c")) == (1, 1, 0)


def test_a_commit_counts_more_samples_than_one_statement_can_name(scenario):
    # PostgreSQL takes at most 65535 parameters in one statement, and each count row has several.
    sample_ids = [f"t{index}" for index in range(20_000)]
    commit = training_commit(1, *(consumed_sample_id(sample_id) for sample_id in sample_ids))
    scenario.session.commit_step(expected_step=0, commit=commit)
    counts = scenario.session.training_state.train_counts(sample_ids)
    assert len(counts) == len(sample_ids) and counts["t19999"] == {"": 1}


def test_train_counts_survive_an_item_that_leaves_and_comes_back(scenario):
    scenario.data.create("acme", [upload("t1", math="train"), upload("t2", math="train")])
    scenario.session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1"), component="a"))
    last = scenario.samples.page()[-1].sequence
    scenario.data.update("acme", ["t1"], {"scenarios": {"math": None}})
    assert scenario.ids() == ["t2"]
    scenario.data.update("acme", ["t1"], {"scenarios": {"math": "train"}})
    (again,) = scenario.samples.get(("t1",))
    assert again.train_counts == {"a": 1}
    assert scenario.samples.page()[-1].sample_id == "t1" and scenario.samples.page()[-1].sequence > last


def test_a_commit_that_fails_counts_no_sample(scenario, monkeypatch):
    session = scenario.session
    scenario.data.create("acme", [upload("t1", math="train")])
    history = session.commit_history
    append = history.append

    def append_then_fail(record: CommitRecord) -> None:
        append(record)
        raise RuntimeError("fails after the counts")

    monkeypatch.setattr(history, "append", append_then_fail)
    with pytest.raises(RuntimeError, match="after the counts"):
        session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1")))
    assert train_counts(scenario.samples) == {"t1": {}}
    assert session.history() == ()


def test_imported_tasks_are_the_tasks_that_no_stage_made(scenario):
    samples = scenario.samples
    scenario.data.create("acme", [upload("t1", math="train")])
    scenario.task("task-a", "train")
    scenario.task("held", "eval", 9001)
    samples.for_trainer("").add_record_samples(
        None, (), (("generated-tasks/g", made_task("g", metadata={"task": {"name": "g"}})),)
    )
    assert [(task.sample_id, task.split) for task in samples.imported_tasks()] == [
        ("task-a", "train"),
        ("held", "eval"),
    ]
    (held,) = samples.imported_tasks(split="eval")
    assert held.sample_id == "held" and (held.sample.task_path / "instruction.md").is_file()
    train = scenario.data.read("acme", "task-a")
    (imported,) = samples.imported_tasks(split="train")
    assert imported.sample.metadata == {"task": {"name": "task-a", "files_digest": train.digest, "split": "train"}}
    assert scenario.ids(kind="task") == ["task-a", "held", "math.generated-tasks.g"]


def legacy_sample_tables(connection, schema: str | None, storage_id: str | None) -> None:
    """Make the ``training_sample`` and ``sample_record`` tables of the sample store before RFC #1140, with an upload
    and a sample that the trainer ``a`` made."""

    def scope() -> list[Column]:
        return [] if storage_id is None else [Column("storage_id", String(32), nullable=False)]

    samples = Table(
        "training_sample",
        MetaData(schema=schema),
        Column("sequence", Integer, primary_key=True),
        *scope(),
        Column("sample_id", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("split", String(8), nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
        Column("component", Text, nullable=True),
        Column("group_key", Text, nullable=True),
        Column("record_ids_json", Text, nullable=False, server_default="[]"),
    )
    records = Table(
        "sample_record",
        MetaData(schema=schema),
        *scope(),
        Column("sample_id", Text, primary_key=True),
        Column("agent_record_id", Text, primary_key=True),
    )
    samples.create(connection)
    records.create(connection)
    key = {} if storage_id is None else {"storage_id": storage_id}
    row = {"kind": "trajectory", "split": "train", "content_json": "{}", "content_sha256": "0" * 64}
    connection.execute(
        samples.insert(),
        [
            {**key, **row, "sample_id": "upload", "component": None},
            {**key, **row, "sample_id": "a:r1:0", "component": "a"},
        ],
    )
    connection.execute(records.insert().values(**key, sample_id="a:r1:0", agent_record_id="r1"))


def test_a_start_drops_the_sample_store_and_the_train_counts_of_its_stage_samples(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        consumed = (consumed_sample_id("upload"), consumed_sample_id("a:r1:0"))
        session.commit_step(expected_step=0, commit=training_commit(1, *consumed, component="a"))
        state = session.training_state
        schema = state.counts.table.schema
        with state.database._transaction(write=True) as connection:
            storage_id = state.counts.scope.get("storage_id")
            legacy_sample_tables(connection, schema, storage_id)

    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        state = session.training_state
        assert state.train_counts(["upload", "a:r1:0"]) == {"upload": {"a": 1}}
        with state.database._transaction(write=False) as connection:
            assert not inspect(connection).has_table("training_sample", schema=schema)
            assert not inspect(connection).has_table("sample_record", schema=schema)


def test_a_start_drops_the_manual_buffers_of_the_older_schema(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        state = session.training_state
        schema = state.counts.table.schema
        with state.database._transaction(write=True) as connection:
            legacy = Table(
                "manual_buffer",
                MetaData(schema=schema),
                *(Column(key, Text, primary_key=True) for key in ("trainer", "sample_id")),
            )
            legacy.create(connection)
            connection.execute(legacy.insert().values(trainer="", sample_id="t1"))

    with (
        closing(open_durable_storage()) as storage,
        closing(storage.open("math")) as session,
        session.training_state.database._transaction(write=False) as connection,
    ):
        assert not inspect(connection).has_table("manual_buffer", schema=schema)
