from __future__ import annotations

import hashlib
import json
from contextlib import closing

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, inspect, select, update
from tests.support.task_service_stubs import task_for

from reef.artifact import ArtifactRef
from reef.core.job_data import TaskSample, TrajectorySample
from reef.core.tasks import write_harbor_task
from reef.storage.commits import CommitRecord
from reef.storage.postgres import PostgresScenarioStorage, postgres_url
from reef.storage.samples import SampleConflict, consumed_sample_id
from reef.storage.sql_samples import sample_content
from reef.storage.sqlite import SQLiteScenarioStorage


def trajectory(text: str, **fields) -> TrajectorySample:
    return TrajectorySample(
        {"schema_version": "ATIF-v1.4", "agent": {"name": "fixture"}, "steps": [{"step_id": 1, "message": text}]},
        **fields,
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresScenarioStorage(url, tmp_path, schema=schema)
    else:
        opened = SQLiteScenarioStorage(None if request.param == "memory" else tmp_path)
    with closing(opened):
        yield opened


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path / "files", schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path / "files")


def test_uploaded_trajectories_keep_upload_order_per_scenario(storage):
    grouped = trajectory("b", group_id="g", source_agent_record_ids=("r1",))
    with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
        assert session.samples.add_trajectories((("t1", trajectory("a")), ("t2", grouped), ("t3", trajectory("c"))))
        stored = session.samples.page()
        assert [item.sample_id for item in stored] == ["t1", "t2", "t3"]
        assert stored[0].sequence < stored[1].sequence < stored[2].sequence
        assert [item.sample for item in stored] == [trajectory("a"), grouped, trajectory("c")]
        page = session.samples.page(after_sequence=stored[0].sequence, limit=1)
        assert [item.sample_id for item in page] == ["t2"]
        assert session.samples.count() == 3
        assert session.samples.count(after_sequence=stored[2].sequence) == 0
        assert other.samples.count() == 0


def test_a_retry_changes_nothing_and_a_conflict_saves_no_trajectory(storage):
    with closing(storage.open("math")) as session:
        assert session.samples.add_trajectories((("t1", trajectory("a")),))
        assert not session.samples.add_trajectories((("t1", trajectory("a")),))
        with pytest.raises(SampleConflict, match="other content"):
            session.samples.add_trajectories((("t2", trajectory("c")), ("t1", trajectory("b"))))
        with pytest.raises(ValueError, match="non-empty"):
            session.samples.add_trajectories((("t3", trajectory("c")), (" ", trajectory("d"))))
        assert [item.sample_id for item in session.samples.page()] == ["t1"]


def test_uploaded_trajectories_survive_a_restart_and_archive_retires_them(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.samples.add_trajectories((("t1", trajectory("a")), ("t2", trajectory("b"))))

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            assert [item.sample_id for item in session.samples.page()] == ["t1", "t2"]
        storage.archive("math")
        with closing(storage.open("math")) as session:
            assert session.samples.count() == 0


def training_commit(step: int, *consumed: str, component: str | None = None) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        artifact_ref=ArtifactRef(f"content:{step}", f"release:{step}", "base"),
        checkpoint=True,
        algorithm_state={"step": step},
        high_water_sequence=step,
        high_water_offset=step,
        consumed_ids=frozenset(consumed),
        recorded_at=1000.0 + step,
        component=component,
    )


def train_counts(session) -> dict[str, dict[str, int]]:
    return {item.sample_id: item.train_counts for item in session.samples.page()}


def test_a_commit_counts_each_sample_it_consumed_once_for_its_trainer(storage):
    with closing(storage.open("math")) as session:
        session.samples.add_trajectories((("t1", trajectory("a")), ("t2", trajectory("b"))))
        session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1"), "record-1"))
        assert train_counts(session) == {"t1": {"": 1}, "t2": {}}
        # A retry of the stored commit counts nothing again; a later commit counts again.
        session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1"), "record-1"))
        session.commit_step(
            expected_step=1, commit=training_commit(2, consumed_sample_id("t1"), consumed_sample_id("t2"))
        )
        assert train_counts(session) == {"t1": {"": 2}, "t2": {"": 1}}
        # The commit of one component of several counts for that component only.
        session.commit_step(expected_step=2, commit=training_commit(3, consumed_sample_id("t2"), component="b"))
        (_, second) = session.samples.page()
        assert second.train_counts == {"": 1, "b": 1}
        assert (second.train_count(None), second.train_count("b"), second.train_count("c")) == (1, 1, 0)


def test_a_commit_that_fails_counts_no_sample(storage, monkeypatch):
    with closing(storage.open("math")) as session:
        session.samples.add_trajectories((("t1", trajectory("a")),))
        history = session.commit_history
        append = history.append

        def append_then_fail(record: CommitRecord) -> None:
            append(record)
            raise RuntimeError("fails after the counts")

        monkeypatch.setattr(history, "append", append_then_fail)
        with pytest.raises(RuntimeError, match="after the counts"):
            session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1")))
        assert train_counts(session) == {"t1": {}}
        assert session.history() == ()


def write_legacy_trajectories(connection, *, schema: str | None, storage_id: str | None) -> None:
    """Make the uploaded_trajectory table of the older schema hold t1 and t2, and two commits list t1 as consumed.

    The first commit names no component, and the second names the component ``b``.
    """
    scope = [] if storage_id is None else [Column("storage_id", String(32), nullable=False)]
    legacy = Table(
        "uploaded_trajectory",
        MetaData(schema=schema),
        Column("sequence", Integer, primary_key=True),
        *scope,
        Column("trajectory_id", Text, nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
    )
    legacy.create(connection)
    for name, text in (("t1", "a"), ("t2", "b")):
        content = sample_content(trajectory(text))
        row = {
            "trajectory_id": name,
            "content_json": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
        connection.execute(legacy.insert().values(row if storage_id is None else {**row, "storage_id": storage_id}))
    commits = Table("scenario_commit", MetaData(schema=schema), autoload_with=connection)
    for step, component in ((1, None), (2, "b")):
        encoded = json.dumps(training_commit(step, "trajectory:t1", component=component).to_dict())
        values = {"step": step, "commit_json": encoded}
        connection.execute(
            commits.insert().values(values if storage_id is None else {**values, "storage_id": storage_id})
        )
    samples = Table("training_sample", MetaData(schema=schema), autoload_with=connection)
    connection.execute(samples.delete())


def test_sqlite_moves_the_uploaded_trajectories_of_an_older_schema_with_their_train_counts(tmp_path):
    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")):
        pass
    (database,) = tmp_path.glob("*.sqlite3")
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        write_legacy_trajectories(connection, schema=None, storage_id=None)
    engine.dispose()

    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")) as session:
        assert train_counts(session) == {"t1": {"": 1, "b": 1}, "t2": {}}
        assert [item.sample for item in session.samples.page()] == [trajectory("a"), trajectory("b")]
    engine = create_engine(f"sqlite:///{database}")
    assert not inspect(engine).has_table("uploaded_trajectory")
    engine.dispose()


def test_postgres_moves_the_uploaded_trajectories_of_schema_version_9_with_their_train_counts(
    postgres_config, tmp_path
):
    url, schema = postgres_config
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage, closing(storage.open("math")):
        pass
    engine = create_engine(postgres_url(url))
    with engine.begin() as connection:
        stores = Table("record_store", MetaData(schema=schema), autoload_with=connection)
        storage_id = connection.execute(select(stores.c.storage_id)).scalar_one()
        write_legacy_trajectories(connection, schema=schema, storage_id=storage_id)
        version = Table("schema_version", MetaData(schema=schema), autoload_with=connection)
        connection.execute(update(version).values(version=9))
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as session,
    ):
        assert train_counts(session) == {"t1": {"": 1, "b": 1}, "t2": {}}
    with engine.connect() as connection:
        assert not inspect(connection).has_table("uploaded_trajectory", schema=schema)
        version = Table("schema_version", MetaData(schema=schema), autoload_with=connection)
        assert connection.execute(select(version.c.version)).scalars().all() == [12]
    engine.dispose()


def task_directory(root, name: str, port: int = 9000):
    return write_harbor_task(task_for(name, "record-1", port), root)


def test_tasks_keep_their_split_and_files_and_share_the_ids_of_the_trajectories(storage, tmp_path):
    with closing(storage.open("math")) as session:
        samples = session.samples
        samples.add_trajectories((("t1", trajectory("a")),))
        (train,) = samples.add_tasks((task_directory(tmp_path / "a", "task-a"),), "train")
        samples.add_tasks((task_directory(tmp_path / "b", "held", 9001),), "eval")
        assert isinstance(train.sample, TaskSample) and train.sample.task_path.is_dir()
        assert train.sample.metadata["task"] == {
            "name": "task-a",
            "files_digest": train.files_digest,
            "split": "train",
        }
        assert [(task.sample_id, task.split) for task in samples.tasks()] == [("task-a", "train"), ("held", "eval")]
        assert [task.sample_id for task in samples.tasks(split="eval")] == ["held"]
        assert [item.sample_id for item in samples.page(kind="trajectory")] == ["t1"]
        assert samples.count(kind="task") == 2 and samples.count() == 3
        # The same task again is returned as it is; other files, the other split, or a trajectory's ID conflict.
        assert samples.add_tasks((task_directory(tmp_path / "c", "task-a"),), "train") == (train,)
        with pytest.raises(SampleConflict, match="other content"):
            samples.add_tasks((task_directory(tmp_path / "d", "task-a", 9002),), "train")
        with pytest.raises(SampleConflict, match="stored in train, not eval"):
            samples.add_tasks((task_directory(tmp_path / "e", "task-a"),), "eval")
        with pytest.raises(SampleConflict, match="stored as a trajectory"):
            samples.add_tasks((task_directory(tmp_path / "f", "t1"),), "train")
        assert samples.count() == 3


def test_a_start_moves_the_imported_tasks_of_the_old_task_store_with_their_train_counts(
    open_durable_storage, tmp_path
):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        store = session.tasks.open("imported-tasks")
        for port, (name, split) in enumerate((("task-a", "train"), ("task-b", "train"), ("held", "eval"))):
            store.add_task(directory=task_directory(tmp_path / "tasks", name, 9000 + port), split=split)
        store.mark_trained(task_names=("task-a",))
        session.commit_step(
            expected_step=0,
            commit=training_commit(1, "rollout:math:rollout:0123456789ab/task-a", component="b"),
        )

    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        assert [(task.sample_id, task.split, task.train_counts) for task in session.samples.tasks()] == [
            ("task-a", "train", {"b": 1}),
            ("task-b", "train", {}),
            ("held", "eval", {}),
        ]
        (held,) = session.samples.tasks(split="eval")
        assert isinstance(held.sample, TaskSample) and (held.sample.task_path / "instruction.md").is_file()
        assert session.tasks.open("imported-tasks").tasks() == ()


def test_record_samples_belong_to_their_trainer_and_consume_their_records(storage):
    with closing(storage.open("math")) as session:
        samples = session.samples
        samples.add_trajectories((("upload", trajectory("u")),))
        grid = (("a:r1:0", trajectory("a1")), ("a:r2:0", trajectory("a2")))
        assert samples.add_record_samples("a", "a:r1", ("r1", "i1", "r2"), grid)
        assert not samples.add_record_samples("a", "a:r1", ("r1", "i1", "r2"), grid), "a retry changes nothing"
        samples.add_record_samples("b", None, ("r3",), (("b:r3:0", trajectory("b")),))
        with pytest.raises(SampleConflict, match="other content"):
            samples.add_record_samples("a", "a:r1", ("r1",), (("a:r1:0", trajectory("other")),))
        with pytest.raises(ValueError, match="one or more samples"):
            samples.add_record_samples("a", None, ("r9",), ())

        assert [item.sample_id for item in samples.page(trainer="a")] == ["upload", "a:r1:0", "a:r2:0"]
        assert [item.sample_id for item in samples.page(trainer="b")] == ["upload", "b:r3:0"]
        assert samples.count(trainer="a") == 3 and samples.count() == 4
        (_, first, _) = samples.page(trainer="a")
        assert (first.component, first.group, first.record_ids) == ("a", "a:r1", ("i1", "r1", "r2"))
        assert samples.page()[0].component is None and samples.page()[0].record_ids == ()
        assert samples.consumed_record_ids("a") == {"r1", "i1", "r2"}
        assert samples.consumed_record_ids("b") == {"r3"} and samples.consumed_record_ids("") == frozenset()


def test_sqlite_adds_the_record_sample_columns_to_an_older_sample_table(tmp_path):
    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")) as session:
        session.samples.add_trajectories((("t1", trajectory("a")),))
    (database,) = tmp_path.glob("*.sqlite3")
    engine = create_engine(f"sqlite:///{database}")
    with engine.begin() as connection:
        for column in ("component", "group_key", "record_ids_json"):
            connection.exec_driver_sql(f"ALTER TABLE training_sample DROP COLUMN {column}")
    engine.dispose()

    with closing(SQLiteScenarioStorage(tmp_path)) as storage, closing(storage.open("math")) as session:
        (stored,) = session.samples.page()
        assert (stored.sample_id, stored.component, stored.group, stored.record_ids) == ("t1", None, None, ())
        session.samples.add_record_samples("", None, ("r1",), (("r1:0", trajectory("b")),))
        assert session.samples.consumed_record_ids("") == {"r1"}


def test_a_trainers_manual_buffer_keeps_store_order_and_a_commit_removes_what_it_trained(storage):
    with closing(storage.open("math")) as session:
        samples = session.samples
        samples.add_trajectories((("t1", trajectory("a")), ("t2", trajectory("b")), ("t3", trajectory("c"))))
        samples.insert_into_buffer("", ("t3", "t1"))
        samples.insert_into_buffer("", ("t1",))
        samples.insert_into_buffer("other", ("t2",))
        assert samples.manual_buffer("") == ("t1", "t3")
        assert [item.sample_id for item in samples.get(("t3", "t1", "missing"))] == ["t1", "t3"]
        samples.remove_from_buffer("", ("t3", "missing"))
        assert samples.manual_buffer("") == ("t1",)
        # The commit of the lone trainer trains t1: it leaves that trainer's buffer, not another's.
        samples.insert_into_buffer("", ("t2",))
        session.commit_step(expected_step=0, commit=training_commit(1, consumed_sample_id("t1")))
        assert samples.manual_buffer("") == ("t2",) and samples.manual_buffer("other") == ("t2",)
