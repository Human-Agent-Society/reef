from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.batches import TrajectoryItem
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.storage.trajectories import TrajectoryConflict


def trajectory(text: str, **fields) -> TrajectoryItem:
    return TrajectoryItem(
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
        assert session.trajectories.add((("t1", trajectory("a")), ("t2", grouped), ("t3", trajectory("c"))))
        stored = session.trajectories.page()
        assert [item.trajectory_id for item in stored] == ["t1", "t2", "t3"]
        assert stored[0].sequence < stored[1].sequence < stored[2].sequence
        assert [item.trajectory for item in stored] == [trajectory("a"), grouped, trajectory("c")]
        page = session.trajectories.page(after_sequence=stored[0].sequence, limit=1)
        assert [item.trajectory_id for item in page] == ["t2"]
        assert session.trajectories.count() == 3
        assert session.trajectories.count(after_sequence=stored[2].sequence) == 0
        assert other.trajectories.count() == 0


def test_a_retry_changes_nothing_and_a_conflict_saves_no_trajectory(storage):
    with closing(storage.open("math")) as session:
        assert session.trajectories.add((("t1", trajectory("a")),))
        assert not session.trajectories.add((("t1", trajectory("a")),))
        with pytest.raises(TrajectoryConflict, match="other content"):
            session.trajectories.add((("t2", trajectory("c")), ("t1", trajectory("b"))))
        with pytest.raises(ValueError, match="non-empty"):
            session.trajectories.add((("t3", trajectory("c")), (" ", trajectory("d"))))
        assert [item.trajectory_id for item in session.trajectories.page()] == ["t1"]


def test_uploaded_trajectories_survive_a_restart_and_archive_retires_them(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.trajectories.add((("t1", trajectory("a")), ("t2", trajectory("b"))))

    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session:
            assert [item.trajectory_id for item in session.trajectories.page()] == ["t1", "t2"]
        storage.archive("math")
        with closing(storage.open("math")) as session:
            assert session.trajectories.count() == 0
