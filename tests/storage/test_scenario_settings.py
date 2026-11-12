"""Scenario settings live in the storage backend, and an upgrade drops the retired model setting."""

from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.serving_config import ServingConfig
from reef.core.snapshot import SnapshotRef
from reef.storage.commits import CommitHistoryError, CommitRecord
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.scenario import ScenarioSettings
from reef.storage.sqlite import SQLiteScenarioStorage

EVALUATION = {"harness": {"name": "terminus-2"}}


def committed(step: int = 1) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        commit_number=step,
        snapshot=SnapshotRef(f"release:{step}", "base"),
        checkpoint=True,
        algorithm_state={"step": step},
        consumed_ids=frozenset({f"record-{step}"}),
        high_water_sequence=step,
        high_water_offset=step,
        recorded_at=1000.0 + step,
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def open_storage(request, tmp_path):
    """Open storages over one location; memory storage keeps settings for its own lifetime only."""
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path, schema=schema)
    if request.param == "sqlite":
        return lambda: SQLiteScenarioStorage(tmp_path)
    memory = SQLiteScenarioStorage(None)
    return lambda: memory


def test_settings_default_replace_and_stay_per_scenario(open_storage):
    storage = open_storage()
    assert storage.read_settings("math") == ScenarioSettings()
    settings = ScenarioSettings(
        training_mode="manual",
        artifact_evaluation=EVALUATION,
        activation={"module": "evaluated", "config": {"min_mean_score": 0.5}},
        serving_config=ServingConfig({"weights": {"temperature": 0.6, "min_replicas": 1}}),
        record2sample_config={"records": {"max_staleness": 2, "steps": [{"keep_score": {"min": 1.0}}]}},
    )
    storage.write_settings("math", settings)
    assert open_storage().read_settings("math") == settings
    assert open_storage().read_settings("code") == ScenarioSettings()
    storage.write_settings("math", ScenarioSettings(training_mode="auto"))
    assert open_storage().read_settings("math") == ScenarioSettings(training_mode="auto")


@pytest.mark.parametrize(
    "config",
    [[], {"": {}}, {"records": "max_staleness=2"}, {"records": {"limit": float("nan")}}],
)
def test_settings_refuse_a_record2sample_config_that_is_not_json_objects_by_component(config):
    with pytest.raises(ValueError, match="record2sample_config"):
        ScenarioSettings(record2sample_config=config)


def test_postgres_session_of_an_archived_scenario_refuses_commit_reads(postgres_config, tmp_path):
    url, schema = postgres_config
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage:
        store = storage.open("math")
        store.append_commit(expected_commit_number=0, commit=committed())
        storage.archive("math")
        with pytest.raises(RuntimeError, match="archived"):
            store.history()
        # The records are record items, and the archive removed the scenario's uses of them.
        assert store.records.count("math") == 0
        store.close()


def test_archive_retires_settings_with_records_and_commits(open_storage):
    storage = open_storage()
    with closing(storage.open("math")) as store:
        store.append_commit(expected_commit_number=0, commit=committed())
    storage.write_settings("math", ScenarioSettings(artifact_evaluation=EVALUATION))
    storage.archive("math")
    assert storage.read_settings("math") == ScenarioSettings()
    with closing(storage.open("math")) as store:
        assert store.history() == ()


def test_reading_settings_of_an_unknown_scenario_creates_nothing(tmp_path):
    storage = SQLiteScenarioStorage(tmp_path)
    assert storage.read_settings("math") == ScenarioSettings()
    # Only the storage's own data store is there.
    assert [path.name for path in tmp_path.iterdir()] == ["data"]


@pytest.mark.parametrize("value", [["not", "an", "object"], 3])
def test_settings_reject_values_of_the_wrong_type(value):
    with pytest.raises(ValueError):
        ScenarioSettings(artifact_evaluation=value)
    with pytest.raises(ValueError):
        ScenarioSettings(training_mode=value)


def test_postgres_restart_on_an_empty_directory_finds_commits_and_settings(postgres_config, tmp_path):
    url, schema = postgres_config
    settings = ScenarioSettings(artifact_evaluation=EVALUATION, training_mode="manual")
    with closing(PostgresScenarioStorage(url, tmp_path / "first", schema=schema)) as storage:
        with closing(storage.open("math")) as store:
            store.append_commit(expected_commit_number=0, commit=committed())
            store.append_commit(expected_commit_number=1, commit=committed(2))
        storage.write_settings("math", settings)
    assert [path for path in (tmp_path / "first").rglob("*") if path.is_file()] == []
    with closing(PostgresScenarioStorage(url, tmp_path / "second", schema=schema)) as storage:
        assert storage.read_settings("math") == settings
        with closing(storage.open("math")) as store:
            assert store.history() == (committed(), committed(2))
            assert store.training_run_position() == (0, 2)
            assert store.recover(checkpoint=None) == committed(2)


def test_postgres_commit_table_keeps_identical_retries_and_rejects_different_content(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as store,
    ):
        store.append_commit(expected_commit_number=0, commit=committed())
        commits = store.commit_history
        commits.append(CommitRecord.from_dict({**committed().to_dict(), "recorded_at": 5.0}))
        with pytest.raises(CommitHistoryError, match="conflicts with its existing record"):
            commits.append(CommitRecord.from_dict({**committed().to_dict(), "checkpoint": False}))
        assert commits.records() == (committed(),)
