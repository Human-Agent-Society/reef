"""Scenario settings live in the storage backend, and an upgrade drops the retired model setting."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing

import pytest
from sqlalchemy import select

from reef.core.artifact_ref import ArtifactRef
from reef.storage.commits import CommitHistoryError, CommitRecord
from reef.storage.postgres import POSTGRES_SCHEMA_VERSION, PostgresRecordDatabase, PostgresScenarioStorage
from reef.storage.scenario import ScenarioSettings
from reef.storage.sqlite import SQLiteScenarioStorage

EVALUATION = {"harness": {"name": "terminus-2"}}


def committed(step: int = 1) -> CommitRecord:
    return CommitRecord(
        scenario="math",
        step=step,
        artifact_ref=ArtifactRef(f"content:{step}", f"release:{step}", "base"),
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
    )
    storage.write_settings("math", settings)
    assert open_storage().read_settings("math") == settings
    assert open_storage().read_settings("code") == ScenarioSettings()
    storage.write_settings("math", ScenarioSettings(training_mode="auto"))
    assert open_storage().read_settings("math") == ScenarioSettings(training_mode="auto")


def test_postgres_session_of_an_archived_scenario_refuses_commit_reads(postgres_config, tmp_path):
    url, schema = postgres_config
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage:
        store = storage.open("math")
        store.commit_step(expected_step=0, commit=committed())
        storage.archive("math")
        with pytest.raises(RuntimeError, match="archived"):
            store.history()
        with pytest.raises(RuntimeError, match="archived"):
            store.records.count("math")
        store.close()


def test_archive_retires_settings_with_records_and_commits(open_storage):
    storage = open_storage()
    with closing(storage.open("math")) as store:
        store.commit_step(expected_step=0, commit=committed())
    storage.write_settings("math", ScenarioSettings(artifact_evaluation=EVALUATION))
    storage.archive("math")
    assert storage.read_settings("math") == ScenarioSettings()
    with closing(storage.open("math")) as store:
        assert store.history() == ()


def test_reading_settings_of_an_unknown_scenario_creates_nothing(tmp_path):
    storage = SQLiteScenarioStorage(tmp_path)
    assert storage.read_settings("math") == ScenarioSettings()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [["not", "an", "object"], 3])
def test_settings_reject_values_of_the_wrong_type(value):
    with pytest.raises(ValueError):
        ScenarioSettings(artifact_evaluation=value)
    with pytest.raises(ValueError):
        ScenarioSettings(training_mode=value)


def test_sqlite_upgrade_drops_a_stored_model_setting(tmp_path):
    """A database of an older version keeps a ``model_config`` row; the next open deletes it and keeps the rest."""
    SQLiteScenarioStorage(tmp_path).write_settings("math", ScenarioSettings(training_mode="manual"))
    database = tmp_path / f"{hashlib.sha256(b'math').hexdigest()}.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "INSERT INTO scenario_setting (name, value_json) VALUES ('model_config', ?)",
            (json.dumps({"url": "http://model.example", "model": "served"}),),
        )
    assert SQLiteScenarioStorage(tmp_path).read_settings("math") == ScenarioSettings(training_mode="manual")
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT name FROM scenario_setting").fetchall() == [("training_mode",)]


def test_postgres_restart_on_an_empty_directory_finds_commits_and_settings(postgres_config, tmp_path):
    url, schema = postgres_config
    settings = ScenarioSettings(artifact_evaluation=EVALUATION, training_mode="manual")
    with closing(PostgresScenarioStorage(url, tmp_path / "first", schema=schema)) as storage:
        with closing(storage.open("math")) as store:
            store.commit_step(expected_step=0, commit=committed())
            store.commit_step(expected_step=1, commit=committed(2))
        storage.write_settings("math", settings)
    assert [path for path in (tmp_path / "first").rglob("*") if path.is_file()] == []
    with closing(PostgresScenarioStorage(url, tmp_path / "second", schema=schema)) as storage:
        assert storage.read_settings("math") == settings
        with closing(storage.open("math")) as store:
            assert store.history() == (committed(), committed(2))
            assert store.training_run_position() == (0, 2)
            assert store.recover(checkpoint=None) == committed(2)


def test_postgres_upgrade_from_version_15_drops_a_stored_model_setting(postgres_config, tmp_path):
    url, schema = postgres_config
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage:
        storage.write_settings("math", ScenarioSettings(training_mode="manual"))
    with closing(PostgresRecordDatabase(url, schema=schema)) as database:
        settings = database.tables.settings
        storage_id = database.active_storage_id("math")
        with database.transaction() as connection:
            # Write the row that version 15 kept for a scenario's model override.
            connection.execute(
                settings.insert().values(
                    storage_id=storage_id,
                    name="model_config",
                    value_json=json.dumps({"url": "http://model.example", "model": "served"}),
                )
            )
            version = settings.metadata.tables[f"{schema}.schema_version"]
            connection.execute(version.update().values(version=15))
    with closing(PostgresRecordDatabase(url, schema=schema)) as database, database.transaction() as connection:
        names = connection.execute(select(database.tables.settings.c.name)).scalars().all()
        version = database.tables.settings.metadata.tables[f"{schema}.schema_version"]
        assert connection.execute(select(version.c.version)).scalar_one() == POSTGRES_SCHEMA_VERSION
    assert names == ["training_mode"]
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage:
        assert storage.read_settings("math") == ScenarioSettings(training_mode="manual")


def test_postgres_commit_table_keeps_identical_retries_and_rejects_different_content(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as store,
    ):
        store.commit_step(expected_step=0, commit=committed())
        commits = store.commit_history
        commits.append(CommitRecord.from_dict({**committed().to_dict(), "recorded_at": 5.0}))
        with pytest.raises(CommitHistoryError, match="conflicts with its existing record"):
            commits.append(CommitRecord.from_dict({**committed().to_dict(), "checkpoint": False}))
        assert commits.records() == (committed(),)
