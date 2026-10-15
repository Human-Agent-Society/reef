"""Scenario settings live in the storage backend, and older local files are imported once."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing

import pytest

from reef.core.artifact_ref import ArtifactRef
from reef.storage.commit_log import CommitLog
from reef.storage.commits import CommitLogError, CommitRecord
from reef.storage.postgres import PostgresRecordDatabase, PostgresScenarioStorage
from reef.storage.scenario import ScenarioSettings
from reef.storage.sqlite import SQLiteScenarioStorage

MODEL = {"url": "http://model.example", "model": "served", "api": "openai", "api_key": "scoped-key"}


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
    settings = ScenarioSettings(model_config=MODEL, training_mode="manual")
    storage.write_settings("math", settings)
    assert open_storage().read_settings("math") == settings
    assert open_storage().read_settings("code") == ScenarioSettings()
    storage.write_settings("math", ScenarioSettings(training_mode="hybrid"))
    assert open_storage().read_settings("math") == ScenarioSettings(training_mode="hybrid")


def test_archive_retires_settings_with_records_and_commits(open_storage):
    storage = open_storage()
    with closing(storage.open("math")) as store:
        store.commit_step(expected_step=0, commit=committed())
    storage.write_settings("math", ScenarioSettings(model_config=MODEL))
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
        ScenarioSettings(model_config=value)
    with pytest.raises(ValueError):
        ScenarioSettings(training_mode=value)


def test_sqlite_imports_model_configuration_files_once(tmp_path):
    key = hashlib.sha256(b"math").hexdigest()
    (tmp_path / f"{key}-model.json").write_text(json.dumps(MODEL))
    (tmp_path / f"{hashlib.sha256(b'reset').hexdigest()}-model.json").write_text("null")
    storage = SQLiteScenarioStorage(tmp_path)
    assert storage.read_settings("math") == ScenarioSettings(model_config=MODEL)
    assert storage.read_settings("reset") == ScenarioSettings()
    assert list(tmp_path.glob("*-model.json")) == []
    assert len(list((tmp_path / "archived").glob("imported-*/*-model.json"))) == 2
    assert (tmp_path / f"{key}.sqlite3").stat().st_mode & 0o777 == 0o600
    assert SQLiteScenarioStorage(tmp_path).read_settings("math") == ScenarioSettings(model_config=MODEL)


def test_postgres_restart_on_an_empty_directory_finds_commits_and_settings(postgres_config, tmp_path):
    url, schema = postgres_config
    settings = ScenarioSettings(model_config=MODEL, training_mode="manual")
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


def test_postgres_imports_commit_logs_and_model_files_of_its_own_schema(postgres_config, tmp_path):
    url, schema = postgres_config
    with closing(PostgresRecordDatabase(url, schema=schema)) as database:
        storage_id = database.storage_id("math")
    log = CommitLog(tmp_path / f"postgres-{storage_id}.commits.jsonl")
    log.append(committed())
    log.append(committed(2))
    (tmp_path / f"{hashlib.sha256(b'math').hexdigest()}-model.json").write_text(json.dumps(MODEL))
    other_schema_log = tmp_path / f"postgres-{'0' * 32}.commits.jsonl"
    other_schema_log.write_text("")
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage:
        assert storage.read_settings("math") == ScenarioSettings(model_config=MODEL)
        with closing(storage.open("math")) as store:
            assert store.history() == (committed(), committed(2))
    assert not log.path.exists()
    assert other_schema_log.exists()
    assert len(list((tmp_path / "archived").glob("imported-*/*"))) == 2
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as store,
    ):
        assert store.history() == (committed(), committed(2))


def test_postgres_commit_table_keeps_identical_retries_and_rejects_different_content(postgres_config, tmp_path):
    url, schema = postgres_config
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as storage,
        closing(storage.open("math")) as store,
    ):
        store.commit_step(expected_step=0, commit=committed())
        commits = store.commit_history
        commits.append(CommitRecord.from_dict({**committed().to_dict(), "recorded_at": 5.0}))
        with pytest.raises(CommitLogError, match="conflicts with its existing record"):
            commits.append(CommitRecord.from_dict({**committed().to_dict(), "checkpoint": False}))
        assert commits.records() == (committed(),)
