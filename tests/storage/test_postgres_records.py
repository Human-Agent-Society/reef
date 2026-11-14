"""Real PostgreSQL coverage for the dialect and deployment lifecycle boundaries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from threading import Barrier

import pytest

from reef.core.errors import ReefError, ScenarioDeletedError
from reef.core.records_types import AgentRecord, RequestType
from reef.core.snapshot import SnapshotRef
from reef.service.assembly import _recipe_owned_settings, build_dispatcher
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.deployment_config import translate_layout
from reef.service.deploy.service_config import ServiceSettings, service_config_arguments, service_config_from_mapping
from reef.storage.commits import CommitRecord
from reef.storage.postgres import PostgresRecordDatabase, PostgresScenarioStorage, postgres_url
from reef.storage.record_items import ItemRecordStore


def record(record_id="first", scenario="math"):
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"text": "中文 🐟", "nested": {"body": record_id}},
        created_at=1789200000.123456,
    )


def test_concurrent_database_initialization(postgres_config):
    url, schema = postgres_config
    barrier = Barrier(2)

    def initialize():
        barrier.wait(timeout=5)
        PostgresRecordDatabase(url, schema=schema).close()

    with ThreadPoolExecutor(2) as pool:
        for result in [pool.submit(initialize) for _ in range(2)]:
            result.result()


def test_schema_version_and_database_close(postgres_config):
    url, schema = postgres_config
    database = PostgresRecordDatabase(url, schema=schema)
    with database.transaction() as connection:
        version = database.settings_table.metadata.tables[f"{schema}.schema_version"]
        connection.execute(version.update().values(version=999))
    database.close()
    database.close()
    with pytest.raises(RuntimeError, match="closed"), database.transaction():
        pass
    with pytest.raises(ReefError, match="unsupported PostgreSQL record schema version"):
        PostgresRecordDatabase(url, schema=schema)


def test_postgres_dependency_is_optional_and_error_is_actionable(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ReefError, match=r"reef-infra\[postgres\]"):
        PostgresRecordDatabase("postgresql:///unused")


def test_factory_commit_recovery_and_archive(postgres_config, tmp_path):
    url, schema = postgres_config
    committed = CommitRecord(
        scenario="math",
        commit_number=1,
        snapshot=SnapshotRef("snapshot:1", "base"),
        checkpoint=True,
        algorithm_state={"step": 1},
        consumed_ids=frozenset({"first"}),
        high_water_sequence=1,
        high_water_offset=1,
        recorded_at=123.0,
    )
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory,
        closing(factory.open("math")) as store,
    ):
        store.records.append(record())
        store.append_commit(expected_commit_number=0, commit=committed)
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory:
        with closing(factory.open("math")) as store:
            assert store.history() == (committed,)
            assert store.recover(checkpoint=None) == committed
            assert store.records.count("math") == 1
            assert store.records.append_result(record()).inserted is False
        archived = factory.archive("math")
        assert archived[0].startswith("postgres://")
        # A deleted scenario's ID is never opened again, also by a new factory.
        with pytest.raises(ScenarioDeletedError):
            factory.open("math")
    with (
        closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory,
        pytest.raises(ScenarioDeletedError),
    ):
        factory.open("math")


@pytest.mark.parametrize(
    "values",
    [
        {"record_backend": "unknown"},
        {"record_backend": "reef.storage.scenario:ScenarioStorage"},
        {"record_backend": {"module": "reef.storage.sqlite:SQLiteScenarioDatabase"}},
        {"record_backend": {"module": "reef.storage.sqlite:SQLiteScenarioStorage", "options": {}}},
        {"record_backend": "postgres"},
        {"record_database_url": "postgresql:///db"},
        {"record_backend": "postgres", "record_database_url": "sqlite:///local"},
        {"record_backend": "postgres", "record_database_url": "postgresql:///db", "record_database_schema": "public"},
        {"record_backend": "postgres", "record_database_url": "postgresql:///db", "record_database_schema": "bad;sql"},
    ],
)
def test_deployment_rejects_invalid_record_backend(values):
    with pytest.raises(ValueError):
        ServiceSettings(recipe="recipe", **values)


def test_deployment_file_names_a_storage_plugin_with_its_config(tmp_path):
    path = tmp_path / "deployment.yaml"
    path.write_text(
        "recipe:\n  implementation: recipe\n"
        "storage:\n  record_backend:\n    module: reef.storage.postgres:PostgresScenarioStorage\n"
        "    config:\n      database_url: postgresql://user:secret@example.invalid/db\n"
        "  coordination_backend: postgres\n  record_database_url: postgresql://user:secret@example.invalid/db\n"
    )
    settings = service_config_from_mapping(translate_layout(load_config(path)))
    assert settings.record_backend == {
        "module": "reef.storage.postgres:PostgresScenarioStorage",
        "config": {"database_url": "postgresql://user:secret@example.invalid/db"},
    }
    argument = next(argument for argument in service_config_arguments() if argument.name == "record_backend")
    assert argument.parse("reef.storage.sqlite:SQLiteScenarioStorage") == "reef.storage.sqlite:SQLiteScenarioStorage"
    assert argument.parse('{"module": "a.b:C"}') == {"module": "a.b:C"}


def test_config_interpolation_and_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_RECORD_URL", "postgresql://user:secret@example.invalid/db")
    path = tmp_path / "deployment.yaml"
    path.write_text(
        "reef:\n  recipe: recipe\n  record_backend: postgres\n"
        "  record_database_url: ${TEST_RECORD_URL}\n  record_database_schema: deployment_one\n"
    )
    settings = service_config_from_mapping(load_config(path))
    assert settings.record_database_url == "postgresql://user:secret@example.invalid/db"
    assert settings.record_database_schema == "deployment_one"
    assert "secret" not in repr(settings)
    assert _recipe_owned_settings(settings) == {}
    assert ServiceSettings(recipe="recipe").record_backend == "sqlite"
    with pytest.raises(ValueError) as error:
        postgres_url("secret is not a url")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("backend", ["sqlite", "postgres", "sqlite-plugin", "postgres-plugin"])
def test_deployment_selects_record_backend(backend, request, tmp_path):
    values: dict = {}
    if backend == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        values = {"record_backend": backend, "record_database_url": url, "record_database_schema": schema}
    elif backend == "sqlite-plugin":
        values = {"record_backend": "reef.storage.sqlite:SQLiteScenarioStorage", "coordination_backend": "sqlite"}
    elif backend == "postgres-plugin":
        url, schema = request.getfixturevalue("postgres_config")
        plugin = {
            "module": "reef.storage.postgres:PostgresScenarioStorage",
            "config": {"database_url": url, "schema": schema},
        }
        # The records are record items, so a new host needs the data store in PostgreSQL too.
        values = {
            "record_backend": plugin,
            "coordination_backend": "postgres",
            "data_store": {
                "module": "reef.storage.sql_data:PostgresDataStore",
                "config": {"database_url": url, "schema": f"{schema}_data"},
            },
            "record_database_url": url,
            "record_database_schema": schema,
        }
    settings = ServiceSettings(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        **values,
    )
    for restart in (False, True):
        if restart and backend.startswith("postgres"):
            # PostgreSQL keeps all scenario state, so a new host starts from an empty record directory.
            settings = replace(settings, agent_record_dir=str(tmp_path / "new-host-records"))
        dispatcher = build_dispatcher(settings)
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            # The records are record items of the data store, which follows the record backend (RFC #1219).
            assert isinstance(scenario.records, ItemRecordStore)
            if restart:
                assert scenario.records.read("math", "first") == record()
            else:
                scenario.records.append(record())
        finally:
            dispatcher.close()
