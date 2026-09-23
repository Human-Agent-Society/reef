"""Real PostgreSQL coverage for the dialect and deployment lifecycle boundaries."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing
from dataclasses import replace
from threading import Barrier, Event

import pytest
from sqlalchemy import select

from reef.core.artifact_ref import ArtifactRef
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType
from reef.service.assembly import _recipe_owned_settings, build_dispatcher
from reef.service.deploy.config_utils import load_config
from reef.service.deploy.service_config import ServiceConfig, service_config_from_mapping
from reef.storage.commits import CommitRecord
from reef.storage.postgres import PostgresRecordDatabase, PostgresRecordStore, PostgresScenarioStorage, postgres_url
from reef.storage.records import RecordConflict, RecordRetention
from reef.storage.sqlite import SQLiteRecordStore


def record(record_id="first", scenario="math"):
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"text": "中文 🐟", "nested": {"body": record_id}},
        created_at=1789200000.123456,
    )


def test_namespaces_restart_and_receipt_isolation(postgres_config):
    url, schema = postgres_config
    original = record()
    with closing(PostgresRecordStore(url, schema=schema, name="one")) as first:
        assert first.append(original) == original
        first.record_consumption("math", frozenset({"first"}), receipt_id="step", metadata={"step": 1})
    with closing(PostgresRecordStore(url, schema=schema, name="two")) as other:
        assert other.existing_receipt(original) is None
        assert other.consumption_receipts("math") == ()
        assert other.append_result(original).inserted
        assert other.get("math", "first") == original
    with closing(PostgresRecordStore(url, schema=schema, name="one")) as reopened:
        assert reopened.append_result(original).inserted is False
        assert reopened.get_for_audit("math", "first").item == original
        assert len(reopened.consumption_receipts("math")) == 1
    with closing(PostgresRecordStore(url, schema=schema, name="two")) as other:
        assert other.count("math") == 1


def test_large_consumption_receipt_and_atomic_conflict(postgres_database):
    with closing(PostgresRecordStore(postgres_database)) as store:
        store.append(record())
        ids = frozenset({"first", *(f"record-{index:06d}" for index in range(5000))})
        store.record_consumption("math", ids, receipt_id="large", metadata={"step": 1})
        store.append(record("later"))
        with pytest.raises(RecordConflict):
            store.record_consumption("math", ids, receipt_id="large", metadata={"step": 2})
        assert store.get("math", "later") is not None
        assert store.consumption_receipts("math")[0]["consumed_ids"] == tuple(sorted(ids))


def test_concurrent_append_retry_and_conflict(postgres_database):
    with (
        closing(PostgresRecordStore(postgres_database)) as first,
        closing(PostgresRecordStore(postgres_database)) as second,
    ):
        barrier = Barrier(2)

        def append(store):
            barrier.wait(timeout=5)
            return store.append_result(record())

        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(append, (first, second)))
        assert sorted(result.inserted for result in results) == [False, True]
        with pytest.raises(RecordConflict):
            second.append(replace(record(), payload={"changed": True}))
        assert first.count("math") == 1


def test_concurrent_database_initialization_and_consumption(postgres_config):
    url, schema = postgres_config
    barrier = Barrier(2)

    def initialize_and_consume(step):
        barrier.wait(timeout=5)
        with closing(PostgresRecordStore(url, schema=schema)) as store:
            store.append(record())
            try:
                store.record_consumption("math", frozenset({"first"}), receipt_id="step", metadata={"step": step})
            except RecordConflict:
                return False
            return True

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(initialize_and_consume, (1, 2))) == [False, True]
    with closing(PostgresRecordStore(url, schema=schema)) as store:
        assert store.count("math") == 1
        assert len(store.consumption_receipts("math")) == 1


def test_schema_version_and_database_close(postgres_config):
    url, schema = postgres_config
    database = PostgresRecordDatabase(url, schema=schema)
    with closing(PostgresRecordStore(database)) as store:
        store.append(record())
    with database.transaction() as connection:
        version = database.tables.records.metadata.tables[f"{schema}.schema_version"]
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
        PostgresRecordStore("postgresql:///unused")


def test_sequence_allocation_waits_for_prior_commit(postgres_database):
    with (
        closing(PostgresRecordStore(postgres_database)) as first,
        closing(PostgresRecordStore(postgres_database)) as second,
    ):
        started = Event()

        def append_second():
            started.set()
            return second.append(record("second"))

        with ThreadPoolExecutor(1) as pool:
            with first._transaction("math", write=True) as connection:
                first._insert(
                    connection, first._tables.records, {**first._encode(record())._asdict(), "body_bytes": 30}
                )
                pending = pool.submit(append_second)
                assert started.wait(5)
                with pytest.raises(TimeoutError):
                    pending.result(timeout=0.2)
                with closing(PostgresRecordStore(postgres_database, name="independent")) as independent:
                    independent.append(record("unblocked"))
            pending.result(timeout=5)
        page = first.replay_page("math")
        assert [item.agent_record_id for _, item in page] == ["first", "second"]
        assert first.replay_page("math", after_sequence=page[0][0]) == (page[1],)


def test_reader_snapshot_and_transaction_rollback(postgres_database):
    with (
        closing(PostgresRecordStore(postgres_database)) as first,
        closing(PostgresRecordStore(postgres_database)) as second,
    ):
        original = first.append(record())
        with first._transaction("math", write=False) as connection:
            query = select(first._tables.consumption.c.receipt_id)
            assert connection.execute(query).all() == []
            second.record_consumption("math", frozenset({"first"}), receipt_id="skip", metadata={})
            assert connection.execute(query).all() == []
        assert len(first.consumption_receipts("math")) == 1
        assert first.get("math", "first") == original
        assert first.get_for_audit("math", "first").item == original
        with pytest.raises(RuntimeError, match="rollback"), first._transaction("math", write=True) as connection:
            first._insert(
                connection, first._tables.records, {**first._encode(record("failed"))._asdict(), "body_bytes": 10}
            )
            raise RuntimeError("rollback")
        assert second.get("math", "failed") is None
        assert second.append_result(record("failed")).inserted


def test_archive_generation_retention_and_closed_sessions(postgres_database):
    old = PostgresRecordStore(postgres_database, name="math")
    try:
        old.append(record())
        old.record_consumption("math", frozenset({"first"}), receipt_id="skip", metadata={})
        old.append(record("active"))
        archived_id = postgres_database.archive("math")
        assert archived_id == old.storage_id
        with pytest.raises(RuntimeError, match="archived"):
            old.append(record("stale"))
        with closing(PostgresRecordStore(postgres_database, name="math")) as new:
            assert new.storage_id != old.storage_id
            assert new.consumption_receipts("math") == ()
            assert new.append_result(record()).inserted
            assert postgres_database.prune(RecordRetention(max_bytes=1)) == 3
            assert new.count("math") == 0
            assert new.loss("math").record_count == 1
            with postgres_database.transaction() as connection:
                rows = connection.execute(select(postgres_database.tables.records.c.agent_record_id)).scalars().all()
            assert rows == []
    finally:
        old.close()
        old.close()
    with pytest.raises(RuntimeError, match="closed"):
        old.count("math")


def test_capacity_counts_all_records_and_preserves_retry_hashes(postgres_database):
    with closing(PostgresRecordStore(postgres_database)) as store:
        for name in ("expired", "oldest", "newest", "active"):
            store.append(record(name))
        for name in ("expired", "oldest", "newest"):
            store.record_consumption("math", frozenset({name}), receipt_id=name, metadata={})
        with postgres_database.transaction() as connection:
            table = postgres_database.tables.records
            newest_bytes = connection.execute(
                select(table.c.body_bytes).where(table.c.agent_record_id == "newest")
            ).scalar_one()
        assert postgres_database.prune(RecordRetention(max_bytes=newest_bytes)) == 3
        assert [row.item.agent_record_id for row in store.audit_page("math")] == ["active"]
        assert store.loss("math").record_count == 3
        assert store.append_result(record("expired")).inserted is False


def test_factory_commit_recovery_and_archive(postgres_config, tmp_path, monkeypatch):
    url, schema = postgres_config
    committed = CommitRecord(
        scenario="math",
        step=1,
        artifact_ref=ArtifactRef("content:1", "release:1", "base"),
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

        store.commit_step(expected_step=0, commit=committed)
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory:
        with closing(factory.open("math")) as store:
            assert store.history() == (committed,)
            assert store.records.count("math") == 1
            assert store.recover(checkpoint=None) == committed
            assert store.records.count("math") == 1
            assert store.records.append_result(record()).inserted is False
        archived = factory.archive("math")
        assert archived[0].startswith("postgres://")
        with closing(factory.open("math")) as store:
            assert store.history() == ()
            assert store.records.append_result(record()).inserted
        assert factory.prune(days=7, max_bytes=1) == 2


def test_archive_move_failure_cannot_replay_old_log(postgres_config, tmp_path, monkeypatch):
    url, schema = postgres_config
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory:
        with closing(factory.open("math")) as store:
            path = store.commit_log.path
            path.touch()

        def failed_move(*args):
            raise OSError("move failed")

        monkeypatch.setattr("reef.storage.postgres.shutil.move", failed_move)
        with pytest.raises(OSError, match="move failed"):
            factory.archive("math")
        with closing(factory.open("math")) as store:
            assert store.commit_log.path != path
            assert store.history() == ()


@pytest.mark.parametrize(
    "values",
    [
        {"record_backend": "unknown"},
        {"record_backend": "postgres"},
        {"record_database_url": "postgresql:///db"},
        {"record_backend": "postgres", "record_database_url": "sqlite:///local"},
        {"record_backend": "postgres", "record_database_url": "postgresql:///db", "record_database_schema": "public"},
        {"record_backend": "postgres", "record_database_url": "postgresql:///db", "record_database_schema": "bad;sql"},
    ],
)
def test_deployment_rejects_invalid_record_backend(values):
    with pytest.raises(ValueError):
        ServiceConfig(recipe="recipe", **values)


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
    assert ServiceConfig(recipe="recipe").record_backend == "sqlite"
    with pytest.raises(ValueError) as error:
        postgres_url("secret is not a url")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_deployment_selects_record_backend(backend, request, tmp_path):
    values = {}
    if backend == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        values = {"record_backend": backend, "record_database_url": url, "record_database_schema": schema}
    settings = ServiceConfig(
        recipe="recipe",
        agent_record_dir=str(tmp_path / "records"),
        artifact_repository=str(tmp_path / "artifacts.git"),
        artifact_work_dir=str(tmp_path / "work"),
        artifact_cache_dir=str(tmp_path / "cache"),
        **values,
    )
    expected = PostgresRecordStore if backend == "postgres" else SQLiteRecordStore
    for restart in (False, True):
        dispatcher = build_dispatcher(settings)
        try:
            scenario = dispatcher.get_or_create_scenario("math")
            assert isinstance(scenario.records, expected)
            if restart:
                assert scenario.records.get("math", "first") == record()
            else:
                scenario.records.append(record())
        finally:
            dispatcher.close()


def test_capacity_metadata_upgrade_preserves_existing_records(postgres_config):
    url, schema = postgres_config
    with closing(PostgresRecordDatabase(url, schema=schema)) as database:
        with closing(PostgresRecordStore(database)) as records:
            records.append(record())
        with database.transaction() as connection:
            database.tables.eviction.drop(connection)
            version = database.tables.records.metadata.tables[f"{schema}.schema_version"]
            connection.execute(version.update().values(version=1))
    with (
        closing(PostgresRecordDatabase(url, schema=schema)) as database,
        closing(PostgresRecordStore(database)) as records,
    ):
        assert records.get("math", "first") == record()
        assert records.loss("math").record_count == 0
        assert database.prune(RecordRetention(max_bytes=1)) == 1
        assert records.loss("math").record_count == 1


def test_legacy_schema_moves_retirement_to_consumption_and_removes_old_tables(postgres_config):
    from sqlalchemy import inspect

    url, schema = postgres_config
    with closing(PostgresRecordDatabase(url, schema=schema)) as database:
        with closing(PostgresRecordStore(database, name="one")) as records:
            records.append(record())
            records.record_consumption(
                "math", frozenset({"earlier"}), receipt_id="skip", metadata={"outcome": "stale"}
            )
        with closing(PostgresRecordStore(database, name="two")) as records:
            records.append(record("other"))
        with database.transaction() as connection:
            connection.exec_driver_sql(f'ALTER TABLE "{schema}".record_consumption RENAME TO compaction_receipts')
            connection.exec_driver_sql(
                f'ALTER TABLE "{schema}".compaction_receipts RENAME CONSTRAINT record_consumption_pkey TO compaction_receipts_pkey'
            )
            connection.exec_driver_sql(
                f'ALTER TABLE "{schema}".compaction_receipts RENAME COLUMN consumed_ids_json TO compacted_ids_json'
            )
            connection.exec_driver_sql(
                f'ALTER TABLE "{schema}".compaction_receipts RENAME COLUMN consumed_ids_sha256 TO compacted_ids_sha256'
            )
            connection.exec_driver_sql(f'ALTER TABLE "{schema}".agent_record ADD COLUMN compacted_at DOUBLE PRECISION')
            connection.exec_driver_sql(f'UPDATE "{schema}".agent_record SET compacted_at=1')
            connection.exec_driver_sql(
                f'CREATE INDEX agent_record_active_type_sequence ON "{schema}".agent_record (scenario, sequence) '
                "WHERE compacted_at IS NULL"
            )
            version = database.tables.records.metadata.tables[f"{schema}.schema_version"]
            connection.execute(version.update().values(version=2))
    for _ in range(2):
        with closing(PostgresRecordDatabase(url, schema=schema)) as database:
            with closing(PostgresRecordStore(database, name="one")) as records:
                assert records.get("math", "first") == record()
                assert {
                    key for receipt in records.consumption_receipts("math") for key in receipt["consumed_ids"]
                } == {"first", "earlier"}
                assert len(records.consumption_receipts("math")) == 2
            with closing(PostgresRecordStore(database, name="two")) as records:
                assert records.get("math", "other") == record("other")
                assert records.consumption_receipts("math")[0]["consumed_ids"] == ("other",)
            with database.transaction() as connection:
                inspector = inspect(connection)
                assert "compaction_receipts" not in inspector.get_table_names(schema=schema)
                assert "compacted_at" not in {
                    column["name"] for column in inspector.get_columns("agent_record", schema=schema)
                }
