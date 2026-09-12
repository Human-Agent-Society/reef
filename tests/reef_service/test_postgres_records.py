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
from reef.service.deploy.config import load_config
from reef.service.deploy.service_config import ServiceSettings, service_settings_from_config
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
        first.compact("math", frozenset({"first"}), receipt_id="step", receipt_metadata={"step": 1})
    with closing(PostgresRecordStore(url, schema=schema, name="two")) as other:
        assert other.existing_receipt(original) is None
        assert other.compaction_receipts("math") == ()
        assert other.append_result(original).inserted
        assert other.get("math", "first") == original
        assert other.purge_compacted("math", before=1e20) == 0
    with closing(PostgresRecordStore(url, schema=schema, name="one")) as reopened:
        assert reopened.append_result(original).inserted is False
        assert reopened.get_for_audit("math", "first").item == original
        assert len(reopened.compaction_receipts("math")) == 1
        assert reopened.purge_compacted("math", before=1e20) == 1
    with closing(PostgresRecordStore(url, schema=schema, name="two")) as other:
        assert other.count("math") == 1


def test_large_compaction_receipt_and_atomic_conflict(postgres_database):
    with closing(PostgresRecordStore(postgres_database)) as store:
        store.append(record())
        ids = frozenset({"first", *(f"record-{index:06d}" for index in range(5000))})
        store.compact("math", ids, receipt_id="large", receipt_metadata={"step": 1})
        store.append(record("later"))
        with pytest.raises(RecordConflict):
            store.compact("math", ids, receipt_id="large", receipt_metadata={"step": 2})
        assert store.get("math", "later") is not None
        assert store.compaction_receipts("math")[0]["compacted_ids"] == tuple(sorted(ids))


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


def test_concurrent_database_initialization_and_compaction(postgres_config):
    url, schema = postgres_config
    barrier = Barrier(2)

    def initialize_and_compact(step):
        barrier.wait(timeout=5)
        with closing(PostgresRecordStore(url, schema=schema)) as store:
            store.append(record())
            try:
                store.compact("math", frozenset({"first"}), receipt_id="step", receipt_metadata={"step": step})
            except RecordConflict:
                return False
            return True

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(initialize_and_compact, (1, 2))) == [False, True]
    with closing(PostgresRecordStore(url, schema=schema)) as store:
        assert store.count("math") == 0
        assert len(store.compaction_receipts("math")) == 1


def test_schema_version_and_database_close(postgres_config):
    url, schema = postgres_config
    database = PostgresRecordDatabase(url, schema=schema)
    with closing(PostgresRecordStore(database)) as store:
        store.append(record())
    with database.transaction() as connection:
        version = database.tables.records.metadata.tables[f"{schema}.schema_version"]
        connection.execute(version.update().values(version=2))
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
            query = select(first._tables.records.c.compacted_at)
            assert connection.execute(query).scalar_one() is None
            second.compact("math", frozenset({"first"}))
            assert connection.execute(query).scalar_one() is None
        assert first.get("math", "first") is None
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
        old.compact("math", frozenset({"first"}))
        old.append(record("active"))
        archived_id = postgres_database.archive("math")
        assert archived_id == old.storage_id
        with pytest.raises(RuntimeError, match="archived"):
            old.append(record("stale"))
        with closing(PostgresRecordStore(postgres_database, name="math")) as new:
            assert new.storage_id != old.storage_id
            assert new.compaction_receipts("math") == ()
            assert new.append_result(record()).inserted
            assert postgres_database.prune(RecordRetention(max_bytes=1)) == 1
            assert new.count("math") == 1
            with postgres_database.transaction() as connection:
                rows = connection.execute(select(postgres_database.tables.records.c.agent_record_id)).scalars().all()
            assert sorted(rows) == ["active", "first"]
    finally:
        old.close()
        old.close()
    with pytest.raises(RuntimeError, match="closed"):
        old.count("math")


def test_retention_age_then_oldest_byte_budget(postgres_database):
    with closing(PostgresRecordStore(postgres_database)) as store:
        for name in ("expired", "oldest", "newest", "active"):
            store.append(record(name))
        for name in ("expired", "oldest", "newest"):
            store.compact("math", frozenset({name}))
        with postgres_database.transaction() as connection:
            table = postgres_database.tables.records
            connection.execute(table.update().where(table.c.agent_record_id == "expired").values(compacted_at=1))
            newest_bytes = connection.execute(
                select(table.c.body_bytes).where(table.c.agent_record_id == "newest")
            ).scalar_one()
        assert postgres_database.prune(RecordRetention(max_bytes=newest_bytes)) == 2
        assert [row.item.agent_record_id for row in store.audit_page("math")] == ["newest", "active"]
        assert store.append_result(record("expired")).inserted is False


def test_factory_commit_recovery_and_archive(postgres_config, tmp_path, monkeypatch):
    url, schema = postgres_config
    committed = CommitRecord(
        scenario="math",
        step=1,
        artifact_ref=ArtifactRef("content:1", "release:1", "base"),
        checkpoint=True,
        algorithm_state={"step": 1},
        compacted_ids=frozenset({"first"}),
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

        def interrupted_compaction(*args, **kwargs):
            raise RuntimeError("compaction interrupted")

        monkeypatch.setattr(store.records, "compact", interrupted_compaction)
        with pytest.raises(RuntimeError, match="compaction interrupted"):
            store.commit_step(expected_step=0, commit=committed)
    with closing(PostgresScenarioStorage(url, tmp_path, schema=schema)) as factory:
        with closing(factory.open("math")) as store:
            assert store.history() == (committed,)
            assert store.records.count("math") == 1
            assert store.recover(checkpoint=None) == committed
            assert store.records.count("math") == 0
            assert store.records.append_result(record()).inserted is False
        archived = factory.archive("math")
        assert archived[0].startswith("postgres://")
        with closing(factory.open("math")) as store:
            assert store.history() == ()
            assert store.records.append_result(record()).inserted
        assert factory.prune(days=7, max_bytes=1) == 1


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
        ServiceSettings(recipe="recipe", **values)


def test_config_interpolation_and_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_RECORD_URL", "postgresql://user:secret@example.invalid/db")
    path = tmp_path / "deployment.yaml"
    path.write_text(
        "reef:\n  recipe: recipe\n  record_backend: postgres\n"
        "  record_database_url: ${TEST_RECORD_URL}\n  record_database_schema: deployment_one\n"
    )
    settings = service_settings_from_config(load_config(path))
    assert settings.record_database_url == "postgresql://user:secret@example.invalid/db"
    assert settings.record_database_schema == "deployment_one"
    assert "secret" not in repr(settings)
    assert _recipe_owned_settings(settings) == {}
    assert ServiceSettings(recipe="recipe").record_backend == "sqlite"
    with pytest.raises(ValueError) as error:
        postgres_url("secret is not a url")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_deployment_selects_record_backend(backend, request, tmp_path):
    values = {}
    if backend == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        values = {"record_backend": backend, "record_database_url": url, "record_database_schema": schema}
    settings = ServiceSettings(
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
