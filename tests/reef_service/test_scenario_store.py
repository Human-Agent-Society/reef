"""Scenario store contracts and the commit log adapter's recovery boundaries."""

from __future__ import annotations

import hashlib
import multiprocessing
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from reef.core.artifact_ref import ArtifactRef
from reef.core.errors import ReefError
from reef.core.records_types import AgentRecord, RequestType
from reef.scenario.state import CommitRecord, RecordProgress, ScenarioSnapshot
from reef.scenario.store import ScenarioStoreConflict
from reef.storage.commit_log import CommitLog, CommitLogScenarioStore
from reef.storage.factory import SQLiteScenarioStoreFactory
from reef.storage.sqlite import SQLiteRecordStore


def record(record_id: str, *, scenario: str = "math") -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario=scenario,
        request_type=RequestType.INFERENCE,
        payload={"text": record_id},
    )


def commit(step: int = 1, *, scenario: str = "math") -> CommitRecord:
    return CommitRecord(
        scenario=scenario,
        step=step,
        artifact_ref=ArtifactRef(f"content:{step}", f"release:{step}", "base"),
        checkpoint=True,
        algorithm_state={"step": step},
        high_water_sequence=step,
        high_water_offset=step,
        compacted_ids=frozenset({f"record-{step}"}),
        consumed_ids=frozenset({f"record-{step}"}),
        recorded_at=1000.0 + step,
    )


def snapshot(step: int = 0) -> ScenarioSnapshot:
    return ScenarioSnapshot(
        scenario="math",
        base_artifact=ArtifactRef("base", "base", None),
        scenario_step=step,
        algorithm_state=None if step == 0 else {"step": step},
        record_progress=(
            None
            if step == 0
            else RecordProgress(step, step, frozenset({f"record-{step}"}), frozenset({f"record-{step}"}))
        ),
        operation="training",
    )


@pytest.fixture(params=[False, True], ids=["memory", "durable"])
def store(request, tmp_path):
    factory = SQLiteScenarioStoreFactory(tmp_path if request.param else None)
    session = factory.open("math")
    yield session
    session.close()
    factory.close()


def test_commit_is_fenced_compacts_only_its_scenario_and_preserves_audit(store):
    store.records.append(record("record-1"))
    store.records.append(record("other", scenario="code"))
    accepted = store.commit_step(expected_step=0, commit=commit())

    assert accepted == commit()
    assert store.history() == (accepted,)
    assert store.records.replay("math") == ()
    assert store.records.get("code", "other") is not None
    assert store.records.get_for_audit("math", "record-1").compacted_at is not None
    assert store.training_run_position() == (0, 1)


def test_retry_returns_original_timestamp_even_after_later_commits(store):
    original = store.commit_step(expected_step=0, commit=commit())
    store.commit_step(expected_step=1, commit=commit(2))

    accepted = store.commit_step(expected_step=0, commit=replace(original, recorded_at=9999.0))

    assert accepted.recorded_at == original.recorded_at
    assert len(store.history()) == 2
    assert store.training_run_position() == (0, 2)


@pytest.mark.parametrize(
    "changed",
    [
        {"algorithm_state": {"step": 99}},
        {"compacted_ids": frozenset({"other"})},
        {"consumed_ids": frozenset({"other"})},
        {"metrics": {"loss": 1.0}},
        {"training_job_id": "another-job"},
        {"checkpoint": False},
    ],
)
def test_retry_rejects_different_committed_content(store, changed):
    original = store.commit_step(expected_step=0, commit=commit())

    with pytest.raises(ScenarioStoreConflict, match="conflicts"):
        store.commit_step(expected_step=0, commit=replace(original, **changed))

    assert store.history() == (original,)


def test_commit_rejects_skips_wrong_expected_step_and_foreign_scenario(store):
    with pytest.raises(ScenarioStoreConflict, match="current step is 0"):
        store.commit_step(expected_step=1, commit=commit(2))
    with pytest.raises(ScenarioStoreConflict, match="commit step must be 1"):
        store.commit_step(expected_step=0, commit=commit(2))
    with pytest.raises(ScenarioStoreConflict, match="scenario 'code'"):
        store.commit_step(expected_step=0, commit=commit(scenario="code"))

    assert store.history() == ()


@pytest.mark.parametrize("expected", [-1, True, 0.5])
def test_commit_validates_expected_step(store, expected):
    with pytest.raises(ValueError, match="expected_step"):
        store.commit_step(expected_step=expected, commit=commit())


def test_retry_repairs_compaction_after_the_commit_point(store, monkeypatch):
    store.records.append(record("record-1"))
    compact = store.records.compact

    def fail_compaction(*args, **kwargs):
        raise OSError("interrupted compaction")

    monkeypatch.setattr(store.records, "compact", fail_compaction)
    with pytest.raises(OSError, match="interrupted compaction"):
        store.commit_step(expected_step=0, commit=commit())
    assert store.history() == (commit(),)
    assert store.records.count("math") == 1

    monkeypatch.setattr(store.records, "compact", compact)
    store.commit_step(expected_step=0, commit=replace(commit(), recorded_at=9999.0))

    assert store.history() == (commit(),)
    assert store.records.count("math") == 0
    assert store.training_run_position() == (0, 1)


def test_recover_adopts_checkpoint_and_preserves_rollback_fields(store):
    store.records.append(record("record-2"))
    checkpoint = replace(
        snapshot(2), operation="rollback", rollback_target_release_id="base", metrics={"quality": 0.5}
    )
    head = store.recover(snapshot=checkpoint, checkpoint_head=commit(2).artifact_ref)

    assert head.step == 2
    assert head.checkpoint is True
    assert head.operation == "rollback"
    assert head.operation_verified is True
    assert head.rollback_target_release_id == "base"
    assert head.metrics == {"quality": 0.5}
    assert head.compacted_ids == frozenset({"record-2"})
    assert head.consumed_ids == frozenset({"record-2"})
    assert store.records.count("math") == 0
    assert store.history() == (head,)
    store.commit_step(expected_step=2, commit=commit(3))
    assert store.training_run_position() == (2, 1)


def test_recover_keeps_unverified_legacy_snapshot_operation(store):
    head = store.recover(snapshot=replace(snapshot(2), operation=None), checkpoint_head=commit(2).artifact_ref)

    assert head.operation == "training"
    assert head.operation_verified is False


def test_recovery_at_creation_is_empty_and_foreign_snapshot_is_rejected(store):
    assert store.recover(snapshot=snapshot(), checkpoint_head=snapshot().base_artifact) is None
    with pytest.raises(ReefError, match="snapshot belongs to scenario 'code'"):
        store.recover(snapshot=replace(snapshot(), scenario="code"), checkpoint_head=snapshot().base_artifact)


def test_training_position_ignores_promote_and_resets_on_rollback(store):
    store.commit_step(expected_step=0, commit=commit())
    store.commit_step(
        expected_step=1, commit=replace(commit(2), operation="promote", rollback_target_release_id="base")
    )
    assert store.training_run_position() == (0, 1)
    store.commit_step(
        expected_step=2, commit=replace(commit(3), operation="rollback", rollback_target_release_id="base")
    )
    store.commit_step(expected_step=3, commit=commit(4))
    assert store.training_run_position() == (3, 1)


def test_close_is_idempotent_and_rejects_further_use(store, monkeypatch):
    calls = 0
    records = store.records
    close_records = records.close

    def close_once():
        nonlocal calls
        calls += 1
        close_records()

    monkeypatch.setattr(records, "close", close_once)
    store.close()
    store.close()

    assert calls == 1
    with pytest.raises(RuntimeError, match="store is closed"):
        store.history()
    with pytest.raises(RuntimeError, match="store is closed"):
        store.commit_step(expected_step=0, commit=commit())
    with pytest.raises(RuntimeError, match="store is closed"):
        store.recover(snapshot=snapshot(), checkpoint_head=snapshot().base_artifact)
    with pytest.raises(RuntimeError, match="store is closed"):
        _ = store.records


def test_recovery_replays_all_compactions_even_before_snapshot(tmp_path):
    journal = CommitLog(tmp_path / "commits.jsonl")
    with closing(CommitLogScenarioStore("math", SQLiteRecordStore(tmp_path / "records.sqlite3"), journal)) as session:
        for step in range(1, 4):
            session.records.append(record(f"record-{step}"))
            journal.append(commit(step))

        head = session.recover(snapshot=snapshot(2), checkpoint_head=commit(2).artifact_ref)

        assert head == commit(3)
        assert session.records.count("math") == 0
        assert len(session.records.audit_page("math")) == 3
        session.commit_step(expected_step=3, commit=commit(4))


def test_recovery_repairs_failed_compaction_after_reopening(tmp_path, monkeypatch):
    factory = SQLiteScenarioStoreFactory(tmp_path)
    with closing(factory.open("math")) as session:
        session.records.append(record("record-1"))

        def fail_compaction(*args, **kwargs):
            raise OSError("interrupted compaction")

        monkeypatch.setattr(session.records, "compact", fail_compaction)
        with pytest.raises(OSError):
            session.commit_step(expected_step=0, commit=commit())

    with closing(factory.open("math")) as recovered:
        assert recovered.recover(snapshot=snapshot(), checkpoint_head=snapshot().base_artifact) == commit()
        assert recovered.records.count("math") == 0
        assert recovered.records.get_for_audit("math", "record-1") is not None
    factory.close()


@pytest.mark.parametrize("durable_append", [False, True])
def test_journal_failure_preserves_the_actual_commit_point(tmp_path, monkeypatch, durable_append):
    journal = CommitLog(tmp_path / "commits.jsonl")
    with closing(CommitLogScenarioStore("math", SQLiteRecordStore(), journal)) as session:
        session.records.append(record("record-1"))
        append = journal.append

        def interrupt_append(value):
            if durable_append:
                append(value)
            raise OSError("interrupted append")

        monkeypatch.setattr(journal, "append", interrupt_append)
        with pytest.raises(OSError, match="interrupted append"):
            session.commit_step(expected_step=0, commit=commit())
        assert session.history() == ((commit(),) if durable_append else ())
        assert session.records.count("math") == 1

        monkeypatch.setattr(journal, "append", append)
        session.commit_step(expected_step=0, commit=replace(commit(), recorded_at=9999.0))
        assert len(session.history()) == 1
        assert session.records.count("math") == 0


@pytest.mark.parametrize("steps,snapshot_step,message", [([2], 0, "resumes at step 2"), ([1, 3], 0, "jumps")])
def test_recovery_rejects_gaps_after_checkpoint(tmp_path, steps, snapshot_step, message):
    journal = CommitLog(tmp_path / "commits.jsonl")
    for step in steps:
        journal.append(commit(step))
    with (
        closing(CommitLogScenarioStore("math", SQLiteRecordStore(), journal)) as session,
        pytest.raises(ReefError, match=message),
    ):
        session.recover(snapshot=snapshot(snapshot_step), checkpoint_head=snapshot().base_artifact)


def test_recovery_rejects_journal_for_another_scenario(tmp_path):
    journal = CommitLog(tmp_path / "commits.jsonl")
    journal.append(commit(scenario="code"))
    with (
        closing(CommitLogScenarioStore("math", SQLiteRecordStore(), journal)) as session,
        pytest.raises(ReefError, match="holds records for 'code'"),
    ):
        session.recover(snapshot=snapshot(), checkpoint_head=snapshot().base_artifact)


def test_existing_initial_step_is_fenced_without_history():
    with closing(CommitLogScenarioStore("math", SQLiteRecordStore(), initial_step=3)) as session:
        with pytest.raises(ScenarioStoreConflict, match="current step is 3"):
            session.commit_step(expected_step=0, commit=commit())
        session.commit_step(expected_step=3, commit=commit(4))


def test_conflicting_sessions_have_only_one_winner(tmp_path):
    first_factory = SQLiteScenarioStoreFactory(tmp_path)
    second_factory = SQLiteScenarioStoreFactory(tmp_path)
    with closing(first_factory.open("math")) as first, closing(second_factory.open("math")) as second:
        barrier = Barrier(2)

        def try_commit(session, label):
            barrier.wait(timeout=10)
            try:
                return session.commit_step(expected_step=0, commit=replace(commit(), metrics={"writer": label}))
            except ScenarioStoreConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(try_commit, session, label) for label, session in enumerate((first, second))]
            outcomes = [attempt.result(timeout=10) for attempt in attempts]

        assert sum(outcome is not None for outcome in outcomes) == 1
        assert first.history() == second.history()
        assert len(first.history()) == 1
    first_factory.close()
    second_factory.close()


def test_session_refreshes_cached_history_after_another_writer(tmp_path):
    factory = SQLiteScenarioStoreFactory(tmp_path)
    with closing(factory.open("math")) as first, closing(factory.open("math")) as second:
        first.commit_step(expected_step=0, commit=commit())
        assert first.history() == (commit(),)
        second.commit_step(expected_step=1, commit=commit(2))
        with pytest.raises(ScenarioStoreConflict, match="conflicts"):
            first.commit_step(expected_step=1, commit=replace(commit(2), metrics={"different": True}))
        assert first.commit_step(expected_step=1, commit=replace(commit(2), recorded_at=9999.0)) == commit(2)
        first.commit_step(expected_step=2, commit=commit(3))
        assert second.history() == first.history()
    factory.close()


def _commit_in_process(directory, started, ready, results, label):
    factory = SQLiteScenarioStoreFactory(Path(directory))
    with closing(factory.open("math")) as session:
        ready.put(True)
        if not started.wait(timeout=20):
            raise RuntimeError("commit process was not started")
        try:
            session.commit_step(expected_step=0, commit=replace(commit(), metrics={"writer": label}))
            results.put("accepted")
        except ScenarioStoreConflict:
            results.put("conflict")
    factory.close()


def test_conflicting_processes_have_only_one_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    ready = context.Queue()
    results = context.Queue()
    # Initialize schema first so this test targets commit fencing specifically.
    factory = SQLiteScenarioStoreFactory(tmp_path)
    factory.open("math").close()
    processes = [
        context.Process(target=_commit_in_process, args=(str(tmp_path), started, ready, results, label))
        for label in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for _ in processes:
            assert ready.get(timeout=30) is True
        started.set()
        assert sorted(results.get(timeout=30) for _ in processes) == ["accepted", "conflict"]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        ready.close()
        results.close()
    with closing(factory.open("math")) as session:
        assert len(session.history()) == 1
    factory.close()


def test_factory_preserves_paths_archives_and_recreates_without_old_history(tmp_path):
    key = hashlib.sha256(b"math").hexdigest()
    factory = SQLiteScenarioStoreFactory(tmp_path)
    with closing(factory.open("math")) as session:
        session.records.append(record("record-1"))
        session.commit_step(expected_step=0, commit=commit())
    assert tmp_path / f"{key}.sqlite3" in factory.state_paths("math")
    assert tmp_path / f"{key}.commits.jsonl" in factory.state_paths("math")

    first_archive = factory.archive("math")
    assert len(first_archive) == 2
    assert all(Path(location).exists() for location in first_archive)
    assert (tmp_path / ".locks" / f"{key}.commits.jsonl.lock").exists()
    with closing(factory.open("math")) as session:
        assert session.history() == ()
        assert session.records.count("math") == 0
        session.commit_step(expected_step=0, commit=commit())
    second_archive = factory.archive("math")
    assert set(first_archive).isdisjoint(second_archive)
    assert all(Path(location).exists() for location in first_archive)
    factory.close()


def test_factory_retention_includes_archives_and_preserves_active_records(tmp_path):
    factory = SQLiteScenarioStoreFactory(tmp_path)
    with closing(factory.open("math")) as session:
        session.records.append(record("record-1"))
        session.records.append(record("active"))
        session.commit_step(expected_step=0, commit=commit())
    archived = factory.archive("math")

    assert factory.prune(days=7.0, max_bytes=1) == 1

    database = next(Path(path) for path in archived if path.endswith(".sqlite3"))
    with SQLiteRecordStore(database) as records:
        assert records.get_for_audit("math", "record-1") is None
        assert records.get("math", "active") is not None
    factory.close()


def test_factory_close_preserves_existing_session_and_rejects_new_operations():
    factory = SQLiteScenarioStoreFactory()
    with closing(factory.open("math")) as session:
        assert factory.durable is False
        assert factory.archive("math") == ()
        assert factory.prune(days=7.0, max_bytes=1) == 0
        factory.close()
        factory.close()
        session.commit_step(expected_step=0, commit=commit())
        with pytest.raises(RuntimeError, match="factory is closed"):
            factory.open("math")
        with pytest.raises(RuntimeError, match="factory is closed"):
            factory.archive("math")
        with pytest.raises(RuntimeError, match="factory is closed"):
            factory.prune(days=7.0, max_bytes=1)


def test_import_and_in_memory_storage_do_not_require_posix_locks(tmp_path):
    code = textwrap.dedent(
        """
        import sys
        from contextlib import closing
        from pathlib import Path

        sys.modules['fcntl'] = None
        import reef
        from reef.core.errors import ReefError
        from reef.storage.factory import SQLiteScenarioStoreFactory

        with closing(SQLiteScenarioStoreFactory()) as factory:
            with closing(factory.open('math')) as store:
                assert store.history() == ()
                assert store.records.count('math') == 0

        with closing(SQLiteScenarioStoreFactory(Path(sys.argv[1]))) as factory:
            with closing(factory.open('math')) as store:
                try:
                    store.history()
                except ReefError as exc:
                    assert 'require POSIX file locking (fcntl)' in str(exc)
                else:
                    raise AssertionError('durable store must reject unavailable POSIX locking')
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0, result.stderr
