from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from reef.artifact import ArtifactRef, LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.records import RecordConflict, RecordStore


def item(
    agent_record_id: str,
    scenario: str,
    request_type: RequestType = RequestType.INFERENCE,
    *,
    references: tuple[str, ...] = (),
) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=agent_record_id,
        scenario=scenario,
        request_type=request_type,
        payload={"value": agent_record_id},
        created_at=float(len(agent_record_id)),
        references=references,
    )


@pytest.mark.unit
def test_agent_record_replays_in_append_order_per_scenario() -> None:
    records = RecordStore()
    records.append(item("a", "math"))
    records.append(item("b", "code"))
    records.append(item("c", "math", RequestType.REPORT, references=("a",)))

    assert [item.agent_record_id for item in records.replay("math")] == ["a", "c"]
    assert [item.agent_record_id for item in records.replay("code")] == ["b"]
    assert records.get("math", "c").references == ("a",)


@pytest.mark.unit
def test_append_is_idempotent_for_identical_data() -> None:
    records = RecordStore()
    original = item("a", "math")
    retry = item("a", "math")

    assert records.append(original) is original
    assert records.append(retry) is original
    assert records.replay("math") == (original,)


@pytest.mark.unit
def test_duplicate_agent_record_id_rejects_different_content() -> None:
    records = RecordStore()
    records.append(item("same", "math"))

    with pytest.raises(RecordConflict, match="same"):
        records.append(item("same", "code"))


@pytest.mark.unit
def test_lookup_does_not_cross_scenario_boundaries() -> None:
    records = RecordStore()
    records.append(item("a", "math"))

    assert records.get("code", "a") is None


@pytest.mark.unit
def test_inference_data_can_record_release_id() -> None:
    artifact = ArtifactRef("artifact-1", "version-1", "initial")

    inference = AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"model": "reef"},
        artifact_ref=artifact,
    )

    assert inference.artifact_ref == artifact


@pytest.mark.unit
def test_agent_record_persists_across_store_restarts(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    artifact = LiveWeightArtifactRef(
        "artifact-1",
        "version-1",
        "initial",
        runtime_load_id="weights-1",
    )
    original = AgentRecord.create(
        agent_record_id="persisted",
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"messages": [{"role": "user", "content": "你好"}]},
        created_at=123.5,
        references=("parent",),
        artifact_ref=artifact,
    )

    with RecordStore(database) as first:
        first.append(original)

    with RecordStore(database) as second:
        assert second.get("math", "persisted") == original
        assert second.replay("math") == (original,)
        assert second.count("math") == 1


@pytest.mark.unit
def test_agent_record_replay_supports_bounded_pages() -> None:
    records = RecordStore()
    for agent_record_id in ("a", "b", "c", "d"):
        records.append(item(agent_record_id, "math"))

    assert [record.agent_record_id for record in records.replay("math", offset=1, limit=2)] == ["b", "c"]
    assert records.replay("math", offset=4, limit=2) == ()


@pytest.mark.unit
def test_agent_record_keyset_pages_skip_other_scenarios() -> None:
    records = RecordStore()
    records.append(item("a", "math"))
    records.append(item("other", "code"))
    records.append(item("b", "math"))

    first = records.replay_page("math", limit=1)
    second = records.replay_page("math", after_sequence=first[-1][0], limit=1)

    assert [record.agent_record_id for _, record in first] == ["a"]
    assert [record.agent_record_id for _, record in second] == ["b"]


@pytest.mark.unit
def test_conflicting_retry_is_rejected_after_store_restart(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    with RecordStore(database) as first:
        first.append(item("same", "math"))

    with RecordStore(database) as second:
        assert second.append(item("same", "math")).agent_record_id == "same"
        with pytest.raises(RecordConflict, match="same"):
            second.append(item("same", "code"))


@pytest.mark.unit
def test_compact_hides_records_but_retains_retry_tombstones(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    inference = item("inference", "math")
    retained = item("retained", "math")
    report = item("report", "math", RequestType.REPORT, references=("inference",))
    with RecordStore(database) as records:
        for record in (inference, retained, report):
            records.append(record)
        records.compact("math", frozenset({"inference", "report"}))
        assert [record.agent_record_id for record in records.replay("math")] == ["retained"]
        assert records.get("math", "inference") is None
        assert records.get("math", "retained") is not None
        archived = records.get_for_audit("math", "inference")
        assert archived is not None and archived.item == inference
        assert archived.compacted_at is not None
        assert records.get_for_audit("code", "inference") is None

    with RecordStore(database) as recovered:
        assert recovered.append_result(inference).inserted is False
        late = item("late", "math", RequestType.REPORT, references=("inference",))
        assert recovered.append_result(late).inserted is False
        assert recovered.count("math") == 1
        assert recovered.get_for_audit("math", "inference") == archived
        assert recovered.get_for_audit("math", "report").item == report
        assert recovered.existing_receipt(inference).agent_record_id == inference.agent_record_id


@pytest.mark.unit
def test_compacted_record_id_rejects_a_retry_with_different_content(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    original = AgentRecord.create(
        agent_record_id="same",
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"value": "original"},
        created_at=1.0,
    )
    conflicting = AgentRecord.create(
        agent_record_id="same",
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"value": "changed"},
        created_at=2.0,
    )
    with RecordStore(database) as records:
        records.append(original)
        records.compact("math", frozenset({"same"}))

    with RecordStore(database) as recovered, pytest.raises(RecordConflict, match="same"):
        recovered.append(conflicting)


@pytest.mark.unit
def test_discarded_report_keeps_the_stored_content_canonical() -> None:
    records = RecordStore()
    records.append(item("inference", "math"))
    stored = item("report", "math", RequestType.REPORT, references=("inference",))
    records.append(stored)
    records.compact("math", frozenset({"inference"}))

    divergent = AgentRecord.create(
        agent_record_id="report",
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"value": "changed"},
        created_at=6.0,
        references=("inference",),
    )
    with pytest.raises(RecordConflict, match="report"):
        records.append(divergent)

    assert records.get("math", "report") == stored
    assert records.append_result(stored).inserted is False


@pytest.mark.unit
def test_compact_is_a_noop_for_empty_id_set() -> None:
    records = RecordStore()
    records.append(item("a", "math"))
    records.compact("math", frozenset())
    assert records.count("math") == 1


@pytest.mark.unit
def test_compact_skips_unknown_ids_silently() -> None:
    records = RecordStore()
    records.append(item("a", "math"))
    records.compact("math", frozenset({"missing"}))
    assert records.count("math") == 1


@pytest.mark.unit
def test_compact_fingerprints_large_id_sets_in_bounded_queries() -> None:
    records = RecordStore()
    items = [item(f"record-{index}", "math") for index in range(1001)]
    for record in items:
        records.append(record)

    records.compact("math", frozenset(record.agent_record_id for record in items))

    assert records.count("math") == 0
    assert records.append_result(items[-1]).inserted is False
    archived = records.audit_page("math", limit=1001)
    assert len(archived) == 1001
    assert all(record.compacted_at is not None for record in archived)


@pytest.mark.unit
def test_compaction_receipt_is_atomic_with_retirement_and_persists(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    with RecordStore(database) as records:
        records.append(item("inference", "math"))
        records.append(item("report", "math", RequestType.REPORT, references=("inference",)))
        records.compact(
            "math",
            frozenset({"inference", "report"}),
            receipt_id="batch-1",
            receipt_metadata={
                "outcome": "stale",
                "metrics": {
                    "staleness/samples_dropped": 1,
                    "staleness/source_agent_record_ids": ["inference"],
                },
            },
        )
        assert records.count("math") == 0

    with RecordStore(database) as recovered:
        receipts = recovered.compaction_receipts("math")

        assert len(receipts) == 1
        assert receipts[0]["receipt_id"] == "batch-1"
        assert receipts[0]["compacted_ids"] == ("inference", "report")
        assert receipts[0]["metadata"] == {
            "outcome": "stale",
            "metrics": {
                "staleness/samples_dropped": 1,
                "staleness/source_agent_record_ids": ["inference"],
            },
        }
        assert isinstance(receipts[0]["recorded_at"], float)
        assert all(record.compacted_at is not None for record in recovered.audit_page("math"))


@pytest.mark.unit
def test_compaction_receipt_is_idempotent_and_conflicting_content_fails() -> None:
    records = RecordStore()
    metadata = {"outcome": "stale", "metrics": {"staleness/samples_dropped": 1}}

    records.compact("math", frozenset(), receipt_id="batch-1", receipt_metadata=metadata)
    records.compact("math", frozenset(), receipt_id="batch-1", receipt_metadata=metadata)

    assert len(records.compaction_receipts("math")) == 1
    with pytest.raises(RecordConflict, match="different content"):
        records.compact(
            "math",
            frozenset(),
            receipt_id="batch-1",
            receipt_metadata={"outcome": "stale", "metrics": {"staleness/samples_dropped": 2}},
        )

    records.compact(
        "math",
        frozenset({"other"}),
        receipt_id="batch-1",
        receipt_metadata=metadata,
    )
    assert len(records.compaction_receipts("math")) == 2


@pytest.mark.unit
def test_audit_reads_preserve_trace_content_without_reactivating_it() -> None:
    trace = AgentRecord.create(
        agent_record_id="trace",
        scenario="code",
        request_type=RequestType.INFERENCE,
        created_at=10.0,
        payload={
            "messages": [{"role": "user", "content": "修复登录重试"}],
            "response": {"choices": [{"message": {"role": "assistant", "content": "先复现失败。"}}]},
        },
        artifact_ref=ArtifactRef("artifact", "release", "initial"),
    )
    report = replace(item("feedback", "code", RequestType.REPORT, references=("trace",)), payload={"score": 0.2})
    with RecordStore() as records:
        records.append(trace)
        records.append(report)
        records.compact("code", frozenset({"trace", "feedback"}))

        page = records.audit_page("code")
        assert [entry.item for entry in page] == [trace, report]
        assert all(entry.compacted_at is not None for entry in page)
        assert records.get_for_audit("code", "feedback").item.references == ("trace",)
        assert records.count("code") == 0
        assert records.replay("code") == ()
        assert records.replay_page("code") == ()
        assert records.get("code", "trace") is None
        assert records.append_result(trace).inserted is False
        assert records.append_result(item("late", "code", RequestType.REPORT, references=("trace",))).inserted is False
        assert records.count("code") == 0


@pytest.mark.unit
def test_training_pages_skip_compacted_gaps_and_audit_pages_include_them() -> None:
    with RecordStore() as records:
        for record in (
            item("a", "code"),
            item("other", "math"),
            item("b", "code", RequestType.REPORT),
            item("c", "code"),
            item("d", "code"),
            item("e", "code", RequestType.REPORT),
        ):
            records.append(record)
        records.compact("code", frozenset({"a", "c", "other"}))
        assert records.get("math", "other") is not None
        first = records.replay_page("code", limit=1)
        second = records.replay_page("code", after_sequence=first[0][0], limit=1)
        assert [row.agent_record_id for _, row in first] == ["b"]
        assert [row.agent_record_id for _, row in second] == ["d"]
        assert [row.agent_record_id for row in records.replay("code", offset=1, limit=1)] == ["d"]
        assert records.count("code") == 3
        assert records.count("code", request_type=RequestType.INFERENCE) == 1
        assert records.count("code", request_type=RequestType.REPORT, after_sequence=first[0][0]) == 1
        audit_first = records.audit_page("code", limit=2)
        audit_second = records.audit_page("code", after_sequence=audit_first[-1].sequence, limit=2)
        assert [row.item.agent_record_id for row in audit_first] == ["a", "b"]
        assert [row.item.agent_record_id for row in audit_second] == ["c", "d"]
        assert audit_first[0].compacted_at is not None
        assert audit_first[1].compacted_at is None
        assert records.audit_page("missing") == ()


@pytest.mark.unit
def test_repeated_compaction_preserves_the_first_retirement_time(monkeypatch) -> None:
    with RecordStore() as records:
        records.append(item("a", "math"))
        monkeypatch.setattr("reef.records.time.time", lambda: 100.0)
        records.compact("math", frozenset({"a"}))
        first = records.get_for_audit("math", "a")
        monkeypatch.setattr("reef.records.time.time", lambda: 200.0)
        records.compact("math", frozenset({"a"}))
        assert records.get_for_audit("math", "a") == first
        assert first.compacted_at == 100.0


@pytest.mark.unit
def test_old_schema_migrates_without_losing_live_records_or_reviving_deleted_bodies(tmp_path) -> None:
    database = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE agent_record (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_record_id TEXT NOT NULL UNIQUE, scenario TEXT NOT NULL,
                request_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                created_at REAL NOT NULL, references_json TEXT NOT NULL, artifact_json TEXT
            );
            CREATE TABLE consumed_agent_record (
                agent_record_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL
            );
            INSERT INTO agent_record VALUES (7, 'live', 'math', 'inference', '{"value":"live"}', 4, '[]', NULL);
            INSERT INTO consumed_agent_record VALUES ('deleted', 'legacy-fingerprint');
            """
        )

    def open_store(_: int) -> None:
        with RecordStore(database) as records:
            assert records.get("math", "live") == item("live", "math")
            entry = records.get_for_audit("math", "live")
            assert entry.sequence == 7 and entry.compacted_at is None
            assert records.get_for_audit("math", "deleted") is None

    # Concurrent openers must see one atomic, idempotent schema upgrade.
    with ThreadPoolExecutor(max_workers=3) as executor:
        tuple(executor.map(open_store, range(3)))
    with RecordStore(database) as records:
        late = item("late", "math", RequestType.REPORT, references=("deleted",))
        assert records.append_result(late).inserted is False
        records.append(item("next", "math"))
        assert records.get_for_audit("math", "next").sequence > 7
        records.compact("math", frozenset({"live"}))
    with RecordStore(database) as records:
        assert records.get("math", "live") is None
        assert records.get_for_audit("math", "live").item == item("live", "math")


@pytest.mark.unit
def test_compaction_failure_rolls_back_body_state_hashes_and_receipt(tmp_path) -> None:
    database = tmp_path / "records.sqlite3"
    with RecordStore(database) as records:
        records.append(item("a", "math"))
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TRIGGER reject_compaction BEFORE UPDATE OF compacted_at ON agent_record "
                "BEGIN SELECT RAISE(ABORT, 'injected compaction failure'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match="injected compaction failure"):
            records.compact("math", frozenset({"a"}), receipt_id="batch", receipt_metadata={"outcome": "stale"})
        assert records.get("math", "a") is not None
        assert records.get_for_audit("math", "a").compacted_at is None
        assert records.compaction_receipts("math") == ()
        assert records.append_result(item("report", "math", RequestType.REPORT, references=("a",))).inserted is True


@pytest.mark.unit
def test_purge_is_bounded_scoped_and_preserves_retry_and_receipt_contracts(tmp_path, monkeypatch) -> None:
    database = tmp_path / "records.sqlite3"
    original = item("a", "math")
    with RecordStore(database) as records:
        for record in (
            original,
            item("b", "math"),
            item("newer", "math"),
            item("active", "math"),
            item("other", "code"),
        ):
            records.append(record)
        monkeypatch.setattr("reef.records.time.time", lambda: 100.0)
        records.compact("math", frozenset({"a", "b"}), receipt_id="batch", receipt_metadata={"outcome": "stale"})
        records.compact("code", frozenset({"other"}))
        monkeypatch.setattr("reef.records.time.time", lambda: 200.0)
        records.compact("math", frozenset({"newer"}))
        assert records.purge_compacted("math", before=100.0) == 0
        assert records.purge_compacted("math", before=150.0, limit=1) == 1
        assert records.get_for_audit("math", "a") is None
        assert records.get_for_audit("math", "b") is not None
        assert records.purge_compacted("math", before=150.0) == 1
        assert records.purge_compacted("math", before=150.0) == 0
        assert records.get_for_audit("math", "newer") is not None
        assert records.get("math", "active") is not None
        assert records.get_for_audit("code", "other") is not None

    with RecordStore(database) as records:
        assert records.append_result(original).inserted is False
        assert records.existing_receipt(original).agent_record_id == "a"
        with pytest.raises(RecordConflict):
            records.append(replace(original, payload={"value": "changed"}))
        assert records.append_result(item("late", "math", RequestType.REPORT, references=("a",))).inserted is False
        assert records.get_for_audit("math", "a") is None
        assert records.compaction_receipts("math")[0]["receipt_id"] == "batch"
        assert records.count("math") == 1


@pytest.mark.unit
@pytest.mark.parametrize("before", [float("nan"), float("inf"), float("-inf")])
def test_purge_rejects_non_finite_cutoffs(before: float) -> None:
    with RecordStore() as records, pytest.raises(ValueError, match="finite"):
        records.purge_compacted("math", before=before)


@pytest.mark.unit
@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_purge_rejects_invalid_limits(limit) -> None:
    with RecordStore() as records, pytest.raises(ValueError, match="positive integer"):
        records.purge_compacted("math", before=100.0, limit=limit)


@pytest.mark.unit
@pytest.mark.parametrize("options", [{"after_sequence": -1}, {"limit": 0}, {"limit": -1}])
def test_audit_page_rejects_invalid_bounds(options) -> None:
    with RecordStore() as records, pytest.raises(ValueError):
        records.audit_page("math", **options)
