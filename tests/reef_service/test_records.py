from __future__ import annotations

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
def test_inference_data_can_record_artifact_version() -> None:
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
        weight_version="weights-1",
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

    with RecordStore(database) as recovered:
        assert recovered.append_result(inference).inserted is False
        late = item("late", "math", RequestType.REPORT, references=("inference",))
        assert recovered.append_result(late).inserted is False
        assert recovered.count("math") == 1


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
def test_compaction_receipt_is_atomic_with_payload_deletion_and_persists(tmp_path) -> None:
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
