"""Read retained records and persisted commit metadata without interpreting learning outcomes."""

from __future__ import annotations

from itertools import islice
from typing import Any

from reef.core.artifact_ref import encode_artifact_ref
from reef.records import StoredRecord
from reef.scenario.scenario import Scenario


def record_metadata(stored: StoredRecord) -> dict[str, Any]:
    item = stored.item
    return {
        "sequence": stored.sequence,
        "agent_record_id": item.agent_record_id,
        "request_type": item.request_type.value,
        "created_at": item.created_at,
        "compacted_at": stored.compacted_at,
        "references": list(item.references),
        "artifact_ref": encode_artifact_ref(item.artifact_ref) if item.artifact_ref else None,
        "score": item.payload.get("score"),
    }


def read_records(scenario: Scenario, *, after_sequence: int, limit: int) -> dict[str, Any]:
    if after_sequence < 0 or not 1 <= limit <= 100:
        raise ValueError("after_sequence must be non-negative and limit must be between 1 and 100")
    retained = scenario.records.audit_page(scenario.name, after_sequence=after_sequence, limit=limit + 1)
    page = retained[:limit]
    return {
        "scenario": scenario.name,
        "records": [record_metadata(stored) for stored in page],
        "next_after_sequence": page[-1].sequence if len(retained) > limit else None,
    }


def read_record(scenario: Scenario, record_id: str) -> dict[str, Any] | None:
    stored = scenario.records.get_for_audit(scenario.name, record_id)
    return None if stored is None else {**record_metadata(stored), "payload": stored.item.payload}


def read_commits(scenario: Scenario, *, after_step: int, limit: int, record_ids: tuple[str, ...]) -> dict[str, Any]:
    """Filter committed consumed_ids by exact record IDs; preserve their stored meaning."""
    if after_step < 0 or not 1 <= limit <= 100:
        raise ValueError("after_step must be non-negative and limit must be between 1 and 100")
    if len(record_ids) > 100 or any(not value or len(value) > 256 for value in record_ids):
        raise ValueError("at most 100 non-empty record_id values of at most 256 characters are accepted")
    requested = frozenset(record_ids)
    commits = scenario.commit_log.records() if scenario.commit_log else ()
    matches = tuple(
        islice(
            (
                commit
                for commit in commits
                if commit.step > after_step and (not requested or not requested.isdisjoint(commit.consumed_ids))
            ),
            limit + 1,
        )
    )
    return {
        "scenario": scenario.name,
        "commits": [
            {
                "step": commit.step,
                "operation": commit.operation,
                "operation_verified": commit.operation_verified,
                "recorded_at": commit.recorded_at,
                "artifact_ref": encode_artifact_ref(commit.artifact_ref),
                "pending": commit.pending,
                "consumed_ids": sorted(commit.consumed_ids),
                "metrics": commit.metrics,
            }
            for commit in matches[:limit]
        ],
        "next_after_step": matches[limit - 1].step if len(matches) > limit else None,
    }
