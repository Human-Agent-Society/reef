"""Read-only projections of retained records and committed learning participation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from reef.records import StoredRecord
from reef.scenario.commit_log import CommitRecord
from reef.scenario.scenario import Scenario


def learning_links(commits: tuple[CommitRecord, ...], record_id: str) -> list[dict[str, Any]]:
    """Only committed consumed_ids prove participation; retirement alone does not."""
    links = []
    for commit in commits:
        if record_id not in commit.consumed_ids:
            continue
        metrics = commit.metrics or {}
        selection = metrics.get("selection")
        selection = selection if isinstance(selection, Mapping) else {}
        links.append(
            {
                "step": commit.step,
                "operation": commit.operation,
                "release_id": commit.artifact_ref.release_id,
                "candidate_id": selection.get("candidate_id"),
                "selected": metrics.get("selected"),
                "pending": commit.pending,
                "recorded_at": commit.recorded_at,
                "reason": selection.get("reason"),
                "gate": {
                    key: metrics[key]
                    for key in ("wins", "losses", "ties", "candidate_score", "current_score")
                    if isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
                },
            }
        )
    return links


def record_summary(stored: StoredRecord, commits: tuple[CommitRecord, ...]) -> dict[str, Any]:
    item = stored.item
    links = learning_links(commits, item.agent_record_id)
    score = item.payload.get("score")
    return {
        "sequence": stored.sequence,
        "agent_record_id": item.agent_record_id,
        "request_type": item.request_type.value,
        "created_at": item.created_at,
        "compacted_at": stored.compacted_at,
        "references": list(item.references),
        "served_by": item.artifact_ref.release_id if item.artifact_ref else None,
        "score": score if isinstance(score, (int, float)) and math.isfinite(score) else None,
        "learning_state": "consumed" if links else "unknown" if stored.compacted_at is not None else "awaiting",
        "reason": (
            "Named in committed consumed_ids. Participation does not imply a candidate was promoted."
            if links
            else (
                "Retired from the active queue; no committed consumption or eligibility reason is recorded."
                if stored.compacted_at is not None
                else "No committed consumption yet. The record may be waiting for feedback, processing, or a batch."
            )
        ),
        "learning_steps": links,
    }


def inspect_learning(scenario: Scenario, *, after_sequence: int, limit: int) -> dict[str, Any]:
    if after_sequence < 0 or not 1 <= limit <= 100:
        raise ValueError("after_sequence must be non-negative and limit must be between 1 and 100")
    retained = scenario.records.audit_page(scenario.name, after_sequence=after_sequence, limit=limit + 1)
    commits = scenario.commit_log.records() if scenario.commit_log else ()
    page = retained[:limit]
    processor = scenario.trainer.processor
    return {
        "scenario": scenario.name,
        "records": [record_summary(stored, commits) for stored in page],
        "next_after_sequence": page[-1].sequence if len(retained) > limit else None,
        "policy": {
            "processor": type(processor).__name__,
            "required_request_types": sorted(rt.value for rt in processor.required_request_types),
            "training_mode": scenario.trainer.training_mode,
            "status": dict(scenario.trainer.processor_status()),
            "historical_decisions_available": False,
            "eval_thresholds_available": False,
        },
    }


def inspect_record(scenario: Scenario, record_id: str) -> dict[str, Any] | None:
    stored = scenario.records.get_for_audit(scenario.name, record_id)
    if stored is None:
        return None
    commits = scenario.commit_log.records() if scenario.commit_log else ()
    return {**record_summary(stored, commits), "payload": stored.item.payload}
