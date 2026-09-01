"""Machine-readable cell and aggregate Meta-Harness reports."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .composition import CompositionCandidate
from .evaluation import EvaluationResult
from .search import CandidateRecord, ProposerSession


def write_cell_report(
    *,
    output_dir: Path,
    cell: str,
    selected: CompositionCandidate,
    population: Sequence[CandidateRecord],
    proposer_sessions: Sequence[ProposerSession],
    test_result: EvaluationResult,
    selected_dev_score: float,
    resumed: bool,
    budget_fill_evaluations: int,
    wall_time_s: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = _private_evidence(output_dir / "private-evaluations")
    target_usage = _aggregate_usage(row.get("usage", {}) for row in evidence)
    proposer_usage = {
        "input_tokens": sum(session.input_tokens for session in proposer_sessions),
        "output_tokens": sum(session.output_tokens for session in proposer_sessions),
    }
    statuses = Counter(str(row.get("status") or "unknown") for row in evidence)
    target_cost = sum(float(row.get("estimated_cost_usd", 0.0)) for row in evidence)
    later_history_read = _later_round_history_read(proposer_sessions, output_dir / "workspace")
    task_checksums = _task_checksums(evidence)
    summary = {
        "cell": cell,
        "selected_candidate_hash": selected.content_hash,
        "selected_dev_score": selected_dev_score,
        "test_score": test_result.score,
        "test_trial_count": len(test_result.trials),
        "candidate_count": len(population),
        "candidate_outcomes": dict(Counter(record.outcome for record in population)),
        "proposer_session_count": len(proposer_sessions),
        "later_round_history_read": later_history_read,
        "target_episode_count": len(evidence),
        "budget_fill_evaluations": budget_fill_evaluations,
        "usage": {"target": target_usage, "proposer": proposer_usage},
        "estimated_cost_usd": {"target": target_cost, "proposer": 0.0, "total": target_cost},
        "trial_statuses": dict(sorted(statuses.items())),
        "task_checksums": task_checksums,
        "wall_time_s": wall_time_s,
        "resumed": resumed,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "selected-candidate.json", selected.to_jsonable())
    _write_json(output_dir / "heldout.json", test_result.to_jsonable())
    _write_json(
        output_dir / "learning-curve.json",
        [
            {
                "round": record.round_index,
                "candidate_hash": record.candidate.content_hash,
                "parent_hashes": list(record.candidate.parent_hashes),
                "train_score": record.train_score,
                "dev_score": record.dev_score,
                "outcome": record.outcome,
                "proposal_id": record.proposal_id,
            }
            for record in population
        ],
    )
    (output_dir / "candidate-parents.dot").write_text(_parent_graph_dot(population), encoding="utf-8")
    return summary


def write_aggregate_report(output_dir: Path, cells: Sequence[str]) -> dict[str, Any]:
    summaries = {cell: json.loads((output_dir / cell / "summary.json").read_text(encoding="utf-8")) for cell in cells}
    episode_counts = {int(summary["target_episode_count"]) for summary in summaries.values()}
    checksum_manifests = [summary.get("task_checksums", {}) for summary in summaries.values()]
    payload = {
        "equal_target_episode_budget": len(episode_counts) == 1,
        "target_episode_counts": {cell: summary["target_episode_count"] for cell, summary in summaries.items()},
        "equal_task_checksums": all(manifest == checksum_manifests[0] for manifest in checksum_manifests),
        "task_checksums": checksum_manifests[0],
        "cells": summaries,
        "test_score_mean": statistics.fmean(float(summary["test_score"]) for summary in summaries.values()),
    }
    _write_json(output_dir / "results.json", payload)
    return payload


def _private_evidence(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("trials/*/round-*/*/*/trial-*/evidence.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError(f"trial evidence is not an object: {path}")
        rows.append(value)
    return rows


def _aggregate_usage(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in items:
        if not isinstance(item, Mapping):
            continue
        for key, value in item.items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def _parent_graph_dot(population: Sequence[CandidateRecord]) -> str:
    lines = ["digraph meta_harness {"]
    for record in population:
        short = record.candidate.content_hash[:12]
        lines.append(f'  "{short}" [label="r{record.round_index} {short}\\n{record.dev_score:.4f}"];')
        lines.extend(f'  "{parent[:12]}" -> "{short}";' for parent in record.candidate.parent_hashes)
    lines.append("}")
    return "\n".join(lines) + "\n"


def _task_checksums(evidence: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for row in evidence:
        verifier = row.get("verifier")
        checksum = verifier.get("task_checksum") if isinstance(verifier, Mapping) else None
        if not checksum:
            continue
        task_id = str(row.get("task_id") or "")
        previous = checksums.setdefault(task_id, str(checksum))
        if previous != checksum:
            raise RuntimeError(f"Terminal-Bench task checksum changed within the cell: {task_id}")
    return dict(sorted(checksums.items()))


def _later_round_history_read(sessions: Sequence[ProposerSession], workspace_dir: Path) -> bool | None:
    if len(sessions) < 2:
        return None
    for session in sessions[1:]:
        if not session.artifact_dir:
            continue
        path = workspace_dir / session.artifact_dir / "metadata.json"
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        evidence = metadata.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("history_accesses"):
            return True
    return False


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
