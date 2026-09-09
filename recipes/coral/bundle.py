"""The result bundle: what a finished CORAL x Reef run leaves behind.

Issue #3's Done-when asks the bundle for score-over-time, the best artifact,
model/skill versions, token/cost accounting, and enough metadata to replay
the run. All of it derives from two artifacts this example already produces:
the call journal (per-call: agent, commit, release id, token usage)
and the sequence of finalized attempt reports (per-attempt: score, status,
parent links). The builder is pure — inputs in, JSON-serializable dict out — so
it runs after the fact on any journal+reports pair, including one recovered
from a crashed run.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from recipes.coral.journal import CallJournal, commit_matches
from recipes.coral.reporter import AttemptReport


def build_result_bundle(
    journal: CallJournal,
    reports: Iterable[AttemptReport],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the run's result bundle.

    ``reports`` is every finalized attempt report, in grading order (the
    order ``score_over_time`` preserves). ``run_id`` filters the journal to
    one run when several share a journal file; reports are trusted as
    pre-filtered by the caller.
    """
    reports = list(reports)
    records = journal.records()
    if run_id is not None:
        records = [r for r in records if r.tags.get("coral-run") == run_id]

    # --- score over time and the best attempt -----------------------------
    score_over_time = [
        {
            "agent_id": report.agent_id,
            "commit_hash": report.commit_hash,
            "parent_hash": report.parent_hash,
            "score": report.score,
            "status": report.status,
        }
        for report in reports
    ]
    scored = [r for r in reports if r.score is not None]
    best = max(scored, key=lambda r: r.score) if scored else None

    # --- correlated calls + serving revisions per attempt -----------------------
    # An attempt's exact calls are the record ids its report references
    # (resolved by the watcher when the attempt finalized). Reports without
    # references — e.g. rebuilt from a partial recovery — fall back to the
    # journal coordinate: the gateway journals each call under the worktree
    # commit it was made from, which is the attempt's *parent*.
    by_record_id = {r.agent_record_id: r for r in records if r.agent_record_id}

    def _calls_for(report: AttemptReport) -> list:
        if report.references:
            return [by_record_id[rid] for rid in report.references if rid in by_record_id]
        if not report.parent_hash:
            return []
        return [
            r for r in records if r.agent_id == report.agent_id and commit_matches(r.commit_hash, report.parent_hash)
        ]

    attempts: list[dict[str, Any]] = []
    for report in reports:
        calls = _calls_for(report)
        attempts.append(
            {
                "agent_id": report.agent_id,
                "commit_hash": report.commit_hash,
                "score": report.score,
                "status": report.status,
                "inference_calls": len(calls),
                "inference_record_ids": [c.agent_record_id for c in calls if c.agent_record_id],
                "release_ids": sorted({c.release_id for c in calls if c.release_id}),
                "prompt_tokens": sum(c.prompt_tokens or 0 for c in calls),
                "completion_tokens": sum(c.completion_tokens or 0 for c in calls),
            }
        )

    # --- run-level accounting ----------------------------------------------
    total_prompt = sum(r.prompt_tokens or 0 for r in records)
    total_completion = sum(r.completion_tokens or 0 for r in records)
    revisions = sorted({r.release_id for r in records if r.release_id})

    return {
        "run_id": run_id,
        "scenario": reports[0].scenario if reports else None,
        "attempt_count": len(reports),
        "score_over_time": score_over_time,
        "best_attempt": (
            None
            if best is None
            else {
                "agent_id": best.agent_id,
                "commit_hash": best.commit_hash,
                "score": best.score,
                "parent_hash": best.parent_hash,
            }
        ),
        "model_revisions_served": revisions,
        "token_accounting": {
            "inference_calls": len(records),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        },
        "attempts": attempts,
        "replay": {
            "journal_path": str(journal.path),
            "correlation": "x-reef-tag-coral-{run,agent,commit} on stored INFERENCE records",
        },
    }
