"""Watch CORAL's finalized attempts and report each one to Reef exactly once.

CORAL's grader daemon finalizes every attempt as a JSON record under the
run's ``.coral`` directory (``public/attempts/*.json``, or
``islands/<id>/attempts/*.json`` in multi-island runs). That on-disk record
is the integration surface: this module reads the JSON directly, so the
adapter package keeps its no-CORAL-import contract while consuming the real
runtime's output.

For each newly finalized attempt the watcher resolves the attempt's
inference references from the call journal and posts one
:class:`~recipes.coral.reporter.AttemptReport`. What gets reported:

- ``real`` attempts, whatever their score — the processor decides what
  trains; the report is the record of what happened.
- ``grader_error`` and ``tune`` attempts are skipped: the former is the
  eval machinery failing (no policy signal in it), the latter is a config
  sweep CORAL itself excludes from optimization budgets.
- an attempt is reported once, in its first terminal state. A later regrade
  under the same commit would produce a conflicting resend of the same
  deterministic report id, which reef rejects loudly — terminal means
  terminal.

Reference resolution: the gateway journals every call with the worktree's
HEAD at call time — the *parent* the agent was editing, not the commit
``coral eval`` creates afterwards. So an attempt's calls are the journal
records carrying its agent id and its parent hash. Consecutive attempts an
agent makes from the same parent (a revert, a rejected eval retried) share
that coordinate; the watcher claims records in journal order, so each
report takes only records not already attributed to an earlier attempt.

Crash safety: the reported/claimed state persists to a JSON state file
after every acknowledged report. A watcher restarted after a crash re-posts
at most the one in-flight report, and the reporter's deterministic
client-supplied id makes that resend dedup server-side.
"""

from __future__ import annotations

import json
import logging
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recipes.coral.journal import CallJournal, commit_matches
from recipes.coral.reporter import AttemptReport, report_attempt

logger = logging.getLogger(__name__)

#: ``metadata.budget_class`` values (CORAL's attempt classification) the
#: watcher reports. Attempts without the key are ``real`` (CORAL default).
_REPORTED_BUDGET_CLASSES = frozenset({"real"})


@dataclass(frozen=True)
class FinalizedAttempt:
    """The slice of CORAL's on-disk attempt record the adapter consumes."""

    commit_hash: str
    agent_id: str
    score: float | None
    status: str
    parent_hash: str | None
    timestamp: str
    title: str = ""
    feedback: str = ""
    budget_class: str = "real"
    archived: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FinalizedAttempt:
        metadata = data.get("metadata") or {}
        budget_class = metadata.get("budget_class")
        if budget_class not in ("real", "grader_error", "tune"):
            budget_class = "real"
        score = data.get("score")
        return cls(
            commit_hash=data["commit_hash"],
            agent_id=data["agent_id"],
            score=float(score) if score is not None else None,
            status=data.get("status", "crashed"),
            parent_hash=data.get("parent_hash"),
            timestamp=data.get("timestamp", ""),
            title=data.get("title", ""),
            feedback=data.get("feedback", ""),
            budget_class=budget_class,
            archived=metadata.get("archived") is True,
        )


def iter_attempt_files(coral_dir: Path) -> Iterator[Path]:
    """Every attempt JSON under a run's ``.coral`` directory.

    Single-island runs keep attempts in ``public/attempts/``; multi-island
    runs shard them under ``islands/<id>/attempts/``. Both layouts are
    scanned so the watcher does not care which mode the run used.
    """
    for attempts_dir in (
        coral_dir / "public" / "attempts",
        *sorted((coral_dir / "islands").glob("*/attempts")),
    ):
        if attempts_dir.is_dir():
            yield from sorted(attempts_dir.glob("*.json"))


def read_finalized_attempts(coral_dir: Path) -> list[FinalizedAttempt]:
    """Parse every terminal, unarchived attempt record, oldest first.

    Pending attempts (grader hasn't scored them yet) and malformed files
    (a write races the scan; CORAL writes atomically, but a foreign file
    could sit in the directory) are skipped, never fatal.
    """
    attempts: list[FinalizedAttempt] = []
    for path in iter_attempt_files(coral_dir):
        try:
            attempt = FinalizedAttempt.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            continue
        if attempt.status == "pending" or attempt.archived:
            continue
        attempts.append(attempt)
    attempts.sort(key=lambda a: a.timestamp)
    return attempts


class AttemptWatcher:
    """Poll a run's attempts and report each finalized one to Reef once.

    Drive it with :meth:`poll_once` from any loop the caller owns (the
    example runs it on a thread beside CORAL's monitor loop, with a final
    drain after the run stops). ``self.reports`` accumulates every
    acknowledged report in post order — the bundle builder's input.
    """

    def __init__(
        self,
        *,
        coral_dir: Path,
        journal: CallJournal,
        reef_url: str,
        scenario: str,
        run_id: str,
        token: str | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.coral_dir = Path(coral_dir)
        self.journal = journal
        self.reef_url = reef_url
        self.scenario = scenario
        self.run_id = run_id
        self.token = token
        self.state_path = Path(state_path) if state_path else self.coral_dir / "reef_reported.json"
        self.reports: list[AttemptReport] = []
        self._reported: set[str] = set()  # commit hashes acknowledged by reef
        self._claimed: set[str] = set()  # journal record ids attributed to a report
        self._load_state()

    # -- state persistence --------------------------------------------------

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._reported = set(state.get("reported_commits", []))
            self._claimed = set(state.get("claimed_record_ids", []))
        except (json.JSONDecodeError, OSError, TypeError):
            logger.warning("unreadable watcher state at %s; starting fresh", self.state_path)

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "reported_commits": sorted(self._reported),
                "claimed_record_ids": sorted(self._claimed),
            }
        )
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(self.state_path)

    # -- reference resolution ------------------------------------------------

    def _resolve_references(self, attempt: FinalizedAttempt) -> tuple[str, ...]:
        """Unclaimed journal record ids at the attempt's (agent, parent) coordinate."""
        if not attempt.parent_hash:
            return ()
        out: list[str] = []
        for record in self.journal.records():
            if (
                record.agent_id == attempt.agent_id
                and record.agent_record_id
                and record.agent_record_id not in self._claimed
                and record.agent_record_id not in out
                and commit_matches(record.commit_hash, attempt.parent_hash)
            ):
                out.append(record.agent_record_id)
        return tuple(out)

    # -- the poll ------------------------------------------------------------

    def poll_once(self) -> list[AttemptReport]:
        """Report every newly finalized attempt; returns the reports posted.

        A report that fails with a transport error stays unreported and is
        retried on the next poll; the deterministic report id keeps a resend
        after a mid-POST crash from double-counting.
        """
        posted: list[AttemptReport] = []
        for attempt in read_finalized_attempts(self.coral_dir):
            if attempt.commit_hash in self._reported:
                continue
            if attempt.budget_class not in _REPORTED_BUDGET_CLASSES:
                self._reported.add(attempt.commit_hash)  # terminal skip, never revisit
                self._save_state()
                continue
            references = self._resolve_references(attempt)
            report = AttemptReport(
                scenario=self.scenario,
                agent_id=attempt.agent_id,
                commit_hash=attempt.commit_hash,
                score=attempt.score,
                status=attempt.status,
                parent_hash=attempt.parent_hash,
                run_id=self.run_id,
                feedback=attempt.feedback or None,
                references=references,
            )
            try:
                ack = report_attempt(self.reef_url, report, token=self.token)
            except urllib.error.HTTPError:
                # Reef rejected the report (e.g. a conflicting resend under the
                # same deterministic id). That's a bug to surface, not retry.
                raise
            except OSError as exc:  # connection refused / timeout — retry next poll
                logger.warning(
                    "report for attempt %s failed (%s); will retry",
                    attempt.commit_hash[:12],
                    exc,
                )
                continue
            self._reported.add(attempt.commit_hash)
            self._claimed.update(references)
            self._save_state()
            self.reports.append(report)
            posted.append(report)
            logger.info(
                "reported attempt %s (agent=%s score=%s refs=%d ack=%s)",
                attempt.commit_hash[:12],
                attempt.agent_id,
                attempt.score,
                len(references),
                ack.get("agent_record_id"),
            )
        return posted

    def run(self, stop_event: Any, interval_s: float = 5.0) -> None:
        """Poll until ``stop_event`` is set, then drain one final time."""
        while not stop_event.is_set():
            self.poll_once()
            stop_event.wait(interval_s)
        self.poll_once()
