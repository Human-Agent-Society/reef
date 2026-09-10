"""Append-only JSONL journal: which Reef records belong to which CORAL attempt.

A crash loses at most the in-flight request; torn lines are skipped on read.
When a receipt did not survive the proxy hop, the tags Reef stored with the
INFERENCE record remain the correlation key.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CallRecord:
    """One provider call as the gateway saw it."""

    request_id: str
    timestamp: str
    scenario: str
    agent_id: str
    commit_hash: str
    path: str
    status_code: int
    agent_record_id: str | None = None
    #: ``x-reef-release-id`` from the response: which serving revision answered.
    release_id: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> CallRecord:
        data = json.loads(line)
        return cls(**data)


class CallJournal:
    """Append-only JSONL journal, safe for concurrent agents behind one gateway."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CallRecord) -> None:
        line = record.to_json() + "\n"
        with self._lock, open(self._path, "a", encoding="utf-8") as f:
            f.write(line)

    def records(self) -> list[CallRecord]:
        if not self._path.exists():
            return []
        out: list[CallRecord] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(CallRecord.from_json(line))
                except (json.JSONDecodeError, TypeError):
                    continue  # torn write from a crash — skip, never fail reads
        return out

    def size(self) -> int:
        """Current record count — a cursor for :meth:`record_ids_since`.

        Two sibling attempts run from the same worktree state, so their calls
        share the (agent, commit) coordinate; a caller that snapshots the
        cursor before an attempt's calls and resolves with it afterwards gets
        exactly that attempt's records.
        """
        return len(self.records())

    def record_ids_since(self, cursor: int, agent_id: str, commit_hash: str) -> list[str]:
        """Captured record ids for one attempt, starting at ``cursor``."""
        seen: dict[str, None] = {}
        for record in self.records()[cursor:]:
            if (
                record.agent_id == agent_id
                and record.commit_hash == commit_hash
                and record.agent_record_id
                and record.agent_record_id not in seen
            ):
                seen[record.agent_record_id] = None
        return list(seen)

    def record_ids_for_attempt(self, agent_id: str, commit_hash: str) -> list[str]:
        """The captured Reef record ids for an attempt, in call order, deduplicated.

        ``commit_hash`` is the worktree commit the calls were made *from* (the
        attempt's parent) — the gateway stamps each request with the worktree's
        HEAD at call time, which is the state the agent was editing, not the
        commit ``coral eval`` creates afterwards. Prefix comparison in either
        direction, because CORAL's middleware truncates the hash to 12 chars
        while attempt records carry the full 40.

        A retried provider call produces two journal lines with two distinct
        record ids — both belong to the attempt (both hit the model); dedup
        only collapses the same receipt seen twice.
        """
        seen: dict[str, None] = {}
        for record in self.records():
            if (
                record.agent_id == agent_id
                and commit_matches(record.commit_hash, commit_hash)
                and record.agent_record_id
                and record.agent_record_id not in seen
            ):
                seen[record.agent_record_id] = None
        return list(seen)


def commit_matches(recorded: str, wanted: str) -> bool:
    """True when two commit identifiers name the same commit.

    Either side may be truncated (the gateway journals 12 chars, attempt
    records carry 40), so the shorter must prefix the longer. Guarded to at
    least 7 chars so placeholder values like ``unknown`` never prefix-match.
    """
    if not recorded or not wanted:
        return False
    if len(recorded) < 7 or len(wanted) < 7:
        return recorded == wanted
    shorter, longer = sorted((recorded, wanted), key=len)
    return longer.startswith(shorter)


def deterministic_report_id(scenario: str, agent_id: str, commit_hash: str) -> str:
    """Client-supplied ``agent_record_id`` for the attempt's report.

    Reef dedups identical resends of the same client-supplied id, so a
    reporter that crashes after POST and runs again does not double-count
    the attempt (and a changed payload under the same id is rejected loudly
    rather than silently duplicated).
    """
    import hashlib

    digest = hashlib.sha256(f"{scenario}\x00{agent_id}\x00{commit_hash}".encode()).hexdigest()
    return f"coral-report-{digest[:24]}"
