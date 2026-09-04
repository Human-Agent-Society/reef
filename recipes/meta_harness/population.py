"""Durable candidate population for the Meta-Harness method.

The population is algorithm state, not a filesystem database.  Reef carries
the value through its scenario commit log and snapshot metadata.  The JSON
file managed by :class:`PopulationStore` is deliberately write-only from the
method's point of view: it is an inspectable post-commit mirror, and recovery
always rewrites it from Reef's committed state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
_OUTCOMES = frozenset({"pending", "retained", "selected"})


def normalize_entries(entries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return stable, JSON-only root entry options with unique ids."""
    if isinstance(entries, (str, bytes)):
        raise ValueError("candidate entries must be a sequence of entry mappings")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"candidate entry {index} must be a mapping")
        try:
            value = json.loads(json.dumps(dict(entry), sort_keys=True))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate entry {index} must be JSON-serializable") from exc
        entry_id = value.get("id")
        kind = value.get("name")
        if not isinstance(entry_id, str) or not entry_id or ":" in entry_id:
            raise ValueError(f"candidate entry {index} requires a non-empty root-level string id")
        if entry_id in seen:
            raise ValueError(f"candidate entries contain duplicate id {entry_id!r}")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"candidate entry {index} requires a non-empty string name")
        seen.add(entry_id)
        normalized.append(value)
    return tuple(normalized)


def content_id(entries: Sequence[Mapping[str, Any]]) -> str:
    """Content-address one complete composition, including entry order."""
    canonical = json.dumps(normalize_entries(entries), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scores(values: Sequence[float]) -> tuple[float, ...]:
    scores = tuple(float(value) for value in values)
    if not scores:
        raise ValueError("candidate score vectors must not be empty")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("candidate score vectors must contain only finite values")
    return scores


@dataclass
class CandidateRecord:
    """One unique composition and the committed evidence collected for it."""

    candidate_id: str
    entries: tuple[dict[str, Any], ...]
    parent_id: str | None
    proposed_at: int
    outcome: str
    scores: tuple[float, ...] | None = None
    hypothesis: str = ""
    changes: str = ""

    def __post_init__(self) -> None:
        self.entries = normalize_entries(self.entries)
        if self.candidate_id != content_id(self.entries):
            raise ValueError("candidate id does not match its composition content")
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id):
            raise ValueError("candidate parent id must be a non-empty string or None")
        if isinstance(self.proposed_at, bool) or not isinstance(self.proposed_at, int) or self.proposed_at < 0:
            raise ValueError("candidate proposed_at must be a non-negative integer")
        if self.outcome not in _OUTCOMES:
            raise ValueError(f"candidate outcome must be one of {tuple(sorted(_OUTCOMES))}")
        if self.scores is not None:
            self.scores = _scores(self.scores)

    @classmethod
    def create(
        cls,
        entries: Sequence[Mapping[str, Any]],
        *,
        parent_id: str | None,
        proposed_at: int,
        outcome: str,
        hypothesis: str = "",
        changes: str = "",
    ) -> CandidateRecord:
        normalized = normalize_entries(entries)
        return cls(
            candidate_id=content_id(normalized),
            entries=normalized,
            parent_id=parent_id,
            proposed_at=proposed_at,
            outcome=outcome,
            hypothesis=hypothesis,
            changes=changes,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRecord:
        entries = value.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError("candidate record entries must be a list")
        if not all(isinstance(entry, Mapping) for entry in entries):
            raise ValueError("candidate record entries must contain only mappings")
        scores = value.get("scores")
        if scores is not None and (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))):
            raise ValueError("candidate record scores must be a list or null")
        parent = value.get("parent_id")
        return cls(
            candidate_id=str(value.get("candidate_id", "")),
            entries=tuple(dict(entry) for entry in entries),
            parent_id=None if parent is None else str(parent),
            proposed_at=int(value.get("proposed_at", 0)),
            outcome=str(value.get("outcome", "")),
            scores=None if scores is None else tuple(float(score) for score in scores),
            hypothesis=str(value.get("hypothesis", "")),
            changes=str(value.get("changes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "entries": [dict(entry) for entry in self.entries],
            "parent_id": self.parent_id,
            "proposed_at": self.proposed_at,
            "outcome": self.outcome,
            "scores": None if self.scores is None else list(self.scores),
            "hypothesis": self.hypothesis,
            "changes": self.changes,
        }


@dataclass
class Population:
    """Every proposed candidate plus the currently served member and budgets."""

    candidates: list[CandidateRecord] = field(default_factory=list)
    served_id: str | None = None
    pending_id: str | None = None
    proposer_calls: int = 0
    episode_calls: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids: set[str] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in ids:
                raise ValueError("population candidates must have unique content ids")
            if candidate.parent_id is not None and candidate.parent_id not in ids:
                raise ValueError("population candidate parent must precede its child")
            ids.add(candidate.candidate_id)
        if self.served_id is not None and self.served_id not in ids:
            raise ValueError("population served candidate is missing")
        if self.pending_id is not None and self.pending_id not in ids:
            raise ValueError("population pending candidate is missing")
        if self.pending_id is not None and self.by_id(self.pending_id).outcome != "pending":
            raise ValueError("population pending candidate must have pending outcome")
        for label, value in (("proposer_calls", self.proposer_calls), ("episode_calls", self.episode_calls)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"population {label} must be a non-negative integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Population:
        if int(value.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
            raise ValueError(f"population schema_version must be {SCHEMA_VERSION}")
        rows = value.get("candidates", ())
        attempts = value.get("attempts", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("population candidates must be a list")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise ValueError("population attempts must be a list")
        candidates = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("population candidate rows must be mappings")
            candidates.append(CandidateRecord.from_dict(row))
        normalized_attempts = []
        for row in attempts:
            if not isinstance(row, Mapping):
                raise ValueError("population attempt rows must be mappings")
            normalized_attempts.append(json.loads(json.dumps(dict(row), sort_keys=True)))
        served = value.get("served_id")
        pending = value.get("pending_id")
        return cls(
            candidates=candidates,
            served_id=None if served is None else str(served),
            pending_id=None if pending is None else str(pending),
            proposer_calls=int(value.get("proposer_calls", 0)),
            episode_calls=int(value.get("episode_calls", 0)),
            attempts=normalized_attempts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "served_id": self.served_id,
            "pending_id": self.pending_id,
            "proposer_calls": self.proposer_calls,
            "episode_calls": self.episode_calls,
            "attempts": [dict(attempt) for attempt in self.attempts],
        }

    def clone(self) -> Population:
        return Population.from_dict(self.to_dict())

    def by_id(self, candidate_id: str) -> CandidateRecord:
        try:
            return next(candidate for candidate in self.candidates if candidate.candidate_id == candidate_id)
        except StopIteration as exc:
            raise KeyError(candidate_id) from exc

    @property
    def served(self) -> CandidateRecord:
        if self.served_id is None:
            raise ValueError("population has no served candidate")
        return self.by_id(self.served_id)

    @property
    def pending(self) -> CandidateRecord:
        if self.pending_id is None:
            raise ValueError("population has no pending candidate")
        return self.by_id(self.pending_id)

    @property
    def generated_candidates(self) -> int:
        return sum(candidate.parent_id is not None for candidate in self.candidates)

    def sync_served(self, entries: Sequence[Mapping[str, Any]], *, step: int) -> CandidateRecord:
        """Make the composition Reef says it serves a known population member."""
        normalized = normalize_entries(entries)
        candidate_id = content_id(normalized)
        if self.candidates:
            if candidate_id != self.served_id:
                raise ValueError("committed Meta-Harness population does not match Reef's served composition")
            candidate = self.by_id(candidate_id)
            candidate.outcome = "selected"
            self.pending_id = None
            return candidate
        candidate = CandidateRecord.create(
            normalized,
            parent_id=None,
            proposed_at=step,
            outcome="selected",
            hypothesis="Reef served composition",
        )
        self.candidates.append(candidate)
        candidate.outcome = "selected"
        self.served_id = candidate.candidate_id
        self.pending_id = None
        return candidate

    def stage_candidate(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        parent_id: str,
        step: int,
        hypothesis: str = "",
        changes: str = "",
    ) -> tuple[CandidateRecord, bool]:
        if self.pending_id is not None:
            raise RuntimeError("population already has a pending candidate")
        self.by_id(parent_id)
        candidate = CandidateRecord.create(
            entries,
            parent_id=parent_id,
            proposed_at=step,
            outcome="pending",
            hypothesis=hypothesis,
            changes=changes,
        )
        try:
            return self.by_id(candidate.candidate_id), False
        except KeyError:
            self.candidates.append(candidate)
            self.pending_id = candidate.candidate_id
            return candidate, True

    def record_decision(
        self,
        *,
        candidate_scores: Sequence[float],
        current_scores: Sequence[float],
        selected: bool,
    ) -> CandidateRecord:
        candidate = self.pending
        served = self.served
        candidate.scores = _scores(candidate_scores)
        if served.scores is None:
            served.scores = _scores(current_scores)
        candidate.outcome = "selected" if selected else "retained"
        if selected:
            self.served_id = candidate.candidate_id
        self.pending_id = None
        self.episode_calls += len(candidate_scores) + len(current_scores)
        return candidate

    def record_attempt(self, **values: Any) -> None:
        row = json.loads(json.dumps(values, sort_keys=True))
        self.attempts.append(row)


class PopulationStore:
    """A committed population plus one disposable per-step transaction."""

    def __init__(self, mirror_path: Path) -> None:
        self.mirror_path = Path(mirror_path)
        self._committed = Population()
        self._staged: Population | None = None

    @property
    def committed(self) -> Population:
        return self._committed

    @property
    def active(self) -> Population:
        if self._staged is None:
            raise RuntimeError("Meta-Harness population has no active step")
        return self._staged

    def begin(self, state: Mapping[str, Any]) -> Population:
        self._staged = None
        incoming = Population.from_dict(state)
        if incoming.to_dict() != self._committed.to_dict():
            raise RuntimeError(
                "Meta-Harness algorithm state differs from the last applied commit; restart the scenario "
                "before continuing"
            )
        self._staged = incoming.clone()
        return self._staged

    def abort(self) -> None:
        self._staged = None

    def restore_committed(self, state: Mapping[str, Any]) -> None:
        self._committed = Population.from_dict(state)
        self._staged = None

    def commit_applied(self, state: Mapping[str, Any]) -> None:
        self.restore_committed(state)
        self.persist()

    def persist(self) -> None:
        """Atomically rewrite the mirror from committed Reef state."""
        self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.mirror_path.with_name(f".{self.mirror_path.name}.tmp")
        temporary.write_text(json.dumps(self._committed.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.mirror_path)


__all__ = [
    "CandidateRecord",
    "Population",
    "PopulationStore",
    "content_id",
    "normalize_entries",
]
