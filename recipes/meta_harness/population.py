"""The Meta-Harness candidate population: every proposed tree, its scores, and the budget spent.

The population is the method state Reef commits with each step
(``Population.to_dict``), not a file of its own. A step reads it from the
committed state, folds in the evaluation of its pending candidate
(``population_for_step``), and returns the next one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from reef.harness.tree import HarnessTree
from reef.train.harness_backend.contract import StepOutcome

SCHEMA_VERSION = 2
#: The method state key that holds the population.
POPULATION_KEY = "population"
OUTCOMES = frozenset({"pending", "retained", "selected"})


def finite_scores(values: Sequence[float | None]) -> tuple[float, ...]:
    """A non-empty score vector of finite values, a failed episode read as zero."""
    scores = tuple(0.0 if value is None else float(value) for value in values)
    if not scores:
        raise ValueError("candidate score vectors must not be empty")
    if any(not math.isfinite(value) for value in scores):
        raise ValueError("candidate score vectors must contain only finite values")
    return scores


@dataclass
class CandidateRecord:
    """One unique tree and the results collected for it.

    Values: ``candidate_id`` is the checksum of the tree ``files`` holds, so
    two proposals of the same tree are one candidate. ``parent_id`` is the
    candidate it was proposed from, ``None`` for a tree the population took
    as served (the seed, a rollback). ``proposed_at`` ≥ 0 is the step that
    proposed or took it. ``outcome`` is ``pending`` until the step that
    proposed it is evaluated, then ``selected`` or ``retained``. ``scores``
    are its finite per-task scores, ``None`` before its evaluation.
    ``hypothesis`` and ``changes`` are the proposer's own words.
    Purpose: ``Population`` holds one per tree; the proposer shows them to
    the model, and the selection reads the served one's scores.
    """

    candidate_id: str
    files: dict[str, str]
    parent_id: str | None
    proposed_at: int
    outcome: str
    scores: tuple[float, ...] | None = None
    hypothesis: str = ""
    changes: str = ""

    def __post_init__(self) -> None:
        if self.candidate_id != HarnessTree(self.files).checksum():
            raise ValueError("candidate id does not match its tree")
        if self.parent_id is not None and (not isinstance(self.parent_id, str) or not self.parent_id):
            raise ValueError("candidate parent id must be a non-empty string or None")
        if isinstance(self.proposed_at, bool) or not isinstance(self.proposed_at, int) or self.proposed_at < 0:
            raise ValueError("candidate proposed_at must be a non-negative integer")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"candidate outcome must be one of {tuple(sorted(OUTCOMES))}")
        if self.scores is not None:
            self.scores = finite_scores(self.scores)

    @property
    def tree(self) -> HarnessTree:
        return HarnessTree(self.files)

    @classmethod
    def create(
        cls,
        tree: HarnessTree,
        *,
        parent_id: str | None,
        proposed_at: int,
        outcome: str,
        hypothesis: str = "",
        changes: str = "",
    ) -> CandidateRecord:
        return cls(
            candidate_id=tree.checksum(),
            files=dict(tree.files),
            parent_id=parent_id,
            proposed_at=proposed_at,
            outcome=outcome,
            hypothesis=hypothesis,
            changes=changes,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRecord:
        files = value.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("candidate record files must be an object of path to text")
        scores = value.get("scores")
        if scores is not None and (not isinstance(scores, Sequence) or isinstance(scores, (str, bytes))):
            raise ValueError("candidate record scores must be a list or null")
        parent = value.get("parent_id")
        return cls(
            candidate_id=str(value.get("candidate_id", "")),
            files={str(path): str(text) for path, text in files.items()},
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
            "files": dict(self.files),
            "parent_id": self.parent_id,
            "proposed_at": self.proposed_at,
            "outcome": self.outcome,
            "scores": None if self.scores is None else list(self.scores),
            "hypothesis": self.hypothesis,
            "changes": self.changes,
        }


@dataclass
class Population:
    """Every proposed candidate, the served one, the pending one with the step that proposed it, and the budgets.

    Values: ``candidates`` have unique ids, each parent before its child.
    ``served_id`` and ``pending_id`` name candidates the list holds, or are
    ``None``; ``pending_id`` and ``pending_step`` are both set or both
    ``None``, and a pending candidate has the ``pending`` outcome.
    ``proposer_calls`` and ``episode_calls`` ≥ 0 count the model calls and
    the evaluation episodes the search spent. ``attempts`` records each
    proposal attempt, kept or not.
    Purpose: the method state of a Meta-Harness scenario under
    ``POPULATION_KEY``; the proposer reads and updates it each step, and the
    selection reads it.
    """

    candidates: list[CandidateRecord] = field(default_factory=list)
    served_id: str | None = None
    pending_id: str | None = None
    pending_step: int | None = None
    proposer_calls: int = 0
    episode_calls: int = 0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids: set[str] = set()
        for candidate in self.candidates:
            if candidate.candidate_id in ids:
                raise ValueError("population candidates must have unique tree checksums")
            if candidate.parent_id is not None and candidate.parent_id not in ids:
                raise ValueError("population candidate parent must precede its child")
            ids.add(candidate.candidate_id)
        if self.served_id is not None and self.served_id not in ids:
            raise ValueError("population served candidate is missing")
        if self.pending_id is not None and self.pending_id not in ids:
            raise ValueError("population pending candidate is missing")
        if (self.pending_id is None) != (self.pending_step is None):
            raise ValueError("population pending_id and pending_step are both set or both None")
        if self.pending_id is not None and self.by_id(self.pending_id).outcome != "pending":
            raise ValueError("population pending candidate must have pending outcome")
        for label, value in (("proposer_calls", self.proposer_calls), ("episode_calls", self.episode_calls)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"population {label} must be a non-negative integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> Population:
        """The population a committed ``to_dict`` snapshot describes; the empty one for none."""
        if value is None:
            return cls()
        if int(value.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
            raise ValueError(f"population schema_version must be {SCHEMA_VERSION}")
        rows = value.get("candidates", ())
        attempts = value.get("attempts", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError("population candidates must be a list")
        if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
            raise ValueError("population attempts must be a list")
        served, pending, pending_step = value.get("served_id"), value.get("pending_id"), value.get("pending_step")
        return cls(
            candidates=[CandidateRecord.from_dict(row) for row in rows],
            served_id=None if served is None else str(served),
            pending_id=None if pending is None else str(pending),
            pending_step=None if pending_step is None else int(pending_step),
            proposer_calls=int(value.get("proposer_calls", 0)),
            episode_calls=int(value.get("episode_calls", 0)),
            attempts=[json.loads(json.dumps(dict(row), sort_keys=True)) for row in attempts],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "served_id": self.served_id,
            "pending_id": self.pending_id,
            "pending_step": self.pending_step,
            "proposer_calls": self.proposer_calls,
            "episode_calls": self.episode_calls,
            "attempts": [dict(attempt) for attempt in self.attempts],
        }

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

    def sync_served(self, tree: HarnessTree, *, step: int) -> CandidateRecord:
        """Make the tree the step serves a known member and the served one.

        A tree the population does not hold (the first step, or an operator's
        rollback) joins it as a parentless candidate.
        """
        candidate_id = tree.checksum()
        try:
            candidate = self.by_id(candidate_id)
        except KeyError:
            candidate = CandidateRecord.create(
                tree, parent_id=None, proposed_at=step, outcome="selected", hypothesis="Reef served tree"
            )
            self.candidates.append(candidate)
        candidate.outcome = "selected"
        self.served_id = candidate_id
        return candidate

    def stage_candidate(
        self, tree: HarnessTree, *, parent_id: str, step: int, hypothesis: str = "", changes: str = ""
    ) -> tuple[CandidateRecord, bool]:
        """The candidate ``tree`` is, staged as ``step``'s pending one when it is new; whether it is new."""
        if self.pending_id is not None:
            raise RuntimeError("population already has a pending candidate")
        self.by_id(parent_id)
        try:
            return self.by_id(tree.checksum()), False
        except KeyError:
            candidate = CandidateRecord.create(
                tree, parent_id=parent_id, proposed_at=step, outcome="pending", hypothesis=hypothesis, changes=changes
            )
            self.candidates.append(candidate)
            self.pending_id, self.pending_step = candidate.candidate_id, step
            return candidate, True

    def discard_pending(self) -> None:
        """Drop the pending candidate: its step proposed it but never had it evaluated."""
        pending_id = self.pending_id
        self.pending_id = self.pending_step = None
        if pending_id is None:
            return
        self.candidates = [
            candidate
            for candidate in self.candidates
            if not (candidate.candidate_id == pending_id and candidate.outcome == "pending")
        ]

    def settle(self, outcome: StepOutcome) -> CandidateRecord:
        """Record the evaluation of the pending candidate: its scores, the served one's first scores, the result."""
        candidate = self.pending
        served = self.served
        candidate.scores = finite_scores(outcome.candidate_scores)
        # Written once, never refreshed: the frontier only moves when a candidate wins, so an incumbent keeps the
        # score it was admitted on. Overwriting it with each batch would let the bar drift downward, which
        # upstream's `if avg > current_best_avg` never does.
        if served.scores is None:
            served.scores = finite_scores(outcome.current_scores)
        candidate.outcome = "selected" if outcome.selected else "retained"
        if outcome.selected:
            self.served_id = candidate.candidate_id
        self.pending_id = self.pending_step = None
        self.episode_calls += len(outcome.candidate_scores) + len(outcome.current_scores)
        return candidate

    def record_attempt(self, **values: Any) -> None:
        self.attempts.append(json.loads(json.dumps(values, sort_keys=True)))


def population_for_step(method_state: Mapping[str, Any], last_outcome: StepOutcome | None) -> Population:
    """The population as a step starts: the committed one, with the evaluation of its pending candidate folded in.

    The pending candidate's evaluation is the outcome of the step that
    proposed it. A pending candidate whose step never reached an evaluation
    has none and is dropped.
    """
    population = Population.from_dict(method_state.get(POPULATION_KEY))
    if population.pending_id is None:
        return population
    if last_outcome is not None and last_outcome.step == population.pending_step:
        population.settle(last_outcome)
    else:
        population.discard_pending()
    return population


__all__ = ["POPULATION_KEY", "CandidateRecord", "Population", "finite_scores", "population_for_step"]
