"""The GEPA candidate pool on disk: fronts, parent sampling, budget.

GEPA keeps its whole search in one in-process state object: every program
it ever proposed, that program's per-instance validation scores, the Pareto
front each instance induces, a round-robin cursor per candidate, and the
metric-call budget spent so far. Reef's algorithm state holds only the
served composition, and the method owns everything else, so that state
lives here - one JSON file per scenario beside the run, rewritten
atomically after every change so an interrupted step resumes instead of
restarting the search.

Two rules are GEPA's and are reproduced exactly:

- the instance fronts (``gepa.core.state``): a strictly better score on a
  task replaces that task's front, an equal score joins it. Recomputing the
  fronts from the recorded scores gives the same sets as GEPA's incremental
  update, so they are derived here rather than stored.
- parent sampling (``gepa.gepa_utils``): drop the candidates whose fronts
  are all covered by other candidates, then sample the survivors weighted
  by how many fronts each one appears on. That is what makes GEPA explore
  from specialists rather than always from the current best.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Candidate:
    """One program GEPA proposed: its texts, where it came from, how it scored."""

    texts: dict[str, str]
    parent: int | None = None
    #: One score per ``evolution.tasks`` entry, in task order. ``None`` until
    #: the selector records the mechanism's validation pass, which is what
    #: makes a candidate eligible to become a parent.
    val_scores: list[float] | None = None
    minibatch_scores: list[float] | None = None
    #: Round-robin index into this candidate's sorted component keys.
    cursor: int = 0
    #: Metric calls charged when this candidate was discovered, GEPA's
    #: ``num_metric_calls_by_discovery``: the x axis of a budget curve.
    discovered_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "texts": dict(self.texts),
            "parent": self.parent,
            "val_scores": None if self.val_scores is None else list(self.val_scores),
            "minibatch_scores": None if self.minibatch_scores is None else list(self.minibatch_scores),
            "cursor": self.cursor,
            "discovered_at": self.discovered_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Candidate:
        scores = data.get("val_scores")
        minibatch = data.get("minibatch_scores")
        parent = data.get("parent")
        return cls(
            texts={str(key): str(value) for key, value in dict(data.get("texts") or {}).items()},
            parent=None if parent is None else int(parent),
            val_scores=None if scores is None else [float(score) for score in scores],
            minibatch_scores=None if minibatch is None else [float(score) for score in minibatch],
            cursor=int(data.get("cursor", 0)),
            discovered_at=int(data.get("discovered_at", 0)),
        )


@dataclass
class Archive:
    """Every candidate the search has produced, and the budget it has spent."""

    path: Path
    candidates: list[Candidate] = field(default_factory=list)
    #: The candidate the mechanism is serving, and the one a proposal is
    #: stated against; ``pending`` is the candidate awaiting the selector's
    #: validation pass, at most one, because a step settles before the next.
    served: int | None = None
    pending: int | None = None
    metric_calls: int = 0
    #: GEPA iterations that reached a child evaluation, and how many of those
    #: the minibatch acceptance test rejected.
    steps: int = 0
    rejections: int = 0
    #: One row per reflection: what the model was shown and what it wrote,
    #: with the minibatch scores on both sides and the verdict. GEPA keeps the
    #: same record in its run directory; without it a search outcome can only
    #: be inferred from the candidate texts after the fact.
    proposals: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.path.exists():
            self._load()

    # -- persistence ---------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the GEPA archive at {self.path}: {error}") from error
        if not isinstance(data, Mapping):
            raise ValueError(f"the GEPA archive at {self.path} must hold a JSON object")
        self.candidates = [Candidate.from_dict(entry) for entry in data.get("candidates", ())]
        served, pending = data.get("served"), data.get("pending")
        self.served = None if served is None else int(served)
        self.pending = None if pending is None else int(pending)
        self.metric_calls = int(data.get("metric_calls", 0))
        self.steps = int(data.get("steps", 0))
        self.rejections = int(data.get("rejections", 0))
        self.proposals = [dict(row) for row in data.get("proposals", ())]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "served": self.served,
            "pending": self.pending,
            "metric_calls": self.metric_calls,
            "steps": self.steps,
            "rejections": self.rejections,
            "proposals": [dict(row) for row in self.proposals],
        }

    def _write(self) -> None:
        """Rewrite the archive atomically: a crash resumes, never truncates."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_name(f"{self.path.name}.tmp")
        scratch.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, self.path)

    # -- growth --------------------------------------------------------------

    def seed(self, texts: Mapping[str, str]) -> int:
        """Append ``texts`` as a parentless candidate and serve it.

        Used for the first boot and for a rollback: when the served tree no
        longer matches the archive's served candidate, whatever the operator
        put there is the new root of the search, not a descendant of it.
        """
        self.candidates.append(Candidate(texts=dict(texts), discovered_at=self.metric_calls))
        self.served = len(self.candidates) - 1
        self.pending = None
        self._write()
        return self.served

    def add(self, texts: Mapping[str, str], parent: int, minibatch_scores: Sequence[float]) -> int:
        """Append an accepted child as the pending candidate and return its index.

        The child inherits its parent's round-robin cursor, as GEPA does, so
        successive descendants keep walking the component list rather than
        all reflecting on the same component.
        """
        self.candidates.append(
            Candidate(
                texts=dict(texts),
                parent=parent,
                minibatch_scores=[float(score) for score in minibatch_scores],
                cursor=self.candidates[parent].cursor,
                discovered_at=self.metric_calls,
            )
        )
        self.pending = len(self.candidates) - 1
        self.steps += 1
        self._write()
        return self.pending

    def reject(self) -> None:
        """Record a GEPA iteration whose child lost the minibatch comparison."""
        self.steps += 1
        self.rejections += 1
        self._write()

    def log_proposal(self, row: Mapping[str, Any]) -> None:
        """Keep one reflection's full record: parent, component, minibatch,
        prompt, reply, child scores, and whether the child was accepted."""
        self.proposals.append({"step": self.steps, **row})
        self._write()

    def record_validation(self, index: int, scores: Sequence[float]) -> None:
        self.candidates[index].val_scores = [float(score) for score in scores]
        self._write()

    def serve(self, index: int) -> None:
        self.served = index
        self._write()

    def clear_pending(self) -> None:
        self.pending = None
        self._write()

    def charge(self, calls: int) -> None:
        """Spend ``calls`` of the metric-call budget: one per scored episode."""
        self.metric_calls += calls
        self._write()

    # -- GEPA's two rules ----------------------------------------------------

    def fronts(self) -> dict[int, set[int]]:
        """Task index to the candidates holding that task's best score."""
        scored = [(index, candidate.val_scores) for index, candidate in enumerate(self.candidates)]
        scored = [(index, scores) for index, scores in scored if scores]
        fronts: dict[int, set[int]] = {}
        best: dict[int, float] = {}
        for index, scores in scored:
            for task, score in enumerate(scores):
                previous = best.get(task)
                if previous is None or score > previous:
                    best[task] = score
                    fronts[task] = {index}
                elif score == previous:
                    fronts[task].add(index)
        return fronts

    def mean_val(self, index: int) -> float:
        scores = self.candidates[index].val_scores
        return sum(scores) / len(scores) if scores else 0.0

    def best(self) -> int:
        """The archive's best candidate by mean validation score, first on ties."""
        if not self.candidates:
            raise ValueError("the GEPA archive holds no candidates")
        means = [self.mean_val(index) for index in range(len(self.candidates))]
        return means.index(max(means))

    def select_parent(self, rng: random.Random) -> int:
        """Sample a parent from the Pareto front, GEPA's selection rule.

        Falls back to the served candidate before any candidate has been
        validated: the seed's own scores only arrive with the mechanism's
        first evaluation pass, and the search has to start somewhere.
        """
        survivors = _remove_dominated(self.fronts(), [self.mean_val(index) for index in range(len(self.candidates))])
        frequency: dict[int, int] = {}
        for front in survivors.values():
            for index in front:
                frequency[index] = frequency.get(index, 0) + 1
        # Sorted so a seeded rng reproduces the same walk across processes;
        # the multiset is what GEPA samples, the order is ours.
        sampling = [index for index, count in sorted(frequency.items()) for _ in range(count)]
        if not sampling:
            if self.served is None:
                raise ValueError("the GEPA archive has no candidate to parent a proposal")
            return self.served
        return rng.choice(sampling)

    def next_component(self, index: int, keys: Sequence[str]) -> str:
        """The next component to reflect on for this candidate, round robin."""
        if not keys:
            raise ValueError("a GEPA candidate needs at least one evolvable component")
        candidate = self.candidates[index]
        position = candidate.cursor % len(keys)
        candidate.cursor = (position + 1) % len(keys)
        self._write()
        return keys[position]


def _is_dominated(candidate: int, others: set[int], fronts: Mapping[int, set[int]]) -> bool:
    """Whether every front holding ``candidate`` also holds one of ``others``."""
    return all(any(other in front for other in others) for front in fronts.values() if candidate in front)


def _remove_dominated(fronts: Mapping[int, set[int]], scores: Sequence[float]) -> dict[int, set[int]]:
    """GEPA's ``remove_dominated_programs``: prune covered candidates.

    Candidates are considered worst-first, so when two of them cover each
    other the lower-scoring one is the one dropped.
    """
    frequency: dict[int, int] = {}
    for front in fronts.values():
        for index in front:
            frequency[index] = frequency.get(index, 0) + 1
    programs = sorted(frequency, key=lambda index: scores[index])
    dominated: set[int] = set()
    removing = True
    while removing:
        removing = False
        for candidate in programs:
            if candidate in dominated:
                continue
            if _is_dominated(candidate, set(programs) - {candidate} - dominated, fronts):
                dominated.add(candidate)
                removing = True
                break
    return {task: {index for index in front if index not in dominated} for task, front in fronts.items()}
