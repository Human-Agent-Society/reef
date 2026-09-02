"""The GEPA candidate pool: fronts, parent sampling, budget, persistence.

GEPA keeps its whole search in one state object: every program
it ever proposed, that program's per-instance validation scores, the Pareto
front each instance induces, a round-robin cursor per candidate, and the
metric-call budget spent so far. Standalone drivers persist every mutation
to JSON. A Reef recipe instead keeps this object transactionally in Reef's
algorithm state and updates the JSON mirror only after the scenario commit
is durable, so a failed activation or commit cannot advance the search.

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
    #: Standalone archives write after every mutation. Reef-bound archives
    #: disable this and persist only from the post-commit hook.
    autosave: bool = True
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
    #: GEPA's one random stream, in its order: a parent draw, then a reshuffle
    #: of the training order at an epoch boundary. Persisted so the same seed
    #: walks the same problems as ``gepa.optimize`` would, across restarts.
    rng_seed: int = 0
    rng_state: list[Any] | None = None
    iteration: int = 0
    order: list[int] = field(default_factory=list)
    epoch: int = -1
    plans: dict[str, dict[str, Any]] = field(default_factory=dict)
    _plan_refreshed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not isinstance(self.autosave, bool):
            raise TypeError("archive autosave must be a boolean")
        if self.path.exists():
            self._load()

    # -- persistence ---------------------------------------------------------

    def _read(self) -> Mapping[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the GEPA archive at {self.path}: {error}") from error
        if not isinstance(data, Mapping):
            raise ValueError(f"the GEPA archive at {self.path} must hold a JSON object")
        return data

    def _load(self) -> None:
        self.restore(self._read())

    def restore(self, data: Mapping[str, Any]) -> None:
        """Replace the in-memory archive from a committed state snapshot."""
        if not isinstance(data, Mapping):
            raise ValueError("GEPA archive state must be a mapping")
        self._plan_refreshed = False
        self.candidates = [Candidate.from_dict(entry) for entry in data.get("candidates", ())]
        served, pending = data.get("served"), data.get("pending")
        self.served = None if served is None else int(served)
        self.pending = None if pending is None else int(pending)
        self.metric_calls = int(data.get("metric_calls", 0))
        self.steps = int(data.get("steps", 0))
        self.rejections = int(data.get("rejections", 0))
        self.proposals = [dict(row) for row in data.get("proposals", ())]
        self.rng_seed = int(data.get("rng_seed", self.rng_seed))
        self.rng_state = data.get("rng_state")
        self.iteration = int(data.get("iteration", 0))
        self.order = [int(index) for index in data.get("order", ())]
        self.epoch = int(data.get("epoch", -1))
        self.plans = {str(key): dict(value) for key, value in (data.get("plans") or {}).items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "served": self.served,
            "pending": self.pending,
            "metric_calls": self.metric_calls,
            "steps": self.steps,
            "rejections": self.rejections,
            "proposals": [dict(row) for row in self.proposals],
            "rng_seed": self.rng_seed,
            "rng_state": self.rng_state,
            "iteration": self.iteration,
            "order": list(self.order),
            "epoch": self.epoch,
            "plans": {key: dict(value) for key, value in self.plans.items()},
        }

    def refresh(self) -> None:
        """Re-read the file: the driver that plans an iteration and the method
        that runs it hold separate objects on one archive, never at once.

        A Reef-bound archive accepts only the driver's scheduling fields from
        disk. Candidate, budget, and serving state must still match the
        committed in-memory snapshot, so a stale or speculative mirror cannot
        replace Reef's canonical state.
        """
        if self.autosave:
            if self.path.exists():
                self._load()
            return
        if self._plan_refreshed:
            return
        self._plan_refreshed = True
        if not self.path.exists():
            return
        data = self._read()
        current = self.to_dict()
        planned_fields = {"rng_state", "order", "epoch", "plans"}
        if any(data.get(key) != value for key, value in current.items() if key not in planned_fields):
            raise ValueError(f"the GEPA archive plan at {self.path} does not match Reef's committed state")
        self.rng_state = data.get("rng_state")
        self.order = [int(index) for index in data.get("order", ())]
        self.epoch = int(data.get("epoch", -1))
        self.plans = {str(key): dict(value) for key, value in (data.get("plans") or {}).items()}

    def persist(self) -> None:
        """Rewrite the JSON mirror atomically from the current state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scratch = self.path.with_name(f"{self.path.name}.tmp")
        scratch.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(scratch, self.path)

    def _write(self) -> None:
        if self.autosave:
            self.persist()

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

    # -- the iteration plan ---------------------------------------------------

    def plan(self, trainset_size: int, valset_size: int, minibatch_size: int) -> dict[str, Any]:
        """The current iteration's parent and minibatch, drawn the way GEPA draws them.

        ``gepa.optimize`` feeds one generator to both its Pareto selector and
        its epoch-shuffled batch sampler, and consumes it in that order: the
        parent is chosen, then the training order is reshuffled when an epoch
        starts. Reproducing the order is what makes a seed mean the same thing
        here as there. The plan is written before anything runs, so a restart
        replays the same iteration instead of drawing a new one.
        """
        self.refresh()
        key = str(self.iteration)
        if key in self.plans:
            return dict(self.plans[key])
        rng = self._rng()
        parent = self.select_parent(rng, valset_size)
        base = self.iteration * minibatch_size
        epoch = 0 if self.epoch == -1 else base // max(len(self.order), 1)
        if not self.order or epoch > self.epoch:
            self.epoch = epoch
            self.order = _shuffled_epoch(trainset_size, minibatch_size, rng)
        start = base % len(self.order)
        plan = {"iteration": self.iteration, "parent": parent, "minibatch": self.order[start : start + minibatch_size]}
        self.plans[key] = plan
        self.rng_state = _state_to_json(rng.getstate())
        self._write()
        return dict(plan)

    def take_parent(self, valset_size: int) -> int:
        """The parent for this iteration and the step to the next one.

        A driver that planned the iteration fixed the parent already; a
        deployment feeding real traffic did not, so the parent is drawn here
        from the same generator, keeping GEPA's order of draws either way.
        """
        self.refresh()
        plan = self.plans.get(str(self.iteration))
        if plan is not None:
            parent = int(plan["parent"])
        else:
            rng = self._rng()
            parent = self.select_parent(rng, valset_size)
            self.rng_state = _state_to_json(rng.getstate())
        self.iteration += 1
        self._write()
        return parent

    def _rng(self) -> random.Random:
        rng = random.Random(self.rng_seed)
        if self.rng_state is not None:
            rng.setstate(_state_from_json(self.rng_state))
        return rng

    def select_parent(self, rng: random.Random, valset_size: int | None = None) -> int:
        """Sample a parent from the Pareto front, GEPA's selection rule.

        The sampling list is built in GEPA's order - fronts in task order,
        each front ascending, candidates in order of first appearance - so
        one draw lands on the same candidate here as there. Before any
        candidate has been validated GEPA has already scored its seed, which
        then sits on every front: the same draw over ``valset_size`` copies of
        the served candidate keeps the generator in step.
        """
        survivors = _remove_dominated(self.fronts(), [self.mean_val(index) for index in range(len(self.candidates))])
        frequency: dict[int, int] = {}
        for task in sorted(survivors):
            for index in sorted(survivors[task]):
                frequency[index] = frequency.get(index, 0) + 1
        sampling = [index for index, count in frequency.items() for _ in range(count)]
        if not sampling:
            # An empty archive is about to be seeded at index 0 by the first
            # proposal; the plan for that iteration is drawn before it exists.
            return rng.choice([self.served if self.served is not None else 0] * (valset_size or 1))
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


def _shuffled_epoch(size: int, minibatch_size: int, rng: random.Random) -> list[int]:
    """GEPA's ``EpochShuffledBatchSampler`` epoch: a shuffle, padded up to a
    whole number of minibatches by repeating the least frequent id, ties to
    the one that appeared last."""
    order = list(range(size))
    rng.shuffle(order)
    frequency = dict.fromkeys(order, 1)
    remainder = size % minibatch_size
    for _ in range(minibatch_size - remainder if remainder else 0):
        least = min(reversed(list(frequency)), key=lambda index: frequency[index])
        order.append(least)
        frequency[least] += 1
    return order


def _state_to_json(state: Any) -> list[Any]:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _state_from_json(state: Sequence[Any]) -> Any:
    version, internal, gauss = state
    return (int(version), tuple(int(value) for value in internal), gauss)


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
