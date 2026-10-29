"""The GEPA candidate pool: fronts, parent sampling, and budget.

GEPA keeps its whole search in one state object: every program
it ever proposed, that program's per-instance validation scores, the Pareto
front each instance induces, a round-robin cursor per candidate, and the
metric-call budget spent so far. The method keeps this object in its
``state/archive.json`` (``Archive.to_state``), which Reef commits with each
step, so a step that fails or is never committed cannot advance the search.

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

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """One program GEPA proposed: its texts, where it came from, how it scored.

    Values: ``texts`` maps each component key to its text. ``parent`` is
    the index of the candidate it was proposed from, ``None`` for the seed.
    ``val_scores`` and ``minibatch_scores`` are its scores, ``None`` until
    measured; ``cursor`` ≥ 0 and ``discovered_at`` ≥ 0 are described below.
    Purpose: ``Archive.candidates`` holds one per program.
    """

    texts: dict[str, str]
    parent: int | None = None
    #: One score per evaluation task, in task order. ``None`` until the next
    #: step settles the candidate's validation pass, which is what makes a
    #: candidate eligible to become a parent.
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
    """Every candidate the search has produced, and the budget it has spent.

    Values: ``served`` and ``pending`` are indexes into ``candidates`` or
    ``None``; ``pending_step`` is set exactly when ``pending`` is, and
    ``pending`` only while ``served`` is. The counts ``metric_calls``,
    ``steps``, ``rejections`` and ``iteration`` are ≥ 0, and the fields below
    say what each one holds. The constructor raises ``ValueError`` for a
    value outside these conditions.
    Purpose: ``GEPA.step`` reads it from ``state/archive.json`` with
    ``from_state``, folds in the latest evaluation with ``settle``, runs one
    iteration on it, and writes ``to_state`` back.
    """

    candidates: list[Candidate] = field(default_factory=list)
    #: The candidate whose texts the served tree holds, the tree each child is
    #: written over; ``pending`` is the candidate awaiting its validation
    #: pass, at most one, because a step settles before the next.
    served: int | None = None
    pending: int | None = None
    #: The step that proposed the pending candidate: only that step's evaluation is its validation pass.
    pending_step: int | None = None
    metric_calls: int = 0
    #: GEPA iterations that reached a child evaluation, and how many of those
    #: the minibatch acceptance test rejected.
    steps: int = 0
    rejections: int = 0
    #: One row per reflection: the parent, the component, the minibatch scores
    #: on both sides and the result. GEPA keeps the same record in its run
    #: directory; without it a search outcome can only be inferred from the
    #: candidate texts after the fact. The step's ``calls.json`` keeps what the
    #: model was shown and what it wrote, so a row leaves both out: every step
    #: commits the archive whole.
    proposals: list[dict[str, Any]] = field(default_factory=list)
    #: GEPA's one random stream, in its order: a parent draw, then a reshuffle
    #: of the training order at an epoch boundary. Committed with the state, so
    #: the same seed walks the same problems as ``gepa.optimize`` would.
    rng_seed: int = 0
    rng_state: list[Any] | None = None
    iteration: int = 0
    order: list[int] = field(default_factory=list)
    epoch: int = -1

    def __post_init__(self) -> None:
        for label, index in (("served", self.served), ("pending", self.pending)):
            if index is not None and not 0 <= index < len(self.candidates):
                raise ValueError(f"Archive.{label} must index one of the {len(self.candidates)} candidates")
        if (self.pending is None) != (self.pending_step is None):
            raise ValueError("Archive.pending_step must be set exactly when Archive.pending is")
        if self.pending is not None and self.served is None:
            raise ValueError("Archive.pending needs a served candidate")
        for label, count in (
            ("metric_calls", self.metric_calls),
            ("steps", self.steps),
            ("rejections", self.rejections),
            ("iteration", self.iteration),
        ):
            if count < 0:
                raise ValueError(f"Archive.{label} must be at least 0")

    @classmethod
    def from_state(cls, data: Mapping[str, Any] | None, *, rng_seed: int = 0) -> Archive:
        """The archive a committed ``to_state`` snapshot describes; an empty one seeded ``rng_seed`` for none."""
        if data is None:
            return cls(rng_seed=rng_seed)
        if not isinstance(data, Mapping):
            raise ValueError("GEPA archive state must be a mapping")
        served, pending = data.get("served"), data.get("pending")
        return cls(
            candidates=[Candidate.from_dict(entry) for entry in data.get("candidates", ())],
            served=None if served is None else int(served),
            pending=None if pending is None else int(pending),
            pending_step=None if data.get("pending_step") is None else int(data["pending_step"]),
            metric_calls=int(data.get("metric_calls", 0)),
            steps=int(data.get("steps", 0)),
            rejections=int(data.get("rejections", 0)),
            proposals=[dict(row) for row in data.get("proposals", ())],
            rng_seed=int(data.get("rng_seed", rng_seed)),
            rng_state=data.get("rng_state"),
            iteration=int(data.get("iteration", 0)),
            order=[int(index) for index in data.get("order", ())],
            epoch=int(data.get("epoch", -1)),
        )

    def to_state(self) -> dict[str, Any]:
        """The JSON snapshot ``from_state`` reads back as an equal archive."""
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "served": self.served,
            "pending": self.pending,
            "pending_step": self.pending_step,
            "metric_calls": self.metric_calls,
            "steps": self.steps,
            "rejections": self.rejections,
            "proposals": [dict(row) for row in self.proposals],
            "rng_seed": self.rng_seed,
            "rng_state": self.rng_state,
            "iteration": self.iteration,
            "order": list(self.order),
            "epoch": self.epoch,
        }

    # -- growth --------------------------------------------------------------

    def seed(self, texts: Mapping[str, str]) -> None:
        """Append ``texts`` as a parentless candidate and serve it.

        Used for the first boot and for a new training start: when the served tree no
        longer matches the archive's served candidate, whatever the operator
        put there is the new root of the search, not a descendant of it.
        """
        self.candidates.append(Candidate(texts=dict(texts), discovered_at=self.metric_calls))
        self.served = len(self.candidates) - 1
        self.clear_pending()

    def add(self, texts: Mapping[str, str], parent: int, minibatch_scores: Sequence[float], step: int) -> int:
        """Append an accepted child as the pending candidate of ``step`` and return its index.

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
        self.pending_step = step
        self.steps += 1
        return self.pending

    def reject(self) -> None:
        """Record a GEPA iteration whose child lost the minibatch comparison."""
        self.steps += 1
        self.rejections += 1

    def log_proposal(self, row: Mapping[str, Any]) -> None:
        """Keep one reflection's record: parent, component, minibatch scores,
        child scores, and whether the child was accepted."""
        self.proposals.append({"step": self.steps, **row})

    def record_validation(self, index: int, scores: Sequence[float]) -> None:
        """Keep ``scores``, one per evaluation task, as the validation scores of candidate ``index``."""
        self.candidates[index].val_scores = [float(score) for score in scores]

    def settle(
        self, candidate_scores: Sequence[float], served_scores: Sequence[float], selected: bool, *, repeats: int
    ) -> None:
        """Fold one evaluated step into the archive: GEPA's validation pass and the update of the served candidate.

        Each score is one task's mean over the ``repeats`` rollouts the
        evaluation ran for it. The pending candidate takes the step's
        candidate scores and is charged one metric call per rollout,
        ``len(candidate_scores) * repeats``; a served candidate that has no
        validation yet (the seed, on the first evaluated step) takes the
        served scores, charged when it was seeded. A selected step serves the
        pending candidate.
        """
        pending = self.pending
        if pending is not None:
            self.record_validation(pending, candidate_scores)
            self.charge(len(candidate_scores) * repeats)
        if self.served is not None and self.candidates[self.served].val_scores is None:
            self.record_validation(self.served, served_scores)
        if selected and pending is not None:
            self.served = pending
        self.clear_pending()

    def clear_pending(self) -> None:
        self.pending = None
        self.pending_step = None

    def charge(self, calls: int) -> None:
        """Spend ``calls`` of the metric-call budget: one per scored rollout."""
        self.metric_calls += calls

    # -- GEPA's two rules ----------------------------------------------------

    def fronts(self) -> dict[int, set[int]]:
        """Task index to the candidates holding that task's best score."""
        scored = [
            (index, candidate.val_scores) for index, candidate in enumerate(self.candidates) if candidate.val_scores
        ]
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
        """The mean validation score of candidate ``index``; 0.0 before it has one."""
        scores = self.candidates[index].val_scores
        return sum(scores) / len(scores) if scores else 0.0

    # -- the iteration plan ---------------------------------------------------

    def plan(self, trainset_size: int, valset_size: int, minibatch_size: int) -> dict[str, Any]:
        """This iteration's parent and minibatch, drawn the way GEPA draws them, and the step to the next iteration.

        ``gepa.optimize`` feeds one generator to both its Pareto selector and
        its epoch-shuffled batch sampler, and consumes it in that order: the
        parent is chosen, then the training order is reshuffled when an epoch
        starts. Reproducing the order is what makes a seed mean the same thing
        here as there. The draw is a function of the committed state, so a
        driver that reads the same state draws the same plan.
        """
        rng = self.generator()
        parent = self.select_parent(rng, valset_size)
        base = self.iteration * minibatch_size
        epoch = 0 if self.epoch == -1 else base // max(len(self.order), 1)
        if not self.order or epoch > self.epoch:
            self.epoch = epoch
            self.order = shuffled_epoch(trainset_size, minibatch_size, rng)
        start = base % len(self.order)
        plan = {"iteration": self.iteration, "parent": parent, "minibatch": self.order[start : start + minibatch_size]}
        self.rng_state = rng_state_to_json(rng.getstate())
        self.iteration += 1
        return plan

    def take_parent(self, valset_size: int) -> int:
        """The parent for this iteration, drawn from GEPA's generator, and the step to the next iteration.

        A deployment fed real traffic has no training order to plan, so only
        the parent is drawn.
        """
        rng = self.generator()
        parent = self.select_parent(rng, valset_size)
        self.rng_state = rng_state_to_json(rng.getstate())
        self.iteration += 1
        return parent

    def generator(self) -> random.Random:
        """GEPA's random stream where the committed state left it; a new one seeded ``rng_seed`` before any draw."""
        rng = random.Random(self.rng_seed)
        if self.rng_state is not None:
            rng.setstate(rng_state_from_json(self.rng_state))
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
        survivors = remove_dominated(self.fronts(), [self.mean_val(index) for index in range(len(self.candidates))])
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
        return keys[position]


def shuffled_epoch(size: int, minibatch_size: int, rng: random.Random) -> list[int]:
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


def rng_state_to_json(state: Any) -> list[Any]:
    """A ``random.Random`` state as JSON: its version, its internal state as a list, and its Gaussian cache."""
    version, internal, gauss = state
    return [version, list(internal), gauss]


def rng_state_from_json(state: Sequence[Any]) -> Any:
    """The ``random.Random`` state that ``rng_state_to_json`` wrote."""
    version, internal, gauss = state
    return (int(version), tuple(int(value) for value in internal), gauss)


def is_dominated(candidate: int, others: set[int], fronts: Mapping[int, set[int]]) -> bool:
    """Whether every front holding ``candidate`` also holds one of ``others``."""
    return all(any(other in front for other in others) for front in fronts.values() if candidate in front)


def remove_dominated(fronts: Mapping[int, set[int]], scores: Sequence[float]) -> dict[int, set[int]]:
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
            if is_dominated(candidate, set(programs) - {candidate} - dominated, fronts):
                dominated.add(candidate)
                removing = True
                break
    return {task: {index for index in front if index not in dominated} for task, front in fronts.items()}
