"""The history of one scenario's harness evolution: every tree it evaluated, every rollout, every decision.

The history is one directory::

    candidates.jsonl   one CandidateRecord per tree, the first time a step proposes or evaluates it; a second
                       line adds the method's summary and notes when a step proposes a tree it scored first
    rollouts.jsonl     one RolloutRecord per rollout that ran
    decisions.jsonl    one DecisionRecord per step whose candidate the evaluation decided
    trees/<tree>/      the files of each tree, written once
    rollouts/<id>/     the record of each rollout: its trajectory files and rollout.json

The rollouts are the score cache. A rollout of the same tree, task, repeat
and evaluation context runs again only when the earlier one failed (the
environment could not run it), or when a recheck asks for fresh rollouts; the
latest one that did not fail is the cached one.
Methods read the history. Only the harness backend writes it, from the
training thread. The history keeps what happened, also in a step that did not
commit (its method, its evaluation or its commit failed): such a step runs
again under the same number, so a record of the current step or a later one
comes from an attempt that did not commit, and a method that counts what it
tried skips it.
Each JSONL file is append only. A crash can leave a torn last line: a reader
skips it, and ``History.open``, which the backend calls before it writes, cuts
it off, so the next append starts a line of its own.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from reef.core.evaluation import RolloutScore
from reef.evaluation.harness.scoring import FailureObservation, ScoredRollout
from reef.harness.tree import HarnessTree

CANDIDATES_FILE = "candidates.jsonl"
ROLLOUTS_FILE = "rollouts.jsonl"
DECISIONS_FILE = "decisions.jsonl"
TREES_DIR = "trees"
ROLLOUTS_DIR = "rollouts"
#: Where a candidate came from: the seed tree, a method's step, a client's proposal, or an earlier release.
ORIGINS = ("seed", "method", "proposal", "release")


@dataclass(frozen=True)
class CandidateRecord:
    """One tree of the history.

    Values: ``tree`` is the tree's checksum (``HarnessTree.checksum``).
    ``step`` ≥ 1 is the step that first proposed or evaluated it. ``origin``
    is one of ``ORIGINS``. ``parent`` is the checksum of the head tree it was
    proposed against; it is ``None`` exactly when ``origin`` is ``seed`` or
    ``release``. ``summary`` and ``notes`` describe the tree when a step
    proposed it: what the method said, or for a ``proposal`` the client's
    reason and no notes. Both are empty for a tree that no step proposed. The
    latest line of a tree in ``candidates.jsonl`` is its record.
    Purpose: the history makes one per tree; a method reads them to choose a
    parent or to show what was tried.
    """

    tree: str
    step: int
    origin: str
    parent: str | None = None
    summary: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tree:
            raise ValueError("CandidateRecord.tree must be a tree checksum")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("CandidateRecord.step must be a positive integer")
        if self.origin not in ORIGINS:
            raise ValueError(f"CandidateRecord.origin must be one of {', '.join(ORIGINS)}")
        if (self.parent is None) != (self.origin in ("seed", "release")):
            raise ValueError("CandidateRecord.parent is None exactly for a seed or a release")
        object.__setattr__(self, "notes", MappingProxyType(dict(self.notes)))

    def to_json(self) -> dict[str, Any]:
        """The JSONL row of this record; ``CandidateRecord(**row)`` reads it back as an equal record."""
        return {
            "tree": self.tree,
            "step": self.step,
            "origin": self.origin,
            "parent": self.parent,
            "summary": self.summary,
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class RolloutRecord:
    """One rollout that ran, and its scored result.

    Values: ``rollout`` names its record directory under ``rollouts/``.
    ``tree`` is the checksum of the tree it ran. ``task`` is the task as the
    evaluation names it (a prompt or a task directory) and ``task_key``
    identifies its content. ``repeat`` ≥ 0 is the sample index of the
    (tree, task) pair. ``context`` is the key of the evaluation context
    (``EvaluationContext``: the served and the named models, the adapter, the
    binary, the rollout executor, the scorer and the rollout settings).
    ``step`` ≥ 1 is the step that ran it. ``score`` is its ``RolloutScore``:
    failed when the environment could not run it, and then ``failure`` says
    why; a rollout that ran may also carry a ``failure`` (no answer, or a
    nonzero exit). The other
    fields are those of ``ScoredRollout``: ``reply``, ``residue`` ≥ 0,
    ``agents``, ``path`` and ``transcript_read``.
    Purpose: the evaluator makes one per rollout; the cache reuses a scored
    one, the evaluation's metrics are built from them, and methods read them.
    """

    rollout: str
    tree: str
    task: str
    task_key: str
    repeat: int
    context: str
    step: int
    score: RolloutScore
    failure: FailureObservation | None = None
    reply: str | None = None
    residue: int = 0
    agents: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    path: Mapping[str, Any] | None = None
    transcript_read: bool = True

    def __post_init__(self) -> None:
        if not (self.rollout and self.tree and self.task_key and self.context):
            raise ValueError("RolloutRecord needs a rollout id, a tree, a task key and a context")
        for label, value, least in (("repeat", self.repeat, 0), ("step", self.step, 1), ("residue", self.residue, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"RolloutRecord.{label} must be an integer of at least {least}")
        if not isinstance(self.score, RolloutScore):
            raise TypeError("RolloutRecord.score must be a RolloutScore")
        if self.score.is_failed and self.failure is None:
            raise ValueError("a failed RolloutRecord must say why it failed")

    @classmethod
    def from_scored(
        cls,
        scored: ScoredRollout,
        *,
        rollout: str,
        tree: str,
        task: str,
        task_key: str,
        repeat: int,
        context: str,
        step: int,
    ) -> RolloutRecord:
        """The record of one scored rollout of ``tree``."""
        return cls(
            rollout=rollout,
            tree=tree,
            task=task,
            task_key=task_key,
            repeat=repeat,
            context=context,
            step=step,
            score=scored.score,
            failure=scored.failure,
            reply=scored.reply,
            residue=scored.residue,
            agents=scored.agents,
            path=scored.path,
            transcript_read=scored.transcript_read,
        )

    def to_json(self) -> dict[str, Any]:
        """The JSONL row of this record; ``from_json`` reads it back as an equal record."""
        row = asdict(self)
        row["score"] = self.score.to_json()
        row["agents"] = {name: dict(work) for name, work in self.agents.items()}
        row["path"] = None if self.path is None else dict(self.path)
        return row

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> RolloutRecord:
        failure = row.get("failure")
        return cls(
            **{
                **row,
                "score": RolloutScore.from_json(row["score"]),
                "failure": None if failure is None else FailureObservation.from_dict(failure),
            }
        )


@dataclass(frozen=True)
class DecisionRecord:
    """The evaluation's decision on one step's candidate.

    Values: ``step`` ≥ 1 is the step; ``tree`` is the candidate's checksum
    and ``head`` the checksum of the head tree the step started from, which
    a comparison measured it against; ``selected`` says whether the step
    kept it, as a new artifact that the activation rule may serve;
    ``policy`` names the publication policy and ``reason`` explains it.
    Purpose: the backend writes one when it settles a step; a method reads
    them to learn what happened to its earlier candidates.
    """

    step: int
    tree: str
    head: str
    selected: bool
    policy: str
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 1:
            raise ValueError("DecisionRecord.step must be a positive integer")
        if not (self.tree and self.head and self.policy and self.reason):
            raise ValueError("DecisionRecord needs a tree, a head, a policy and a reason")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The JSON objects of ``path``, one per line; a missing file has none, and a torn last line is skipped."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                continue
            raise ValueError(f"{path} line {number + 1} is not JSON") from None
    return rows


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object as a line of ``path`` and flush it to disk."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cut_torn_line(path: Path) -> None:
    """Cut ``path`` after its last newline, which removes a torn last line; a missing file stays missing."""
    if not path.is_file():
        return
    with path.open("r+b") as handle:
        content = handle.read()
        if not content or content.endswith(b"\n"):
            return
        handle.truncate(content.rfind(b"\n") + 1)
        handle.flush()
        os.fsync(handle.fileno())


class History:
    """One scenario's history directory, loaded into memory and appended to.

    Values: ``root`` is the directory. ``candidates`` maps each tree's
    checksum to its record; ``rollouts`` and ``decisions`` hold the records
    of their JSONL files in file order; ``scored`` maps (tree, task key,
    repeat, context) to the latest rollout of that key that did not fail, so a recheck's
    fresh rollout replaces the one it measured again. ``next_rollout``
    is above the number of every rollout directory and record, so a new
    rollout never reuses the directory of one that a crash left behind.
    Purpose: the harness backend opens one per scenario and writes it during
    each step; the evaluator reads and fills the score cache through it; a
    method opens one on ``Workspace.history_dir`` to read it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.candidates: dict[str, CandidateRecord] = {
            record.tree: record
            for record in (CandidateRecord(**row) for row in read_jsonl(self.root / CANDIDATES_FILE))
        }
        self.rollouts: list[RolloutRecord] = [
            RolloutRecord.from_json(row) for row in read_jsonl(self.root / ROLLOUTS_FILE)
        ]
        self.decisions: list[DecisionRecord] = [
            DecisionRecord(**row) for row in read_jsonl(self.root / DECISIONS_FILE)
        ]
        self.scored: dict[tuple[str, str, int, str], RolloutRecord] = {}
        for record in self.rollouts:
            self.cache(record)
        numbers = [int(record.rollout) for record in self.rollouts]
        rollouts_dir = self.root / ROLLOUTS_DIR
        if rollouts_dir.is_dir():
            numbers.extend(int(path.name) for path in rollouts_dir.iterdir() if path.name.isdigit())
        self.next_rollout = max(numbers, default=0) + 1

    @classmethod
    def open(cls, root: Path) -> Self:
        """The history under ``root`` for the writer, created empty when it does not exist.

        Each JSONL file loses a torn last line first, so the writer's next append starts a line of its own.
        """
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for name in (CANDIDATES_FILE, ROLLOUTS_FILE, DECISIONS_FILE):
            cut_torn_line(root / name)
        return cls(root)

    # -- reading ---------------------------------------------------------------------------------------------------

    def tree(self, checksum: str) -> HarnessTree:
        """The files of a tree the history holds; ``KeyError`` for an unknown tree."""
        directory = self.root / TREES_DIR / checksum
        if checksum not in self.candidates or not directory.is_dir():
            raise KeyError(f"the history holds no tree {checksum}")
        return HarnessTree.read(directory)

    def rollouts_of(self, checksum: str) -> tuple[RolloutRecord, ...]:
        """The rollouts of one tree, in the order they ran."""
        return tuple(record for record in self.rollouts if record.tree == checksum)

    def scores(self, checksum: str, tasks: Sequence[str], *, repeats: int = 1) -> tuple[float | None, ...]:
        """The score values of one tree on ``tasks``, task by task and repeat by repeat, as a method reads them.

        Each (task, repeat) takes the value of its latest rollout in any context; ``None`` where no rollout ran
        or the latest one failed.
        """
        latest = {
            (record.task, record.repeat): record.score.value for record in self.rollouts if record.tree == checksum
        }
        return tuple(latest.get((task, repeat)) for task in tasks for repeat in range(repeats))

    def decision(self, step: int) -> DecisionRecord | None:
        """The decision on the candidate of ``step``; ``None`` when the step decided no candidate."""
        return next((record for record in reversed(self.decisions) if record.step == step), None)

    def rollout_dir(self, rollout: str) -> Path:
        """The record directory of one rollout."""
        return self.root / ROLLOUTS_DIR / rollout

    def cached(self, checksum: str, task_key: str, repeat: int, context: str) -> RolloutRecord | None:
        """The latest rollout of this tree, task, repeat and context that did not fail; ``None`` when there is none."""
        return self.scored.get((checksum, task_key, repeat, context))

    # -- writing ---------------------------------------------------------------------------------------------------

    def add_candidate(
        self,
        tree: HarnessTree,
        *,
        step: int,
        origin: str,
        parent: str | None = None,
        summary: str = "",
        notes: Mapping[str, Any] | None = None,
    ) -> CandidateRecord:
        """The record of ``tree``: the existing one, or a new one written with the tree's files.

        An existing record without a summary or notes takes the ``summary`` and ``notes`` given here, as a new
        line, so a step that proposes a tree its method scored earlier keeps what the method said about it.
        """
        checksum = tree.checksum()
        existing = self.candidates.get(checksum)
        if existing is not None:
            if (summary or notes) and not existing.summary and not existing.notes:
                # A tree the method scored before its step proposed it: the step adds what the method said.
                described = replace(existing, summary=summary, notes=notes or {})
                append_jsonl(self.root / CANDIDATES_FILE, described.to_json())
                self.candidates[checksum] = described
                return described
            return existing
        record = CandidateRecord(checksum, step, origin, parent, summary, notes or {})
        directory = self.root / TREES_DIR / checksum
        if not directory.is_dir():
            # A crash between the files and the record leaves a complete directory; it is written once.
            staging = self.root / TREES_DIR / f".{checksum}.part"
            shutil.rmtree(staging, ignore_errors=True)
            # The empty tree writes no file, so the directory is made first.
            staging.mkdir(parents=True)
            tree.write(staging)
            staging.rename(directory)
        append_jsonl(self.root / CANDIDATES_FILE, record.to_json())
        self.candidates[checksum] = record
        return record

    def new_rollout(self) -> tuple[str, Path]:
        """A fresh rollout id and its record directory, which does not exist yet."""
        rollout = f"{self.next_rollout:08d}"
        self.next_rollout += 1
        return rollout, self.rollout_dir(rollout)

    def add_rollout(self, record: RolloutRecord) -> None:
        """Append one rollout; one that did not fail joins the cache."""
        if record.tree not in self.candidates:
            raise ValueError(f"rollout {record.rollout} ran a tree the history does not hold")
        append_jsonl(self.root / ROLLOUTS_FILE, record.to_json())
        self.rollouts.append(record)
        self.cache(record)

    def add_decision(self, record: DecisionRecord) -> None:
        """Append the decision on one step's candidate."""
        append_jsonl(self.root / DECISIONS_FILE, asdict(record))
        self.decisions.append(record)

    def cache(self, record: RolloutRecord) -> None:
        """Make a rollout that did not fail the cache entry of its tree, task, repeat and context; the latest one
        wins."""
        if not record.score.is_failed:
            self.scored[(record.tree, record.task_key, record.repeat, record.context)] = record


__all__ = [
    "CANDIDATES_FILE",
    "DECISIONS_FILE",
    "ORIGINS",
    "ROLLOUTS_DIR",
    "ROLLOUTS_FILE",
    "TREES_DIR",
    "CandidateRecord",
    "DecisionRecord",
    "History",
    "RolloutRecord",
]
