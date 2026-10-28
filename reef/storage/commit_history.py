"""Commit history contract and the scenario store that combines it with a record store.

CommitHistory is the contract for the ordered commits of one scenario
generation. ``sql_commits`` implements it as a table in the scenario's record
database. CommitHistoryScenarioStore combines a CommitHistory with an injected
RecordStore. The history append is the durable commit point; recovery restores
consumption progress. Artifact publication is not in this module.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from threading import RLock

from reef.core.errors import ReefError
from reef.storage.commits import RECORD_KIND, CommitHistoryError, CommitRecord
from reef.storage.evaluations import EvaluationStore
from reef.storage.records import RecordStore
from reef.storage.scenario import ScenarioStore, ScenarioStoreConflict
from reef.storage.tasks import TaskStorage
from reef.storage.trajectories import TrajectoryStore


def same_commit(left: CommitRecord, right: CommitRecord) -> bool:
    """Whether two commits are identical apart from ``recorded_at``, the identical-retry rule."""
    return {key: value for key, value in left.to_dict().items() if key != "recorded_at"} == {
        key: value for key, value in right.to_dict().items() if key != "recorded_at"
    }


class CommitHistory(ABC):
    """The ordered commits of one scenario generation.

    Values: ``records()`` is a tuple of ``CommitRecord`` whose steps strictly
    increase, so each step occurs at most once. Steps need not be consecutive;
    ``CommitHistoryScenarioStore.recover`` rejects a gap after a checkpoint.
    The history does not check which scenario a record names; the scenario
    store does. ``label`` names where the history is kept, and ``durable`` is
    true when the records survive a process restart.

    Purpose: a storage adapter makes it, and ``CommitHistoryScenarioStore``
    reads and appends through it. ``append`` is the commit point.

    Operations, for a history whose records are ``H`` before the call:

    - ``append(r)``: if ``e`` in ``H`` has ``e.step == r.step``, the call does
      nothing when ``same_commit(e, r)`` and raises ``CommitHistoryError``
      otherwise. If ``r.step`` is below the last step of ``H``, it raises
      ``CommitHistoryError``. Otherwise ``records()`` is ``(*H, r)`` afterwards.
    - ``locked()``: no other session or process appends while the block runs.
    - ``training_run_position()`` is ``(s, n)``: ``s`` is the step of the last
      rollback in ``H``, or 0 when there is none, and ``n`` is the number of
      training records in ``H`` with a step above ``s``.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """Where the history is kept, for error messages."""

    @property
    @abstractmethod
    def durable(self) -> bool:
        """Whether the commits survive a process restart."""

    @abstractmethod
    def records(self) -> tuple[CommitRecord, ...]:
        """Every commit in increasing step order."""

    @abstractmethod
    def append(self, record: CommitRecord) -> None:
        """Persist one commit before returning."""

    @abstractmethod
    def locked(self) -> AbstractContextManager[None]:
        """Serialize a read-check-append sequence with other sessions and processes."""

    def training_run_position(self) -> tuple[int, int]:
        """Return the latest rollback step and the training steps after it."""
        run_segment = 0
        run_step = 0
        for record in self.records():
            if record.operation == "rollback":
                run_segment = record.step
                run_step = 0
            elif record.operation == "training" and record.step > run_segment:
                run_step += 1
        return run_segment, run_step


class CommitHistoryScenarioStore(ScenarioStore):
    """One open session of a scenario: its records, task stores, uploaded trajectories, evaluations and commits.

    Values: ``scenario`` is a non-empty name. ``records``, ``tasks``,
    ``trajectories``, ``evaluations`` and ``commit_history`` hold that
    scenario's state; the bundled storages keep all five in one database. Every record in the
    history names ``scenario``; a read that finds another name raises
    ``ReefError``. ``initial_step`` is a non-negative step for an existing
    scenario constructed at a saved step. The current step is the last step in
    the history, or ``initial_step`` when the history is empty. ``durable`` is
    ``commit_history.durable``. After ``close``, every operation raises
    ``RuntimeError``.

    Purpose: a ``ScenarioStorage`` makes it in ``open``. The scenario and its
    trainers read and commit through it, under the ``ScenarioStore`` contract.
    """

    def __init__(
        self,
        scenario: str,
        records: RecordStore,
        commit_history: CommitHistory,
        *,
        tasks: TaskStorage,
        trajectories: TrajectoryStore,
        evaluations: EvaluationStore,
        initial_step: int = 0,
    ) -> None:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        if not isinstance(initial_step, int) or isinstance(initial_step, bool) or initial_step < 0:
            raise ValueError("initial_step must be a non-negative integer")
        self._scenario = scenario
        self._records = records
        self._tasks = tasks
        self._trajectories = trajectories
        self._evaluations = evaluations
        self._commit_history = commit_history
        self._initial_step = initial_step
        self._cached_history: tuple[CommitRecord, ...] = ()
        self._commits_by_step: dict[int, CommitRecord] = {}
        self._lock = RLock()
        self._closed = False

    @property
    def records(self) -> RecordStore:
        with self._lock:
            self._ensure_open()
            return self._records

    @property
    def durable(self) -> bool:
        return self._commit_history.durable

    @property
    def tasks(self) -> TaskStorage:
        with self._lock:
            self._ensure_open()
            return self._tasks

    @property
    def trajectories(self) -> TrajectoryStore:
        with self._lock:
            self._ensure_open()
            return self._trajectories

    @property
    def evaluations(self) -> EvaluationStore:
        with self._lock:
            self._ensure_open()
            return self._evaluations

    @property
    def commit_history(self) -> CommitHistory:
        """The commit history, for callers of the concrete storage adapters."""
        with self._lock:
            self._ensure_open()
            return self._commit_history

    def history(self) -> tuple[CommitRecord, ...]:
        # Commits are append-only, so a read needs no lock against other writers.
        with self._lock:
            self._ensure_open()
            return self._history()

    def training_run_position(self) -> tuple[int, int]:
        with self._lock:
            self._ensure_open()
            return self._commit_history.training_run_position()

    def commit_step(self, *, expected_step: int, commit: CommitRecord) -> CommitRecord:
        with self._locked():
            if not isinstance(expected_step, int) or isinstance(expected_step, bool) or expected_step < 0:
                raise ValueError("expected_step must be a non-negative integer")
            if commit.scenario != self._scenario:
                raise ScenarioStoreConflict(f"commit belongs to scenario {commit.scenario!r}, not {self._scenario!r}")
            if commit.step != expected_step + 1:
                raise ScenarioStoreConflict(f"commit step must be {expected_step + 1}, got {commit.step}")
            history = self._history()
            existing = self._commits_by_step.get(commit.step)
            if existing is not None:
                if not same_commit(existing, commit):
                    raise ScenarioStoreConflict(f"commit step {commit.step} conflicts with its existing record")
                return existing
            current_step = history[-1].step if history else self._initial_step
            if expected_step != current_step:
                raise ScenarioStoreConflict(f"expected scenario step {expected_step}, current step is {current_step}")
            self._commit_history.append(commit)
            return commit

    def recover(self, *, checkpoint: CommitRecord | None) -> CommitRecord | None:
        with self._locked():
            if checkpoint is not None:
                if checkpoint.scenario != self._scenario:
                    raise ReefError(f"checkpoint belongs to scenario {checkpoint.scenario!r}, not {self._scenario!r}")
                if not checkpoint.checkpoint or checkpoint.pending:
                    raise ReefError("recovery requires an activated checkpoint commit")
            checkpoint_step = 0 if checkpoint is None else checkpoint.step
            records = self._history()
            history_label = self._commit_history.label
            applicable = [record for record in records if record.step > checkpoint_step]
            for previous, current in itertools.pairwise(applicable):
                if current.step != previous.step + 1:
                    raise ReefError(
                        f"commit history {history_label} jumps from step {previous.step} to {current.step}; "
                        "the history is corrupt"
                    )
            if applicable and applicable[0].step != checkpoint_step + 1:
                raise ReefError(
                    f"commit history {history_label} resumes at step {applicable[0].step}, "
                    f"expected {checkpoint_step + 1} after the checkpointed step "
                    f"{checkpoint_step}; the history is corrupt"
                )
            last_step = records[-1].step if records else 0
            if checkpoint is not None and checkpoint_step > last_step:
                head = checkpoint
                self._commit_history.append(head)
            elif applicable:
                head = applicable[-1]
            elif records and last_step == checkpoint_step:
                head = records[-1]
            else:
                head = None
            self._initial_step = checkpoint_step if head is None else head.step
            return head

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._records.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario store is closed")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._lock:
            self._ensure_open()
            with self._commit_history.locked():
                yield

    def _history(self) -> tuple[CommitRecord, ...]:
        records = self._commit_history.records()
        if records is self._cached_history:
            return records
        previous = self._cached_history
        appended = len(previous) <= len(records) and (not previous or records[len(previous) - 1] is previous[-1])
        if not appended:
            self._commits_by_step.clear()
        for record in records[len(previous) if appended else 0 :]:
            if record.scenario != self._scenario:
                raise ReefError(
                    f"commit history {self._commit_history.label} holds records for {record.scenario!r}, "
                    f"not {self._scenario!r}"
                )
            self._commits_by_step[record.step] = record
        self._cached_history = records
        return records


__all__ = [
    "RECORD_KIND",
    "CommitHistory",
    "CommitHistoryError",
    "CommitHistoryScenarioStore",
    "CommitRecord",
    "same_commit",
]
