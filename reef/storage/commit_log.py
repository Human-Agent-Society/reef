"""JSONL commit storage and its scenario store implementation.

CommitLog owns append, fsync, idempotent retries, torn-tail repair, and caching.
CommitLogScenarioStore combines it with an injected RecordStore. The log
append is the durable commit point; interrupted record compaction is repaired
on retry or recovery. Stable POSIX file locks serialize commit writers across
sessions and processes. Artifact publication remains outside this module.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

from reef.core.errors import ReefError
from reef.scenario.commits import RECORD_KIND, CommitLogError, CommitRecord
from reef.scenario.store import ScenarioStore, ScenarioStoreConflict
from reef.storage.records import RecordStore


class CommitLog:
    """Append-only JSONL commit log for one scenario.

    Appends serialize the record to a single line and fsync before returning;
    that is the commit point. Retrying the same scenario step is a no-op, while
    different content for an existing step is a conflict. Reads tolerate
    exactly one torn tail (a crash mid-append) and reject corruption elsewhere.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._cached_records: list[CommitRecord] | None = None
        self._cached_steps: dict[int, CommitRecord] | None = None
        self._cached_snapshot: tuple[CommitRecord, ...] | None = None
        self._cached_signature: tuple[int, int, int, int] | None = None
        self._run_segment = 0
        self._run_step = 0

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: CommitRecord) -> None:
        encoded = (
            json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        with self._lock:
            self._truncate_partial_tail()
            previous_signature = self._file_signature()
            cached_records = self._cached_records
            records: Sequence[CommitRecord]
            if cached_records is None or previous_signature != self._cached_signature:
                records = self._reload_cache(previous_signature)
            else:
                records = cached_records
            cached_steps = self._cached_steps
            existing = (
                cached_steps.get(record.step)
                if cached_steps is not None
                else next((item for item in records if item.step == record.step), None)
            )
            if existing is not None:
                if self._same_commit(existing, record):
                    return
                raise CommitLogError(f"commit step {record.step} conflicts with its existing record")
            if records and record.step < records[-1].step:
                raise CommitLogError(f"commit step {record.step} precedes existing step {records[-1].step}")
            with open(self._path, "ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            current_signature = self._file_signature()
            cached_records = self._cached_records
            if (
                cached_records is not None
                and previous_signature == self._cached_signature
                and self._is_exclusive_append(previous_signature, current_signature, len(encoded))
            ):
                cached_records.append(record)
                if cached_steps is not None:
                    cached_steps[record.step] = record
                self._cached_snapshot = None
                self._advance_run_position(record)
                self._cached_signature = current_signature
            else:
                self._invalidate_cache()

    @staticmethod
    def _same_commit(left: CommitRecord, right: CommitRecord) -> bool:
        return {key: value for key, value in left.to_dict().items() if key != "recorded_at"} == {
            key: value for key, value in right.to_dict().items() if key != "recorded_at"
        }

    def records(self) -> tuple[CommitRecord, ...]:
        with self._lock:
            signature = self._file_signature()
            cached_records = self._cached_records
            if cached_records is not None and signature == self._cached_signature:
                if self._cached_snapshot is None:
                    self._cached_snapshot = tuple(cached_records)
                return self._cached_snapshot
            return self._reload_cache(signature)

    def training_run_position(self) -> tuple[int, int]:
        """Return the latest rollback step and training steps after it."""
        with self._lock:
            signature = self._file_signature()
            if self._cached_records is not None and signature == self._cached_signature:
                return self._run_segment, self._run_step
            records = self._reload_cache(signature)
            if self._cached_records is not None and signature == self._cached_signature:
                return self._run_segment, self._run_step
            return self._position(records)

    def _file_signature(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return None
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _truncate_partial_tail(self) -> None:
        try:
            with open(self._path, "r+b") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    return
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) == b"\n":
                    return
                handle.seek(0)
                content = handle.read()
                handle.truncate(content.rfind(b"\n") + 1)
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError:
            return

    def _cache(self, records: tuple[CommitRecord, ...], signature: tuple[int, int, int, int] | None) -> None:
        self._cached_records = list(records)
        self._cached_steps = {record.step: record for record in records}
        self._cached_snapshot = records
        self._cached_signature = signature
        self._run_segment, self._run_step = self._position(records)

    @staticmethod
    def _is_exclusive_append(
        previous: tuple[int, int, int, int] | None,
        current: tuple[int, int, int, int] | None,
        appended_bytes: int,
    ) -> bool:
        if current is None:
            return False
        if previous is None:
            return current[2] == appended_bytes
        return current[:2] == previous[:2] and current[2] == previous[2] + appended_bytes

    def _reload_cache(self, signature: tuple[int, int, int, int] | None) -> tuple[CommitRecord, ...]:
        self._invalidate_cache()
        records, complete = self._read_records()
        if complete and signature == self._file_signature():
            self._cache(records, signature)
        return records

    def _invalidate_cache(self) -> None:
        self._cached_records = None
        self._cached_steps = None
        self._cached_snapshot = None
        self._cached_signature = None

    @staticmethod
    def _position(records: tuple[CommitRecord, ...]) -> tuple[int, int]:
        run_segment = 0
        run_step = 0
        for record in records:
            if record.operation == "rollback":
                run_segment = record.step
                run_step = 0
            elif record.operation == "training" and record.step > run_segment:
                run_step += 1
        return run_segment, run_step

    def _advance_run_position(self, record: CommitRecord) -> None:
        if record.operation == "rollback":
            self._run_segment = record.step
            self._run_step = 0
        elif record.operation == "training" and record.step > self._run_segment:
            self._run_step += 1

    def _read_records(self) -> tuple[tuple[CommitRecord, ...], bool]:
        if not self._path.exists():
            return (), True
        records: list[CommitRecord] = []
        with open(self._path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if index == len(lines) - 1:
                    return tuple(records), False  # torn tail from a crash mid-append
                raise CommitLogError(f"commit log {self._path} has a corrupt record at line {index + 1}") from exc
            if not isinstance(value, dict):
                raise CommitLogError(f"commit log {self._path} line {index + 1} is not a record object")
            try:
                records.append(CommitRecord.from_dict(value))
            except CommitLogError as exc:
                raise CommitLogError(f"commit log {self._path} line {index + 1}: {exc}") from exc
        return tuple(records), True


class CommitLogScenarioStore(ScenarioStore):
    """Own one records connection and an optional durable commit log.

    Passing no commit log preserves the existing in-memory configuration. Its
    commits still enforce step fencing and retries for the session lifetime.
    ``initial_step`` supports existing scenarios constructed at a saved step.
    """

    def __init__(
        self,
        scenario: str,
        records: RecordStore,
        commit_log: CommitLog | None = None,
        *,
        initial_step: int = 0,
    ) -> None:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        if not isinstance(initial_step, int) or isinstance(initial_step, bool) or initial_step < 0:
            raise ValueError("initial_step must be a non-negative integer")
        self._scenario = scenario
        self._records = records
        self._commit_log = commit_log
        self._initial_step = initial_step
        self._memory_history: list[CommitRecord] = []
        self._memory_run_segment = 0
        self._memory_run_step = 0
        self._cached_history: tuple[CommitRecord, ...] = ()
        self._commits_by_step: dict[int, CommitRecord] = {}
        self._lock = RLock()
        self._closed = False
        self._lock_path = (
            None if commit_log is None else commit_log.path.parent / ".locks" / f"{commit_log.path.name}.lock"
        )

    @property
    def records(self) -> RecordStore:
        with self._lock:
            self._ensure_open()
            return self._records

    @property
    def durable(self) -> bool:
        return self._commit_log is not None

    @property
    def commit_log(self) -> CommitLog | None:
        """Commit log access for callers of the concrete file adapter."""
        with self._lock:
            self._ensure_open()
            return self._commit_log

    def history(self) -> tuple[CommitRecord, ...]:
        with self._locked():
            return self._history()

    def training_run_position(self) -> tuple[int, int]:
        with self._locked():
            if self._commit_log is not None:
                return self._commit_log.training_run_position()
            return self._memory_run_segment, self._memory_run_step

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
                if not self._same_commit(existing, commit):
                    raise ScenarioStoreConflict(f"commit step {commit.step} conflicts with its existing record")
                self._compact(existing)
                return existing
            current_step = history[-1].step if history else self._initial_step
            if expected_step != current_step:
                raise ScenarioStoreConflict(f"expected scenario step {expected_step}, current step is {current_step}")
            self._append(commit)
            self._compact(commit)
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
            log_label = "<no commit log>" if self._commit_log is None else str(self._commit_log.path)
            applicable = [record for record in records if record.step > checkpoint_step]
            for previous, current in itertools.pairwise(applicable):
                if current.step != previous.step + 1:
                    raise ReefError(
                        f"commit log {log_label} jumps from step {previous.step} to {current.step}; the log is corrupt"
                    )
            if applicable and applicable[0].step != checkpoint_step + 1:
                raise ReefError(
                    f"commit log {log_label} resumes at step {applicable[0].step}, "
                    f"expected {checkpoint_step + 1} after the checkpointed step "
                    f"{checkpoint_step}; the log is corrupt"
                )
            last_step = records[-1].step if records else 0
            if checkpoint is not None and checkpoint_step > last_step:
                head = checkpoint
                self._append(head)
                records = (*records, head)
            elif applicable:
                head = applicable[-1]
            elif records and last_step == checkpoint_step:
                head = records[-1]
            else:
                head = None
            # Every earlier commit can have an interrupted compaction, including
            # commits before the checkpoint from which the trainer is restored.
            for record in records:
                self._compact(record)
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
            if self._lock_path is None:
                yield
                return
            try:
                import fcntl
            except ImportError as exc:
                raise ReefError("durable commit log stores require POSIX file locking (fcntl)") from exc
            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._lock_path, "a+b") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _history(self) -> tuple[CommitRecord, ...]:
        records = tuple(self._memory_history) if self._commit_log is None else self._commit_log.records()
        if records is self._cached_history:
            return records
        previous = self._cached_history
        appended = len(previous) <= len(records) and (not previous or records[len(previous) - 1] is previous[-1])
        if not appended:
            self._commits_by_step.clear()
        for record in records[len(previous) if appended else 0 :]:
            if record.scenario != self._scenario:
                log_label = "<no commit log>" if self._commit_log is None else str(self._commit_log.path)
                raise ReefError(
                    f"commit log {log_label} holds records for {record.scenario!r}, not {self._scenario!r}"
                )
            self._commits_by_step[record.step] = record
        self._cached_history = records
        return records

    def _append(self, commit: CommitRecord) -> None:
        if self._commit_log is None:
            self._memory_history.append(commit)
            if commit.operation == "rollback":
                self._memory_run_segment = commit.step
                self._memory_run_step = 0
            elif commit.operation == "training" and commit.step > self._memory_run_segment:
                self._memory_run_step += 1
        else:
            self._commit_log.append(commit)

    def _compact(self, commit: CommitRecord) -> None:
        if commit.compacted_ids:
            self._records.compact(self._scenario, commit.compacted_ids)

    @staticmethod
    def _same_commit(left: CommitRecord, right: CommitRecord) -> bool:
        return {key: value for key, value in left.to_dict().items() if key != "recorded_at"} == {
            key: value for key, value in right.to_dict().items() if key != "recorded_at"
        }


__all__ = ["RECORD_KIND", "CommitLog", "CommitLogError", "CommitLogScenarioStore", "CommitRecord"]
