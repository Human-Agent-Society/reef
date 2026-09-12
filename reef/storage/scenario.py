"""Assemble record backends and JSONL commits for scenario storage.

Storage services own backend resources, commit log paths, archival, and retention.
Domain code receives a scenario store without depending on these storage choices.
"""

from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from pathlib import Path
from threading import RLock

from reef.scenario.store import ScenarioStorage
from reef.storage.commit_log import CommitLog, CommitLogScenarioStore
from reef.storage.postgres import PostgresRecordDatabase, PostgresRecordStore
from reef.storage.records import RecordRetention
from reef.storage.sqlite import SQLiteRecordRetention, SQLiteRecordStore


class SQLiteScenarioStorage(ScenarioStorage):
    """Open existing hashed SQLite and JSONL paths and manage their lifecycle."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._lock = RLock()
        self._closed = False
        if self._directory is not None:
            self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def durable(self) -> bool:
        return self._directory is not None

    def open(self, scenario: str) -> CommitLogScenarioStore:
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            database = None if self._directory is None else self._directory / f"{key}.sqlite3"
            commit_log = None if self._directory is None else CommitLog(self._directory / f"{key}.commits.jsonl")
            records = SQLiteRecordStore(database)
            try:
                return CommitLogScenarioStore(scenario, records, commit_log)
            except BaseException:
                records.close()
                raise

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            paths = self.state_paths(scenario)
            if self._directory is None:
                return ()
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            destination = self._directory / "archived" / f"{self._scenario_key(scenario)}-{stamp}-{uuid.uuid4().hex}"
            moved: list[str] = []
            for path in paths:
                if path.exists():
                    destination.mkdir(parents=True, exist_ok=True)
                    target = destination / path.name
                    shutil.move(str(path), str(target))
                    moved.append(str(target))
            return tuple(moved)

    def prune(self, *, days: float, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            retention = RecordRetention(days, max_bytes)
            return 0 if self._directory is None else SQLiteRecordRetention(retention).prune(self._directory)

    def state_paths(self, scenario: str) -> tuple[Path, ...]:
        """Existing local state paths; the stable process lock is never moved."""
        with self._lock:
            self._ensure_open()
            key = self._scenario_key(scenario)
            if self._directory is None:
                return ()
            return tuple(
                self._directory / name
                for name in (f"{key}.sqlite3", f"{key}.sqlite3-wal", f"{key}.sqlite3-shm", f"{key}.commits.jsonl")
            )

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")

    @staticmethod
    def _scenario_key(scenario: str) -> str:
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("scenario must be a non-empty string")
        return hashlib.sha256(scenario.encode("utf-8")).hexdigest()


class PostgresScenarioStorage(ScenarioStorage):
    """Combine pooled PostgreSQL records with generation-specific local commit logs."""

    def __init__(self, database_url: str, directory: Path, *, schema: str = "reef_records") -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._database = PostgresRecordDatabase(database_url, schema=schema)
        self._schema = schema
        self._lock = RLock()
        self._closed = False

    @property
    def durable(self) -> bool:
        return True

    def open(self, scenario: str) -> CommitLogScenarioStore:
        with self._lock:
            self._ensure_open()
            records = PostgresRecordStore(self._database, name=scenario)
            try:
                commit_log = CommitLog(self._log_path(records.storage_id))
                return CommitLogScenarioStore(scenario, records, commit_log)
            except BaseException:
                records.close()
                raise

    def archive(self, scenario: str) -> tuple[str, ...]:
        with self._lock:
            self._ensure_open()
            storage_id = self._database.archive(scenario)
            if storage_id is None:
                return ()
            archived = [f"postgres://{self._schema}/record-store/{storage_id}"]
            # The next generation always gets another log name. If moving the
            # old log fails, reopening cannot apply it to the new empty store.
            path = self._log_path(storage_id)
            if path.exists():
                destination = self._directory / "archived" / f"postgres-{storage_id}"
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / path.name
                shutil.move(str(path), str(target))
                archived.append(str(target))
            return tuple(archived)

    def prune(self, *, days: float, max_bytes: int) -> int:
        with self._lock:
            self._ensure_open()
            return self._database.prune(RecordRetention(days, max_bytes))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._database.close()

    def _log_path(self, storage_id: str) -> Path:
        return self._directory / f"postgres-{storage_id}.commits.jsonl"

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("scenario storage is closed")


__all__ = ["PostgresScenarioStorage", "SQLiteScenarioStorage"]
