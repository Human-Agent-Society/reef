"""Task stores in the record database of a scenario, for each SQL backend.

The task tables are in the same database as the records. Each database
operation uses one transaction of the record store. The task directories are
in a ``TaskFileStore``. The adapters build the tables with :func:`task_tables`
and supply their namespace. This module implements the task store behavior once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from sqlalchemy import BigInteger, Boolean, Column, ForeignKey, MetaData, String, Table, Text, and_, func, select
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.storage.sql_records import SQLRecordStore
from reef.storage.tasks import (
    SPLIT_NAMES,
    GenerationState,
    SplitName,
    StoredTask,
    TaskCandidate,
    TaskConflict,
    TaskFileStore,
    TaskStorage,
    TaskStore,
    TaskStoreError,
)


@dataclass(frozen=True)
class TaskTables:
    """The task tables of one database, and the namespace values that ``scope`` gives them.

    ``generations`` holds one row for each generation. ``candidates`` holds one row for each saved
    candidate, with the digest of its task files and its outcome.
    ``published`` holds one row for each published task name, with its
    position in publication order and the digest of its task files.
    """

    generations: Table
    candidates: Table
    published: Table
    scope: Mapping[str, str] = field(default_factory=dict)


def task_tables(metadata: MetaData, *, foreign_key: str | None = None) -> TaskTables:
    """Define the task tables in ``metadata``.

    Each table has the key column ``store_name``. If ``foreign_key`` names a
    column, each table also has the key column ``storage_id``, which refers to
    that column. The adapter then gives ``storage_id`` in ``TaskTables.scope``.
    """

    def key_columns() -> list[Column[str]]:
        columns: list[Column[str]] = [Column("store_name", Text, primary_key=True)]
        if foreign_key is not None:
            columns.insert(0, Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True))
        return columns

    return TaskTables(
        generations=Table(
            "task_generation",
            metadata,
            *key_columns(),
            Column("generation", BigInteger, primary_key=True),
            Column("source_record_ids_json", Text, nullable=False),
            Column("candidates_saved", Boolean, nullable=False),
            Column("is_complete", Boolean, nullable=False),
        ),
        candidates=Table(
            "candidate_task",
            metadata,
            *key_columns(),
            Column("generation", BigInteger, primary_key=True),
            Column("candidate_index", BigInteger, primary_key=True),
            Column("task_name", Text, nullable=False),
            Column("files_digest", String(64), nullable=False),
            Column("split", Text),
            Column("error", Text, nullable=False),
        ),
        published=Table(
            "published_task",
            metadata,
            *key_columns(),
            Column("task_name", Text, primary_key=True),
            Column("sequence", BigInteger, nullable=False),
            Column("files_digest", String(64), nullable=False),
            Column("split", Text, nullable=False),
        ),
    )


class SQLTaskStorage(TaskStorage):
    """Opens the task stores of one scenario in the database of its record store."""

    def __init__(self, records: SQLRecordStore, tables: TaskTables, files: TaskFileStore, scenario: str) -> None:
        self.records = records
        self.tables = tables
        self.files = files
        self.scenario = scenario

    def open(self, name: str) -> SQLTaskStore:
        return SQLTaskStore(self, name)


class SQLTaskStore(TaskStore):
    """One named task store. Each database operation is one transaction of the record database."""

    def __init__(self, storage: SQLTaskStorage, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a task store name must be a non-empty string")
        self.records = storage.records
        self.tables = storage.tables
        self.files = storage.files
        self.scenario = storage.scenario
        self.name = name
        self.key = {**self.tables.scope, "store_name": name}

    def begin_generation(self, *, generation: int, source_record_ids: tuple[str, ...]) -> GenerationState:
        requested = GenerationState(generation=generation, source_record_ids=source_record_ids)
        with self.records._transaction(write=True) as connection:
            self.records._insert_if_absent(
                connection,
                self.tables.generations,
                {
                    **self.key,
                    "generation": generation,
                    "source_record_ids_json": json.dumps(list(source_record_ids), ensure_ascii=False),
                    "candidates_saved": False,
                    "is_complete": False,
                },
            )
            state = self.require_generation(connection, generation)
        if state.source_record_ids != requested.source_record_ids:
            raise TaskStoreError(f"generation {generation} of task store {self.name!r} has other source records")
        return state

    def save_candidates(self, *, generation: int, directories: tuple[Path, ...]) -> GenerationState:
        # Save the files before the rows, so that a saved candidate always has its files. The file
        # store can be slow or remote, so the copies run outside a database transaction.
        candidates = tuple(
            TaskCandidate(directory.name, self.files.put(self.scenario, directory)) for directory in directories
        )
        with self.records._transaction(write=True) as connection:
            state = self.require_generation(connection, generation)
            if state.candidates is not None:
                # A saved candidate can have an outcome already, so a retry compares the files only.
                if [(saved.name, saved.files_digest) for saved in state.candidates] != [
                    (candidate.name, candidate.files_digest) for candidate in candidates
                ]:
                    raise TaskStoreError(f"generation {generation} of task store {self.name!r} has other candidates")
                return state
            if candidates:
                connection.execute(
                    self.tables.candidates.insert(),
                    [
                        {
                            **self.key,
                            "generation": generation,
                            "candidate_index": index,
                            "task_name": candidate.name,
                            "files_digest": candidate.files_digest,
                            "split": None,
                            "error": "",
                        }
                        for index, candidate in enumerate(candidates)
                    ],
                )
            connection.execute(
                self.tables.generations.update()
                .where(self.matching(self.tables.generations, generation=generation))
                .values(candidates_saved=True)
            )
        return replace(state, candidates=candidates)

    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        if split not in SPLIT_NAMES:
            raise ValueError("split must be 'train' or 'eval'")
        with self.records._transaction(write=True) as connection:
            candidate = self.require_candidate(connection, generation, index)
            if candidate["error"]:
                raise TaskStoreError(f"candidate {index} of generation {generation} is rejected")
            published = self.publish(connection, candidate["task_name"], candidate["files_digest"], split)
            if candidate["split"] is None:
                self.set_outcome(connection, generation, index, split=split)
        return published

    def add_task(self, *, directory: Path, split: SplitName) -> StoredTask:
        if split not in SPLIT_NAMES:
            raise ValueError("split must be 'train' or 'eval'")
        digest = self.files.put(self.scenario, directory)
        with self.records._transaction(write=True) as connection:
            return self.publish(connection, directory.name, digest, split)

    def reject_task(self, *, generation: int, index: int, reason: str) -> None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("a rejection needs a reason")
        with self.records._transaction(write=True) as connection:
            candidate = self.require_candidate(connection, generation, index)
            if candidate["split"] is not None:
                raise TaskStoreError(f"candidate {index} of generation {generation} is published")
            if candidate["error"] == reason:
                return
            if candidate["error"]:
                raise TaskStoreError(f"candidate {index} of generation {generation} is rejected for another reason")
            self.set_outcome(connection, generation, index, error=reason)

    def complete_generation(self, *, generation: int) -> GenerationState:
        with self.records._transaction(write=True) as connection:
            state = self.require_generation(connection, generation)
            if state.candidates is None or state.pending_indices:
                raise TaskStoreError(
                    f"generation {generation} of task store {self.name!r} has candidates without outcome"
                )
            connection.execute(
                self.tables.generations.update()
                .where(self.matching(self.tables.generations, generation=generation))
                .values(is_complete=True)
            )
        return replace(state, is_complete=True)

    def generation(self, generation: int) -> GenerationState | None:
        with self.records._transaction(write=False) as connection:
            return self.read_generation(connection, generation)

    def generations(self) -> tuple[GenerationState, ...]:
        generations = self.tables.generations
        with self.records._transaction(write=False) as connection:
            numbers = connection.execute(
                select(generations.c.generation).where(self.matching(generations)).order_by(generations.c.generation)
            ).scalars()
            return tuple(self.require_generation(connection, number) for number in numbers.all())

    def tasks(self, *, split: SplitName | None = None) -> tuple[StoredTask, ...]:
        with self.records._transaction(write=False) as connection:
            return self.published_tasks(connection) if split is None else self.published_tasks(connection, split=split)

    def path(self, files_digest: str) -> Path:
        return self.files.path(self.scenario, files_digest)

    def publish(self, connection: Connection, name: str, digest: str, split: SplitName) -> StoredTask:
        """Record ``name`` as published with the files ``digest`` to ``split``, and return the published task.

        If ``name`` is published with the same files and split, return that
        task. Raise ``TaskConflict`` if it is published with other files or in
        the other split.
        """
        existing = self.published_tasks(connection, task_name=name)
        if existing:
            (task,) = existing
            if task.files_digest != digest:
                raise TaskConflict(f"task {name!r} is published with other files")
            if task.split != split:
                raise TaskConflict(f"task {name!r} is published to {task.split}, not {split}")
            return task
        published = self.tables.published
        last_sequence = connection.execute(
            select(func.coalesce(func.max(published.c.sequence), 0)).where(self.matching(published))
        ).scalar_one()
        connection.execute(
            published.insert().values(
                **self.key, task_name=name, sequence=last_sequence + 1, files_digest=digest, split=split
            )
        )
        (task,) = self.published_tasks(connection, task_name=name)
        return task

    def published_tasks(self, connection: Connection, **columns: object) -> tuple[StoredTask, ...]:
        """Read the published tasks of this store that have the given column values, in publication order."""
        published = self.tables.published
        rows = connection.execute(
            select(published.c.task_name, published.c.split, published.c.files_digest)
            .where(self.matching(published, **columns))
            .order_by(published.c.sequence)
        ).mappings()
        return tuple(StoredTask(row["task_name"], row["split"], row["files_digest"]) for row in rows)

    def matching(self, table: Table, **columns: object) -> ColumnElement[bool]:
        """Select the rows of this store in ``table`` that have the given column values."""
        return and_(*(table.c[column] == value for column, value in {**self.key, **columns}.items()))

    def read_generation(self, connection: Connection, generation: int) -> GenerationState | None:
        """Read one generation and its candidates in the transaction of the caller."""
        row = (
            connection.execute(
                select(self.tables.generations).where(self.matching(self.tables.generations, generation=generation))
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        candidates = self.tables.candidates
        candidate_rows = (
            connection.execute(
                select(candidates)
                .where(self.matching(candidates, generation=generation))
                .order_by(candidates.c.candidate_index)
            )
            .mappings()
            .all()
        )
        return generation_state(row, candidate_rows)

    def require_generation(self, connection: Connection, generation: int) -> GenerationState:
        """Read one generation. Raise ``TaskStoreError`` if it does not exist."""
        state = self.read_generation(connection, generation)
        if state is None:
            raise TaskStoreError(f"generation {generation} of task store {self.name!r} does not exist")
        return state

    def require_candidate(self, connection: Connection, generation: int, index: int) -> RowMapping:
        """Read one candidate row. Raise ``TaskStoreError`` if it does not exist."""
        candidates = self.tables.candidates
        row = (
            connection.execute(
                select(candidates).where(self.matching(candidates, generation=generation, candidate_index=index))
            )
            .mappings()
            .first()
        )
        if row is None:
            raise TaskStoreError(f"generation {generation} of task store {self.name!r} has no candidate {index}")
        return row

    def set_outcome(self, connection: Connection, generation: int, index: int, **outcome: str) -> None:
        """Write the ``split`` or ``error`` of one candidate."""
        candidates = self.tables.candidates
        connection.execute(
            candidates.update()
            .where(self.matching(candidates, generation=generation, candidate_index=index))
            .values(**outcome)
        )


def generation_state(row: RowMapping, candidate_rows: Sequence[RowMapping]) -> GenerationState:
    """Build a generation checkpoint from its row and its candidate rows in index order."""
    candidates = tuple(
        TaskCandidate(candidate["task_name"], candidate["files_digest"], candidate["split"], candidate["error"])
        for candidate in candidate_rows
    )
    return GenerationState(
        generation=int(row["generation"]),
        source_record_ids=tuple(json.loads(row["source_record_ids_json"])),
        candidates=candidates if row["candidates_saved"] else None,
        is_complete=bool(row["is_complete"]),
    )
