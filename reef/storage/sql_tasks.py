"""Task stores in the record database of a scenario, for each SQL backend.

The task tables are in the same database as the records. Each database
operation uses one transaction of the record store. The task directories are
in a ``TaskFileStore``. The adapters build the tables with :func:`task_tables`
and supply their namespace. This module implements the task store behavior once.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from sqlalchemy import BigInteger, Boolean, Column, Float, ForeignKey, MetaData, String, Table, Text, and_, select
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.tasks import GeneratedTask, HarborTask, PlayArm, write_harbor_task
from reef.storage.sql_records import SQLRecordStore
from reef.storage.tasks import (
    SPLIT_NAMES,
    GenerationState,
    SplitName,
    SplitSettings,
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

    ``stores`` holds the split settings of each store. ``generations`` holds
    one row for each generation. ``candidates`` holds one row for each saved
    candidate, with its outcome. ``published`` holds one row for each
    published task name, with the digest of its task files.
    """

    stores: Table
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
        stores=Table(
            "task_store",
            metadata,
            *key_columns(),
            Column("eval_fraction", Float(53), nullable=False),
            Column("seed", BigInteger, nullable=False),
        ),
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
            "task_candidate",
            metadata,
            *key_columns(),
            Column("generation", BigInteger, primary_key=True),
            Column("candidate_index", BigInteger, primary_key=True),
            Column("task_name", Text, nullable=False),
            Column("generated_json", Text, nullable=False),
            Column("split", Text),
            Column("error", Text, nullable=False),
        ),
        published=Table(
            "generated_task",
            metadata,
            *key_columns(),
            Column("task_name", Text, primary_key=True),
            Column("files_digest", String(64), nullable=False),
            Column("split", Text, nullable=False),
            Column("generation", BigInteger, nullable=False),
            Column("candidate_index", BigInteger, nullable=False),
        ),
    )


class SQLTaskStorage(TaskStorage):
    """Opens the task stores of one scenario in the database of its record store."""

    def __init__(self, records: SQLRecordStore, tables: TaskTables, files: TaskFileStore, scenario: str) -> None:
        self.records = records
        self.tables = tables
        self.files = files
        self.scenario = scenario

    def open(self, name: str, *, split: SplitSettings) -> SQLTaskStore:
        return SQLTaskStore(self, name, split=split)


class SQLTaskStore(TaskStore):
    """One named task store. Each database operation is one transaction of the record database."""

    def __init__(self, storage: SQLTaskStorage, name: str, *, split: SplitSettings) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("a task store name must be a non-empty string")
        self.records = storage.records
        self.tables = storage.tables
        self.files = storage.files
        self.scenario = storage.scenario
        self.name = name
        self.key = {**self.tables.scope, "store_name": name}
        stores = self.tables.stores
        with self.records._transaction(write=True) as connection:
            self.records._insert_if_absent(
                connection, stores, {**self.key, "eval_fraction": split.eval_fraction, "seed": split.seed}
            )
            row = connection.execute(select(stores).where(self.matching(stores))).mappings().one()
        stored = SplitSettings(eval_fraction=row["eval_fraction"], seed=row["seed"])
        if stored != split:
            raise TaskStoreError(
                f"task store {name!r} has eval_fraction {stored.eval_fraction} and seed {stored.seed}, "
                f"not eval_fraction {split.eval_fraction} and seed {split.seed}"
            )

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

    def save_candidates(self, *, generation: int, candidates: tuple[GeneratedTask, ...]) -> GenerationState:
        documents = [generated_document(item) for item in candidates]
        with self.records._transaction(write=True) as connection:
            state = self.require_generation(connection, generation)
            if state.candidates is not None:
                if [generated_document(saved.generated) for saved in state.candidates] != documents:
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
                            "task_name": item.task.name,
                            "generated_json": document,
                            "split": None,
                            "error": "",
                        }
                        for index, (item, document) in enumerate(zip(candidates, documents, strict=True))
                    ],
                )
            connection.execute(
                self.tables.generations.update()
                .where(self.matching(self.tables.generations, generation=generation))
                .values(candidates_saved=True)
            )
        return replace(state, candidates=tuple(TaskCandidate(item) for item in candidates))

    def publish_task(self, *, generation: int, index: int, split: SplitName) -> StoredTask:
        if split not in SPLIT_NAMES:
            raise ValueError("split must be 'train' or 'eval'")
        with self.records._transaction(write=False) as connection:
            candidate = self.require_candidate(connection, generation, index)
        if candidate["error"]:
            raise TaskStoreError(f"candidate {index} of generation {generation} is rejected")
        # Save the files before the row, so that a published row always has its files. The file
        # store can be slow or remote, so the copy runs outside a database transaction.
        with tempfile.TemporaryDirectory(prefix="reef-task-") as staging:
            task = generated_from_document(candidate["generated_json"]).task
            digest = self.files.put(self.scenario, write_harbor_task(task, Path(staging)))
        published = self.tables.published
        name = candidate["task_name"]
        with self.records._transaction(write=True) as connection:
            existing = (
                connection.execute(select(published).where(self.matching(published, task_name=name)))
                .mappings()
                .first()
            )
            if existing is None:
                connection.execute(
                    published.insert().values(
                        **self.key,
                        task_name=name,
                        files_digest=digest,
                        split=split,
                        generation=generation,
                        candidate_index=index,
                    )
                )
                first_candidate = candidate
            else:
                if existing["files_digest"] != digest:
                    raise TaskConflict(f"task {name!r} is published with other files")
                if existing["split"] != split:
                    raise TaskConflict(f"task {name!r} is published to {existing['split']}, not {split}")
                first_candidate = self.require_candidate(
                    connection, existing["generation"], existing["candidate_index"]
                )
            if candidate["split"] is None:
                self.set_outcome(connection, generation, index, split=split)
        return StoredTask(
            generated=generated_from_document(first_candidate["generated_json"]),
            split=split,
            generation=int(first_candidate["generation"]),
            index=int(first_candidate["candidate_index"]),
            files_digest=digest,
        )

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
        candidates = self.tables.candidates
        published = self.tables.published
        statement = (
            select(
                published.c.split,
                published.c.files_digest,
                candidates.c.generation,
                candidates.c.candidate_index,
                candidates.c.generated_json,
            )
            .join(
                candidates,
                and_(
                    *(candidates.c[column] == published.c[column] for column in self.key),
                    candidates.c.generation == published.c.generation,
                    candidates.c.candidate_index == published.c.candidate_index,
                ),
            )
            .where(self.matching(published) if split is None else self.matching(published, split=split))
            .order_by(published.c.generation, published.c.candidate_index)
        )
        with self.records._transaction(write=False) as connection:
            rows = connection.execute(statement).mappings().all()
        return tuple(
            StoredTask(
                generated=generated_from_document(row["generated_json"]),
                split=row["split"],
                generation=int(row["generation"]),
                index=int(row["candidate_index"]),
                files_digest=row["files_digest"],
            )
            for row in rows
        )

    def path(self, task: StoredTask) -> Path:
        return self.files.path(self.scenario, task.files_digest)

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
        TaskCandidate(generated_from_document(candidate["generated_json"]), candidate["split"], candidate["error"])
        for candidate in candidate_rows
    )
    return GenerationState(
        generation=int(row["generation"]),
        source_record_ids=tuple(json.loads(row["source_record_ids_json"])),
        candidates=candidates if row["candidates_saved"] else None,
        is_complete=bool(row["is_complete"]),
    )


def generated_document(generated: GeneratedTask) -> str:
    """Encode a generated task as canonical JSON text. An identical retry gives the same text."""
    return json.dumps(
        {
            "task": generated.task.document(),
            "measurement_arms": [
                {"name": arm.name, "count": arm.count, "extra_instruction_files": list(arm.extra_instruction_files)}
                for arm in generated.measurement_arms
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def generated_from_document(text: str) -> GeneratedTask:
    """Decode the text of :func:`generated_document`. The constructors validate the content again."""
    document = json.loads(text)
    task = document["task"]
    return GeneratedTask(
        task=HarborTask(**{**task, "source_agent_record_ids": tuple(task["source_agent_record_ids"])}),
        measurement_arms=tuple(
            PlayArm(arm["name"], arm["count"], tuple(arm["extra_instruction_files"]))
            for arm in document["measurement_arms"]
        ),
    )
