"""The sample store in the record database of a scenario, for each SQL backend.

The sample table and the train count table are in the same database as the
records and the commits. Each database operation uses one transaction of the
record store. The commit history counts the samples that a commit consumed in
the transaction of the commit (``SQLCommitHistory``). The adapters build the
tables with :func:`sample_table` and :func:`sample_train_count_table` and supply
their namespace.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from sqlalchemy import (
    BigInteger,
    Column,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
    or_,
    select,
)
from sqlalchemy.engine import Connection, RowMapping
from sqlalchemy.sql.elements import ColumnElement

from reef.core.job_data import TaskSample, TrajectorySample
from reef.storage.samples import SampleConflict, SampleKind, SampleStore, StoredSample
from reef.storage.sql_records import ScopedTable, SQLRecordStore
from reef.storage.tasks import SPLIT_NAMES, SplitName, TaskFileStore


def sample_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``training_sample`` table in ``metadata``.

    ``kind`` is the kind of the sample, ``trajectory`` or ``task``, and
    ``split`` is ``train`` or ``eval``; a trajectory is always ``train``.
    ``content_json`` holds the sample as canonical JSON (``sample_content``),
    and ``content_sha256`` its SHA-256. ``component`` is the trainer that made
    the sample from records (the component its commits name, or ``""`` for the
    only trainer of a scenario), or NULL for a sample that every trainer reads,
    such as an upload. ``group_key`` is the group that trains together, or
    NULL. ``record_ids_json`` is the JSON list of the records that the sample's
    unit consumed. If ``foreign_key`` names a column, the
    table also has the column ``storage_id``, which refers to it. The adapter
    then gives ``storage_id`` in ``ScopedTable.scope``. Sample IDs are unique in
    each namespace.
    """
    columns = (
        Column("sample_id", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("split", String(8), nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
        Column("component", Text, nullable=True),
        Column("group_key", Text, nullable=True),
        Column("record_ids_json", Text, nullable=False, server_default="[]"),
    )
    if foreign_key is None:
        return Table(
            "training_sample",
            metadata,
            Column("sequence", Integer, primary_key=True),
            *columns,
            UniqueConstraint("sample_id"),
            sqlite_autoincrement=True,
        )
    return Table(
        "training_sample",
        metadata,
        Column("sequence", BigInteger, Identity(), primary_key=True),
        Column("storage_id", String(32), ForeignKey(foreign_key), nullable=False),
        *columns,
        UniqueConstraint("storage_id", "sample_id"),
        Index("training_sample_storage_sequence", "storage_id", "sequence"),
    )


def sample_train_count_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``sample_train_count`` table in ``metadata``: how many committed jobs of each trainer held a sample.

    A row ``(sample_id, component, train_count)`` says that ``train_count``
    committed jobs of the trainer ``component`` listed
    ``consumed_sample_id(sample_id)``. ``component`` is the component that
    the commits name, or ``""`` for the only trainer of a scenario, whose
    commits name none. A missing row is a count of 0. ``foreign_key`` adds
    the ``storage_id`` column as for ``sample_table``.
    """
    columns = (
        Column("sample_id", Text, primary_key=True),
        Column("component", Text, primary_key=True),
        Column("train_count", Integer, nullable=False),
    )
    if foreign_key is None:
        return Table("sample_train_count", metadata, *columns)
    return Table(
        "sample_train_count",
        metadata,
        Column("storage_id", String(32), ForeignKey(foreign_key), primary_key=True),
        *columns,
    )


def sample_content(sample: TrajectorySample) -> str:
    """The canonical JSON of a trajectory that the sample table stores; ``stored_sample`` reads it back."""
    content = {
        "trajectory": dict(sample.trajectory),
        "group_id": sample.group_id,
        "source_agent_record_ids": list(sample.source_agent_record_ids),
    }
    return canonical_json(content)


def task_content(files_digest: str) -> str:
    """The canonical JSON of a task that the sample table stores: the digest of its directory's files."""
    return canonical_json({"files_digest": files_digest})


def canonical_json(content: Mapping[str, object]) -> str:
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_json_list(values: tuple[str, ...]) -> str:
    """The canonical JSON of a list of strings, in sorted order without repeats."""
    return json.dumps(sorted(set(values)), ensure_ascii=False, separators=(",", ":"))


class SQLSampleStore(SampleStore):
    """The training samples of one scenario in its record database.

    ``table`` is the sample table and ``train_counts`` the train count table
    (``sample_train_count_table``), both in the namespace of the scenario.
    ``files`` keeps the directories of the scenario's tasks, under
    ``scenario``.
    """

    def __init__(
        self,
        records: SQLRecordStore,
        table: ScopedTable,
        train_counts: ScopedTable,
        files: TaskFileStore,
        scenario: str,
    ) -> None:
        self.records = records
        self.table = table
        self.train_counts = train_counts
        self.files = files
        self.scenario = scenario

    def add_trajectories(self, trajectories: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        rows = []
        for sample_id, trajectory in trajectories:
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("a sample ID must be a non-empty string")
            rows.append((sample_id, sample_content(trajectory)))
        with self.records._transaction(write=True) as connection:
            inserted = [
                self.insert(connection, sample_id, "trajectory", "train", content) for sample_id, content in rows
            ]
        return any(inserted)

    def add_record_samples(
        self,
        trainer: str,
        group: str | None,
        record_ids: tuple[str, ...],
        trajectories: tuple[tuple[str, TrajectorySample], ...],
    ) -> bool:
        if not isinstance(trainer, str):
            raise TypeError("trainer must be a string")
        if group is not None and (not isinstance(group, str) or not group):
            raise ValueError("a sample group must be a non-empty string or None")
        if not trajectories or any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
            raise ValueError("record samples need one or more samples and non-empty record IDs")
        rows = [(sample_id, sample_content(trajectory)) for sample_id, trajectory in trajectories]
        consumed = canonical_json_list(record_ids)
        with self.records._transaction(write=True) as connection:
            inserted = [
                self.insert(
                    connection,
                    sample_id,
                    "trajectory",
                    "train",
                    content,
                    component=trainer,
                    group_key=group,
                    record_ids_json=consumed,
                )
                for sample_id, content in rows
            ]
        return any(inserted)

    def consumed_record_ids(self, trainer: str) -> frozenset[str]:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            rows = connection.execute(
                select(table.c.record_ids_json).where(self.table.condition(), table.c.component == trainer)
            ).scalars()
            return frozenset(record_id for encoded in rows for record_id in json.loads(encoded))

    def add_tasks(self, directories: tuple[Path, ...], split: SplitName) -> tuple[StoredSample, ...]:
        if split not in SPLIT_NAMES:
            raise ValueError("split must be 'train' or 'eval'")
        rows = [(directory.name, task_content(self.files.put(self.scenario, directory))) for directory in directories]
        with self.records._transaction(write=True) as connection:
            for sample_id, content in rows:
                self.insert(connection, sample_id, "task", split, content)
            return self.read(connection, self.table.table.c.sample_id.in_([sample_id for sample_id, _ in rows]))

    def insert(
        self,
        connection: Connection,
        sample_id: str,
        kind: SampleKind,
        split: SplitName,
        content: str,
        *,
        component: str | None = None,
        group_key: str | None = None,
        record_ids_json: str = "[]",
    ) -> bool:
        """Store one sample unless it is stored as it is; return whether it is new. ``SampleConflict`` otherwise.

        ``component``, ``group_key`` and ``record_ids_json`` are the columns of ``sample_table``.
        """
        table = self.table.table
        content_sha256 = hashlib.sha256(content.encode()).hexdigest()
        stored = connection.execute(
            select(table.c.kind, table.c.split, table.c.content_sha256).where(
                self.table.condition(), table.c.sample_id == sample_id
            )
        ).first()
        if stored is None:
            connection.execute(
                table.insert().values(
                    **self.table.scope,
                    sample_id=sample_id,
                    kind=kind,
                    split=split,
                    content_json=content,
                    content_sha256=content_sha256,
                    component=component,
                    group_key=group_key,
                    record_ids_json=record_ids_json,
                )
            )
            return True
        if stored.kind != kind:
            raise SampleConflict(f"sample {sample_id!r} is already stored as a {stored.kind}")
        if stored.content_sha256 != content_sha256:
            raise SampleConflict(f"sample {sample_id!r} is already stored with other content")
        if stored.split != split:
            raise SampleConflict(f"sample {sample_id!r} is already stored in {stored.split}, not {split}")
        return False

    def page(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 256,
        kind: SampleKind | None = None,
        trainer: str | None = None,
    ) -> tuple[StoredSample, ...]:
        if after_sequence < 0 or limit <= 0:
            raise ValueError("after_sequence must be non-negative and limit positive")
        with self.records._transaction(write=False) as connection:
            return self.read(connection, *self.conditions(after_sequence, kind, trainer), limit=limit)

    def count(self, *, after_sequence: int = 0, kind: SampleKind | None = None, trainer: str | None = None) -> int:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            statement = (
                select(func.count())
                .select_from(table)
                .where(self.table.condition(), *self.conditions(after_sequence, kind, trainer))
            )
            return int(connection.execute(statement).scalar_one())

    def conditions(
        self, after_sequence: int, kind: SampleKind | None, trainer: str | None
    ) -> tuple[ColumnElement[bool], ...]:
        """The conditions of the samples after ``after_sequence``, only ``kind``'s, and only those that ``trainer``
        reads, if given; ``read`` adds the scope."""
        table = self.table.table
        conditions: list[ColumnElement[bool]] = [table.c.sequence > after_sequence]
        if kind is not None:
            conditions.append(table.c.kind == kind)
        if trainer is not None:
            conditions.append(or_(table.c.component.is_(None), table.c.component == trainer))
        return tuple(conditions)

    def read(
        self, connection: Connection, *conditions: ColumnElement[bool], limit: int | None = None
    ) -> tuple[StoredSample, ...]:
        """The stored samples that match ``conditions``, in arrival order, at most ``limit``, with their counts."""
        table = self.table.table
        counts = self.train_counts.table
        statement = select(table).where(self.table.condition(), *conditions).order_by(table.c.sequence)
        rows = connection.execute(statement if limit is None else statement.limit(limit)).mappings().all()
        train_counts: dict[str, dict[str, int]] = {row["sample_id"]: {} for row in rows}
        if train_counts:
            for count in connection.execute(
                select(counts.c.sample_id, counts.c.component, counts.c.train_count).where(
                    self.train_counts.condition(), counts.c.sample_id.in_(list(train_counts))
                )
            ):
                train_counts[count.sample_id][count.component] = int(count.train_count)
        return tuple(self.stored_sample(row, train_counts[row["sample_id"]]) for row in rows)

    def stored_sample(self, row: RowMapping, train_counts: Mapping[str, int]) -> StoredSample:
        """Return the stored sample of one table row, with its train counts by component."""
        content = json.loads(row["content_json"])
        split = row["split"]
        if row["kind"] == "task":
            digest = content["files_digest"]
            task = TaskSample(
                self.files.path(self.scenario, digest),
                metadata={"task": {"name": row["sample_id"], "files_digest": digest, "split": split}},
            )
            return StoredSample(int(row["sequence"]), row["sample_id"], split, task, digest, train_counts)
        trajectory = TrajectorySample(
            content["trajectory"],
            group_id=content["group_id"],
            source_agent_record_ids=tuple(content["source_agent_record_ids"]),
        )
        return StoredSample(
            int(row["sequence"]),
            row["sample_id"],
            split,
            trajectory,
            None,
            train_counts,
            component=row["component"],
            group=row["group_key"],
            record_ids=tuple(json.loads(row["record_ids_json"])),
        )
