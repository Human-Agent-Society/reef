"""The sample store in the record database of a scenario, for each SQL backend.

The sample table is in the same database as the records and the commits. Each
database operation uses one transaction of the record store. The commit history
counts the samples that a commit consumed in the transaction of the commit
(``SQLCommitHistory``). The adapters build the table with :func:`sample_table`
and supply their namespace.
"""

from __future__ import annotations

import hashlib
import json

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
    select,
)
from sqlalchemy.engine import RowMapping

from reef.core.job_data import TrajectorySample
from reef.storage.samples import SampleConflict, SampleStore, StoredSample
from reef.storage.sql_records import ScopedTable, SQLRecordStore


def sample_table(metadata: MetaData, *, foreign_key: str | None = None) -> Table:
    """Define the ``training_sample`` table in ``metadata``.

    ``kind`` is the kind of the sample, ``trajectory`` or ``task``, and
    ``split`` is ``train`` or ``eval``; a trajectory is always ``train``.
    ``content_json`` holds the sample as canonical JSON (``sample_content``),
    and ``content_sha256`` its SHA-256. ``train_count`` is the number of
    committed jobs that trained the sample. If ``foreign_key`` names a column,
    the table also has the column ``storage_id``, which refers to it. The
    adapter then gives ``storage_id`` in ``ScopedTable.scope``. Sample IDs are
    unique in each namespace.
    """
    columns = (
        Column("sample_id", Text, nullable=False),
        Column("kind", String(16), nullable=False),
        Column("split", String(8), nullable=False),
        Column("content_json", Text, nullable=False),
        Column("content_sha256", String(64), nullable=False),
        Column("train_count", Integer, nullable=False, server_default="0"),
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


def sample_content(sample: TrajectorySample) -> str:
    """The canonical JSON of ``sample`` that the sample table stores; ``stored_sample`` reads it back."""
    content = {
        "trajectory": dict(sample.trajectory),
        "group_id": sample.group_id,
        "source_agent_record_ids": list(sample.source_agent_record_ids),
    }
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class SQLSampleStore(SampleStore):
    """The training samples of one scenario in its record database."""

    def __init__(self, records: SQLRecordStore, table: ScopedTable) -> None:
        self.records = records
        self.table = table

    def add(self, samples: tuple[tuple[str, TrajectorySample], ...]) -> bool:
        rows = []
        for sample_id, sample in samples:
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("a sample ID must be a non-empty string")
            content_json = sample_content(sample)
            rows.append((sample_id, content_json, hashlib.sha256(content_json.encode()).hexdigest()))
        table = self.table.table
        inserted = False
        with self.records._transaction(write=True) as connection:
            for sample_id, content_json, content_sha256 in rows:
                stored = connection.execute(
                    select(table.c.content_sha256).where(self.table.condition(), table.c.sample_id == sample_id)
                ).scalar_one_or_none()
                if stored is not None:
                    if stored != content_sha256:
                        raise SampleConflict(f"sample {sample_id!r} is already stored with other content")
                    continue
                connection.execute(
                    table.insert().values(
                        **self.table.scope,
                        sample_id=sample_id,
                        kind="trajectory",
                        split="train",
                        content_json=content_json,
                        content_sha256=content_sha256,
                    )
                )
                inserted = True
        return inserted

    def page(self, *, after_sequence: int = 0, limit: int = 256) -> tuple[StoredSample, ...]:
        if after_sequence < 0 or limit <= 0:
            raise ValueError("after_sequence must be non-negative and limit positive")
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            rows = (
                connection.execute(
                    select(table)
                    .where(self.table.condition(), table.c.sequence > after_sequence)
                    .order_by(table.c.sequence)
                    .limit(limit)
                )
                .mappings()
                .all()
            )
        return tuple(stored_sample(row) for row in rows)

    def count(self, *, after_sequence: int = 0) -> int:
        table = self.table.table
        with self.records._transaction(write=False) as connection:
            statement = select(func.count()).select_from(table)
            return int(
                connection.execute(
                    statement.where(self.table.condition(), table.c.sequence > after_sequence)
                ).scalar_one()
            )


def stored_sample(row: RowMapping) -> StoredSample:
    """Return the stored sample of one table row."""
    content = json.loads(row["content_json"])
    sample = TrajectorySample(
        content["trajectory"],
        group_id=content["group_id"],
        source_agent_record_ids=tuple(content["source_agent_record_ids"]),
    )
    return StoredSample(
        sequence=int(row["sequence"]),
        sample_id=row["sample_id"],
        sample=sample,
        train_count=int(row["train_count"]),
    )
