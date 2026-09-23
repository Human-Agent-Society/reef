"""One-time upgrades of older record stores and persisted progress formats."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Mapping

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, LargeBinary, MetaData, Table, cast, func, inspect, select
from sqlalchemy.engine import Connection


def read_consumed_ids(value: Mapping[str, object], *, context: str) -> frozenset[str]:
    """Normalize old progress into the current consumption-only contract."""
    consumed: set[str] = set()
    for key in ("compacted_ids", "consumed_ids"):
        ids = value.get(key, [] if key == "compacted_ids" else None)
        if not isinstance(ids, list) or any(not isinstance(record_id, str) for record_id in ids):
            raise ValueError(f"{context} record_progress.{key} must be a list of strings")
        consumed.update(ids)
    return frozenset(consumed)


def migrate_record_storage(connection: Connection, records: Table, consumption: Table) -> None:
    """Upgrade in the caller's schema transaction, paging IDs without loading bodies.

    Old retired rows become consumption receipts. Original bodies remain readable;
    deleting them is exclusively a capacity decision after this migration.
    """
    schema = records.schema
    inspector = inspect(connection)
    columns = {column["name"] for column in inspector.get_columns(records.name, schema=schema)}
    operations = Operations(MigrationContext.configure(connection))
    if "body_bytes" not in columns:
        operations.add_column(
            records.name,
            Column("body_bytes", records.c.body_bytes.type, nullable=False, server_default="0"),
            schema=schema,
        )
        connection.execute(
            records.update().values(
                body_bytes=func.length(cast(records.c.payload_json, LargeBinary))
                + func.length(cast(records.c.references_json, LargeBinary))
                + func.coalesce(func.length(cast(records.c.artifact_json, LargeBinary)), 0)
            )
        )

    def _insert(
        scenario: str,
        receipt_id: str,
        ids: frozenset[str],
        metadata: dict[str, object],
        recorded_at: float,
        storage_id: str | None,
    ) -> None:
        encoded_ids = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
        row: dict[str, object] = {
            "scenario": scenario,
            "receipt_id": receipt_id,
            "consumed_ids_json": encoded_ids,
            "consumed_ids_sha256": hashlib.sha256(encoded_ids.encode()).hexdigest(),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
            "recorded_at": recorded_at,
        }
        if storage_id is not None:
            row["storage_id"] = storage_id
        connection.execute(consumption.insert().values(row))

    if inspector.has_table("compaction_receipts", schema=schema):
        legacy = Table("compaction_receipts", MetaData(), schema=schema, autoload_with=connection)
        offset = 0
        while True:
            rows = (
                connection.execute(select(legacy).order_by(*legacy.primary_key).offset(offset).limit(256))
                .mappings()
                .all()
            )
            if not rows:
                break
            for row in rows:
                metadata = json.loads(row["metadata_json"])
                ids = read_consumed_ids(
                    {
                        "consumed_ids": metadata.pop("consumed_ids", []),
                        "compacted_ids": json.loads(row["compacted_ids_json"]),
                    },
                    context="legacy receipt",
                )
                _insert(
                    row["scenario"],
                    "legacy:" + row["receipt_id"],
                    ids,
                    metadata,
                    row["recorded_at"],
                    row.get("storage_id"),
                )
            offset += len(rows)
        legacy.drop(connection)

    if "compacted_at" in columns:
        legacy_records = Table(records.name, MetaData(), schema=schema, autoload_with=connection)
        after_sequence = 0
        while True:
            selected = [legacy_records.c.sequence, legacy_records.c.scenario, legacy_records.c.agent_record_id]
            if "storage_id" in legacy_records.c:
                selected.append(legacy_records.c.storage_id)
            rows = (
                connection.execute(
                    select(*selected)
                    .where(legacy_records.c.compacted_at.is_not(None), legacy_records.c.sequence > after_sequence)
                    .order_by(legacy_records.c.sequence)
                    .limit(256)
                )
                .mappings()
                .all()
            )
            if not rows:
                break
            groups: dict[tuple[str | None, str], set[str]] = defaultdict(set)
            for row in rows:
                groups[(row.get("storage_id"), row["scenario"])].add(row["agent_record_id"])
            after_sequence = rows[-1]["sequence"]
            for (storage_id, scenario), record_ids in groups.items():
                _insert(scenario, f"retired:{after_sequence}", frozenset(record_ids), {}, time.time(), storage_id)
        for index in inspector.get_indexes(records.name, schema=schema):
            # Includes the old partial indexes whose predicate references this column.
            index_name = index["name"]
            if index_name is not None and (
                "compacted_at" in index["column_names"] or index_name.startswith("agent_record_active_")
            ):
                operations.drop_index(index_name, table_name=records.name, schema=schema)
        operations.drop_column(records.name, "compacted_at", schema=schema)
