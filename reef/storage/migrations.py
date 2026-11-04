"""One-time upgrades of older record stores and persisted progress formats."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, LargeBinary, MetaData, Table, Text, cast, func, inspect, literal, select, tuple_, update
from sqlalchemy.engine import Connection

from reef.core.atif_payloads import call_shape, canonical_json
from reef.core.trajectory_index import call_key, sha256_text
from reef.storage.commits import read_consumed_ids
from reef.storage.sql_records import SEGMENT_COLUMNS, ScopedTable, SQLRecordStore, TrajectoryNodes
from reef.storage.work_queue import FINAL_STATES

logger = logging.getLogger(__name__)


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


def migrate_record_trajectories(connection: Connection, records: Table, nodes: Table) -> None:
    """Give each stored record with a message list its trajectory node, in ``nodes`` (RFC #1054).

    Run in the caller's schema transaction, after ``migrate_record_storage`` and after ``create_all`` made ``nodes``.
    A record table that has ``message_field`` is left unchanged. Otherwise add ``message_field`` and
    ``payload_sha256``, then rewrite each record whose payload has a message list, oldest first and 256 at a time,
    as an append writes it: the record's node keeps the payload as ATIF, with the entries after its parent's, and
    the row keeps the payload's hash. Its ``body_bytes`` counts the new row. Each node goes into the namespace of its
    record's store generation (``storage_id``), or of the database when the table has no such column.

    Training requests are no longer records (``reef.train.training_requests``): the migration deletes the rows of
    request type ``train``. A commit that consumed one records it in ``metrics["training_request"]``; a request
    that no commit consumed is gone, and its client files it again.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(records.name, schema=records.schema)}
    if "message_field" in columns:
        return
    deleted = connection.execute(records.delete().where(records.c.request_type == "train")).rowcount
    if deleted:
        logger.warning(
            "deleted %d training request record(s) of %s: requests now live in the training request queue",
            deleted,
            records.fullname,
        )
    operations = Operations(MigrationContext.configure(connection))
    for name in ("message_field", "payload_sha256"):
        operations.add_column(records.name, Column(name, records.c[name].type, nullable=True), schema=records.schema)
    scope_names = ("storage_id",) if "storage_id" in records.c else ()
    after_sequence = 0
    while True:
        rows = (
            connection.execute(
                select(records).where(records.c.sequence > after_sequence).order_by(records.c.sequence).limit(256)
            )
            .mappings()
            .all()
        )
        if not rows:
            return
        after_sequence = rows[-1]["sequence"]
        for row in rows:
            payload = json.loads(row["payload_json"])
            key = call_key(payload)
            shape = call_shape(payload)
            if key is None or shape is None:
                continue
            encoded = SQLRecordStore.row_content(row)._replace(
                payload_json="{}",
                message_field=shape.message_field,
                payload_sha256=sha256_text(canonical_json(payload)),
            )
            connection.execute(
                update(records)
                .where(records.c.sequence == row["sequence"])
                .values(
                    payload_json=encoded.payload_json,
                    message_field=encoded.message_field,
                    payload_sha256=encoded.payload_sha256,
                    body_bytes=encoded.body_bytes,
                )
            )
            scope = {name: str(row[name]) for name in scope_names}
            TrajectoryNodes(ScopedTable(nodes, scope)).add(
                connection, int(row["sequence"]), row["scenario"], row["agent_record_id"], payload, key
            )


def migrate_data_segments(connection: Connection, items: Table, nodes: Table) -> None:
    """Let the rows of ``items`` and ``nodes`` keep their content in a segment (``reef.storage.segments``).

    Run in the caller's schema transaction, after ``create_all``. Add the ``SEGMENT_COLUMNS`` and their index to a
    table that has no ``segment_id``, and let a node's ``entries_json`` and ``call_json`` be NULL. A table that has
    ``segment_id`` is left unchanged.
    """
    inspector = inspect(connection)
    operations = Operations(MigrationContext.configure(connection))
    for table in (items, nodes):
        if "segment_id" in {column["name"] for column in inspector.get_columns(table.name, schema=table.schema)}:
            continue
        with operations.batch_alter_table(table.name, schema=table.schema) as batch:
            for name in SEGMENT_COLUMNS:
                batch.add_column(Column(name, table.c[name].type, nullable=True))
            if table is nodes:
                batch.alter_column("entries_json", existing_type=table.c.entries_json.type, nullable=True)
                batch.alter_column("call_json", existing_type=table.c.call_json.type, nullable=True)
        index = next(index for index in table.indexes if "segment_id" in index.columns)
        index.create(connection)


def migrate_data_uses(connection: Connection, items: Table, uses: Table) -> None:
    """Give each use of an item by a scenario its own sequence and its split (RFC #1140).

    Run in the caller's schema transaction, after ``create_all``. A use table that has ``sequence`` is left
    unchanged. Otherwise read its uses, drop it, create it again, and insert the uses in the order of their items,
    each with its item's split for the scenario.
    """
    if "sequence" in {column["name"] for column in inspect(connection).get_columns(uses.name, schema=uses.schema)}:
        return
    old = Table(uses.name, MetaData(), schema=uses.schema, autoload_with=connection)
    rows = connection.execute(
        select(old.c.owner, old.c.item_id, old.c.scenario, items.c.metadata_json)
        .join(items, (items.c.owner == old.c.owner) & (items.c.item_id == old.c.item_id))
        .order_by(items.c.sequence)
    ).all()
    old.drop(connection)
    uses.create(connection)
    if rows:
        connection.execute(
            uses.insert(),
            [
                {
                    "owner": row.owner,
                    "item_id": row.item_id,
                    "scenario": row.scenario,
                    "split": json.loads(row.metadata_json)["scenarios"].get(row.scenario, "train"),
                }
                for row in rows
            ],
        )


def migrate_data_indexes(connection: Connection, items: Table) -> None:
    """Create each index of ``items`` that the table does not have yet, such as ``data_item_by_kind``.

    Run in the caller's schema transaction, after ``create_all``, which makes no index for a table that exists.
    """
    for index in items.indexes:
        index.create(connection, checkfirst=True)


def migrate_data_trajectories(connection: Connection, items: Table, nodes: Table) -> None:
    """Give each stored record item with a message list its trajectory node, in ``nodes`` (RFC #1054).

    Run in the caller's schema transaction, after ``create_all`` made ``nodes``. An item table that has
    ``message_field`` is left unchanged. Otherwise add ``message_field``, then rewrite each record item whose
    payload has a message list, in arrival order and 256 at a time, as ``create`` writes it: its content keeps an
    empty payload, and the item's node, in the namespace of its owner, keeps the payload as ATIF, with the entries
    after its parent's.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(items.name, schema=items.schema)}
    if "message_field" in columns:
        return
    operations = Operations(MigrationContext.configure(connection))
    operations.add_column(
        items.name, Column("message_field", items.c.message_field.type, nullable=True), schema=items.schema
    )
    after_sequence = 0
    while True:
        rows = (
            connection.execute(
                select(items.c.sequence, items.c.owner, items.c.item_id, items.c.metadata_json, items.c.content_json)
                .where(items.c.kind == "record", items.c.sequence > after_sequence)
                .order_by(items.c.sequence)
                .limit(256)
            )
            .mappings()
            .all()
        )
        if not rows:
            return
        after_sequence = rows[-1]["sequence"]
        for row in rows:
            content = json.loads(row["content_json"])
            payload = content["payload"]
            key = call_key(payload)
            shape = call_shape(payload)
            if key is None or shape is None:
                continue
            content["payload"] = {}
            connection.execute(
                update(items)
                .where(items.c.sequence == row["sequence"])
                .values(content_json=canonical_json(content), message_field=shape.message_field)
            )
            TrajectoryNodes(ScopedTable(nodes, {"owner": row["owner"]})).add(
                connection,
                int(row["sequence"]),
                json.loads(row["metadata_json"])["origin"]["scenario"],
                row["item_id"],
                payload,
                key,
            )


def migrate_work_item_columns(connection: Connection, work_items: Table) -> None:
    """Add to an existing ``work_item`` table the columns that it lacks: ``started_at``, ``ended_at`` and
    ``metadata_json``. A final item that was stored before ended at its last change (``changed_at``), since a
    final item did not change; the start of an earlier item is not known, so it stays NULL. Earlier items have no
    metadata.

    Run in the caller's schema transaction, after ``create_all``; a table that ``create_all`` made has them all.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(work_items.name, schema=work_items.schema)}
    operations = Operations(MigrationContext.configure(connection))
    for name in ("started_at", "ended_at", "metadata_json"):
        if name not in columns:
            column = work_items.c[name]
            operations.add_column(
                work_items.name,
                Column(name, column.type, nullable=column.nullable, server_default=column.server_default),
                schema=work_items.schema,
            )
    if "ended_at" not in columns:
        connection.execute(
            update(work_items)
            .where(work_items.c.state.in_(sorted(FINAL_STATES)))
            .values(ended_at=work_items.c.changed_at)
        )


def migrate_training_request_items(connection: Connection, work_items: Table) -> None:
    """Move the queued training requests of an older version to their new queue and kind: the queue
    ``instructions`` becomes ``training_requests``, and the kind ``train_instruction:<scenario>`` becomes
    ``training_request:<scenario>`` (``reef.train.training_requests``). Items of other queues stay as they are.

    Run in the caller's schema transaction, after ``create_all``.
    """
    old = "train_instruction:"
    connection.execute(
        update(work_items)
        .where(work_items.c.queue == "instructions")
        .values(
            queue="training_requests",
            kind=literal("training_request:", Text).concat(func.substr(work_items.c.kind, len(old) + 1)),
        )
    )


def migrate_scenario_settings(connection: Connection, settings: Table) -> None:
    """Delete the ``model_config`` rows of ``settings``, the ``scenario_setting`` table: the per-scenario model
    override that Reef no longer has (RFC #1062). A scenario that had one calls the deployment's model after this.

    Run in the caller's schema transaction, after ``create_all``. The other settings stay as they are.
    """
    connection.execute(settings.delete().where(settings.c.name == "model_config"))


def migrate_training_samples(connection: Connection, train_counts: Table) -> None:
    """Drop the sample store of RFC #1009 (``training_sample``) and the records that its samples held
    (``sample_record``): every training sample is a data item (RFC #1140).

    Run in the caller's schema transaction. Delete the train counts of the samples that a trainer's stage made: the
    stage makes them again, as items, from the records that no commit consumed. The counts of uploads and imported
    tasks name their items, and stay. A schema without the old tables is left unchanged.
    """
    schema = train_counts.schema
    inspector = inspect(connection)
    if inspector.has_table("training_sample", schema=schema):
        legacy = Table("training_sample", MetaData(), schema=schema, autoload_with=connection)
        made = legacy.c.component.is_not(None)
        if "storage_id" in legacy.c:
            key = tuple_(train_counts.c.storage_id, train_counts.c.sample_id)
            connection.execute(
                train_counts.delete().where(key.in_(select(legacy.c.storage_id, legacy.c.sample_id).where(made)))
            )
        else:
            connection.execute(
                train_counts.delete().where(train_counts.c.sample_id.in_(select(legacy.c.sample_id).where(made)))
            )
        legacy.drop(connection)
    if inspector.has_table("sample_record", schema=schema):
        Table("sample_record", MetaData(), schema=schema, autoload_with=connection).drop(connection)


def migrate_manual_buffers(connection: Connection, schema: str | None) -> None:
    """Drop the old ``manual_buffer`` table: a training request names its samples (RFC #1140), and its queued
    buffers are lost. Run in the caller's schema transaction. A schema without the table is left unchanged."""
    if inspect(connection).has_table("manual_buffer", schema=schema):
        Table("manual_buffer", MetaData(), schema=schema, autoload_with=connection).drop(connection)


def migrate_published_tasks(connection: Connection, schema: str | None) -> None:
    """Drop the old ``published_task`` table: a published task is a candidate with a split, and each published
    candidate already has its split. Run in the caller's schema transaction. A schema without the table is left
    unchanged."""
    if inspect(connection).has_table("published_task", schema=schema):
        Table("published_task", MetaData(), schema=schema, autoload_with=connection).drop(connection)
