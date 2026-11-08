"""One-time upgrades of older record stores and persisted progress formats."""

from __future__ import annotations

import json
import logging

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, MetaData, Table, Text, bindparam, delete, func, inspect, literal, select, tuple_, update
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateIndex

from reef.core.atif_payloads import call_shape, canonical_json
from reef.core.trajectory_index import call_key
from reef.storage.sql_records import SEGMENT_COLUMNS, ScopedTable, TrajectoryNodes
from reef.storage.work_queue import FINAL_STATES

logger = logging.getLogger(__name__)

#: The rows that one page of ``migrate_data_request_types`` reads.
MIGRATION_BATCH_SIZE = 1000


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


def migrate_data_request_types(connection: Connection, items: Table) -> None:
    """Add ``request_type`` to ``items``, and fill it for each record item from its content (RFC #1219).

    Run in the caller's schema transaction, after ``create_all``. A table that has ``request_type`` is left
    unchanged. A record item's content stays in its row, so the fill reads no segment.
    """
    if "request_type" in {
        column["name"] for column in inspect(connection).get_columns(items.name, schema=items.schema)
    }:
        return
    with Operations(MigrationContext.configure(connection)).batch_alter_table(
        items.name, schema=items.schema
    ) as batch:
        batch.add_column(Column("request_type", items.c.request_type.type, nullable=True))
    after = 0
    while rows := connection.execute(
        select(items.c.sequence, items.c.content_json)
        .where(items.c.kind == "record", items.c.sequence > after)
        .order_by(items.c.sequence)
        .limit(MIGRATION_BATCH_SIZE)
    ).all():
        after = rows[-1].sequence
        connection.execute(
            update(items).where(items.c.sequence == bindparam("row_sequence")).values(request_type=bindparam("value")),
            [{"row_sequence": row.sequence, "value": json.loads(row.content_json)["request_type"]} for row in rows],
        )


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
    ``IF NOT EXISTS`` checks by name, because the inspector does not list an index on an expression, such as
    ``data_item_by_evaluation``.
    """
    for index in items.indexes:
        connection.execute(CreateIndex(index, if_not_exists=True))


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


def remove_record_copies(connection: Connection, work_items: Table) -> None:
    """Delete the ``record_copy`` items of an older version, which no worker takes (RFC #1219), and the
    ``coordination_mark`` table, which kept how far each copy came.

    Run in the caller's schema transaction, after ``create_all``.
    """
    connection.execute(delete(work_items).where(work_items.c.kind == "record_copy"))
    Table("coordination_mark", MetaData(), schema=work_items.schema).drop(connection, checkfirst=True)


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
