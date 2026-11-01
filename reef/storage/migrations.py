"""One-time upgrades of older record stores and persisted progress formats."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, LargeBinary, MetaData, Table, cast, func, inspect, select, update
from sqlalchemy.engine import Connection

from reef.core.atif_payloads import call_shape, canonical_json
from reef.core.errors import ReefError
from reef.core.trajectory_index import call_key, sha256_text
from reef.storage.commits import read_consumed_ids
from reef.storage.sql_records import ScopedTable, SQLRecordStore, TrajectoryNodes
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

    Training instructions are no longer records (``reef.train.instructions``): the migration deletes the rows of
    request type ``train``. A commit that consumed one records it in ``metrics["training_request"]``; an instruction
    that no commit consumed is gone, and its client files it again.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(records.name, schema=records.schema)}
    if "message_field" in columns:
        return
    deleted = connection.execute(records.delete().where(records.c.request_type == "train")).rowcount
    if deleted:
        logger.warning(
            "deleted %d training instruction record(s) of %s: instructions now live in the instructions queue",
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


def migrate_sample_columns(connection: Connection, samples: Table) -> None:
    """Add to an existing ``training_sample`` table the columns that it lacks: ``component``, ``group_key`` and
    ``record_ids_json``. The samples that were stored before have no component, no group and no records.

    Run in the caller's schema transaction, after ``create_all``; a table that ``create_all`` made has them all.
    """
    columns = {column["name"] for column in inspect(connection).get_columns(samples.name, schema=samples.schema)}
    operations = Operations(MigrationContext.configure(connection))
    for name in ("component", "group_key", "record_ids_json"):
        if name not in columns:
            column = samples.c[name]
            operations.add_column(
                samples.name,
                Column(name, column.type, nullable=column.nullable, server_default=column.server_default),
                schema=samples.schema,
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


def migrate_uploaded_trajectories(connection: Connection, samples: Table, train_counts: Table, commits: Table) -> None:
    """Move the rows of the old ``uploaded_trajectory`` table into ``samples``, then drop the old table.

    Run in the caller's schema transaction, after ``create_all`` made ``samples`` and ``train_counts``. Each
    trajectory keeps its ID, its content and its upload order. For each trainer, its row in ``train_counts`` is
    the number of that trainer's commits in ``commits`` that list the old consumed ID,
    ``trajectory:<trajectory_id>``: the commit's ``component``, or ``""`` for a commit that names none. A schema
    without the old table is left unchanged.
    """
    schema = samples.schema
    if not inspect(connection).has_table("uploaded_trajectory", schema=schema):
        return
    legacy = Table("uploaded_trajectory", MetaData(), schema=schema, autoload_with=connection)
    scoped = "storage_id" in legacy.c
    counted: dict[tuple[str | None, str, str], int] = defaultdict(int)
    for row in connection.execute(select(commits)).mappings():
        commit = json.loads(row["commit_json"])
        component = commit.get("component") or ""
        for consumed in read_consumed_ids(commit.get("record_progress", {}), context="commit"):
            if consumed.startswith("trajectory:"):
                counted[(row["storage_id"] if scoped else None, consumed.removeprefix("trajectory:"), component)] += 1
    after_sequence = 0
    while True:
        rows = (
            connection.execute(
                select(legacy).where(legacy.c.sequence > after_sequence).order_by(legacy.c.sequence).limit(256)
            )
            .mappings()
            .all()
        )
        if not rows:
            break
        for row in rows:
            scope = {"storage_id": row["storage_id"]} if scoped else {}
            connection.execute(
                samples.insert().values(
                    **scope,
                    sample_id=row["trajectory_id"],
                    kind="trajectory",
                    split="train",
                    content_json=row["content_json"],
                    content_sha256=row["content_sha256"],
                )
            )
        after_sequence = rows[-1]["sequence"]
    for (storage_id, sample_id, component), count in counted.items():
        scope = {} if storage_id is None else {"storage_id": storage_id}
        connection.execute(
            train_counts.insert().values(**scope, sample_id=sample_id, component=component, train_count=count)
        )
    legacy.drop(connection)


def legacy_task_trained(connection: Connection, schema: str | None) -> Table | None:
    """The old ``task_trained`` table, the trained marks of the task stores, or None when the schema has none."""
    if not inspect(connection).has_table("task_trained", schema=schema):
        return None
    return Table("task_trained", MetaData(), schema=schema, autoload_with=connection)


def migrate_imported_tasks(
    connection: Connection, samples: Table, train_counts: Table, commits: Table, published: Table
) -> None:
    """Move the tasks of the old ``imported-tasks`` task store into ``samples``, then delete them from the task tables.

    Run in the caller's schema transaction, after ``create_all``. ``published`` is the ``published_task`` table of
    the task stores; the trained marks come from the old ``task_trained`` table. Each task keeps its name as its sample ID,
    its split, the digest of its files, and its publication order after the samples already stored. A trained
    task gets a train count of 1 for each commit of a trainer (the commit's ``component``, or ``""``) whose
    consumed IDs list a rollout of the task, ``rollout:<task use>`` whose task use ends in ``/<name>``. A trained
    task that no commit lists gets the count 1 for ``""``. The task files stay in the task file store. Raise
    ``ReefError`` when a task has the ID of a stored sample, since the store keeps one ID once.
    """
    store = published.c.store_name == "imported-tasks"
    rows = connection.execute(select(published).where(store).order_by(published.c.sequence)).mappings().all()
    if not rows:
        return
    scoped = "storage_id" in published.c
    trained = legacy_task_trained(connection, published.schema)
    trained_names = (
        set()
        if trained is None
        else {
            (row["storage_id"] if scoped else None, row["task_name"])
            for row in connection.execute(select(trained).where(trained.c.store_name == "imported-tasks")).mappings()
        }
    )
    counted: dict[tuple[str | None, str, str], int] = defaultdict(int)
    for row in connection.execute(select(commits)).mappings():
        commit = json.loads(row["commit_json"])
        storage_id = row["storage_id"] if scoped else None
        for consumed in read_consumed_ids(commit.get("record_progress", {}), context="commit"):
            name = consumed.rpartition("/")[2]
            if consumed.startswith("rollout:") and (storage_id, name) in trained_names:
                counted[(storage_id, name, commit.get("component") or "")] += 1
    for storage_id, name in trained_names:
        if not any(key[:2] == (storage_id, name) for key in counted):
            counted[(storage_id, name, "")] = 1
    for row in rows:
        storage_id = row["storage_id"] if scoped else None
        scope = {} if storage_id is None else {"storage_id": storage_id}
        stored = select(samples.c.sample_id).where(samples.c.sample_id == row["task_name"])
        if storage_id is not None:
            stored = stored.where(samples.c.storage_id == storage_id)
        if connection.execute(stored).first() is not None:
            raise ReefError(f"imported task {row['task_name']!r} has the ID of an uploaded trajectory of its scenario")
        content = json.dumps({"files_digest": row["files_digest"]}, separators=(",", ":"), sort_keys=True)
        connection.execute(
            samples.insert().values(
                **scope,
                sample_id=row["task_name"],
                kind="task",
                split=row["split"],
                content_json=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            )
        )
    for (storage_id, name, component), count in counted.items():
        scope = {} if storage_id is None else {"storage_id": storage_id}
        connection.execute(
            train_counts.insert().values(**scope, sample_id=name, component=component, train_count=count)
        )
    if trained is not None:
        connection.execute(trained.delete().where(trained.c.store_name == "imported-tasks"))
    connection.execute(published.delete().where(store))


def migrate_generated_task_marks(connection: Connection, train_counts: Table) -> None:
    """Turn the trained marks of the generated-task stores into train counts, then drop the old ``task_trained`` table.

    Run in the caller's schema transaction, after ``migrate_imported_tasks``. A mark of task ``n`` in the store
    ``generated-tasks`` is a count of 1 of sample ``generated-tasks/<n>`` for the scenario's only trainer (``""``);
    a mark in ``<c>/generated-tasks`` is a count of 1 of sample ``<c>:generated-tasks/<n>`` for the component
    ``c``. The generated-task stage stores those samples on its next start, and they then read as trained. A
    count that exists stays. A schema without the old table is left unchanged.
    """
    trained = legacy_task_trained(connection, train_counts.schema)
    if trained is None:
        return
    scoped = "storage_id" in trained.c
    for row in connection.execute(select(trained).where(trained.c.store_name.like("%generated-tasks"))).mappings():
        component, separator, store_name = str(row["store_name"]).rpartition("/")
        if store_name != "generated-tasks" or (separator and not component):
            continue
        sample_id = (
            f"{component}:generated-tasks/{row['task_name']}" if component else f"generated-tasks/{row['task_name']}"
        )
        scope = {"storage_id": row["storage_id"]} if scoped else {}
        exists = select(train_counts.c.sample_id).where(
            train_counts.c.sample_id == sample_id,
            train_counts.c.component == component,
            *(train_counts.c[name] == value for name, value in scope.items()),
        )
        if connection.execute(exists).first() is None:
            connection.execute(
                train_counts.insert().values(**scope, sample_id=sample_id, component=component, train_count=1)
            )
    trained.drop(connection)


def migrate_scenario_settings(connection: Connection, settings: Table) -> None:
    """Delete the ``model_config`` rows of ``settings``, the ``scenario_setting`` table: the per-scenario model
    override that Reef no longer has (RFC #1062). A scenario that had one calls the deployment's model after this.

    Run in the caller's schema transaction, after ``create_all``. The other settings stay as they are.
    """
    connection.execute(settings.delete().where(settings.c.name == "model_config"))
