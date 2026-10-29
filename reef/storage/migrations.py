"""One-time upgrades of older record stores and persisted progress formats."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Column, LargeBinary, MetaData, Table, cast, func, inspect, select
from sqlalchemy.engine import Connection

from reef.core.errors import ReefError
from reef.storage.commits import read_consumed_ids


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


def legacy_model_configs(directory: Path) -> dict[str, tuple[Mapping[str, object] | None, Path]]:
    """Read the model configuration files that older versions kept beside the records.

    The result maps each scenario key (the SHA-256 of the scenario name) to the
    stored value and its file. A ``null`` value selected the recipe's model.
    """
    configs: dict[str, tuple[Mapping[str, object] | None, Path]] = {}
    for path in sorted(directory.glob("*-model.json")):
        if re.fullmatch(r"[0-9a-f]{64}-model\.json", path.name) is None:
            continue
        value = json.loads(path.read_text())
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"model configuration file {path} must hold an object or null")
        configs[path.name.removesuffix("-model.json")] = (value, path)
    return configs


def archive_imported_files(directory: Path, paths: Sequence[Path]) -> tuple[str, ...]:
    """Move files whose content is now in the storage backend under ``archived/``; return the new locations."""
    if not paths:
        return ()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    destination = directory / "archived" / f"imported-{stamp}-{uuid.uuid4().hex}"
    destination.mkdir(parents=True, exist_ok=True)
    moved = []
    for path in paths:
        target = destination / path.name
        shutil.move(str(path), str(target))
        moved.append(str(target))
    return tuple(moved)
