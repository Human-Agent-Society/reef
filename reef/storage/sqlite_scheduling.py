"""Transactional SQLite persistence for scheduler metadata and bounded active indexes."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import astuple, replace
from pathlib import Path
from threading import RLock
from typing import cast

from reef.storage.scheduling import (
    ScenarioAdmission,
    SchedulerScenarioState,
    SchedulingSnapshot,
    SchedulingStore,
    SchedulingStoreConflict,
    TenantAllocation,
    TerminalState,
)


def read_scenario_admission(row: sqlite3.Row) -> ScenarioAdmission:
    """Decode the same persisted contract for recovery and historical lookups."""
    return ScenarioAdmission(
        scenario_id=row["scenario_id"],
        tenant_id=row["tenant_id"],
        service_id=row["service_id"],
        recipe=row["recipe"],
        scenario_name=row["scenario_name"],
        release_id=row["release_id"],
        model_config_ref=row["model_config_ref"],
        execution_units=row["execution_units"],
        state=cast(SchedulerScenarioState, row["state"]),
        generation=row["generation"],
        created_at=row["created_at"],
        service_config_ref=row["service_config_ref"],
        queue_sequence=row["queue_sequence"],
        idle_since=row["idle_since"],
        reclaim_before=row["reclaim_before"],
        has_started=bool(row["has_started"]),
        execution_allocated=bool(row["execution_allocated"]),
        failure_reason=row["failure_reason"],
        terminal_state=cast(TerminalState, row["terminal_state"]),
    )


class SQLiteSchedulingStore(SchedulingStore):
    """One scheduler authority with transactional version checks and incremental writes.

    Only changed records are written. Completed records stay on disk; the cached
    comparison snapshot contains active records only. No provider bodies or secrets
    belong in this database.
    """

    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        tenant_id_map: Mapping[str, int] | None = None,
        service_id_map: Mapping[str, int] | None = None,
        registration_fingerprint_map: Mapping[str, str] | None = None,
    ) -> None:
        database = str(path)
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.connection = sqlite3.connect(database, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        try:
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                existing = self.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'scheduler_metadata'"
                ).fetchone()
                if existing is None:
                    self.create_tables()
                    self.connection.execute("INSERT INTO scheduler_metadata VALUES (1, 3, 0, 1, 1)")
                else:
                    schema = self.connection.execute("SELECT schema_version FROM scheduler_metadata").fetchone()
                    if schema["schema_version"] == 1:
                        self.migrate_version_one(
                            tenant_id_map or {}, service_id_map or {}, registration_fingerprint_map or {}
                        )
                    elif schema["schema_version"] not in (2, 3):
                        raise ValueError("unsupported scheduler storage schema version")
                    if schema["schema_version"] in (1, 2):
                        self.connection.execute(
                            "UPDATE scenario_admission SET state = 'sleep', idle_since = NULL, reclaim_before = NULL "
                            "WHERE state = 'queued' AND has_started = 1 AND execution_allocated = 0"
                        )
                        self.connection.execute(
                            "UPDATE scheduler_metadata SET schema_version = 3, version = version + 1"
                        )
            self.saved_snapshot = self.load()
        except (sqlite3.Error, ValueError):
            self.connection.close()
            raise

    def create_tables(self) -> None:
        self.connection.execute(
            "CREATE TABLE scheduler_metadata (singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
            "schema_version INTEGER NOT NULL, version INTEGER NOT NULL, next_scenario_id INTEGER NOT NULL, "
            "next_queue_sequence INTEGER NOT NULL)"
        )
        self.connection.execute(
            "CREATE TABLE tenant_allocation (tenant_id INTEGER PRIMARY KEY CHECK (tenant_id > 0), "
            "allocated_execution_units INTEGER NOT NULL CHECK (allocated_execution_units > 0))"
        )
        self.connection.execute(
            "CREATE TABLE scenario_admission (scenario_id INTEGER PRIMARY KEY, tenant_id INTEGER NOT NULL, "
            "service_id INTEGER NOT NULL, recipe TEXT NOT NULL, scenario_name TEXT NOT NULL, release_id TEXT, "
            "model_config_ref TEXT, execution_units INTEGER NOT NULL CHECK (execution_units > 0), state TEXT NOT NULL, "
            "generation INTEGER NOT NULL, created_at REAL NOT NULL, service_config_ref TEXT NOT NULL, "
            "queue_sequence INTEGER NOT NULL, idle_since REAL, reclaim_before REAL, has_started INTEGER NOT NULL, "
            "execution_allocated INTEGER NOT NULL, failure_reason TEXT, terminal_state TEXT NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX scenario_admission_name ON scenario_admission(service_id, scenario_name, scenario_id DESC)"
        )
        self.connection.execute(
            "CREATE INDEX scenario_admission_active ON scenario_admission(queue_sequence) "
            "WHERE state NOT IN ('deleted', 'failed')"
        )

    def migrate_version_one(
        self,
        tenant_id_map: Mapping[str, int],
        service_id_map: Mapping[str, int],
        registration_fingerprint_map: Mapping[str, str],
    ) -> None:
        """Migrate IDs only through explicit, collision-free administrator mappings.

        Already requested version-one cleanup finishes as deletion: its remote
        archival may have happened before migration and cannot safely become reclaim.
        """
        metadata = self.connection.execute("SELECT * FROM scheduler_metadata").fetchone()
        tenants = self.connection.execute("SELECT * FROM tenant_allocation").fetchall()
        scenarios = self.connection.execute("SELECT * FROM scenario_admission ORDER BY scenario_id").fetchall()
        tenant_names = {row["tenant_id"] for row in (*tenants, *scenarios)}
        service_names = {row["service_id"] for row in scenarios}
        for label, names, mapping in (
            ("tenant_id", tenant_names, tenant_id_map),
            ("service_id", service_names, service_id_map),
        ):
            if not names.issubset(mapping):
                raise ValueError(f"schema version 1 requires an explicit {label}_map for every stored ID")
            values = list(mapping.values())
            if any(
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**63 - 1
                for value in values
            ):
                raise ValueError(f"{label}_map values must be positive signed 64-bit integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{label}_map must not merge different identities")
        converted = []
        for row in scenarios:
            active = row["state"] not in {"deleted", "failed"}
            fingerprint = registration_fingerprint_map.get(row["service_config_ref"])
            if active and fingerprint is None:
                raise ValueError("active version-one scenario requires its unchanged registration fingerprint mapping")
            converted.append(
                ScenarioAdmission(
                    scenario_id=row["scenario_id"],
                    tenant_id=tenant_id_map[row["tenant_id"]],
                    service_id=service_id_map[row["service_id"]],
                    recipe=row["recipe"],
                    scenario_name=row["scenario_name"],
                    release_id=row["release_id"],
                    model_config_ref=row["model_config_ref"],
                    execution_units=row["execution_units"],
                    state=cast(SchedulerScenarioState, row["state"]),
                    generation=row["generation"],
                    created_at=row["created_at"],
                    service_config_ref=fingerprint or row["service_config_ref"],
                    queue_sequence=row["scenario_id"],
                    has_started=row["state"] != "queued",
                    execution_allocated=row["state"] in {"starting", "admitted", "cleanup_pending"},
                    failure_reason=row["failure_reason"],
                    terminal_state=cast(TerminalState, row["terminal_state"]),
                )
            )
        self.connection.execute("DROP TABLE scenario_admission")
        self.connection.execute("DROP TABLE tenant_allocation")
        self.connection.execute("DROP TABLE scheduler_metadata")
        self.create_tables()
        next_sequence = max(metadata["next_scenario_id"], max((row.scenario_id + 1 for row in converted), default=1))
        self.connection.execute(
            "INSERT INTO scheduler_metadata VALUES (1, 2, ?, ?, ?)",
            (metadata["version"] + 1, metadata["next_scenario_id"], next_sequence),
        )
        self.connection.executemany(
            "INSERT INTO tenant_allocation VALUES (?, ?)",
            ((tenant_id_map[row["tenant_id"]], row["execution_units"]) for row in tenants),
        )
        self.connection.executemany(
            "INSERT INTO scenario_admission VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (astuple(value) for value in converted),
        )

    def load(self) -> SchedulingSnapshot:
        with self.lock, self.connection:
            self.connection.execute("BEGIN")
            metadata = self.connection.execute("SELECT * FROM scheduler_metadata").fetchone()
            tenants = tuple(
                TenantAllocation(row["tenant_id"], row["allocated_execution_units"])
                for row in self.connection.execute("SELECT * FROM tenant_allocation ORDER BY tenant_id")
            )
            scenarios = tuple(
                read_scenario_admission(row)
                for row in self.connection.execute(
                    "SELECT * FROM scenario_admission WHERE state NOT IN ('deleted', 'failed') ORDER BY queue_sequence"
                )
            )
            self.saved_snapshot = SchedulingSnapshot(
                metadata["version"], metadata["next_scenario_id"], tenants, scenarios, metadata["next_queue_sequence"]
            )
            return self.saved_snapshot

    def save(self, snapshot: SchedulingSnapshot) -> SchedulingSnapshot:
        with self.lock:
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                cursor = self.connection.execute(
                    "UPDATE scheduler_metadata SET version = version + 1, next_scenario_id = ?, next_queue_sequence = ? "
                    "WHERE singleton = 1 AND version = ?",
                    (snapshot.next_scenario_id, snapshot.next_queue_sequence, snapshot.version),
                )
                if cursor.rowcount != 1:
                    raise SchedulingStoreConflict("scheduler state changed in another authority")
                previous = self.saved_snapshot
                if previous.version != snapshot.version:
                    raise SchedulingStoreConflict("reload the current snapshot before saving")
                old_tenants = {value.tenant_id: value for value in previous.tenants}
                new_tenants = {value.tenant_id: value for value in snapshot.tenants}
                self.connection.executemany(
                    "DELETE FROM tenant_allocation WHERE tenant_id = ?",
                    ((identifier,) for identifier in old_tenants.keys() - new_tenants.keys()),
                )
                self.connection.executemany(
                    "INSERT OR REPLACE INTO tenant_allocation VALUES (?, ?)",
                    (
                        astuple(value)
                        for identifier, value in new_tenants.items()
                        if old_tenants.get(identifier) != value
                    ),
                )
                old_scenarios = {value.scenario_id: value for value in previous.scenarios}
                self.connection.executemany(
                    "INSERT OR REPLACE INTO scenario_admission VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (astuple(value) for value in snapshot.scenarios if old_scenarios.get(value.scenario_id) != value),
                )
            self.saved_snapshot = replace(
                snapshot,
                version=snapshot.version + 1,
                scenarios=tuple(value for value in snapshot.scenarios if value.state not in {"deleted", "failed"}),
            )
            return self.saved_snapshot

    def get_scenario_admission(self, scenario_id: int) -> ScenarioAdmission | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM scenario_admission WHERE scenario_id = ?", (scenario_id,)
            ).fetchone()
            return None if row is None else read_scenario_admission(row)

    def find_latest_scenario_admission(self, service_id: int, scenario_name: str) -> ScenarioAdmission | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM scenario_admission WHERE service_id = ? AND scenario_name = ? ORDER BY scenario_id DESC LIMIT 1",
                (service_id, scenario_name),
            ).fetchone()
            return None if row is None else read_scenario_admission(row)

    def close(self) -> None:
        with self.lock:
            self.connection.close()
