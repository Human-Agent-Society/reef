"""Versioned runtime configuration for one Reef service process.

Only this manager writes configuration. Consumers keep immutable snapshots
for the lifetime of an operation; submitted patches become active at an
owner-selected boundary. SQLite makes acceptance and activation recoverable.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from reef.core.errors import ReefError


class ConfigConflict(ReefError):
    """The caller's configuration revision is stale."""


class ConfigDeferred(Exception):
    """The owner must finish existing work before applying this update."""


class ConfigValidator(Protocol):
    def __call__(self, values: Mapping[str, Any], /) -> None:
        """Validate the complete candidate without changing runtime state."""
        ...


class ConfigAction(Protocol):
    def __call__(self) -> None:
        """Activate prepared state or discard its unused resources."""
        ...


@dataclass(frozen=True)
class ConfigSnapshot:
    scope: str
    revision: int
    values: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedConfigChange:
    """Prepared resources; activation must only swap already validated state.

    The owner excludes operations across prepare/persist/activate. Resource
    cleanup belongs in discard if persistence fails, or after activation.
    """

    activate: ConfigAction
    discard: ConfigAction = lambda: None


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _merge(values: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(values)
    for key, value in patch.items():
        merged[key] = (
            _merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


class ConfigManager:
    """Authoritative active snapshots and FIFO updates, scoped by owner.

    Validators are pure and registered by the component that owns a scope.
    A revision identifies a submitted update (including failed updates);
    active_revision identifies the snapshot operations actually use.
    """

    def __init__(self, database: Path | None = None) -> None:
        if database is not None:
            database.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(database) if database is not None else ":memory:", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS configuration (scope TEXT PRIMARY KEY, state TEXT NOT NULL)")
        self._db.commit()
        self._lock = RLock()
        self._validators: dict[str, ConfigValidator] = {}

    def _read(self, scope: str) -> dict[str, Any]:
        row = self._db.execute("SELECT state FROM configuration WHERE scope=?", (scope,)).fetchone()
        if row is None:
            raise ValueError(f"unknown configuration scope {scope!r}")
        return json.loads(row[0])

    def _write(self, scope: str, state: dict[str, Any]) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO configuration(scope, state) VALUES (?, ?)",
                (scope, json.dumps(state, allow_nan=False)),
            )

    def register(
        self,
        scope: str,
        initial: Mapping[str, Any],
        validator: ConfigValidator,
        *,
        restart_values: Mapping[str, Any] | None = None,
    ) -> ConfigSnapshot:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM configuration WHERE scope=?", (scope,)).fetchone()
            if row is None:
                initial_values = json.loads(json.dumps(dict(initial), allow_nan=False))
                validator(initial_values)
                self._write(scope, {"revision": 0, "active_revision": 0, "active": initial_values, "updates": []})
            else:
                state = self._read(scope)
                # Static fields come from the deployment at boot. Managed
                # fields retain their persisted values. Never silently rebase
                # pending updates onto a different startup configuration.
                active = _merge(state["active"], restart_values or {})
                validator(active)
                if active != state["active"]:
                    if scope in self._validators:
                        raise ConfigConflict("startup configuration cannot change in a running scope")
                    if any(update["status"] == "pending" for update in state["updates"]):
                        raise ConfigConflict(
                            "apply pending configuration updates before restarting with different static settings"
                        )
                    revision = state["revision"] + 1
                    state.update(revision=revision, active_revision=revision, active=active)
                    state["updates"].append(
                        {
                            "id": str(revision),
                            "revision": revision,
                            "patch": dict(restart_values or {}),
                            "status": "applied",
                            "source": "startup",
                        }
                    )
                    self._write(scope, state)
            self._validators[scope] = validator
            return self.snapshot(scope)

    def snapshot(self, scope: str) -> ConfigSnapshot:
        with self._lock:
            state = self._read(scope)
            return ConfigSnapshot(scope, state["active_revision"], _freeze(state["active"]))

    def status(self, scope: str) -> dict[str, Any]:
        with self._lock:
            return {"scope": scope, **self._read(scope)}

    def _desired(self, state: dict[str, Any]) -> dict[str, Any]:
        values = state["active"]
        for update in state["updates"]:
            if update["status"] == "pending":
                values = _merge(values, update["patch"])
        return values

    def desired(self, scope: str) -> Mapping[str, Any]:
        with self._lock:
            return _freeze(self._desired(self._read(scope)))

    def submit(self, scope: str, patch: Mapping[str, Any], *, expected_revision: int | None = None) -> dict[str, Any]:
        with self._lock:
            state = self._read(scope)
            if expected_revision is not None and expected_revision != state["revision"]:
                raise ConfigConflict(f"configuration revision is {state['revision']}, not {expected_revision}")
            if not patch:
                raise ValueError("configuration patch must not be empty")
            detached_patch = json.loads(json.dumps(dict(patch), allow_nan=False))
            candidate = _merge(self._desired(state), detached_patch)
            self._validators[scope](candidate)
            revision = state["revision"] + 1
            update = {"id": str(revision), "revision": revision, "patch": detached_patch, "status": "pending"}
            state["revision"] = revision
            state["updates"].append(update)
            self._write(scope, state)
            return {"scope": scope, **update}

    def apply_next(self, scope: str, prepare: Callable[[ConfigSnapshot], PreparedConfigChange]) -> bool:
        """Apply one update at the caller's safe boundary; False if deferred/empty.

        Preparation failure is durable and leaves the active snapshot intact.
        A persistence failure discards the candidate and propagates, so the
        worker cannot continue under a configuration that was not recorded.
        """
        with self._lock:
            state = self._read(scope)
            update = next((item for item in state["updates"] if item["status"] == "pending"), None)
            if update is None:
                return False
            values = _merge(state["active"], update["patch"])
            snapshot = ConfigSnapshot(scope, update["revision"], _freeze(values))
            try:
                self._validators[scope](values)
                change = prepare(snapshot)
            except ConfigDeferred as exc:
                if update.get("waiting_for") != str(exc):
                    update["waiting_for"] = str(exc)
                    self._write(scope, state)
                return False
            except Exception as exc:
                update.pop("waiting_for", None)
                update.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                self._write(scope, state)
                return True
            state.update(active=values, active_revision=update["revision"])
            update["status"] = "applied"
            update.pop("waiting_for", None)
            try:
                self._write(scope, state)
            except BaseException:
                change.discard()
                raise
            change.activate()
            return True

    def close(self) -> None:
        with self._lock:
            self._db.close()
