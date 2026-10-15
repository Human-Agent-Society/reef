"""Atomic JSON storage for service-local scheduler control operations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from reef.storage.admissions import ScenarioAdmissionStore


class JSONScenarioAdmissionStore(ScenarioAdmissionStore):
    """Persist control operations by name independently of admitted instances."""

    def __init__(self, directory: Path | None, *, tenant_id_map: Mapping[str, int] | None = None) -> None:
        self.directory = directory
        self.tenant_id_map = dict(tenant_id_map or {})
        if any(
            isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**63 - 1
            for value in self.tenant_id_map.values()
        ):
            raise ValueError("tenant_id_map values must be positive integers")
        if len(set(self.tenant_id_map.values())) != len(self.tenant_id_map):
            raise ValueError("tenant_id_map must not merge different tenants")

    @property
    def durable(self) -> bool:
        return self.directory is not None

    def read(self, scenario_name: str) -> object:
        if self.directory is None:
            return None
        key = hashlib.sha256(scenario_name.encode()).hexdigest()
        path = self.directory / f"{key}.json"
        try:
            value = json.loads(path.read_text())
            if isinstance(value, dict) and isinstance(value.get("tenant_id"), str):
                old_id = value["tenant_id"]
                if old_id not in self.tenant_id_map:
                    raise ValueError("legacy scenario tenant ID requires an explicit tenant_id_map entry")
                value["tenant_id"] = self.tenant_id_map[old_id]
                self.write(scenario_name, value)
            return value
        except FileNotFoundError:
            return None

    def write(self, scenario_name: str, value: Mapping[str, object]) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(scenario_name.encode()).hexdigest()
        path = self.directory / f"{key}.json"
        descriptor, temporary = tempfile.mkstemp(dir=self.directory, prefix=".assignment-")
        try:
            with os.fdopen(descriptor, "w") as output:
                json.dump(dict(value), output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def scenario_names(self) -> tuple[str, ...]:
        if self.directory is None or not self.directory.exists():
            return ()
        names: list[str] = []
        for path in self.directory.glob("*.json"):
            value = json.loads(path.read_text())
            name = value.get("scenario_name") if isinstance(value, dict) else None
            if not isinstance(name, str) or not name:
                raise ValueError("stored scenario assignment has no name")
            names.append(name)
        return tuple(sorted(names))
