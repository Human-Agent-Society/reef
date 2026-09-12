"""Durable per-scenario history for a bridge that trains several adapters.

The bridge's marker records one job at a time. When several scenarios share
the training group, each also needs its own publication history — the
serving runtime load ID is engine-global and advances whenever *any*
scenario publishes, so a scenario's staleness is the number of *its own*
publications since a rollout, not the global sequence gap — and the
checkpoint that last captured its adapter, which retention must not delete
while the scenario is alive.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reef.runtime.names import SCENARIO_HISTORY_FILENAME
from reef.runtime.runtime_load_id import RuntimeLoadId
from reef.runtime.training_job.durable_io import read_json, write_json

HISTORY_FILENAME = SCENARIO_HISTORY_FILENAME
HISTORY_FORMAT = 1


class ScenarioHistory:
    """Publication history and latest checkpoint per training scenario."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = {}
        value = read_json(path)
        if value is None:
            return
        if not isinstance(value, Mapping) or value.get("format") != HISTORY_FORMAT:
            raise RuntimeError(f"invalid scenario history: {path}")
        scenarios = value.get("scenarios")
        if not isinstance(scenarios, Mapping):
            raise RuntimeError(f"invalid scenario history: {path}")
        for scenario, entry in scenarios.items():
            if not isinstance(scenario, str) or not scenario or not isinstance(entry, Mapping):
                raise RuntimeError(f"invalid scenario history entry: {path}")
            publications = entry.get("publications", [])
            if not isinstance(publications, list) or not all(isinstance(item, str) for item in publications):
                raise RuntimeError(f"invalid scenario history publications: {path}")
            self._entries[scenario] = {
                "publications": list(publications),
                "adapter": entry.get("adapter"),
                "rollout_id": entry.get("rollout_id"),
                "steps": int(entry.get("steps", 0)),
            }

    @property
    def path(self) -> Path:
        return self._path

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry(self, scenario: str) -> Mapping[str, Any] | None:
        entry = self._entries.get(scenario)
        return None if entry is None else dict(entry)

    def adapter(self, scenario: str) -> str | None:
        entry = self._entries.get(scenario)
        adapter = None if entry is None else entry.get("adapter")
        return adapter if isinstance(adapter, str) and adapter else None

    def last_publication(self, scenario: str) -> str | None:
        entry = self._entries.get(scenario)
        if entry is None or not entry["publications"]:
            return None
        return str(entry["publications"][-1])

    def lag(self, scenario: str, producing: RuntimeLoadId) -> int | None:
        """How many of ``scenario``'s publications postdate ``producing``.

        ``None`` means the producing version belongs to another incarnation
        (a previous training-group lifetime), which is never admissible.
        """
        entry = self._entries.get(scenario)
        publications = [] if entry is None else entry["publications"]
        lag = 0
        for value in publications:
            published = RuntimeLoadId.parse(value)
            if published.incarnation != producing.incarnation:
                continue
            if published.sequence > producing.sequence:
                lag += 1
        if publications:
            latest = RuntimeLoadId.parse(publications[-1])
            if latest.incarnation != producing.incarnation:
                return None
        return lag

    def protected_rollouts(self) -> set[int]:
        """Global rollout ids whose checkpoints carry a scenario's latest adapter."""
        return {
            int(entry["rollout_id"])
            for entry in self._entries.values()
            if isinstance(entry.get("rollout_id"), int) and not isinstance(entry["rollout_id"], bool)
        }

    def record_checkpoint(self, scenario: str, rollout_id: int) -> None:
        entry = self._entries.setdefault(
            scenario, {"publications": [], "adapter": None, "rollout_id": None, "steps": 0}
        )
        entry["rollout_id"] = int(rollout_id)
        entry["steps"] = int(entry.get("steps", 0)) + 1
        self._write()

    def record_publication(self, scenario: str, runtime_load_id: str, adapter: str) -> None:
        entry = self._entries.setdefault(
            scenario, {"publications": [], "adapter": None, "rollout_id": None, "steps": 0}
        )
        if not entry["publications"] or entry["publications"][-1] != runtime_load_id:
            entry["publications"].append(runtime_load_id)
        entry["adapter"] = adapter
        self._write()

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            scenario: {
                "runtime_load_id": self.last_publication(scenario),
                "adapter": self.adapter(scenario),
                "publications": len(entry["publications"]),
                "rollout_id": entry.get("rollout_id"),
                "steps": entry.get("steps", 0),
            }
            for scenario, entry in sorted(self._entries.items())
        }

    def _write(self) -> None:
        write_json(self._path, {"format": HISTORY_FORMAT, "scenarios": self._entries})


def history_path(hf_template: str) -> Path:
    """The history sits beside the job marker, in the HF checkpoint directory."""
    return Path(hf_template.format(rollout_id=0)).expanduser().parent / HISTORY_FILENAME


__all__ = ["HISTORY_FILENAME", "HISTORY_FORMAT", "ScenarioHistory", "history_path"]
