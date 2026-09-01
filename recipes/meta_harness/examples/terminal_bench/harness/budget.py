"""Persistent observed-cost guard for resumable live experiments."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path


class SpendCapReached(RuntimeError):
    """The recorded target-model spend reached the operator's local cap."""


class ObservedCostLedger:
    """Idempotently account completed trials and stop before the next one at cap."""

    def __init__(self, path: Path, max_observed_cost_usd: float) -> None:
        if not math.isfinite(max_observed_cost_usd) or max_observed_cost_usd <= 0:
            raise ValueError("observed cost cap must be finite and positive")
        self.path = Path(path).resolve()
        self.cap = float(max_observed_cost_usd)
        if self.path.is_file():
            state = self._read()
            spent = float(state.get("observed_cost_usd", 0.0))
            if not math.isfinite(spent) or spent < 0:
                raise RuntimeError(f"invalid recorded spend in observed-cost ledger: {self.path}")
            if self.cap < spent:
                raise ValueError(f"observed cost cap {self.cap:.2f} is below already recorded spend {spent:.2f}")
            state["max_observed_cost_usd"] = self.cap
            self._write(state)
        else:
            self._write(
                {
                    "schema_version": 1,
                    "max_observed_cost_usd": self.cap,
                    "observed_cost_usd": 0.0,
                    "trials": {},
                }
            )

    @property
    def observed_cost_usd(self) -> float:
        return float(self._read()["observed_cost_usd"])

    def before_trial(self, identity: str) -> None:
        state = self._read()
        if identity in state["trials"]:
            return
        if float(state["observed_cost_usd"]) >= self.cap:
            raise SpendCapReached(f"recorded target-model cost reached ${self.cap:.2f}; no new trial was started")

    def record_trial(self, identity: str, cost_usd: float) -> None:
        if not math.isfinite(cost_usd) or cost_usd < 0:
            raise ValueError("recorded trial cost must be finite and non-negative")
        state = self._read()
        previous = state["trials"].get(identity)
        if previous is not None:
            if float(previous) != float(cost_usd):
                raise RuntimeError(f"trial cost changed across restart: {identity}")
            return
        state["trials"][identity] = float(cost_usd)
        state["observed_cost_usd"] = sum(float(value) for value in state["trials"].values())
        self._write(state)

    def _read(self) -> dict:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("trials"), dict):
            raise RuntimeError(f"invalid observed-cost ledger: {self.path}")
        return value

    def _write(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
