"""Per-scenario publication ledger behind a multi-adapter bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioLedger, ledger_path
from reef.train.slime_backend.reef_adapters.weight_version import WeightVersion


def test_ledger_round_trips_and_computes_per_scenario_lag(tmp_path: Path) -> None:
    path = ledger_path(str(tmp_path / "hf" / "{rollout_id}"))
    ledger = ScenarioLedger(path)
    assert ledger.scenarios == () and ledger.lag("a", WeightVersion("inc", 1)) == 0
    ledger.record_checkpoint("a", 0)
    ledger.record_publication("a", "inc:1", "name-1")
    ledger.record_checkpoint("b", 1)
    ledger.record_publication("b", "inc:2", "name-b")
    ledger.record_checkpoint("a", 2)
    ledger.record_publication("a", "inc:3", "name-3")
    # b's publication (inc:2) does not age a's rollouts; a's own do.
    assert ledger.lag("a", WeightVersion("inc", 1)) == 1
    assert ledger.lag("a", WeightVersion("inc", 2)) == 1
    assert ledger.lag("a", WeightVersion("inc", 3)) == 0
    assert ledger.lag("b", WeightVersion("inc", 1)) == 1
    assert ledger.lag("b", WeightVersion("inc", 3)) == 0
    assert ledger.lag("a", WeightVersion("other", 3)) is None
    assert ledger.protected_rollouts() == {1, 2}
    reloaded = ScenarioLedger(path)
    assert reloaded.status() == ledger.status()
    assert reloaded.adapter("a") == "name-3" and reloaded.last_publication("b") == "inc:2"
    assert reloaded.status()["a"]["steps"] == 2


def test_ledger_rejects_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "reef_scenarios.json"
    path.write_text('{"format": 1, "scenarios": {"a": {"publications": [1]}}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="publications"):
        ScenarioLedger(path)
