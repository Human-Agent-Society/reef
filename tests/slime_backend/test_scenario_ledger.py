"""Per-scenario publication history behind the engine-global runtime load ID."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioLedger, ledger_path


def test_lag_counts_only_the_scenarios_own_publications(tmp_path: Path) -> None:
    ledger = ScenarioLedger(tmp_path / "ledger.json")
    ledger.record_publication("a", "inc:2", "adapter-a-2")
    ledger.record_publication("b", "inc:3", "adapter-b-3")
    ledger.record_publication("a", "inc:4", "adapter-a-4")
    # A rollout produced at inc:3 trails one publication of a (inc:4) and none of b.
    assert ledger.lag("a", RuntimeLoadId.parse("inc:3")) == 1
    assert ledger.lag("b", RuntimeLoadId.parse("inc:3")) == 0
    assert ledger.lag("a", RuntimeLoadId.parse("inc:1")) == 2
    assert ledger.lag("never-published", RuntimeLoadId.parse("inc:1")) == 0
    assert ledger.lag("a", RuntimeLoadId.parse("other:9")) is None


def test_ledger_round_trips_and_protects_latest_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ScenarioLedger(path)
    ledger.record_checkpoint("a", 0)
    ledger.record_publication("a", "inc:1", "adapter-a-1")
    ledger.record_checkpoint("b", 1)
    ledger.record_publication("b", "inc:2", "adapter-b-2")
    ledger.record_checkpoint("a", 2)
    ledger.record_publication("a", "inc:3", "adapter-a-3")
    ledger.record_publication("a", "inc:3", "adapter-a-3")  # idempotent replay

    reloaded = ScenarioLedger(path)
    assert reloaded.scenarios == ("a", "b")
    assert reloaded.protected_rollouts() == {1, 2}
    assert reloaded.adapter("a") == "adapter-a-3" and reloaded.last_publication("b") == "inc:2"
    assert reloaded.status()["a"] == {
        "runtime_load_id": "inc:3",
        "adapter": "adapter-a-3",
        "publications": 2,
        "rollout_id": 2,
        "steps": 2,
    }


def test_ledger_rejects_a_foreign_file(tmp_path: Path) -> None:
    (tmp_path / "ledger.json").write_text('{"format": 99}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid scenario ledger"):
        ScenarioLedger(tmp_path / "ledger.json")


def test_ledger_sits_beside_the_marker() -> None:
    assert ledger_path("/ckpt/hf/{rollout_id}") == Path("/ckpt/hf/reef_scenarios.json")
