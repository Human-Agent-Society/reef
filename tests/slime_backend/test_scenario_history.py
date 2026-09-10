"""Per-scenario publication history behind the engine-global runtime load ID."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.train.slime_backend.reef_adapters.runtime_load_id import RuntimeLoadId
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioHistory, history_path


def test_lag_counts_only_the_scenarios_own_publications(tmp_path: Path) -> None:
    history = ScenarioHistory(tmp_path / "history.json")
    history.record_publication("a", "inc:2", "adapter-a-2")
    history.record_publication("b", "inc:3", "adapter-b-3")
    history.record_publication("a", "inc:4", "adapter-a-4")
    # A rollout produced at inc:3 trails one publication of a (inc:4) and none of b.
    assert history.lag("a", RuntimeLoadId.parse("inc:3")) == 1
    assert history.lag("b", RuntimeLoadId.parse("inc:3")) == 0
    assert history.lag("a", RuntimeLoadId.parse("inc:1")) == 2
    assert history.lag("never-published", RuntimeLoadId.parse("inc:1")) == 0
    assert history.lag("a", RuntimeLoadId.parse("other:9")) is None


def test_history_round_trips_and_protects_latest_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = ScenarioHistory(path)
    history.record_checkpoint("a", 0)
    history.record_publication("a", "inc:1", "adapter-a-1")
    history.record_checkpoint("b", 1)
    history.record_publication("b", "inc:2", "adapter-b-2")
    history.record_checkpoint("a", 2)
    history.record_publication("a", "inc:3", "adapter-a-3")
    history.record_publication("a", "inc:3", "adapter-a-3")  # idempotent replay

    reloaded = ScenarioHistory(path)
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


def test_history_rejects_a_foreign_file(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text('{"format": 99}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid scenario history"):
        ScenarioHistory(tmp_path / "history.json")


def test_history_sits_beside_the_marker() -> None:
    assert history_path("/ckpt/hf/{rollout_id}") == Path("/ckpt/hf/reef_scenarios.json")
