"""Per-scenario publication history behind a multi-adapter bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.runtime.runtime_load_id import RuntimeLoadId
from reef.train.slime_backend.reef_adapters.training_job.scenarios import ScenarioHistory, history_path


def test_history_round_trips_and_computes_per_scenario_lag(tmp_path: Path) -> None:
    path = history_path(str(tmp_path / "hf" / "{rollout_id}"))
    history = ScenarioHistory(path)
    assert history.scenarios == () and history.lag("a", RuntimeLoadId("inc", 1)) == 0
    history.record_checkpoint("a", 0)
    history.record_publication("a", "inc:1", "name-1")
    history.record_checkpoint("b", 1)
    history.record_publication("b", "inc:2", "name-b")
    history.record_checkpoint("a", 2)
    history.record_publication("a", "inc:3", "name-3")
    # b's publication (inc:2) does not age a's rollouts; a's own do.
    assert history.lag("a", RuntimeLoadId("inc", 1)) == 1
    assert history.lag("a", RuntimeLoadId("inc", 2)) == 1
    assert history.lag("a", RuntimeLoadId("inc", 3)) == 0
    assert history.lag("b", RuntimeLoadId("inc", 1)) == 1
    assert history.lag("b", RuntimeLoadId("inc", 3)) == 0
    assert history.lag("a", RuntimeLoadId("other", 3)) is None
    assert history.protected_rollouts() == {1, 2}
    reloaded = ScenarioHistory(path)
    assert reloaded.status() == history.status()
    assert reloaded.adapter("a") == "name-3" and reloaded.last_publication("b") == "inc:2"
    assert reloaded.status()["a"]["steps"] == 2


def test_history_rejects_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "reef_scenarios.json"
    path.write_text('{"format": 1, "scenarios": {"a": {"publications": [1]}}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="publications"):
        ScenarioHistory(path)
