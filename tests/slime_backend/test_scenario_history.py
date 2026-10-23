"""Per-scenario publication history behind the engine-global runtime load ID."""

from __future__ import annotations

from pathlib import Path

import pytest

from reef.runtime.recovery import ScenarioHistory, history_path
from reef.surface.adapter import adapter_name


def test_history_round_trips_and_protects_latest_checkpoints(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = ScenarioHistory(path)
    history.record_checkpoint("a", 0)
    history.record_publication("a", "inc:1", adapter_name("a", "inc:1"), checkpointed=True)
    history.record_checkpoint("b", 1)
    history.record_publication("b", "inc:2", adapter_name("b", "inc:2"), checkpointed=True)
    history.record_checkpoint("a", 2)
    history.record_publication("a", "inc:3", adapter_name("a", "inc:3"), checkpointed=True)
    history.record_publication("a", "inc:3", adapter_name("a", "inc:3"), checkpointed=True)  # idempotent replay

    reloaded = ScenarioHistory(path)
    assert reloaded.scenarios == ("a", "b")
    assert reloaded.protected_rollouts() == {1, 2}
    assert reloaded.adapter("a") == adapter_name("a", "inc:3") and reloaded.last_publication("b") == "inc:2"
    assert reloaded.status()["a"] == {
        "runtime_load_id": "inc:3",
        "adapter": adapter_name("a", "inc:3"),
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
