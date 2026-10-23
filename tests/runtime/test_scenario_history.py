"""Per-scenario publication history behind a multi-adapter bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.runtime.recovery import ScenarioHistory, history_path
from reef.surface.adapter import adapter_name


def test_history_round_trips_per_scenario_publications(tmp_path: Path) -> None:
    path = history_path(str(tmp_path / "hf" / "{rollout_id}"))
    history = ScenarioHistory(path)
    assert history.scenarios == ()
    history.record_checkpoint("a", 0)
    history.record_publication("a", "inc:1", adapter_name("a", "inc:1"), checkpointed=True)
    history.record_checkpoint("b", 1)
    history.record_publication("b", "inc:2", adapter_name("b", "inc:2"), checkpointed=True)
    history.record_checkpoint("a", 2)
    history.record_publication("a", "inc:3", adapter_name("a", "inc:3"), checkpointed=True)
    assert history.protected_rollouts() == {1, 2}
    reloaded = ScenarioHistory(path)
    assert reloaded.status() == history.status()
    assert reloaded.adapter("a") == adapter_name("a", "inc:3") and reloaded.last_publication("b") == "inc:2"
    assert reloaded.status()["a"]["steps"] == 2


def test_a_restart_serves_each_scenarios_checkpoint_adapter(tmp_path: Path) -> None:
    path = tmp_path / "reef_scenarios.json"
    history = ScenarioHistory(path)
    history.record_publication("a", "inc:1", adapter_name("a", "inc:1"), checkpointed=False)
    history.record_publication("b", "inc:2", adapter_name("b", "inc:2"), checkpointed=False)
    history.record_checkpoint("b", 2)
    history.record_publication("b", "inc:3", adapter_name("b", "inc:3"), checkpointed=True)
    history.record_publication("b", "inc:4", adapter_name("b", "inc:4"), checkpointed=False)
    # Until a restart, the engines serve each scenario's last publication.
    assert history.checkpoint_adapter("a") is None and history.adapter("a") == adapter_name("a", "inc:1")
    assert history.checkpoint_adapter("b") == adapter_name("b", "inc:3")
    assert history.status()["b"]["runtime_load_id"] == "inc:4"

    restarted = ScenarioHistory(path)
    restarted.serve_checkpoint_adapters()
    # A scenario without a checkpoint serves no adapter; b serves its saved version, not the newer one.
    assert restarted.adapter("a") is None and restarted.status()["a"]["runtime_load_id"] is None
    assert restarted.adapter("b") == adapter_name("b", "inc:3")
    assert restarted.status()["b"]["runtime_load_id"] == "inc:3"
    assert restarted.last_publication("b") == "inc:4"
    assert ScenarioHistory(path).status() == restarted.status()


def test_a_format_1_history_saved_each_publication(tmp_path: Path) -> None:
    path = tmp_path / "reef_scenarios.json"
    entry = {"publications": ["inc:1"], "adapter": adapter_name("a", "inc:1"), "rollout_id": 0, "steps": 1}
    path.write_text(json.dumps({"format": 1, "scenarios": {"a": entry}}), encoding="utf-8")
    history = ScenarioHistory(path)
    assert history.checkpoint_adapter("a") == adapter_name("a", "inc:1")
    history.serve_checkpoint_adapters()
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == 2
    assert history.adapter("a") == adapter_name("a", "inc:1")


def test_history_rejects_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "reef_scenarios.json"
    path.write_text('{"format": 2, "scenarios": {"a": {"publications": [1]}}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="publications"):
        ScenarioHistory(path)
