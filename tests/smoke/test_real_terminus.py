"""The terminus adapter against a real Terminal-Bench task, end to end.

Opt-in: this runs a Docker container and spends real model calls, so it is
gated on a task and a model rather than a stub. Everything cheaper — that the
episode reaches the runner with its tree (``test_terminus_episode.py``) and
that Harbor accepts what the runner builds (``test_terminus_agent.py``) —
runs in CI.

    REEF_REAL_TERMINUS_TASK=recipes/basic/harbor \\
    REEF_REAL_TERMINUS_MODEL=openai/gpt-4o \\
    OPENAI_API_KEY=... \\
    python -m pytest tests/smoke/test_real_terminus.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.render import render_composition
from reef.harness.trajectory import read_terminus_atif

MODEL = os.environ.get("REEF_REAL_TERMINUS_MODEL", "")
TASK = os.environ.get("REEF_REAL_TERMINUS_TASK", "")
BASE_URL = os.environ.get("REEF_REAL_TERMINUS_BASE_URL", "https://api.openai.com/v1")

pytestmark = pytest.mark.skipif(
    not (TASK and MODEL),
    reason="REEF_REAL_TERMINUS_TASK and REEF_REAL_TERMINUS_MODEL must name a real task and model",
)


def test_real_terminus_runs_a_task_and_reports_a_verifier_reward(tmp_path: Path, monkeypatch) -> None:
    descriptor = get_adapter("terminus")
    nodes = [
        ("rules", {"text": "Work quickly and stop as soon as the task is done."}),
        ("config", {"data": {"max_turns": 8, "model_name": MODEL, "api_base": BASE_URL}}),
    ]
    root = tmp_path / "root"
    for path, text in render_composition(nodes, descriptor).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    sessions = root / "terminus" / "sessions"
    monkeypatch.setenv("REEF_TERMINUS_DIR", str(root))
    monkeypatch.setenv("REEF_TERMINUS_SESSION_DIR", str(sessions))
    monkeypatch.setenv("REEF_TERMINUS_TRIALS_DIR", str(root / "terminus" / "trials"))

    from reef.harness.terminus.runner import run, trial_slug

    exit_code = run(TASK)

    slug = trial_slug(TASK)
    assert sorted(sessions.glob("*.json")) == [sessions / f"{slug}.json"]
    events = read_terminus_atif(sessions)
    assert events and events[0]["type"] == "verifier"
    # The verifier has to have run. Accepting an empty reward here would let
    # this pass on an episode whose container never built.
    assert not events[0]["error"], events[0]["error"]
    assert events[0]["rewards"], "the verifier scored nothing"
    assert exit_code == 0
    # The agent's own turns came back, not just the score.
    assert any(event["type"] == "step" for event in events)
