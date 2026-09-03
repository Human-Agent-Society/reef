"""The terminus adapter against a real Terminal-Bench task, end to end.

Opt-in: this one runs a Docker container and spends real model calls, so it is
gated on both a dataset and a model rather than a stub. Everything cheaper —
that harbor validates the agent spec, imports the class, and binds the tree —
lives in ``tests/reef_service/test_terminus_agent.py`` and runs in CI.

    REEF_REAL_TERMINUS_DATASET=recipes/basic \\
    REEF_REAL_TERMINUS_TASK=harbor \\
    REEF_REAL_TERMINUS_MODEL=openai/gpt-4o \\
    REEF_REAL_TERMINUS_BASE_URL=https://api.openai.com/v1 \\
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

DATASET = os.environ.get("REEF_REAL_TERMINUS_DATASET", "")
MODEL = os.environ.get("REEF_REAL_TERMINUS_MODEL", "")
TASK = os.environ.get("REEF_REAL_TERMINUS_TASK", "harbor")
BASE_URL = os.environ.get("REEF_REAL_TERMINUS_BASE_URL", "https://api.openai.com/v1")

pytestmark = pytest.mark.skipif(
    not (DATASET and MODEL and Path(DATASET).is_dir()),
    reason="REEF_REAL_TERMINUS_DATASET and REEF_REAL_TERMINUS_MODEL must name a real dataset and model",
)

RULES_MARKER = "Reef terminus smoke marker."


def test_real_terminus_renders_runs_and_reports_a_verifier_reward(tmp_path: Path, monkeypatch) -> None:
    descriptor = get_adapter("terminus")
    nodes = [
        ("rules", {"text": RULES_MARKER}),
        ("config", {"data": {"max_turns": 8, "model_name": MODEL, "api_base": BASE_URL}}),
    ]
    root = tmp_path / "root"
    for path, text in render_composition(nodes, descriptor).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    sessions, trials = root / "terminus" / "sessions", root / "terminus" / "trials"
    monkeypatch.setenv("REEF_TERMINUS_DIR", str(root))
    monkeypatch.setenv("REEF_TERMINUS_SESSION_DIR", str(sessions))
    monkeypatch.setenv("REEF_TERMINUS_TRIALS_DIR", str(trials))
    monkeypatch.setenv("REEF_TERMINUS_DATASET", DATASET)

    from reef.harness.terminus.runner import run

    exit_code = run(TASK)

    # The runner wrote exactly one trial where the adapter's reader looks, and
    # the reader turns it into a verifier event plus the agent's ATIF steps.
    written = sorted(sessions.glob("*.json"))
    assert written == [sessions / f"{TASK}.json"]
    events = read_terminus_atif(sessions)
    assert events and events[0]["type"] == "verifier"
    assert events[0]["task"] == TASK
    # The verifier has to have run. Accepting an empty reward here would let
    # this pass on an episode whose container never built.
    assert not events[0]["error"], events[0]["error"]
    assert events[0]["rewards"], "the verifier scored nothing"
    assert exit_code == 0
    # recipes/basic/harbor is deterministic: 17 * 23 into answer.txt.
    if TASK == "harbor" and DATASET.endswith("basic"):
        assert events[0]["reward"] == 1.0
