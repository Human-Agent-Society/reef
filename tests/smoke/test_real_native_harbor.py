"""The native_harbor adapter on a real Harbor task, end to end: Docker builds the task image, a real model plays.

Opt-in: this builds a container and spends real model calls. Everything cheaper (the agent over a fake environment,
the runner's verifier row, the validator) runs in ``tests/reef_service/test_native_harbor.py``. The task in
``data/native_harbor_task`` has python3 and git in its image; its verifier rewards 1 only when both files hold the
answers and the workdir has no ``.git``. Run it where Docker runs linux/amd64 images:

    REEF_REAL_NATIVE_HARBOR_MODEL=qwen/qwen3-coder \\
    REEF_REAL_NATIVE_HARBOR_API_KEY=... \\
    python -m pytest tests/smoke/test_real_native_harbor.py

``REEF_REAL_NATIVE_HARBOR_BASE_URL`` defaults to OpenRouter, without ``/v1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import run_episode
from reef.harness.runners.native.seed import SEED_NODES
from reef.harness.tree.render import render_composition
from reef.train.cordis_backend.backend import tree_files

MODEL = os.environ.get("REEF_REAL_NATIVE_HARBOR_MODEL", "")
API_KEY = os.environ.get("REEF_REAL_NATIVE_HARBOR_API_KEY", "")
BASE_URL = os.environ.get("REEF_REAL_NATIVE_HARBOR_BASE_URL", "https://openrouter.ai/api")
TASK = Path(__file__).parent / "data" / "native_harbor_task"

pytestmark = pytest.mark.skipif(
    not (MODEL and API_KEY),
    reason="REEF_REAL_NATIVE_HARBOR_MODEL and REEF_REAL_NATIVE_HARBOR_API_KEY must name a real model and its key",
)

PEER = {
    "name": "peer",
    "prompt": (
        "You are one of two peers. peer.1 writes sum.txt and peer.2 writes product.txt, with relative paths: your "
        "workdir is your own copy of the task's. First send the other peer your part of the plan with team_send, "
        "then write your file with write_file, then answer in one line."
    ),
}
TEAM_GRAPH = {
    "name": "main",
    "start": "plan",
    "max_steps": 6,
    "stages": {
        "plan": {"kind": "model"},
        "act": {"kind": "tools"},
        "crew": {"kind": "subagent", "mode": "team", "agents": ["peer", "peer"], "workspace": "own"},
        "done": {"kind": "end", "reason": "completed"},
        "quit": {"kind": "end", "reason": "gave_up"},
    },
    "edges": [
        {"from": "plan", "when": "tool_calls", "to": "act"},
        {"from": "plan", "when": "text", "to": "crew"},
        {"from": "act", "when": "done", "to": "plan"},
        {"from": "crew", "when": "completed", "to": "done"},
        {"from": "crew", "when": "gave_up", "to": "quit"},
        {"from": "crew", "when": "budget", "to": "quit"},
        {"from": "crew", "when": "ask", "to": "quit"},
    ],
}


def run(entries: list[dict]) -> dict[str, list[dict]]:
    """One native_harbor episode on the task over ``entries``; each agent's events by the name its header gives."""
    descriptor = get_adapter("native_harbor")
    binding = ModelBinding(base_url=BASE_URL, model=MODEL, api_key=API_KEY, max_output_tokens=4096)
    nodes = [*((entry["name"], entry["config"]) for entry in entries), *binding.compose_nodes(descriptor)]
    files = {**render_composition(nodes, descriptor), **tree_files(descriptor, entries)}
    result = run_episode(descriptor, files, str(TASK), timeout=1800.0)
    (row,) = [event for event in result.trajectory if event["type"] == "verifier"]
    assert row["task"] == str(TASK)
    # The verifier has to have run: an empty reward would pass on an episode whose image never built.
    assert not row["failed"] and not row["error"] and row["rewards"], json.dumps(row)
    by_agent: dict[str, list[dict]] = {}
    agent = "root"
    for event in result.trajectory:
        if event["type"] == "session":
            agent = str(event["data"].get("agent") or "root")
        by_agent.setdefault(agent, []).append(event)
    return by_agent


def test_one_native_agent_runs_a_harbor_task_and_gets_a_verifier_reward() -> None:
    by_agent = run(list(SEED_NODES))
    assert sorted(by_agent) == ["root"]
    assert any(event["type"] == "tool/result" for event in by_agent["root"])


def test_two_peers_message_each_other_and_their_worktrees_merge_before_the_verifier() -> None:
    entries = [
        *(entry for entry in SEED_NODES if entry["name"] != "native_graph"),
        {"id": "peer", "name": "native_agent", "config": PEER},
        {"id": "main", "name": "native_graph", "config": TEAM_GRAPH},
    ]
    by_agent = run(entries)
    assert sorted(by_agent) == ["peer.1", "peer.2", "root"]
    for instance in ("peer.1", "peer.2"):
        found = by_agent[instance]
        assert any(event["type"] == "team/send" for event in found), instance
        assert any(
            event["type"] == "user/message" and event["data"]["source"]["kind"] == "message" for event in found
        ), instance
    assert any(event["type"] == "team/merge" for event in by_agent["root"])
