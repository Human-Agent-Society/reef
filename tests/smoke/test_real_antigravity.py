"""Smoke tests for Antigravity adapter: descriptor, harness facts, composition render, and trajectory reading."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from reef.harness.adapters import available_adapters, get_adapter
from reef.harness.adapters.harness_facts import harness_facts
from reef.harness.episodes.trajectory import final_assistant_text, reader_for
from reef.harness.tree.render import render_composition


def test_antigravity_adapter_registration() -> None:
    """Antigravity must be registered among builtin adapters with expected attributes."""
    assert "antigravity" in available_adapters()
    desc = get_adapter("antigravity")
    assert desc.name == "antigravity"
    assert desc.binary == "agy"
    assert desc.trajectory_format == "antigravity-session-jsonl"
    assert desc.trajectory_path == "antigravity"


def test_antigravity_harness_facts() -> None:
    """Antigravity harness facts describe the CLI tools and prompt syntax."""
    facts = harness_facts("antigravity")
    assert facts is not None
    assert facts.title == "Antigravity"
    assert "read_file" in facts.tools
    assert "run_command" in facts.tools


def test_antigravity_composition_render() -> None:
    """Rendering rules and skills produces valid AGENTS.md and SKILL.md with frontmatter."""
    desc = get_adapter("antigravity")
    nodes = [
        ("rules", {"text": "Always verify options delta hedging invariants."}),
        ("skill", {"name": "gamma-scalp", "text": "Execute high gamma rebalancing on market open."}),
    ]
    files = render_composition(nodes, desc)
    assert "antigravity/AGENTS.md" in files
    assert "Always verify options delta hedging invariants." in files["antigravity/AGENTS.md"]
    assert "antigravity/skills/gamma-scalp/SKILL.md" in files
    skill_text = files["antigravity/skills/gamma-scalp/SKILL.md"]
    assert skill_text.startswith("---\nname: gamma-scalp")
    assert "Execute high gamma rebalancing on market open." in skill_text


def test_antigravity_trajectory_reader() -> None:
    """AntigravitySessionReader parses JSONL events and normalizes assistant text."""
    reader = reader_for("antigravity-session-jsonl")
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / "brain" / "conv-456" / ".system_generated" / "logs"
        log_dir.mkdir(parents=True)
        transcript = log_dir / "transcript.jsonl"
        transcript.write_text(
            json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "Ping"}) + "\n" +
            json.dumps({"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "content": "Pong"}) + "\n"
        )
        events = reader(Path(tmpdir))
        assert len(events) == 2
        assert events[0]["role"] == "user"
        assert events[1]["role"] == "assistant"
        assert events[1]["content"] == "Pong"
        assert final_assistant_text(events) == "Pong"
