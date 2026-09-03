"""The terminus agent against the real harbor package.

Skipped unless the ``terminus`` extra is installed. These are the contract
checks that no amount of hermetic mapping can make: that harbor validates the
agent spec the runner builds, that harbor can import the class that spec
names, and that constructing it actually applies the tree. They need harbor,
but not Docker and not a model, so they run in any job that installs the extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.render import render_composition
from reef.harness.terminus.runner import AGENT_IMPORT_PATH, agent_spec

pytest.importorskip("harbor", reason="the terminus extra is not installed")

ASSEMBLE = 'def assemble(state, request, files):\n    return [{"role": "user", "content": "rewritten"}]\n'


def _rendered(tmp_path: Path) -> Path:
    nodes = [
        ("rules", {"text": "Be brief."}),
        ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
        ("code_extension", {"name": "assemble", "code": ASSEMBLE}),
        (
            "config",
            {"data": {"max_turns": 7, "model_name": "openai/stub", "api_base": "http://127.0.0.1:1/v1"}},
        ),
    ]
    for path, text in render_composition(nodes, get_adapter("terminus")).items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.unit
def test_harbor_validates_the_agent_spec_the_runner_builds(tmp_path: Path) -> None:
    from harbor.models.trial.config import TaskConfig, TrialConfig

    spec = agent_spec({"terminus/config.json": json.dumps({"model_name": "openai/stub"})}, "/episode/terminus")
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"path": Path("recipes/basic/harbor")}).model_dump(),
            "trials_dir": tmp_path,
            "agent": spec,
        }
    )
    assert config.agent.name == AGENT_IMPORT_PATH
    assert config.agent.model_name == "openai/stub"
    assert config.agent.kwargs == {"reef_dir": "/episode/terminus"}


@pytest.mark.unit
def test_harbor_can_import_the_agent_the_spec_names() -> None:
    from harbor.agents.base import BaseAgent
    from harbor.agents.factory import import_class

    agent = import_class(AGENT_IMPORT_PATH, base=BaseAgent, label="agent")
    assert issubclass(agent, BaseAgent)
    assert agent.name() == "reef-terminus-2"


@pytest.mark.unit
def test_constructing_the_agent_binds_every_seam_to_the_tree(tmp_path: Path) -> None:
    from reef.harness.terminus.agent import ReefTerminus

    agent = ReefTerminus(tmp_path / "logs", reef_dir=str(_rendered(tmp_path / "tree")))
    # Rules and the skill body join the instruction; the frontmatter the quirk
    # wrote stays on disk rather than reaching the prompt.
    assert agent._reef_instruction == "Be brief.\n\n# Notes\n\nTake notes."
    assert "---" not in agent._reef_instruction
    # The context seam loaded and rewrites the pending call.
    assert agent._reef_context is not None
    assert agent._reef_context.assemble({"messages": []}) == [{"role": "user", "content": "rewritten"}]


@pytest.mark.unit
def test_an_empty_tree_constructs_stock_terminus(tmp_path: Path) -> None:
    from reef.harness.terminus.agent import ReefTerminus

    root = tmp_path / "tree"
    for path, text in render_composition(
        [("config", {"data": {"model_name": "openai/stub"}})], get_adapter("terminus")
    ).items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    agent = ReefTerminus(tmp_path / "logs", reef_dir=str(root))
    # The equivalence the measured baseline rests on.
    assert agent._reef_instruction == ""
    assert agent._reef_context is None
