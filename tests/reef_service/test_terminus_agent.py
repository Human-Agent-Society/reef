"""The terminus adapter against the real Harbor package.

Skipped unless the ``terminus`` extra is installed. These are the contract
checks no hermetic mapping can make: that Harbor accepts the agent spec and
the trial overrides the runner builds, that the constructor arguments the
render quirk admits are real Terminus 2 parameters, that litellm sends the
bound model name to the bound endpoint, and that Harbor loads the reefine
health task. They need Harbor, but not Docker and not a model, so they run in
any job that installs the extra.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.adapters.terminus.quirks import _ALLOWED_KNOBS, _BINDING_KNOBS
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.runners.terminus import instruction_paths, skill_roots
from reef.harness.runners.terminus.runner import AGENT_NAME, agent_spec
from reef.harness.tree.render import render_composition
from reef.recipe.reefine.evolution import HEALTH_TASK_DIRECTORY

pytest.importorskip("harbor", reason="harbor is not installed")

NODES = [
    ("rules", {"text": "Be brief."}),
    ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
    ("agent_command", {"name": "summarize", "text": "Summarize."}),
    ("config", {"data": {"model_name": "openai/stub", "max_turns": 12}}),
]


def _rendered(root: Path) -> dict[str, str]:
    files = render_composition(NODES, get_adapter("terminus"))
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return files


@pytest.mark.unit
def test_harbor_validates_the_trial_the_runner_builds(tmp_path: Path) -> None:
    from harbor.models.trial.config import TaskConfig, TrialConfig

    files = _rendered(tmp_path / "root")
    config = TrialConfig.model_validate(
        {
            "task": TaskConfig.model_validate({"path": Path("recipes/basic/harbor")}).model_dump(),
            "trials_dir": tmp_path / "trials",
            "agent": agent_spec(str(tmp_path / "root"), files),
            "extra_instruction_paths": instruction_paths(tmp_path / "root", files),
        }
    )
    # Harbor's own agent, configured rather than subclassed.
    assert config.agent.name == AGENT_NAME
    assert config.agent.model_name == "openai/stub"
    assert config.agent.kwargs == {"max_turns": 12}
    assert config.extra_instruction_paths == [tmp_path / "root" / "terminus/AGENTS.md"]


@pytest.mark.unit
def test_harbor_resolves_both_skill_roots_the_tree_renders(tmp_path: Path) -> None:
    from harbor.skills import resolve_skills

    files = _rendered(tmp_path / "root")
    roots = skill_roots(tmp_path / "root", files)
    resolved = resolve_skills([str(path) for path in roots])
    # One skill and one command, each discovered as its own skill directory,
    # so Harbor keeps progressive loading instead of pasting bodies inline.
    assert len(resolved) == 2


@pytest.mark.unit
def test_every_admitted_knob_is_a_real_terminus_2_argument() -> None:
    import inspect

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    parameters = set(inspect.signature(Terminus2.__init__).parameters)
    unknown = sorted((_ALLOWED_KNOBS | _BINDING_KNOBS) - parameters)
    assert unknown == [], f"the quirk admits arguments Terminus 2 does not take: {unknown}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "served", ["qwen/qwen3-coder", "deepseek/deepseek-chat-v3.1", "openai/gpt-4o-mini", "gemma4:26b"]
)
def test_litellm_sends_the_served_model_to_the_bound_endpoint(served: str) -> None:
    """A vendor prefix litellm does not know has no route, and one it knows picks that vendor's client and drops
    the prefix; the binding's model_name reaches api_base with the served name unchanged either way."""
    from litellm import get_llm_provider

    descriptor = get_adapter("terminus")
    binding = ModelBinding(base_url="https://openrouter.ai/api", model=served, api_key="k-1")
    config = json.loads(render_composition([*binding.compose_nodes(descriptor)], descriptor)["terminus/config.json"])
    model, provider, _, _ = get_llm_provider(config["model_name"], api_base=config["api_base"])
    assert (provider, model) == ("openai", served)


@pytest.mark.unit
def test_harbor_loads_the_reefine_health_task_directory() -> None:
    from harbor.models.task.task import Task

    assert Task.is_valid_dir(HEALTH_TASK_DIRECTORY)
    task = Task(HEALTH_TASK_DIRECTORY)
    assert "echo reef-ok" in task.instruction
    assert task.paths.test_path.is_file() and (task.paths.environment_dir / "Dockerfile").is_file()


@pytest.mark.unit
def test_extension_must_remain_a_terminus_agent(monkeypatch) -> None:
    from reef.harness.episodes.executor import ISOLATION_ENV
    from reef.harness.runners.terminus.tree import ENVIRONMENT_ENV, TerminusTreeError

    monkeypatch.setenv(ISOLATION_ENV, "bwrap")
    monkeypatch.setenv(ENVIRONMENT_ENV, "e2b")
    files = render_composition(
        [*NODES, ("code_extension", {"name": "agent", "code": "class Agent: pass\n"})], get_adapter("terminus")
    )
    with pytest.raises(TerminusTreeError, match="must inherit Harbor's Terminus2"):
        agent_spec("/root", files)
