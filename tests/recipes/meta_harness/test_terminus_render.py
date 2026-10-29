"""The Terminal-Bench campaign's terminus adapter renders a tree as Terminus 2 reads it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.tree import HarnessTree, RenderError, render_tree

GOLDEN = Path(__file__).parent / "data" / "terminus_golden"
TERMINUS = get_adapter("terminus")
COMMON = {
    "rules.md": "Answer briefly.\n\nPrefer the standard library.\n",
    "skills/notes/SKILL.md": "# Notes skill\n\nKeep short notes.",
    "commands/summarize.md": "Summarize $1.",
}


def config(data: Mapping[str, object]) -> dict[str, str]:
    """The tree file of the primary config target."""
    return {"config/primary.json": json.dumps(data)}


# terminus refuses code extensions, and its config holds Terminus 2 constructor arguments.
SOURCE = {**COMMON, **config({"max_turns": 40})}


def render(
    files: Mapping[str, str], descriptor: AdapterDescriptor = TERMINUS, binding: ModelBinding | None = None
) -> dict[str, str]:
    """``files`` as a tree rendered for terminus, pointed at ``binding`` when one is given."""
    overlays = () if binding is None else binding.config_overlays(descriptor)
    return render_tree(HarnessTree(dict(files)), descriptor, overlays)


def test_render_matches_the_golden_tree() -> None:
    golden = {
        path.relative_to(GOLDEN).as_posix(): path.read_text() for path in sorted(GOLDEN.rglob("*")) if path.is_file()
    }
    assert render(SOURCE) == golden


def test_terminus_folds_commands_into_a_second_skill_root() -> None:
    files = render(SOURCE)
    # Both roots carry a SKILL.md, and the renderer synthesized frontmatter on each.
    assert files["terminus/skills/notes/SKILL.md"].startswith("---\nname: notes\n")
    assert files["terminus-commands/summarize/SKILL.md"].startswith("---\nname: summarize\n")


def test_terminus_rejects_config_keys_that_are_not_constructor_arguments() -> None:
    with pytest.raises(
        RenderError, match=r"key 'tmux_pane_width' in terminus/config\.json: each key must be a Terminus 2 argument"
    ):
        render(config({"tmux_pane_width": 200}))


@pytest.mark.parametrize("turns", [0, -1, True, "many"])
def test_terminus_rejects_a_max_turns_that_is_not_a_positive_integer(turns) -> None:
    with pytest.raises(RenderError, match="max_turns must be a positive integer"):
        render(config({"max_turns": turns}))


def test_terminus_extension_requires_an_agent_class() -> None:
    with pytest.raises(RenderError, match="must define class Agent"):
        render({"extensions/assemble.py": "def assemble(s, r, f): return None\n"})


def test_terminus_renders_one_extension_without_executing_it() -> None:
    code = "raise RuntimeError('must not run')\nclass Agent: pass\n"
    assert render({"extensions/agent.py": code})["terminus/context/agent.py"] == code
    with pytest.raises(RenderError, match="exactly one extension"):
        render({"extensions/agent.py": code, "extensions/second.py": code})


@pytest.mark.parametrize("model", ["m1", "qwen/qwen3-coder"])
def test_terminus_binding_renders_the_litellm_provider(model: str) -> None:
    binding = ModelBinding(base_url="http://127.0.0.1:9", model=model, api_key="k-1")
    config_json = json.loads(render({}, binding=binding)["terminus/config.json"])
    # The served name stays the model name Harbor looks the context limit up under; litellm_proxy routes litellm
    # to api_base whatever vendor prefix that name carries, with the tree's call arguments in the request body.
    assert config_json["model_name"] == model
    assert config_json["api_base"] == "http://127.0.0.1:9/v1"
    assert config_json["llm_kwargs"] == {"api_key": "k-1", "custom_llm_provider": "litellm_proxy"}


def test_terminus_keeps_the_docker_cli_settings_and_a_bind_mounted_root() -> None:
    """Its runner starts Docker task containers, so the episode keeps the service's Docker variables."""
    assert TERMINUS.host_env == {"DOCKER_HOST": "", "DOCKER_CONTEXT": "", "DOCKER_CONFIG": "{home}/.docker"}
    assert TERMINUS.is_root_bind_mounted
