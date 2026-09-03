"""The terminus runner's tree mapping: node kinds to Terminus 2 seams.

Stdlib only, like the module under test: these run wherever the suite runs,
with no harbor and no Docker, so the mapping a gated tree change depends on is
checked without the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.render import render_composition
from reef.harness.terminus import TerminusTreeError, context_policy, instruction_text, load_tree, terminus_kwargs

ASSEMBLE = """
def assemble(state, request, files):
    state["turns"] = state.get("turns", 0) + 1
    return [{"role": "user", "content": f"turn {state['turns']}"}]
"""


def _tree(**nodes: str) -> dict[str, str]:
    return dict(nodes)


@pytest.mark.unit
def test_an_empty_tree_is_stock_terminus() -> None:
    # The equivalence the baseline rests on: no node, no override.
    assert terminus_kwargs({}) == {}
    assert instruction_text({}) == ""
    assert context_policy({}) is None


@pytest.mark.unit
def test_config_becomes_constructor_arguments() -> None:
    tree = _tree(**{"terminus/config.json": json.dumps({"max_turns": 12, "temperature": 0.2})})
    assert terminus_kwargs(tree) == {"max_turns": 12, "temperature": 0.2}


@pytest.mark.unit
def test_a_config_that_is_not_an_object_is_a_tree_defect() -> None:
    with pytest.raises(TerminusTreeError, match="must be an object"):
        terminus_kwargs(_tree(**{"terminus/config.json": "[1, 2]"}))
    with pytest.raises(TerminusTreeError, match="not valid JSON"):
        terminus_kwargs(_tree(**{"terminus/config.json": "{"}))


@pytest.mark.unit
def test_rules_skills_and_commands_join_one_instruction_in_that_order() -> None:
    tree = _tree(
        **{
            "terminus/AGENTS.md": "Be brief.\n",
            "terminus/skills/notes/SKILL.md": "Take notes.",
            "terminus-commands/summarize/SKILL.md": "Summarize.",
        }
    )
    text = instruction_text(tree)
    assert text.index("Be brief.") < text.index("Take notes.") < text.index("Summarize.")
    # Terminus has no command surface, so a command says what it is instead.
    assert "User-invocable. Summarize." in text


@pytest.mark.unit
def test_the_context_policy_carries_state_across_calls() -> None:
    policy = context_policy(_tree(**{"terminus/context/assemble.py": ASSEMBLE}))
    assert policy is not None
    assert policy.assemble({"messages": []}) == [{"role": "user", "content": "turn 1"}]
    assert policy.assemble({"messages": []}) == [{"role": "user", "content": "turn 2"}]


@pytest.mark.unit
@pytest.mark.parametrize(
    "returned",
    ["return None", "return []", 'return "text"', "return [{'content': 'no role'}]"],
)
def test_a_policy_that_returns_no_usable_messages_leaves_stock_assembly_alone(returned: str) -> None:
    policy = context_policy(_tree(**{"terminus/context/assemble.py": f"def assemble(s, r, f):\n    {returned}\n"}))
    assert policy is not None
    assert policy.assemble({"messages": []}) is None


@pytest.mark.unit
def test_a_module_without_assemble_is_refused_at_load() -> None:
    with pytest.raises(TerminusTreeError, match="must define a callable assemble"):
        context_policy(_tree(**{"terminus/context/x.py": "value = 1\n"}))


@pytest.mark.unit
def test_a_module_that_raises_at_import_is_refused_at_load() -> None:
    with pytest.raises(TerminusTreeError, match="raised while loading"):
        context_policy(_tree(**{"terminus/context/x.py": "raise ValueError('boom')\n"}))


@pytest.mark.unit
def test_load_tree_reads_a_rendered_tree_back_the_way_render_keyed_it(tmp_path: Path) -> None:
    descriptor = get_adapter("terminus")
    files = render_composition([("rules", {"text": "Be brief."})], descriptor)
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    assert load_tree(tmp_path) == files


@pytest.mark.unit
def test_load_tree_refuses_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(TerminusTreeError, match="is not a directory"):
        load_tree(tmp_path / "missing")
