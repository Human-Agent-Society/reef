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
from reef.harness.terminus import (
    TerminusTreeError,
    context_policy,
    instruction_text,
    load_tree,
    runner,
    terminus_kwargs,
)
from reef.harness.trajectory import read_terminus_atif

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


def _trajectory(steps):
    return json.dumps({"schema_version": "ATIF-v1.4", "steps": steps})


@pytest.mark.unit
def test_atif_steps_reads_continuations_in_order(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    # Terminus 2 opens a new continuation file after each summarization pass.
    (trial / "trajectory.json").write_text(_trajectory([{"n": 1}]))
    (trial / "trajectory.cont-1.json").write_text(_trajectory([{"n": 2}]))
    assert runner.atif_steps(trial) == [{"n": 1}, {"n": 2}]


@pytest.mark.unit
def test_a_torn_trajectory_costs_its_steps_not_the_trial(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trajectory.json").write_text(_trajectory([{"n": 1}]))
    (trial / "trajectory.cont-1.json").write_text("{ truncated")
    assert runner.atif_steps(trial) == [{"n": 1}]


@pytest.mark.unit
def test_a_trial_record_carries_the_verifier_rewards(tmp_path: Path) -> None:
    record = runner.trial_record("hello-world", {"accuracy": 1.0}, tmp_path)
    assert record["task"] == "hello-world"
    assert record["rewards"] == {"accuracy": 1.0}
    assert record["reward"] == 1.0
    assert record["failed"] is False


@pytest.mark.unit
def test_a_trial_the_verifier_never_scored_is_recorded_as_failed(tmp_path: Path) -> None:
    record = runner.trial_record("hello-world", None, tmp_path)
    assert record["failed"] is True
    assert record["rewards"] == {} and record["reward"] is None


@pytest.mark.unit
def test_the_written_trial_is_what_the_reader_parses(tmp_path: Path) -> None:
    trials, sessions = tmp_path / "trials", tmp_path / "sessions"
    trials.mkdir()
    (trials / "trajectory.json").write_text(_trajectory([{"n": 1}]))
    runner.write_trial(runner.trial_record("hello-world", {"accuracy": 1.0}, trials), sessions)

    events = read_terminus_atif(sessions)
    assert [event["type"] for event in events] == ["verifier", "step"]
    assert events[0]["reward"] == 1.0 and events[0]["task"] == "hello-world"
    assert events[1]["n"] == 1


@pytest.mark.unit
def test_the_agent_spec_names_the_class_and_carries_the_tree_location(tmp_path: Path) -> None:
    tree = _tree(**{"terminus/config.json": json.dumps({"model_name": "gpt-4o"})})
    spec = runner.agent_spec(tree, "/episode/terminus")
    assert spec == {
        "name": "reef.harness.terminus.agent:ReefTerminus",
        "model_name": "gpt-4o",
        "kwargs": {"reef_dir": "/episode/terminus"},
    }


@pytest.mark.unit
def test_a_tree_without_a_model_name_cannot_run() -> None:
    with pytest.raises(TerminusTreeError, match="carries no model_name"):
        runner.agent_spec({}, "/episode/terminus")


@pytest.mark.unit
def test_task_path_resolves_a_task_inside_the_dataset(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "hello-world").mkdir()
    monkeypatch.setenv("REEF_TERMINUS_DATASET", str(tmp_path))
    assert runner.task_path("hello-world") == str(tmp_path / "hello-world")
    with pytest.raises(TerminusTreeError, match="holds no task"):
        runner.task_path("absent")


@pytest.mark.unit
def test_atif_steps_orders_continuations_numerically(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    for index, step in ((0, 1), (2, 3), (10, 11)):
        name = "trajectory.json" if index == 0 else f"trajectory.cont-{index}.json"
        (trial / name).write_text(_trajectory([{"n": step}]))
    assert runner.atif_steps(trial) == [{"n": 1}, {"n": 3}, {"n": 11}]


@pytest.mark.unit
def test_skill_frontmatter_does_not_reach_the_instruction() -> None:
    # The quirk writes the header so the file is a well-formed skill on disk;
    # Terminus 2 reads this text as instruction, so the delimiters would be
    # literal noise in the prompt.
    rendered = render_composition(
        [("skill", {"name": "notes", "text": "# Notes\n\nTake notes."})], get_adapter("terminus")
    )
    assert rendered["terminus/skills/notes/SKILL.md"].startswith("---\n")
    text = instruction_text(rendered)
    assert text == "# Notes\n\nTake notes."
    assert "---" not in text and "description:" not in text


@pytest.mark.unit
def test_a_skill_that_already_had_frontmatter_keeps_only_its_body() -> None:
    tree = _tree(**{"terminus/skills/n/SKILL.md": "---\nname: n\ndescription: d\n---\nBody here.\n"})
    assert instruction_text(tree) == "Body here."
