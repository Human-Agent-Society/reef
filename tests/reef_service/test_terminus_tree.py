"""The terminus runner's tree mapping: node kinds to Harbor's native inputs.

Stdlib only, like the module under test: these run wherever the suite runs,
with no Harbor and no Docker, so the mapping a gated tree change depends on is
checked without the benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.episodes.trajectory import read_terminus_atif
from reef.harness.runners.terminus import (
    TerminusTreeError,
    instruction_paths,
    load_tree,
    runner,
    skill_roots,
    terminus_kwargs,
)
from reef.harness.tree.render import render_composition


def _tree(**nodes: str) -> dict[str, str]:
    return dict(nodes)


def _render(tmp_path: Path, nodes: list) -> dict[str, str]:
    files = render_composition(nodes, get_adapter("terminus"))
    for path, text in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return files


@pytest.mark.unit
def test_an_empty_tree_is_stock_terminus() -> None:
    # The equivalence the baseline rests on: no node, no override.
    assert terminus_kwargs({}) == {}
    assert instruction_paths("/root", {}) == []
    assert skill_roots("/root", {}) == []


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
def test_rules_become_an_extra_instruction_path() -> None:
    tree = _tree(**{"terminus/AGENTS.md": "Be brief.\n"})
    assert instruction_paths("/root", tree) == [Path("/root/terminus/AGENTS.md")]
    # Whitespace is not a rules node; Harbor would append an empty file.
    assert instruction_paths("/root", _tree(**{"terminus/AGENTS.md": "  \n"})) == []


@pytest.mark.unit
def test_skills_and_commands_become_two_skill_roots() -> None:
    tree = _tree(
        **{
            "terminus/skills/notes/SKILL.md": "Take notes.",
            "terminus-commands/summarize/SKILL.md": "Summarize.",
        }
    )
    assert skill_roots("/root", tree) == [Path("/root/terminus/skills"), Path("/root/terminus-commands")]


@pytest.mark.unit
def test_an_absent_skill_root_is_omitted_because_harbor_rejects_an_empty_one() -> None:
    tree = _tree(**{"terminus/skills/notes/SKILL.md": "Take notes."})
    assert skill_roots("/root", tree) == [Path("/root/terminus/skills")]


@pytest.mark.unit
def test_the_agent_spec_names_harbors_own_agent_and_carries_the_tree(tmp_path: Path) -> None:
    files = _render(
        tmp_path,
        [
            ("config", {"data": {"model_name": "openai/gpt-4o", "max_turns": 12}}),
            ("skill", {"name": "notes", "text": "# Notes\n\nTake notes."}),
        ],
    )
    spec = runner.agent_spec(str(tmp_path), files)
    # Harbor's terminus-2, not a Reef subclass: no Reef code runs in the agent.
    assert spec["name"] == "terminus-2"
    assert spec["model_name"] == "openai/gpt-4o"
    assert spec["kwargs"] == {"max_turns": 12}
    assert spec["skills"] == [str(tmp_path / "terminus/skills")]


@pytest.mark.unit
def test_a_tree_without_a_model_name_cannot_run() -> None:
    with pytest.raises(TerminusTreeError, match="carries no model_name"):
        runner.agent_spec("/root", {})


@pytest.mark.unit
@pytest.mark.parametrize("task", [".", "..", "", ".hidden", "we*ird?"])
def test_a_task_that_cannot_name_a_trial_file_is_refused(task: str) -> None:
    # The task arrives in the episode prompt, and a promoted task can come from
    # recorded traffic, so it never becomes a path component unchecked.
    with pytest.raises(TerminusTreeError):
        runner.trial_slug(task)


@pytest.mark.unit
def test_a_task_path_keeps_only_its_last_segment_for_the_trial_file() -> None:
    # The task may legitimately be a task directory, so a path is not an
    # attack; only its last segment ever becomes a filename.
    assert runner.trial_slug("/data/terminal-bench/hello-world") == "hello-world"
    assert runner.trial_slug("hello-world") == "hello-world"
    assert runner.trial_slug("data\\tb\\hello-world") == "hello-world"


@pytest.mark.unit
@pytest.mark.parametrize("task", ["hello-world", "/data/tb/hello-world", "../../pwned", "/etc/passwd"])
def test_the_trial_file_always_lands_inside_the_session_directory(tmp_path: Path, task: str) -> None:
    sessions = tmp_path / "sessions"
    written = runner.write_trial(runner.trial_record(task, {"acc": 1.0}, tmp_path), sessions)
    assert written.resolve().parent == sessions.resolve()


def _trajectory(steps: list) -> str:
    return json.dumps({"schema_version": "ATIF-v1.4", "steps": steps})


@pytest.mark.unit
def test_atif_steps_orders_continuations_numerically(tmp_path: Path) -> None:
    # Terminus 2 opens a new continuation file after each summarization pass;
    # name order puts cont-1 before the base file and cont-10 before cont-2.
    for index, step in ((0, 1), (2, 3), (10, 11)):
        name = "trajectory.json" if index == 0 else f"trajectory.cont-{index}.json"
        (tmp_path / name).write_text(_trajectory([{"n": step}]))
    assert runner.atif_steps(tmp_path) == [{"n": 1}, {"n": 3}, {"n": 11}]


@pytest.mark.unit
def test_a_torn_trajectory_costs_its_steps_not_the_trial(tmp_path: Path) -> None:
    (tmp_path / "trajectory.json").write_text(_trajectory([{"n": 1}]))
    (tmp_path / "trajectory.cont-1.json").write_text("{ truncated")
    assert runner.atif_steps(tmp_path) == [{"n": 1}]


@pytest.mark.unit
def test_a_trial_record_carries_the_verifier_rewards(tmp_path: Path) -> None:
    record = runner.trial_record("hello-world", {"accuracy": 1.0}, tmp_path)
    assert record["reward"] == 1.0 and record["failed"] is False


@pytest.mark.unit
def test_an_episode_that_never_ran_records_why(tmp_path: Path) -> None:
    # A container that failed to build scores nothing. Without the reason, the
    # trial reads as an agent that failed rather than a run that never happened.
    record = runner.trial_record("hello-world", {}, tmp_path, "docker compose build failed")
    assert record["failed"] is True and record["error"] == "docker compose build failed"


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
def test_load_tree_reads_both_roots_and_skips_the_run_s_own_output(tmp_path: Path) -> None:
    # terminus-commands is a sibling of terminus, and the episode root also
    # holds the task workspace and, once running, sessions and trials.
    files = _render(
        tmp_path,
        [("rules", {"text": "Be brief."}), ("agent_command", {"name": "summarize", "text": "Summarize."})],
    )
    for noise in ("workspace/answer.txt", "terminus/sessions/x.json", "terminus/trials/t/trajectory.json"):
        target = tmp_path / noise
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")

    loaded = load_tree(tmp_path)
    assert loaded == files
    assert skill_roots(tmp_path, loaded) == [tmp_path / "terminus-commands"]


@pytest.mark.unit
def test_load_tree_refuses_a_root_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(TerminusTreeError, match="is not a directory"):
        load_tree(tmp_path / "missing")
