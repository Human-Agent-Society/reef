"""ACE on the workspace protocol: traces grow a playbook in rules.md, with no episode and no correct answer.

A queued fake model is the Reflector and the Curator. The first test runs two steps through the real harness
backend under the ``none`` policy, with no scorer. The other step tests run the method once on a workspace that
``Workspace.create`` builds over the pi seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support._trajectories import recorded_trajectory
from tests.support.harness_methods import QueueChat
from tests.support.harness_service import backend_step
from tests.support.harness_steps import COMPONENT, PI_SEED, TreeReleases, runtime, step_tools

from recipes.ace import ACE, ACESettings
from reef.core.job_data import TrajectorySample
from reef.evaluation.harness import EvaluationSettings, NoEvaluationPolicy
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBindings
from reef.harness.tree import HarnessTree, validate_tree
from reef.recipe import load_recipe_config
from reef.recipe.harness import EvolutionSettings, HarnessEvolutionRecipe
from reef.train.backend import ComponentReleases
from reef.train.harness_backend import StepResult, Workspace
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData

PI = get_adapter("pi")
EXAMPLE = Path(__file__).resolve().parents[3] / "recipes" / "ace" / "examples" / "serve.yaml"


def session(record: str, task: str, score: float | None = None) -> TrajectorySample:
    """One recorded request and its reply, with a score only when ``score`` is given."""
    payload = {
        "messages": [{"role": "user", "content": task}],
        "response": {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
    }
    return recorded_trajectory(record, payload, score)


def reflection(*tags: tuple[str, str]) -> str:
    """A Reflector reply that gives each ``(id, tag)``."""
    keys = ("reasoning", "error_identification", "root_cause_analysis", "correct_approach", "key_insight")
    return json.dumps({**dict.fromkeys(keys, "..."), "bullet_tags": [{"id": key, "tag": tag} for key, tag in tags]})


def curation(*operations: dict[str, str]) -> str:
    return json.dumps({"reasoning": "...", "operations": list(operations)})


def add(section: str, content: str) -> dict[str, str]:
    return {"type": "ADD", "section": section, "content": content}


def playbook_state(*bullets: tuple[str, str, int, int]) -> dict[str, str]:
    """``state/`` with a playbook of ``(section, content, helpful, harmful)`` bullets that step 1 added."""
    rows = [
        {
            "id": f"ace-{number:05d}",
            "section": section,
            "content": content,
            "helpful": helpful,
            "harmful": harmful,
            "step": 1,
        }
        for number, (section, content, helpful, harmful) in enumerate(bullets, start=1)
    ]
    return {"playbook.json": json.dumps({"next_number": len(rows) + 1, "bullets": rows})}


def ace_step(
    tmp_path: Path, chat: QueueChat, settings: ACESettings, *samples: TrajectorySample, state: dict[str, str]
) -> tuple[Workspace, StepResult]:
    """One ACE step on a workspace over the pi seed, with ``chat`` as the served model."""
    workspace = Workspace.create(
        tmp_path / "step",
        history_dir=tmp_path / "history",
        step=2,
        tree=PI_SEED,
        descriptor=PI,
        tasks=(),
        repeats=0,
        samples=samples,
        state_files=state,
    )
    return workspace, ACE(settings).step(workspace, step_tools(workspace, ModelBindings(served=chat)))


def test_two_steps_grow_the_playbook_that_pi_serves_with_no_episode(tmp_path: Path) -> None:
    chat = QueueChat(
        reflection(),
        curation(add("Strategies and insights", "Read the failing test before you edit the code.")),
        reflection(("ace-00001", "helpful")),
        curation(add("Common mistakes to avoid", "Do not report a fix before the tests pass.")),
    )
    recipe = HarnessEvolutionRecipe(
        method=ACE(ACESettings(model="ace")),
        evolution=EvolutionSettings(state_dir=str(tmp_path / "harness")),
        evaluation=EvaluationSettings(policy="none"),
        seed=PI_SEED,
        models={"ace": chat},
        runtime=runtime(),
    )
    releases = TreeReleases(recipe.seed)
    backend = recipe.build_backend("demo", ComponentReleases(releases, COMPONENT), None)
    # The first trace has no score and no feedback: ACE needs no correct answer.
    first = backend_step(
        backend,
        JobData("demo:1", (session("i1", "Fix the failing test."),)),
        backend.initial_state(),
        NoEvaluationPolicy(),
    )
    assert first.metrics["selected"] is True and first.metrics["method_episodes"] == 0
    assert "[BEGIN recorded trace" in chat.prompts[0] and "[user] Fix the failing test." in chat.prompts[0]
    tree = HarnessTree.read(first.artifact.local_path)
    assert validate_tree(tree, PI).passed
    assert tree.files["rules.md"].startswith("Answer briefly.\n\n<!-- ace playbook: begin -->\n## Playbook\n")
    releases.publish("r1", tree)

    second = backend_step(
        backend, JobData("demo:2", (session("i2", "Fix the lint error.", 0.0),)), first.state, NoEvaluationPolicy()
    )
    # The second Reflector read the playbook that the first step committed, and the trace's score.
    assert "- [ace-00001] helpful=0 harmful=0 :: Read the failing test" in chat.prompts[2]
    assert "score: 0.0" in chat.prompts[2]
    rules = HarnessTree.read(second.artifact.local_path).files["rules.md"]
    assert "### Strategies and insights\n- [ace-00001] helpful=1 harmful=0 :: Read the failing test" in rules
    assert "### Common mistakes to avoid\n- [ace-00002] helpful=0 harmful=0 :: Do not report a fix" in rules
    saved = json.loads(HarnessState.from_json(second.state).method_files["playbook.json"])
    assert [(bullet["id"], bullet["step"]) for bullet in saved["bullets"]] == [("ace-00001", 1), ("ace-00002", 2)]
    assert backend.history.episodes == []


def test_tags_count_and_the_merge_drops_duplicates_harmful_bullets_and_the_excess(tmp_path: Path) -> None:
    state = playbook_state(
        ("Strategies and insights", "Keep answers short.", 0, 1),
        ("Strategies and insights", "Name the file you changed.", 3, 0),
        ("Others", "Prefer tabs.", 0, 0),
    )
    chat = QueueChat(
        reflection(
            ("ace-00001", "harmful"), ("[ace-00002]", "helpful"), ("ace-00003", "useful"), ("ace-00009", "helpful")
        ),
        curation(
            add("strategies AND insights", "keep  answers SHORT."),
            add("Strategies and insights", "Run the linter before you commit."),
            add("Common mistakes to avoid", "Check the diff\nbefore you push."),
            add("Unknown", "Use a section that exists."),
            {"type": "UPDATE", "section": "Others", "content": "Prefer spaces."},
            add("Others", "Log in with sk-" + "0" * 24 + "."),
            # A marker line of the block in a bullet would end the block early when the next step reads rules.md.
            add("Others", "Keep the block with <!--  ace playbook: end --> as its last line."),
        ),
    )
    settings = ACESettings(max_bullets=3, prune_margin=2)
    workspace, result = ace_step(tmp_path, chat, settings, session("i1", "Fix it."), state=state)
    # ace-00001 now has two harmful tags more than helpful ones. Of the four bullets left, the cap of three drops
    # the oldest bullet with the lowest helpful minus harmful count.
    saved = json.loads((workspace.state / "playbook.json").read_text())
    assert saved["next_number"] == 6
    assert [(bullet["id"], bullet["content"], bullet["helpful"]) for bullet in saved["bullets"]] == [
        ("ace-00002", "Name the file you changed.", 4),
        ("ace-00004", "Run the linter before you commit.", 0),
        ("ace-00005", "Check the diff before you push.", 0),
    ]
    assert {key: value for key, value in result.notes.items() if key != "skipped"} == {
        "traces": 1,
        "runs": 1,
        "read": 1,
        "helpful_tags": 1,
        "harmful_tags": 1,
        "added": 2,
        "duplicates": 1,
        "refused": 2,
        "ignored": 2,
        "pruned": 1,
        "capped": 1,
        "bullets": 3,
    }
    assert workspace.tree().files["rules.md"] == (
        "Answer briefly.\n\n"
        "<!-- ace playbook: begin -->\n"
        "## Playbook\n\n"
        "Lessons from earlier sessions: use the ones that fit your task. "
        "The counts say how often a review found a lesson helpful or harmful.\n\n"
        "### Strategies and insights\n"
        "- [ace-00002] helpful=4 harmful=0 :: Name the file you changed.\n"
        "- [ace-00004] helpful=0 harmful=0 :: Run the linter before you commit.\n\n"
        "### Common mistakes to avoid\n"
        "- [ace-00005] helpful=0 harmful=0 :: Check the diff before you push.\n"
        "<!-- ace playbook: end -->\n"
    )


def test_a_reply_that_is_not_the_asked_json_skips_its_trace(tmp_path: Path) -> None:
    chat = QueueChat(
        "I cannot judge this session.",
        "My analysis:\n```json\n" + reflection(("ace-00001", "helpful")) + "\n```",
        '{"reasoning": "nothing new", "operations": "none"}',
        '{"key_insight": "Check the exit code."}',
        curation(add("Others", "Check the exit code of each command.")),
    )
    state = playbook_state(("Others", "Prefer tabs.", 0, 0))
    traces = (session("i1", "First task."), session("i2", "Second task."), session("i3", "Third task."))
    workspace, result = ace_step(tmp_path, chat, ACESettings(), *traces, state=state)
    assert result.notes["skipped"] == [
        {"trace": "i1", "reason": "the Reflector's reply is not the asked JSON"},
        {"trace": "i2", "reason": "the Curator's reply is not the asked JSON"},
    ]
    # The second reflection was read from its code fence, so its tag counts although the Curator's reply failed.
    # The third reflection judged no bullet, and its lesson still became a bullet.
    assert (
        workspace.tree()
        .files["rules.md"]
        .endswith(
            "### Others\n"
            "- [ace-00001] helpful=1 harmful=0 :: Prefer tabs.\n"
            "- [ace-00002] helpful=0 harmful=0 :: Check the exit code of each command.\n"
            "<!-- ace playbook: end -->\n"
        )
    )


def test_the_example_serves_ace_on_pi_with_no_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    # reef serve runs the example from the repository root, where its paths start.
    monkeypatch.chdir(EXAMPLE.parents[3])
    recipe = HarnessEvolutionRecipe.from_environment({}, config=load_recipe_config(EXAMPLE), runtime=runtime())
    assert isinstance(recipe.method, ACE) and recipe.method.settings == ACESettings(max_bullets=100, prune_margin=2)
    assert (recipe.evolution.adapter, recipe.evaluation.policy, recipe.scorer) == ("pi", "none", None)
    assert set(recipe.seed.files) == {"rules.md"}
