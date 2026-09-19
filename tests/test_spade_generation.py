"""SPADE's experience section, its play records, and what a generation's report keeps."""

from __future__ import annotations

from pathlib import Path

import pytest

from recipes.beta.spade import PlayRecord, experience_for, experience_text, load_experience
from recipes.beta.spade.generation import (
    GenerationError,
    GenerationRecord,
    ProposalRecord,
    TaskMeasure,
    mean_reward,
    recorded_generations,
    write_generation_report,
)
from reef.harness.client.tasks import TaskPlay


def record(name: str, without: float, with_hint: float, code: str = "", skill: str | None = "deduction") -> PlayRecord:
    return PlayRecord(
        name=name, skill=skill, return_without_hint=without, return_with_hint=with_hint, instruction_excerpt=code
    )


def test_the_experience_is_sorted_into_frontier_mastered_and_out_of_reach_with_the_frontier_by_regret() -> None:
    text = experience_text(
        (
            record("harbor-00001-000-deduction", 0.3, 0.6, "Find the port the service wrote."),
            record("harbor-00001-004-deduction", 0.5, 0.9, "Repair the broken cron entry."),
            record("harbor-00001-001-deduction", 0.95, 1.0),
            record("harbor-00001-002-deduction", 0.0, 1.0),
            record("harbor-00001-003-deduction", 0.05, 0.6),
        )
    )
    frontier = text.index("Within reach but not mastered")
    mastered = text.index("Mastered without any hint")
    out_of_reach = text.index("Out of reach or broken")
    assert frontier < mastered < out_of_reach
    # The frontier lists the higher regret first, and only frontier records show their instruction.
    assert frontier < text.index("harbor-00001-004-deduction") < text.index("harbor-00001-000-deduction") < mastered
    assert "Repair the broken cron entry." in text and "Find the port the service wrote." in text
    assert "earlier instruction" in text
    assert mastered < text.index("harbor-00001-001-deduction") < out_of_reach
    assert out_of_reach < text.index("harbor-00001-002-deduction") < text.index("harbor-00001-003-deduction")
    assert "without hint +0.00, with hint +1.00" in text


def test_only_frontier_records_show_their_instruction_and_a_long_excerpt_is_cut() -> None:
    assert "List the files." not in experience_text(
        (record("harbor-00001-001-deduction", 1.0, 1.0, "List the files."),)
    )
    text = experience_text((record("harbor-00001-000-deduction", 0.3, 0.9, "y" * 2000),))
    assert "y" * 1200 in text and "y" * 1201 not in text
    assert "nothing recorded yet" in experience_text(())
    assert "  harbor-00001-000: without hint +0.30, with hint +0.60" in experience_text(
        (record("harbor-00001-000", 0.3, 0.6, skill=None),)
    )


@pytest.mark.parametrize(
    ("without", "with_hint", "outcome", "regret"),
    [
        (0.3, 0.6, "frontier", 0.3),
        (0.5, 0.25, "frontier", -0.25),
        (0.15, 0.15, "frontier", 0.0),
        (0.9, 0.9, "frontier", 0.0),
        (0.95, 1.0, "mastered", 0.05),
        (1.0, 0.0, "mastered", -1.0),
        (0.0, 1.0, "out_of_reach", 1.0),
        (0.05, 0.6, "out_of_reach", 0.55),
        (-1.0, 0.1, "out_of_reach", 1.1),
    ],
)
def test_a_play_record_knows_its_outcome_and_regret(
    without: float, with_hint: float, outcome: str, regret: float
) -> None:
    played = record("harbor-00001-000-deduction", without, with_hint)
    assert played.outcome == outcome and played.regret == pytest.approx(regret)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"return_with_hint": 1.5}, "return_with_hint"),
        ({"name": "bad name\nwith a newline"}, "task name"),
        ({"name": ""}, "task name"),
        ({"skill": "Deduction"}, "skill"),
        ({"instruction_excerpt": 3}, "instruction_excerpt"),
    ],
)
def test_a_bad_play_record_is_refused(fields: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "name": "harbor-00001-000-deduction",
        "skill": "deduction",
        "return_without_hint": 0.0,
        "return_with_hint": 1.0,
    }
    values.update(fields)
    with pytest.raises(ValueError, match=message):
        PlayRecord(**values)  # type: ignore[arg-type]


def test_experience_for_puts_this_skill_and_the_frontier_first_and_caps_the_records() -> None:
    records = [
        record("t-mastered", 0.95, 1.0, skill="inspection"),
        record("t-other-skill", 0.5, 0.9, skill="repair"),
        record("t-frontier-low", 0.5, 0.6, skill="inspection"),
        record("t-frontier-high", 0.3, 0.9, skill="inspection"),
        record("t-out", 0.0, 0.2, skill="inspection"),
    ]
    chosen = experience_for(records, "inspection")
    assert [item.name for item in chosen] == [
        "t-frontier-high",
        "t-frontier-low",
        "t-out",
        "t-mastered",
        "t-other-skill",
    ]
    many = [record(f"t-{index:02d}", 0.5, 0.6, skill="inspection") for index in range(20)]
    assert len(experience_for(many, "inspection")) == 12
    unskilled = [
        record("t-a", 0.5, 0.6, skill=None),
        record("t-b", 0.0, 0.2, skill=None),
        record("t-c", 0.5, 0.9, skill="repair"),
    ]
    assert [item.name for item in experience_for(unskilled, None)] == ["t-c", "t-a", "t-b"]


def test_mean_reward_counts_an_unscored_run_as_a_loss_and_skips_an_episode_that_never_ran(tmp_path: Path) -> None:
    def play(reward: float | None, error: str = "") -> TaskPlay:
        return TaskPlay(tmp_path, "t", "e", reward, {}, error, (), 0, (), None)

    assert mean_reward([play(1.0), play(None)]) == 0.5 and mean_reward([]) == 0.0
    assert mean_reward([play(1.0), play(None, "the trial raised")]) == 1.0
    assert mean_reward([play(None, "the trial raised")]) == 0.0


def test_a_generation_report_keeps_the_proposals_the_measures_and_the_experience(tmp_path: Path) -> None:
    measure = TaskMeasure(
        name="harbor-00002-000",
        skill=None,
        task_path=tmp_path / "tasks" / "harbor-00002-000",
        digest="ab" * 32,
        plain_rewards=(0.0, 1.0),
        hint_rewards=(1.0,),
        record=record("harbor-00002-000", 0.5, 1.0, skill=None),
    )
    generation = GenerationRecord(
        generation=2,
        proposals=(
            ProposalRecord(0, None, "d1", "harbor-00002-000", "", "r1"),
            ProposalRecord(1, None, "d2", None, "reply refused: no json", "r2"),
        ),
        measures=(measure,),
        manifest_path=tmp_path / "tasks" / "manifest-00002.json",
    )
    path = write_generation_report(tmp_path / "state", generation)
    assert path == tmp_path / "state" / "generation-00002.json"
    assert recorded_generations(tmp_path / "state") == (2,) and recorded_generations(tmp_path / "none") == ()
    assert load_experience(path) == generation.experience == (measure.record,)
    (tmp_path / "state" / "generation-x.json").write_text("{}")
    assert recorded_generations(tmp_path / "state") == (2,)


def test_load_experience_refuses_a_file_that_is_not_a_report(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(GenerationError, match="holds no experience"):
        load_experience(tmp_path / "x.json")
    with pytest.raises(GenerationError, match="is not a generation report"):
        load_experience(tmp_path / "missing.json")
    (tmp_path / "y.json").write_text('{"experience": [{"name": "t"}]}')
    with pytest.raises(GenerationError, match="cannot take"):
        load_experience(tmp_path / "y.json")
