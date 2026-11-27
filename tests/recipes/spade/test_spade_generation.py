"""SPADE's experience section, its play records, the Designer's score and advantages, and what a generation's report
keeps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.support.task_service_stubs import generator_state

from recipes.beta.spade import (
    PlayRecord,
    ScoreSettings,
    centered_advantages,
    designer_score,
    experience_for,
    experience_text,
    load_experience,
    plateau_reward,
)
from recipes.beta.spade.generation import (
    GenerationError,
    GenerationRecord,
    ProposalRecord,
    TaskMeasure,
    generation_report,
    mean_reward,
)
from recipes.beta.spade.generator import saved_designer_advantages


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


def test_mean_reward_is_the_mean_of_the_rewards_that_ran_and_zero_without_any() -> None:
    assert mean_reward((1.0, 0.0)) == 0.5 and mean_reward(()) == 0.0


def test_a_generation_report_keeps_the_proposals_the_measures_and_the_experience() -> None:
    measure = TaskMeasure(
        name="harbor-00002-000",
        skill=None,
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
    )
    report = generation_report(generation)
    document = json.loads(report)
    assert document["generation"] == 2 and document["tasks"][0]["regret"] == 0.5
    assert [proposal["refusal"] for proposal in document["proposals"]] == ["", "reply refused: no json"]
    assert load_experience(report) == generation.experience == (measure.record,)


def test_load_experience_refuses_a_document_that_is_not_a_report() -> None:
    with pytest.raises(GenerationError, match="holds no experience"):
        load_experience("{}")
    with pytest.raises(GenerationError, match="is not a generation report"):
        load_experience("not json")
    with pytest.raises(GenerationError, match="cannot take"):
        load_experience('{"experience": [{"name": "t"}]}')


# The reference code's paper runs (cmd/slime/*/0617/train_plateau_6skill.sh): 0.6 * plateau + 0.4 * floored regret.
PAPER_4B = ScoreSettings(
    plateau_low=0.2, plateau_high=0.4, plateau_ramp=0.2, plateau_weight=0.6, regret_weight=0.4, regret_scale=0.15
)


@pytest.mark.parametrize(
    ("band", "win_rate", "expected"),
    [
        # Qwen3-4B: full reward on [0.2, 0.4]; 0.5 at a win rate of 0.5; 0 at 0.0 and at 0.6.
        ((0.2, 0.4, 0.2), 0.0, 0.0),
        ((0.2, 0.4, 0.2), 0.1, 0.5),
        ((0.2, 0.4, 0.2), 0.2, 1.0),
        ((0.2, 0.4, 0.2), 0.4, 1.0),
        ((0.2, 0.4, 0.2), 0.5, 0.5),
        ((0.2, 0.4, 0.2), 0.6, 0.0),
        ((0.2, 0.4, 0.2), 0.9, 0.0),
        # Qwen3-30B-A3B and the reference function's defaults: 1 on [0.4, 0.6], 0 at or below 0.15 and at or
        # above 0.85.
        ((0.4, 0.6, 0.25), 0.15, 0.0),
        ((0.4, 0.6, 0.25), 0.25, 0.4),
        ((0.4, 0.6, 0.25), 0.4, 1.0),
        ((0.4, 0.6, 0.25), 0.6, 1.0),
        ((0.4, 0.6, 0.25), 0.7, 0.6),
        ((0.4, 0.6, 0.25), 0.85, 0.0),
        ((0.4, 0.6, 0.25), 1.0, 0.0),
    ],
)
def test_the_plateau_reward_matches_the_reference_code_at_the_band_edges(
    band: tuple[float, float, float], win_rate: float, expected: float
) -> None:
    assert plateau_reward(win_rate, *band) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("settings", "regret", "win_rate", "expected"),
    [
        (PAPER_4B, -0.3, 0.3, 0.6),  # the floor: a negative regret adds nothing
        (PAPER_4B, 0.45, 0.3, 1.0),  # the clip: regret / 0.15 = 3 counts as 1
        (PAPER_4B, 0.06, 0.5, 0.46),  # 0.6 * 0.5 + 0.4 * 0.4
        (PAPER_4B, 0.15, 0.0, 0.4),  # out of the band, full regret
        (ScoreSettings(), 0.5, 0.25, 0.5),  # the defaults score the regret alone
        (ScoreSettings(), -0.25, 0.25, 0.0),
        (ScoreSettings(), 1.5, 0.25, 1.0),
    ],
)
def test_the_score_is_the_weighted_plateau_plus_the_floored_and_clipped_regret(
    settings: ScoreSettings, regret: float, win_rate: float, expected: float
) -> None:
    assert designer_score(regret, win_rate, settings) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"plateau_low": 0.5, "plateau_high": 0.4}, "plateau band"),
        ({"plateau_high": 1.5}, "plateau band"),
        ({"plateau_ramp": 0.0}, "positive"),
        ({"regret_scale": -0.1}, "positive"),
        ({"plateau_weight": -0.6}, "non-negative"),
        ({"regret_weight": float("nan")}, "finite"),
    ],
)
def test_score_settings_refuse_a_band_or_a_weight_that_is_out_of_range(fields: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ScoreSettings(**fields)


def scored(name: str, skill: str | None, score: float, plays: int) -> TaskMeasure:
    """A measured train task with ``plays`` training rollouts and the Designer's score ``score``."""
    return TaskMeasure(
        name=name,
        skill=skill,
        digest="ab" * 32,
        plain_rewards=(0.5,) * plays,
        hint_rewards=(),
        record=record(name, 0.5, 0.5, skill=skill),
        score=score,
    )


def test_advantages_are_centered_per_skill_over_the_train_proposals_that_ran() -> None:
    proposals = (
        ProposalRecord(0, "inspection", "d-0", "t-a", ""),
        ProposalRecord(1, "inspection", "d-1", "t-b", ""),
        ProposalRecord(2, "repair", "d-2", "t-c", ""),
        ProposalRecord(3, "repair", "d-3", None, "reply refused: no json"),
        ProposalRecord(4, "repair", "d-4", "t-eval", "", split="eval"),
        ProposalRecord(5, "repair", "d-5", None, "the sandbox provider is down", infrastructure_failure=True),
    )
    measures = (
        scored("t-a", "inspection", 0.9, 2),
        scored("t-b", "inspection", 0.3, 2),
        scored("t-c", "repair", 0.5, 2),
    )
    advantages = centered_advantages(proposals, measures, max_scale=50.0)
    # Six plays over three replies: scale 2. The lone repair reply centers to 0, the provable null.
    assert advantages == pytest.approx({"d-0": 0.6, "d-1": -0.6, "d-2": 0.0})
    assert centered_advantages(proposals[3:], (), max_scale=50.0) == {}


@pytest.mark.parametrize(("plays", "scale"), [(8, 8.0), (32, 32.0), (64, 50.0)])
def test_the_scale_is_the_plays_of_each_reply_capped_at_max_scale(plays: int, scale: float) -> None:
    proposals = tuple(ProposalRecord(index, None, f"d-{index}", f"t-{index:02d}", "") for index in range(24))
    measures = tuple(scored(f"t-{index:02d}", None, float(index % 2), plays) for index in range(24))
    advantages = centered_advantages(proposals, measures, max_scale=50.0)
    assert advantages["d-0"] == pytest.approx(-0.5 * scale) and advantages["d-1"] == pytest.approx(0.5 * scale)


def test_a_report_keeps_the_designer_advantages_of_the_recipes_generation(tmp_path: Path) -> None:
    generation = GenerationRecord(
        generation=0,
        proposals=(ProposalRecord(0, None, "d1", "harbor-00007-000", ""),),
        measures=(scored("harbor-00007-000", None, 0.25, 2),),
        recipe_generation=7,
        designer_advantages={"d1": 0.0},
    )
    state = generator_state(tmp_path, "spade")
    state.write("report/00000", generation_report(generation))
    assert saved_designer_advantages(state, 7) == {"d1": 0.0}
    assert saved_designer_advantages(state, 0) == {}, "the report belongs to the recipe's generation 7"
    state.write("report/00001", "not json")
    with pytest.raises(GenerationError, match="is not a generation report"):
        saved_designer_advantages(state, 8)
