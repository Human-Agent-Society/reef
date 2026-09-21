"""The skill-stream example: the stages' shared rules (training order, the two scorers) and the stream plan."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "recipes" / "sdft" / "examples" / "skill_stream"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def skills():
    """``skills.py`` as the stage container runs it; both stage tasks ship the same file."""
    tooluse = EXAMPLE / "harbor" / "tooluse" / "environment" / "skills.py"
    science = EXAMPLE / "harbor" / "science" / "environment" / "skills.py"
    assert tooluse.read_bytes() == science.read_bytes()
    return _load("skills", tooluse)


def test_wave_schedule_covers_each_epoch_once_and_drops_only_the_tail(skills) -> None:
    schedule = skills.wave_schedule(2674, 2, 32, 42)

    assert len(schedule) == 167
    assert all(len(step) == 32 for step in schedule)
    flat = [index for step in schedule for index in step]
    # Steps run across the epoch boundary; the first 2674 prompts are one shuffle of the epoch.
    assert sorted(flat[:2674]) == list(range(2674))
    assert len(set(flat[2674:])) == 167 * 32 - 2674
    # The order is fixed by the seed and differs between epochs and seeds.
    assert schedule == skills.wave_schedule(2674, 2, 32, 42)
    assert flat[:32] != flat[2674 : 2674 + 32]
    assert skills.wave_schedule(2674, 2, 32, 7) != schedule
    # Tool Use: 4046 prompts, 252 steps at 32 per step over two epochs (28 prompts dropped).
    assert len(skills.wave_schedule(4046, 2, 32, 42)) == 252


def test_science_answer_rule_follows_the_reference_scorer(skills) -> None:
    text = "<reasoning>\nsome steps\n</reasoning>\n<answer>\nB\n</answer>"
    assert skills.extract_answer(text) == "B"
    # The last tag wins, and text without a tag is scored as is.
    assert skills.extract_answer("<answer>A</answer> ... <answer> C </answer>") == "C"
    assert skills.extract_answer("D") == "D"
    assert skills.is_correct("science", text, "B")
    assert not skills.is_correct("science", text, "A")


def test_tooluse_rule_follows_the_reference_scorer(skills) -> None:
    gold = [{"Action": "getWeather", "Action_Input": '{"city": "Boston", "unit": "F"}'}]
    response = (
        "Thought: I need the weather.\nAction: getWeather\n"
        'Action Input: {"city": "Boston"}\nObservation: ...\nAction Input: {"unit": "F"}'
    )
    assert skills.tool_call(response) == ({"getWeather": 1}, {"city": "Boston", "unit": "F"})
    assert skills.is_correct("tooluse", response, gold)
    # The action multiset and the merged inputs must both match; a malformed
    # JSON block is skipped rather than failing the whole response.
    assert not skills.is_correct("tooluse", response + "\nAction: getWeather", gold)
    assert not skills.is_correct("tooluse", 'Action: getWeather\nAction Input: {"city": "Boston"}', gold)
    assert skills.is_correct("tooluse", response + "\nAction Input: {not json}", gold)


def test_training_examples_and_evaluation_windows_follow_the_reference(skills) -> None:
    assert skills.TASKS == ("tooluse", "science")
    assert skills.EVAL_MAX_TOKENS == {"tooluse": 1024, "science": 2048}
    with pytest.raises(ValueError, match="unknown task"):
        skills.load_split("medical", "train")
