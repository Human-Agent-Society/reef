"""The SciKnowEval example: the training order, the reference's scorer, and the requests the stage builds.

Torch/ray free and offline: the container-side module is loaded directly and
its dataset directory is pointed at a fixture, the way
``test_sdft_skill_stream.py`` loads the skill stream's.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "recipes" / "sdpo" / "examples" / "sciknoweval"
#: One row of the reference's split, with its fixed system prompt.
SYSTEM_PROMPT = "Given a question and four options, please select the right answer.\n<answer>\nA\n</answer>\n"
ROW = {
    "idx": 1522,
    "kind": "mcq",
    "dataset": "sciknoweval",
    "answer": "B",
    "prompt": "Which molecule?\n\nA: one\nB: two\nC: three\nD: four\nPlease reason step by step.",
    "system": SYSTEM_PROMPT,
}


@pytest.fixture(scope="module")
def chemistry(tmp_path_factory: pytest.TempPathFactory):
    """``chemistry.py`` as the task container runs it, with a two-row split beside it."""
    data = tmp_path_factory.mktemp("chemistry")
    for split, rows in (("train", [ROW, {**ROW, "idx": 2, "answer": "C"}]), ("test", [ROW])):
        (data / f"{split}.json").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    # The module imports reef_client at module scope; the tests never call it.
    sys.modules.setdefault("reef_client", ModuleType("reef_client")).ReefClient = object
    path = EXAMPLE / "harbor" / "chemistry" / "environment" / "chemistry.py"
    spec = importlib.util.spec_from_file_location("sdpo_chemistry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["sdpo_chemistry"] = module
    spec.loader.exec_module(module)
    module.DATA_DIR = data
    return module


@pytest.mark.unit
def test_the_split_loads_as_the_reference_writes_it(chemistry) -> None:
    train = chemistry.load_split("train")
    assert [row["answer"] for row in train] == ["B", "C"]
    assert len(chemistry.load_split("test")) == 1
    with pytest.raises(FileNotFoundError, match="clones the reference"):
        chemistry.load_split("validation")


@pytest.mark.unit
def test_the_request_is_the_reference_system_prompt_then_the_question(chemistry) -> None:
    assert chemistry.messages(ROW) == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ROW["prompt"]},
    ]


@pytest.mark.unit
def test_the_answer_rule_follows_the_reference_scorer(chemistry) -> None:
    text = "<reasoning>\nsome steps\n</reasoning>\n<answer>\nB\n</answer>"
    assert chemistry.extract_answer(text) == "B"
    # The last tag wins, and a response without one is scored as it stands.
    assert chemistry.extract_answer("<answer>A</answer> then <answer> C </answer>") == "C"
    assert chemistry.extract_answer("D") == "D"
    assert chemistry.is_correct(text, "B")
    assert not chemistry.is_correct(text, "A")
    # The reference compares the letter alone, so a restated answer is wrong.
    assert not chemistry.is_correct("<answer>\nB: two\n</answer>", "B")


@pytest.mark.unit
def test_the_question_schedule_shuffles_each_epoch_and_drops_only_the_tail(chemistry) -> None:
    schedule = chemistry.question_schedule(1890, 30, 32, 42)

    assert all(len(step) == 32 for step in schedule)
    assert len(schedule) == 1890 * 30 // 32
    flat = [index for step in schedule for index in step]
    # Steps run across the epoch boundary; the first epoch is one shuffle of the split.
    assert sorted(flat[:1890]) == list(range(1890))
    assert sorted(flat[1890:3780]) == list(range(1890))
    # The order is fixed by the seed, and differs between epochs and seeds.
    assert schedule == chemistry.question_schedule(1890, 30, 32, 42)
    assert flat[:32] != flat[1890 : 1890 + 32]
    assert chemistry.question_schedule(1890, 30, 32, 7) != schedule


@pytest.mark.unit
def test_evaluation_is_unrecorded_and_averages_over_questions(chemistry, monkeypatch) -> None:
    """avg@n asks for n choices in one unrecorded request and averages the per-question fractions."""
    asked = []

    def inference(scenario, path, payload):
        asked.append(payload)
        letters = ["B", "B", "A", "D"] if payload["messages"][1]["content"] == ROW["prompt"] else ["A"] * 4
        return {"choices": [{"message": {"content": f"<answer>\n{letter}\n</answer>"}} for letter in letters]}

    client = SimpleNamespace(inference=inference, inference_with_record=None)
    other = {**ROW, "prompt": "Another question?", "answer": "A"}
    result = chemistry.evaluate(client, [ROW, other], n=4, concurrency=2)

    # The first question scores 2 of 4, the second 4 of 4.
    assert result["avg_at_n"] == pytest.approx(0.75)
    assert (result["n"], result["questions"]) == (4, 2)
    assert all(payload["n"] == 4 for payload in asked)
    assert all(payload["temperature"] == 0.6 and payload["top_p"] == 0.95 for payload in asked)
