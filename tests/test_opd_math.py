"""Evaluation and batching safeguards for the OPD math campaign."""

import json
from argparse import Namespace

import pytest

from recipes.opd.examples.math.run import Campaign, boxed_answer


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"The answer is \boxed{007}.", 7),
        (r"\boxed{12} is intermediate; finally \boxed{42}.", 42),
        (r"\boxed{\text{42}}", 42),
        (r"\boxed{42}, revised to \boxed{x}.", None),
        (r"\boxed{1234}", None),
        ("My intermediate calculation is 42.", None),
    ],
)
def test_only_final_boxed_integer_is_scored(response, expected) -> None:
    assert boxed_answer(response) == expected


def test_driver_rejects_batch_mismatch_before_creating_output(tmp_path) -> None:
    config = tmp_path / "serve.yaml"
    config.write_text("recipe:\n  config:\n    batch-size: 32\ntraining:\n  config:\n    global_batch_size: 32\n")
    output = tmp_path / "results"
    args = Namespace(config=config, steps=1, prompts_per_step=512, samples_per_prompt=4, output=output)
    with pytest.raises(ValueError, match="Driver batch must match"):
        Campaign(args)
    assert not output.exists()


def test_evaluation_never_submits_training_reports(tmp_path) -> None:
    class Client:
        def __init__(self):
            self.calls = []

        def post(self, path, scenario, payload):
            self.calls.append((path, scenario, payload))
            return {
                "choices": [{"message": {"content": r"\boxed{42}"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 8},
            }, {"x-reef-agent-record-id": "receipt"}

        def get(self, path):
            assert path.endswith("/records/receipt")
            return {"agent_record_id": "receipt", "artifact_ref": {"release_id": "sft-baseline"}}

        def report(self, *args, **kwargs):
            raise AssertionError("Held-out evaluation must not train")

    from recipes.opd.examples.math.run import Question

    campaign = object.__new__(Campaign)
    campaign.output = tmp_path
    campaign.client = Client()
    campaign.args = Namespace(scenario="opd", model="qwen35", seed=0, eval_repeats=2, concurrency=1, eval_tokens=64000)
    campaign.evaluate([Question("q0", "A problem", "42")], 0)
    records = [json.loads(row) for row in (tmp_path / "eval-0000.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {row["seed"] for row in records} == {0, 1}
    assert all(row["evaluation"] and row["correct"] for row in records)
    assert {row["release_id"] for row in records} == {"sft-baseline"}
    assert [call[2]["max_tokens"] for call in campaign.client.calls] == [64000, 64000]


def test_acceptance_uses_final_checkpoint_and_paired_questions(tmp_path) -> None:
    from recipes.opd.examples.math.analyze import analyze

    for step, values in [(0, [False, False]), (20, [True, True]), (200, [True, False])]:
        rows = [
            {"question_id": str(i), "seed": 0, "evaluation": True, "correct": value, "release_id": str(step)}
            for i, value in enumerate(values)
        ]
        (tmp_path / f"eval-{step:04d}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    result = analyze(tmp_path, questions=2, repeats=1)
    assert result["absolute_improvement"] == 0.5
    assert result["target_met"] is True
    assert result["paired_question_bootstrap_95_interval"] == [0.0, 1.0]
    assert [point["step"] for point in result["curve"]] == [0, 20, 200]
    with pytest.raises(ValueError, match="acceptance is pending"):
        analyze(tmp_path, final_step=100, questions=2, repeats=1)


def test_paired_interval_rejects_different_eval_questions() -> None:
    from recipes.opd.examples.math.analyze import paired_interval

    with pytest.raises(ValueError, match="question/seed sets differ"):
        paired_interval({("q0", 0): False}, {("q1", 0): True})
