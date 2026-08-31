from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipes.tttd.examples.tttd.harness.agent import ReefTTTDiscoverHarness
from recipes.tttd.examples.tttd.harness.sandbox import execute_program
from recipes.tttd.examples.tttd.harness.search import (
    Candidate,
    PUCTArchive,
    ScoredSolution,
    TTTDChatRequestBuilder,
    TTTDiscoverHarness,
    build_prompt,
    extract_solution,
)


def test_puct_records_top_children_and_backpropagates_visits():
    archive = PUCTArchive(max_size=10, top_children=2)
    root = archive.add_seed("root", 1.0)
    children = archive.record_expansion(root.candidate_id, [("low", 2.0, 2.0), ("best", 4.0, 4.0), ("mid", 3.0, 3.0)])

    assert [child.reward for child in children] == [4.0, 3.0]
    assert root.best_child_reward == 4.0
    assert root.q == 4.0
    assert root.visits == 3
    assert archive.total_expansions == 3

    archive.record_expansion(children[0].candidate_id, [("grandchild", 5.0, 5.0)])
    assert root.visits == 4
    assert children[0].visits == 1


def test_puct_archive_snapshot_round_trip_preserves_search_state():
    archive = PUCTArchive(max_size=10, top_children=2)
    root = archive.add_seed("seed-solution", 1.0)
    archive.record_expansion(
        root.candidate_id,
        [
            ("first", 3.0, 3.0),
            ("second", 2.0, 2.0),
        ],
        failed_rollouts=1,
    )
    snapshot = archive.state_dict()

    restored = PUCTArchive()
    restored.load_state_dict(snapshot)

    assert restored.state_dict() == snapshot
    assert restored.scores() == archive.scores()
    assert restored.best().solution == "first"


def test_puct_pruning_always_retains_seeds():
    archive = PUCTArchive(max_size=2)
    seed = archive.add_seed("seed", -10.0)
    archive.record_expansion(seed.candidate_id, [("one", 1.0, 1.0), ("two", 2.0, 2.0)])

    assert seed.candidate_id in {candidate.candidate_id for candidate in archive.candidates}
    assert len(archive.candidates) == 2
    assert archive.best().solution == "two"


def test_parent_prompt_reuses_the_normalized_code_block_without_nesting():
    solution = "```python\nprint('parent')\n```"
    parent = Candidate("state-1", solution, 2.5, 2.5, output="finished")

    prompt = build_prompt("Solve the task.", parent)[0]["content"]

    assert "## Selected parent (reward=2.500000)" in prompt
    assert prompt.count("```python") == 1
    assert "```\n```python" not in prompt
    assert "--- Previous Program Output ---\nfinished\n--- End Output ---" in prompt


def test_solution_extraction_matches_official_unclosed_final_block_behavior():
    response = "Reasoning first.\n```python\nprint('complete even without a closing fence')"

    assert extract_solution(response) == "```python\nprint('complete even without a closing fence')\n```"


@pytest.mark.parametrize("task", ["circle_packing_26", "circle_packing_32"])
def test_packing_instruction_does_not_claim_the_dynamic_parent_is_empty(task):
    instruction = (REPO_ROOT / f"recipes/tttd/examples/tttd/harbor/{task}/instruction.md").read_text()

    assert "No previous code available" not in instruction
    assert "Current sum of radii (higher is better): 0.000000" not in instruction


def test_erdos_instruction_and_judge_use_the_same_program_budget():
    task = REPO_ROOT / "recipes/tttd/examples/tttd/harbor/erdos_min_overlap"
    instruction = (task / "instruction.md").read_text()
    scorer = (task / "environment/score.py").read_text()

    assert "budget_s=1000" in instruction
    assert "budget_s=1000" in scorer
    assert "budget_s=900" not in scorer


def _erdos_reward(result):
    _h_values, c5_bound, n_points = result
    c5_bound = float(c5_bound)
    if c5_bound <= 0 or math.isnan(c5_bound) or math.isinf(c5_bound):
        raise ValueError("C5 bound must be positive and finite")
    reward = 1.0 / (1e-8 + c5_bound)
    return reward, -c5_bound, {"c5_bound": c5_bound, "n_points": int(n_points)}


def test_erdos_scorer_scores_valid_construction():
    from recipes.tttd.examples.tttd.harness.scorer import ProgramScorer

    scorer = ProgramScorer(_erdos_reward, eval_timeout_s=30, num_cpus=1, prelude="import numpy as np")

    values = [0.4] * 20 + [0.6] * 20
    solution = (
        "```python\n"
        "def run(seed=42, budget_s=1000, **kwargs):\n"
        f"    h_values = np.array({values!r})\n"
        "    dx = 2.0 / len(h_values)\n"
        "    c5_bound = float(np.max(np.correlate(h_values, 1-h_values, mode='full') * dx))\n"
        "    return h_values, c5_bound, len(h_values)\n"
        "```"
    )
    scored = scorer(solution)

    assert scored.reward == pytest.approx(1.0 / (1e-8 + scored.metrics["c5_bound"]))
    assert scored.value == pytest.approx(-scored.metrics["c5_bound"])
    assert scored.metrics["n_points"] == 40


def test_erdos_scorer_rejects_invalid_solution():
    from recipes.tttd.examples.tttd.harness.scorer import ProgramScorer

    scorer = ProgramScorer(_erdos_reward, eval_timeout_s=30)
    with pytest.raises(ValueError, match="cannot extract Python code"):
        scorer("no code here")


def _scorer():
    """A simple scorer for harness tests: extracts a number from a code block."""

    def score(solution: str) -> ScoredSolution:
        match = re.search(r"```python\s+(\d+)\s*```", solution)
        if not match:
            raise ValueError("no number in code block")
        value = int(match.group(1))
        return ScoredSolution(solution=solution, reward=float(value), value=float(value))

    return score


class _Model:
    def __init__(self):
        self.index = 0
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        self.index += 1
        content = "bad" if self.index == 2 else f"```python\n{self.index}\n```"
        return {"choices": [{"message": {"content": content}}]}


def test_ordinary_harness_calls_model_evaluates_and_archives():
    model = _Model()
    harness = TTTDiscoverHarness(
        model,
        _scorer(),
        "Improve the number.",
        model="test-model",
        groups_per_step=1,
        rollouts_per_group=3,
        max_workers=1,
    )

    results = harness.run_step(0)

    assert [result.reward for result in results] == [1.0, 0.0, 3.0]
    assert results[1].solution == ""
    assert len(model.payloads) == 3
    assert model.payloads[0]["model"] == "test-model"
    assert harness.archive.best().reward == 3.0


class _Client:
    def __init__(self):
        self.index = 0
        self.reports = []

    def inference_with_record(self, scenario, path, payload, *, recipe=None, artifact_version=None):
        assert (scenario, path, recipe, artifact_version) == (
            "discovery",
            "/v1/chat/completions",
            "tttd",
            "checkpoint-v1",
        )
        self.index += 1
        content = "bad" if self.index == 2 else f"```python\n{self.index}\n```"
        return {"choices": [{"message": {"content": content}}]}, f"agent-record-{self.index}"

    def report(self, scenario, payload, *, recipe=None, artifact_version=None):
        self.reports.append((scenario, payload, recipe, artifact_version))
        return {}


def test_reef_harness_reports_each_result_against_exact_inference():
    client = _Client()
    harness = ReefTTTDiscoverHarness(
        client,
        _scorer(),
        "Improve the number.",
        scenario="discovery",
        artifact_version="checkpoint-v1",
        model="reef",
        groups_per_step=1,
        rollouts_per_group=3,
        max_workers=1,
    )

    results = harness.run_step(2)

    assert [result.reward for result in results] == [1.0, 0.0, 3.0]
    assert [report[1]["references"] for report in client.reports] == [
        ["agent-record-1"],
        ["agent-record-2"],
        ["agent-record-3"],
    ]
    assert {report[1]["metadata"]["comparison_set"] for report in client.reports} == {"tttd-step-2-group-0"}
    assert [report[1]["metadata"]["rollout"] for report in client.reports] == [0, 1, 2]
    assert all(report[1]["metadata"]["groups_per_step"] == 1 for report in client.reports)
    assert all(report[1]["metadata"]["rollouts_per_group"] == 3 for report in client.reports)
    assert client.reports[1][1]["feedback"] == "ValueError: response does not contain a Python code block"


def test_tttd_chat_request_builder_leaves_token_capture_to_backend():
    builder = TTTDChatRequestBuilder(max_new_tokens=64)

    payload = builder("qwen", [{"role": "user", "content": "discover"}], {})

    assert payload["model"] == "qwen"
    assert payload["messages"] == [{"role": "user", "content": "discover"}]
    assert payload["max_completion_tokens"] == 64
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == -1
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    # Reef's weight surface names the served adapter; the harness never does.
    assert "lora_path" not in payload


def test_tttd_chat_request_builder_rejects_adapter_override():
    builder = TTTDChatRequestBuilder()

    with pytest.raises(ValueError, match="weight surface"):
        builder("model", [{"role": "user", "content": "hi"}], {"lora_path": None})


def test_sandbox_environment_excludes_operator_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "leak-me")
    source = "import os\ndef entrypoint():\n    return {'token': os.environ.get('HF_TOKEN'), 'path': bool(os.environ.get('PATH'))}\n"
    result = execute_program(source, entrypoint="entrypoint", timeout_s=30, max_cpus=1)
    assert result.result["token"] is None
    assert result.result["path"] is True
