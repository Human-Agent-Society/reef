"""Memory ablation uses the real backend history, persists its cadence, and pairs successful runs."""

import json
from dataclasses import replace

import pytest
from tests.support.harness_service import backend_step
from tests.support.harness_steps import PI_FAKE, PI_SEED, TreeArtifacts, build_harness_backend, harness_recipe

from recipes.reasoning_bank import ReasoningBank
from recipes.reasoning_bank.evaluation import paired_result
from recipes.reasoning_bank.reading import RUNTIME_PATHS, without_memory
from reef.core.evaluation import RolloutScore
from reef.evaluation.harness import RunRecord
from reef.evaluation.harness.scoring import FailureObservation
from reef.harness.tree import HarnessTree
from reef.train.harness_backend.state import HarnessState
from reef.train.types import JobData


@pytest.mark.parametrize("policy", ["none", "floor"])
def test_ablation_survives_restart_and_records_a_loss_even_when_tree_is_unchanged(tmp_path, policy):
    bank = [
        {
            "slug": "check",
            "title": "Check",
            "description": "Use for checking.",
            "content": "Check the result.",
            "outcome": "success",
            "step": 1,
        }
    ]
    method = ReasoningBank.from_config({"ablate_every": 2, "reading": {"reader": "all"}})
    recipe = harness_recipe(
        tmp_path,
        method,
        evaluation={"policy": policy},
        score=lambda task, result: 1.0 if result.trajectory[-1]["rules"] == "memory" else 2.0,
    )
    # The stand-in agent reports whether the published reader was rendered into its package.
    binary = tmp_path / "fake-pi"
    binary.write_text(
        PI_FAKE.replace(
            'rules_path.read_text() if rules_path.exists() else ""',
            '"memory" if (package / "extensions" / "reasoning_bank.ts").exists() else ""',
        )
    )
    artifacts = TreeArtifacts(PI_SEED)
    backend = build_harness_backend(recipe, artifacts=artifacts)
    state = HarnessState(method_files={"memory.json": json.dumps(bank)}).to_json()
    first = backend_step(backend, JobData("demo:1", ()), state, recipe.build_policy(backend))
    assert "memory_ablation" not in first.metrics["method_notes"]
    if policy == "none":
        assert not backend.history.runs
    assert first.artifact is not None
    head = HarnessTree.read(first.artifact.local_path)
    artifacts.publish("one", head)
    # A newly built backend resumes the committed counter, not an in-memory counter.
    backend = build_harness_backend(recipe, artifacts=artifacts)
    second = backend_step(backend, JobData("demo:2", ()), first.state, recipe.build_policy(backend), scenario_step=1)
    result = second.metrics["method_notes"]["memory_ablation"]
    assert result["status"] == "harmful" and result["delta"] == -1.0
    assert result["pairs"] == 1 and result["excluded_pairs"] == 0
    assert second.metrics["skipped"] == "the method left the head tree unchanged"
    saved = json.loads(HarnessState.from_json(second.state).method_files["memory_evaluation.json"])
    assert saved == {"steps": 2, "last_result": result}
    runs = {run.run_id: run for run in backend.history.runs}
    assert runs[result["with_runs"][0]].tree == head.checksum()
    assert runs[result["without_runs"][0]].tree == PI_SEED.checksum()
    assert artifacts.head("harness").files == head.files


def test_ablation_requires_tasks_and_a_scorer_before_learning(tmp_path):
    method = ReasoningBank.from_config({"ablate_every": 1})
    recipe = harness_recipe(tmp_path, method, evaluation={"policy": "none"}, tasks=())
    backend = build_harness_backend(recipe)
    with pytest.raises(ValueError, match="evaluation tasks and a scorer"):
        backend_step(backend, JobData("demo:1", ()), HarnessState().to_json())


def test_empty_bank_records_no_comparison_and_launches_no_runs(tmp_path):
    method = ReasoningBank.from_config({"ablate_every": 1})
    recipe = harness_recipe(tmp_path, method, evaluation={"policy": "none"})
    backend = build_harness_backend(recipe)
    result = backend_step(backend, JobData("demo:1", ()), HarnessState().to_json())
    assert result.metrics["method_notes"]["memory_ablation"] == {
        "status": "empty",
        "pairs": 0,
        "delta": None,
        "step": 1,
    }
    assert not backend.history.runs


def test_ablation_removes_only_owned_memory_files():
    keep = {
        "rules.md": "Original rules",
        "skills/review/SKILL.md": "Review",
        "extensions/other.ts": "export default () => {}",
    }
    files = {**keep, "skills/rb-check/SKILL.md": "Memory", **dict.fromkeys(RUNTIME_PATHS, "reader")}
    assert without_memory(HarnessTree(files)) == HarnessTree(keep)


def test_pairing_excludes_failed_or_unmatched_runs_from_both_means():
    with_run = RunRecord("with-1", "bank", "one", "one", 0, "context", 1, RolloutScore.scored(0.5))
    without_run = replace(with_run, run_id="without-1", tree="empty", score=RolloutScore.scored(1))
    failure = replace(
        with_run,
        run_id="failed",
        task_key="two",
        score=RolloutScore.failed(),
        failure=FailureObservation("two", "launch", "unavailable"),
    )
    unmatched = replace(with_run, run_id="unmatched", task_key="three", score=RolloutScore.scored(100))
    without_two = replace(without_run, run_id="without-2", task_key="two")
    result = paired_result([with_run, failure, unmatched], [without_run, without_two])
    assert (result["pairs"], result["excluded_pairs"], result["delta"], result["status"]) == (1, 2, -0.5, "harmful")
    assert paired_result([failure], [without_two])["status"] == "inconclusive"
    with pytest.raises(ValueError, match="unique"):
        paired_result([with_run, with_run], [without_run])
    with pytest.raises(ValueError, match="same evaluation context"):
        paired_result([with_run], [replace(without_run, context="changed-model")])
