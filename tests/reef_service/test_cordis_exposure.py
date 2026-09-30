"""Eval tasks stay held out (#698): once a consumed batch names an eval task, by its digest or by one of its source
records, the evaluation skips it for good, and eval failures never reach the proposer's failure manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service._trajectories import recorded_trajectory
from reef_service.test_failure_manifest import manifest_state
from reef_service.test_harness_recipe import MODEL, RULES, batch, evaluate, make_binary, run_backend_step, runtime

from reef.core.tasks import HarborTask, TaskSplit, read_harbor_task, write_harbor_task, write_split_manifest
from reef.harness.adapters import get_adapter
from reef.recipe.cordis import CordisRecipe
from reef.train.cordis_backend import (
    CordisBackend,
    EvalSplitTask,
    FailureManifest,
    HarnessCandidate,
    Mutation,
    ScoreComparisonPlugin,
)
from reef.train.cordis_backend.backend import exposed_eval_digests
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.types import TrainingBatch, trajectories

TASK_ONE_DIGEST = "a1" * 32
TASK_TWO_DIGEST = "b2" * 32
EVAL_SPLIT_TASKS = {
    "task one": EvalSplitTask(TASK_ONE_DIGEST, frozenset({"source-1"})),
    "task two": EvalSplitTask(TASK_TWO_DIGEST, frozenset({"source-2"})),
}


def exposure_backend(tmp_path: Path, propose, *, binary: str | None = None) -> CordisBackend:
    return CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one", "task two"),
        models=MODEL,
        binary=binary or str(make_binary(tmp_path)),
        step_record_dir=tmp_path / "record",
        eval_split_tasks=EVAL_SPLIT_TASKS,
    )


def propose_marker(nodes, samples, models):
    """A new rules entry each step, so a later step proposes too."""
    return Mutation("create", f"r{len(nodes) + 1}", {"name": "rules", "config": {"text": "marker rules"}})


def played_batch(digest: str) -> TrainingBatch:
    """A batch of one played episode whose report named the task with ``digest``, as the task player sends it."""
    sample = recorded_trajectory("played-1", {"messages": []}, 1.0).with_metadata(
        task={"name": "played", "path": "/tasks/played", "digest": digest}
    )
    return TrainingBatch("demo:trace:played", (sample,))


def test_a_batch_that_names_an_eval_task_drops_it_from_this_and_every_later_evaluation(tmp_path: Path) -> None:
    b = exposure_backend(tmp_path, propose_marker)
    prepared = b.prepare_step(played_batch(TASK_ONE_DIGEST), b.initial_state(), 0)
    candidate = prepared.candidate
    assert isinstance(candidate, HarnessCandidate)
    assert candidate.evaluation_tasks == ("task two",)
    assert prepared.metrics["not_run_tasks"] == 1
    assert prepared.state["exposed_task_digests"] == [TASK_ONE_DIGEST]
    plugin = ScoreComparisonPlugin(b)
    evaluation = plugin.evaluate(candidate)
    assert len(evaluation.metrics["candidate_scores"]) == 1
    result = b.settle_step(prepared, plugin.decide(candidate, evaluation))
    assert result.state["exposed_task_digests"] == [TASK_ONE_DIGEST]

    # The next batch names nothing, and task one stays out.
    later = b.prepare_step(batch(), result.state, 0)
    assert isinstance(later.candidate, HarnessCandidate) and later.candidate.evaluation_tasks == ("task two",)
    assert later.metrics["not_run_tasks"] == 1 and later.state["exposed_task_digests"] == [TASK_ONE_DIGEST]


def test_an_eval_tasks_source_record_exposes_it_and_a_fully_exposed_split_skips_the_step(tmp_path: Path) -> None:
    calls: list[object] = []

    def propose(nodes, samples, models):
        calls.append(samples)
        return propose_marker(nodes, samples, models)

    b = exposure_backend(tmp_path, propose)
    # The trace was served from the record task two was made from.
    source_batch = TrainingBatch("demo:trace:source", (recorded_trajectory("source-2", {"messages": []}, 0.0),))
    state = {"steps": 1, "entries": [], "exposed_task_digests": [TASK_ONE_DIGEST]}
    prepared = b.prepare_step(source_batch, state, 0)
    assert prepared.outcome == "skip" and prepared.metrics["skipped"] == "every evaluation task is exposed"
    assert prepared.metrics["not_run_tasks"] == 2
    assert prepared.state["exposed_task_digests"] == sorted([TASK_ONE_DIGEST, TASK_TWO_DIGEST])
    # The skip lands before the proposer and before a step record directory is claimed.
    assert calls == [] and not any((tmp_path / "record").iterdir())


def test_only_eval_split_digests_count_as_exposed() -> None:
    samples = (
        recorded_trajectory("other", {"messages": []}, 1.0).with_metadata(task={"name": "t", "digest": "c3" * 32}),
        # A report's metadata is the client's; a digest that is no string names nothing.
        recorded_trajectory("odd", {"messages": []}, 1.0).with_metadata(
            task={"name": "t", "digest": [TASK_TWO_DIGEST]}
        ),
        recorded_trajectory("source-1", {"messages": []}, 1.0),
    )
    assert exposed_eval_digests(samples, EVAL_SPLIT_TASKS) == {TASK_ONE_DIGEST}
    assert exposed_eval_digests(trajectories(batch()), EVAL_SPLIT_TASKS) == set()


def test_eval_failures_stay_out_of_the_proposers_failure_manifest(tmp_path: Path) -> None:
    received: list[FailureManifest | None] = []

    def propose(nodes, samples, models, *, manifest=None):
        received.append(manifest)
        return Mutation("create", f"r{len(received)}", RULES)

    # Every episode fails at launch; a state written before the task manifest still carries a failure manifest.
    b = exposure_backend(tmp_path, propose, binary=str(tmp_path / "no-such-binary"))
    first = run_backend_step(b, batch(), {"steps": 0, "entries": [], "failure_manifest": manifest_state()})
    assert received == [None]
    assert "failure_manifest" not in first.state and "failures" not in first.metrics
    # The failures stay in the evaluation record the pages read.
    failures = first.metrics["selection"]["evaluation"]["metrics"]["current_failures"]
    assert [failure["stage"] for failure in failures] == ["launch", "launch"]
    second = run_backend_step(b, batch(), dict(first.state))
    assert received == [None, None] and "failure_manifest" not in second.state


def test_the_backend_refuses_an_eval_split_that_is_not_its_task_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_split_tasks must name exactly the evaluation tasks"):
        CordisBackend(
            descriptor=get_adapter("pi"),
            propose=resolve_proposer(propose_marker),
            score_episode=resolve_episode_scorer(evaluate),
            tasks=("task one",),
            models=MODEL,
            binary=str(make_binary(tmp_path)),
            eval_split_tasks=EVAL_SPLIT_TASKS,
        )


def test_a_manifest_recipe_carries_the_eval_split_by_task_path(tmp_path: Path, monkeypatch) -> None:
    module = tmp_path / "demo_evolution.py"
    module.write_text(
        "def propose(nodes, samples, model):\n    return None\n\ndef evaluate(task, result):\n    return 0.0\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    root = tmp_path / "tasks"
    for name, record_id in (("held-1", "r1"), ("held-2", "r2"), ("train-1", "r3")):
        write_harbor_task(
            HarborTask(
                name=name,
                instruction=f"task {name}",
                tests={"test.sh": "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"},
                environment={"Dockerfile": "FROM python:3.12-slim\n"},
                source_agent_record_ids=(record_id,),
            ),
            root,
        )
    write_split_manifest(tmp_path / "split.json", TaskSplit(("train-1",), ("held-1", "held-2"), 0, 0.5))
    evolution = {
        "propose": "demo_evolution:propose",
        "evaluate": "demo_evolution:evaluate",
        "task_manifest": str(tmp_path / "split.json"),
        "tasks_root": str(root),
        "adapter": "terminus",
    }
    built = CordisRecipe.from_environment({}, config={"evolution": evolution}, runtime=runtime())
    assert built.eval_split_tasks == {
        str(root / name): EvalSplitTask(read_harbor_task(root / name).digest, frozenset({record_id}))
        for name, record_id in (("held-1", "r1"), ("held-2", "r2"))
    }
    assert built._backend_kwargs()["eval_split_tasks"] == built.eval_split_tasks
    prompts = {"propose": "demo_evolution:propose", "evaluate": "demo_evolution:evaluate", "tasks": ["x"]}
    assert CordisRecipe.from_environment({}, config={"evolution": prompts}, runtime=runtime()).eval_split_tasks is None
