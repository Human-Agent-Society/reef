"""Eval tasks stay held out (#698): once a consumed batch names an eval task, by its name or by one of its source
records, the evaluation skips it and every eval task sharing a source record with it for good, and eval failures
never reach the proposer's failure manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from reef_service._trajectories import recorded_trajectory
from reef_service.test_failure_manifest import manifest_state
from reef_service.test_harness_recipe import MODEL, RULES, batch, evaluate, make_binary, run_backend_step, runtime

from reef.core.tasks import HarborTask, TaskSplit, read_harbor_task, write_harbor_task, write_split_manifest
from reef.harness.adapters import get_adapter
from reef.recipe import RecipeConfigError
from reef.recipe.cordis import CordisRecipe
from reef.train.cordis_backend import (
    CordisBackend,
    EvalSplitTask,
    FailureManifest,
    HarnessCandidate,
    Mutation,
    ScoreComparisonPlugin,
)
from reef.train.cordis_backend.backend import exposed_eval_tasks, named_tasks
from reef.train.cordis_backend.strategies import resolve_episode_scorer, resolve_proposer
from reef.train.types import TrainingBatch, trajectories

EVAL_SPLIT_TASKS = {
    "task one": EvalSplitTask("held-1", frozenset({"source-1"})),
    "task two": EvalSplitTask("held-2", frozenset({"source-2"})),
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


def played_batch(name: str, **task: str) -> TrainingBatch:
    """A batch of one played episode whose report named the task ``name``, as the task player sends it."""
    sample = recorded_trajectory("played-1", {"messages": []}, 1.0).with_metadata(task={"name": name, **task})
    return TrainingBatch("demo:trace:played", (sample,))


def test_a_batch_that_names_an_eval_task_drops_it_from_this_and_every_later_evaluation(tmp_path: Path) -> None:
    b = exposure_backend(tmp_path, propose_marker)
    prepared = b.prepare_step(played_batch("held-1", path="/tasks/held-1", digest="a1" * 32), b.initial_state(), 0)
    candidate = prepared.candidate
    assert isinstance(candidate, HarnessCandidate)
    assert candidate.evaluation_tasks == ("task two",)
    assert prepared.metrics["not_run_tasks"] == 1
    assert prepared.state["consumed_task_names"] == ["held-1"]
    plugin = ScoreComparisonPlugin(b)
    evaluation = plugin.evaluate(candidate)
    assert len(evaluation.metrics["candidate_scores"]) == 1
    result = b.settle_step(prepared, plugin.decide(candidate, evaluation))
    assert result.state["consumed_task_names"] == ["held-1"]

    # The next batch names nothing, and task one stays out.
    later = b.prepare_step(batch(), result.state, 0)
    assert isinstance(later.candidate, HarnessCandidate) and later.candidate.evaluation_tasks == ("task two",)
    assert later.metrics["not_run_tasks"] == 1 and later.state["consumed_task_names"] == ["held-1"]


def test_a_report_without_a_digest_names_its_task_too(tmp_path: Path) -> None:
    """An older task player reports a task's name and path only, and a Harbor report its task_name."""
    b = exposure_backend(tmp_path, propose_marker)
    older = b.prepare_step(played_batch("held-2", path="/tasks/held-2"), b.initial_state(), 0)
    assert isinstance(older.candidate, HarnessCandidate) and older.candidate.evaluation_tasks == ("task one",)
    # A task named before it joined the eval split is exposed once a manifest puts it there.
    joined = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose_marker),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one", "task two", "task three"),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
        eval_split_tasks={**EVAL_SPLIT_TASKS, "task three": EvalSplitTask("held-3")},
    )
    state = {"steps": 1, "entries": [], "consumed_task_names": ["held-2", "held-3"]}
    later = joined.prepare_step(batch(), state, 0)
    assert isinstance(later.candidate, HarnessCandidate) and later.candidate.evaluation_tasks == ("task one",)


def test_an_eval_tasks_source_record_exposes_it_and_a_fully_exposed_split_skips_the_step(tmp_path: Path) -> None:
    calls: list[object] = []

    def propose(nodes, samples, models):
        calls.append(samples)
        return propose_marker(nodes, samples, models)

    b = exposure_backend(tmp_path, propose)
    # The trace was served from the record task two was made from.
    source_batch = TrainingBatch("demo:trace:source", (recorded_trajectory("source-2", {"messages": []}, 0.0),))
    state = {"steps": 1, "entries": [], "consumed_task_names": ["held-1"]}
    prepared = b.prepare_step(source_batch, state, 0)
    assert prepared.outcome == "skip" and prepared.metrics["skipped"] == "every evaluation task is exposed"
    assert prepared.metrics["not_run_tasks"] == 2
    assert prepared.state["consumed_task_names"] == ["held-1"]
    # The skip lands before the proposer and before a step record directory is claimed.
    assert calls == [] and not any((tmp_path / "record").iterdir())


def test_a_source_record_an_earlier_commit_consumed_exposes_its_eval_task(tmp_path: Path) -> None:
    """Tasks are made from served records after those records trained: the trainer's recovery tells the backend
    what every earlier commit consumed, and the task made from one of those records is not run."""
    b = exposure_backend(tmp_path, propose_marker)
    b.observe_consumed_records(frozenset({"source-2", "unrelated"}))
    prepared = b.prepare_step(batch(), b.initial_state(), 0)
    assert isinstance(prepared.candidate, HarnessCandidate) and prepared.candidate.evaluation_tasks == ("task one",)
    assert prepared.metrics["not_run_tasks"] == 1
    # A backend without an eval split keeps nothing.
    plain = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(propose_marker),
        score_episode=resolve_episode_scorer(evaluate),
        tasks=("task one",),
        models=MODEL,
        binary=str(make_binary(tmp_path)),
    )
    plain.observe_consumed_records(frozenset({"source-2"}))
    assert plain.consumed_record_ids == set()


def test_exposure_counts_eval_split_names_and_spreads_to_tasks_sharing_a_source_record() -> None:
    samples = (
        recorded_trajectory("other", {"messages": []}, 1.0).with_metadata(task={"name": "not-held", "digest": "c3"}),
        # A report's metadata is the client's; a name that is no string names nothing.
        recorded_trajectory("odd", {"messages": []}, 1.0).with_metadata(task={"name": ["held-2"]}),
        recorded_trajectory("plain", {"messages": []}, 1.0),
    )
    assert named_tasks(samples) == {"not-held"}
    assert exposed_eval_tasks(EVAL_SPLIT_TASKS, named_tasks(samples), set()) == set()
    assert exposed_eval_tasks(EVAL_SPLIT_TASKS, set(), {"source-1"}) == {"task one"}
    assert exposed_eval_tasks(EVAL_SPLIT_TASKS, named_tasks(trajectories(batch())), set()) == set()
    # A reworded sibling shares the exposed task's record, so the proposer has seen its content too.
    siblings = {
        **EVAL_SPLIT_TASKS,
        "task three": EvalSplitTask("held-3", frozenset({"source-2", "source-3"})),
        "task four": EvalSplitTask("held-4", frozenset({"source-3"})),
    }
    assert exposed_eval_tasks(siblings, {"held-4"}, set()) == {"task two", "task three", "task four"}


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
        str(root / name): EvalSplitTask(read_harbor_task(root / name).name, frozenset({record_id}))
        for name, record_id in (("held-1", "r1"), ("held-2", "r2"))
    }
    assert built._backend_kwargs()["eval_split_tasks"] == built.eval_split_tasks
    # A records batch carries no report, so nothing in it could name an eval task.
    with pytest.raises(RecipeConfigError, match=r"batch_policy 'records' cannot take evolution\.task_manifest"):
        CordisRecipe.from_environment(
            {}, config={"data": {"batch_policy": "records"}, "evolution": evolution}, runtime=runtime()
        )
    prompts = {"propose": "demo_evolution:propose", "evaluate": "demo_evolution:evaluate", "tasks": ["x"]}
    assert CordisRecipe.from_environment({}, config={"evolution": prompts}, runtime=runtime()).eval_split_tasks is None
