from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest
from tests.support.harness_steps import COMPONENT, TreeReleases, harness_recipe

from reef.core.evaluation import PolicyHarness, PolicyModel, RolloutRequest
from reef.evaluation.harness import EvaluationWorkerPool, TaskScorer, evaluation_selection
from reef.runtime.executor import ExecutorFailedError, WorkerSpec
from reef.runtime.executor.config import ExecutorSettings
from reef.train.backend import ComponentReleases


def run_requests(evaluator, *pairings):
    """One run request for each ``(files, task)``, on the evaluator's adapter and served model."""
    return [
        RolloutRequest(PolicyHarness(evaluator.descriptor.name, files), evaluator.served_model, task)
        for files, task in pairings
    ]


class StatefulWorker:
    """Counts its calls; each run is a ``(value, delay)`` pair, and the models are not read."""

    def __init__(self):
        self.calls = 0

    def run(self, task_run, models):
        value, delay = task_run
        time.sleep(delay)
        self.calls += 1
        if value == "bad":
            raise ValueError("scorer rejected task")
        if value == "crash":
            os._exit(7)
        return os.getpid(), self.calls, value


class Scorer(TaskScorer):
    def score(self, task, result):
        return 1.0


def make_pool(backend):
    selection, requirements = evaluation_selection(Scorer(), ExecutorSettings(backend, workers=2))
    return EvaluationWorkerPool(selection, requirements, WorkerSpec(StatefulWorker))


@pytest.mark.parametrize("backend", ["mp", "auto"])
def test_pool_reuses_workers_preserves_order_and_drains_business_failures(backend):
    pool = make_pool(backend)
    assert pool.executor is None
    try:
        first = pool.run([("first", 0), ("second", 0)], None)
        executor = pool.executor
        assert [row[1:] for row in first] == [(1, "first"), (1, "second")]
        with pytest.raises(ValueError, match="scorer rejected"):
            pool.run([("bad", 0), ("drained", 0.2), ("also drained", 0)], None)
        second = pool.run([("again", 0), ("again", 0)], None)
        assert pool.executor is executor
        assert executor.failure is None
        assert [row[0] for row in second] == [row[0] for row in first]
        assert [row[1] for row in second] == [4, 3]
    finally:
        pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.run([("new", 0)], None)
    assert not {row[0] for row in first} & {child.pid for child in multiprocessing.active_children()}


def test_pool_failure_is_not_restarted_or_replayed():
    pool = make_pool("mp")
    try:
        pool.run([("one", 0)], None)
        executor = pool.executor
        with pytest.raises(ExecutorFailedError):
            pool.run([("crash", 0), ("blocked", 120)], None)
        with pytest.raises(ExecutorFailedError):
            pool.run([("not replayed", 0)], None)
        assert pool.executor is executor
    finally:
        pool.close()


def test_unused_pool_close_does_not_launch_workers():
    pool = make_pool("mp")
    pool.close()
    assert pool.executor is None
    with pytest.raises(RuntimeError, match="closed"):
        pool.run([("never", 0)], None)


class IsolationScorer(TaskScorer):
    def __init__(self):
        self.calls = 0

    def score(self, task, result):
        row = result.trajectory[-1]
        if Path(row["workspace"]).exists() or row["rules"] != task:
            raise ValueError("run files leaked or were not cleaned")
        self.calls += 1
        return float(self.calls)


@pytest.mark.parametrize("executor_backend, workers", [("uni", 1), ("mp", 2)])
def test_backend_reuses_scorers_but_isolates_real_run_files(tmp_path, executor_backend, workers):
    binary = tmp_path / "isolation-pi"
    roots = tmp_path / "roots"
    binary.write_text(
        "#!/usr/bin/env python3\nimport os,json\nfrom pathlib import Path\n"
        "session=Path(os.environ['PI_CODING_AGENT_SESSION_DIR'])\n"
        "session.mkdir(parents=True,exist_ok=True)\n"
        "rules=Path(os.environ['PI_CODING_AGENT_DIR'])/'AGENTS.md'\n"
        "row={'type':'agent_end','workspace':os.getcwd(),'rules':rules.read_text() if rules.exists() else ''}\n"
        "(session/'session.jsonl').write_text(json.dumps(row)+'\\n')\n"
        f"with open({str(roots)!r}, 'a') as log: log.write(os.getcwd()+'\\n')\n"
    )
    binary.chmod(0o755)
    recipe = harness_recipe(
        tmp_path,
        lambda workspace, services: None,
        score=IsolationScorer(),
        tasks=("one",),
        worker_executor=ExecutorSettings(executor_backend, workers=workers),
        binary=str(binary),
    )
    backend = recipe.build_backend("test", ComponentReleases(TreeReleases(recipe.seed), COMPONENT), None)
    pool = backend.evaluator.worker_pool
    try:
        for cycle in range(2):
            requests = run_requests(
                backend.evaluator,
                ({"pi-agent/AGENTS.md": "candidate"}, "candidate"),
                ({"pi-agent/AGENTS.md": "current"}, "current"),
            )
            scores = backend.evaluator.run_rollouts(requests)
            expected = [cycle * 2 + 1, cycle * 2 + 2] if workers == 1 else [cycle + 1, cycle + 1]
            assert [row.score.value for row in scores] == expected
        evaluator = backend.evaluator
        for request, message in (
            (RolloutRequest(PolicyHarness("claude"), evaluator.served_model, "task"), "adapter"),
            (RolloutRequest(PolicyHarness("pi"), PolicyModel("another-model"), "task"), "serves"),
        ):
            with pytest.raises(ValueError, match=message):
                evaluator.run_rollouts([request])
        paths = roots.read_text().splitlines()
        assert len(paths) == len(set(paths)) == 4
        assert all(not Path(path).exists() for path in paths)
    finally:
        # The harness training component closes each backend it made; a second close changes nothing.
        backend.close()
    backend.close()
    assert pool.closed
    with pytest.raises(RuntimeError, match="closed"):
        backend.evaluator.run_rollouts(run_requests(backend.evaluator, ({}, "current")))
