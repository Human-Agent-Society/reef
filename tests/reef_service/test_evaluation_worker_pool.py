from __future__ import annotations

import multiprocessing
import os
import time
from pathlib import Path

import pytest

from reef.harness.adapters import get_adapter
from reef.harness.model_binding import ModelBinding
from reef.records import RecordStore
from reef.runtime.executor import ExecutorFailedError, WorkerSpec
from reef.runtime.executor.config import ExecutorSettings
from reef.train import ProcessorContext
from reef.train.cordis_backend.backend import CordisBackend
from reef.train.cordis_backend.execution import EvaluationWorkerPool, evaluation_selection
from reef.train.cordis_backend.strategies import EpisodeScorer, resolve_proposer
from reef.train.processors import DataProcessor
from reef.train.trainer import Trainer


class StatefulWorker:
    def __init__(self):
        self.calls = 0

    def run(self, value, delay=0):
        time.sleep(delay)
        self.calls += 1
        if value == "bad":
            raise ValueError("scorer rejected task")
        if value == "crash":
            os._exit(7)
        return os.getpid(), self.calls, value


class Scorer(EpisodeScorer):
    def __call__(self, task, result):
        return 1.0


def make_pool(backend):
    selection, requirements = evaluation_selection(Scorer(), 2, ExecutorSettings(backend))
    return EvaluationWorkerPool(selection, requirements, WorkerSpec(StatefulWorker))


@pytest.mark.parametrize("backend", ["mp", "local"])
def test_pool_reuses_workers_preserves_order_and_drains_business_failures(backend):
    pool = make_pool(backend)
    assert pool._executor is None
    try:
        first = pool.evaluate([("first",), ("second",)])
        executor = pool._executor
        assert [row[1:] for row in first] == [(1, "first"), (1, "second")]
        with pytest.raises(ValueError, match="scorer rejected"):
            pool.evaluate([("bad",), ("drained", 0.2), ("also drained",)])
        second = pool.evaluate([("again",), ("again",)])
        assert pool._executor is executor
        assert executor.failure is None
        assert [row[0] for row in second] == [row[0] for row in first]
        assert [row[1] for row in second] == [4, 3]
    finally:
        pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.evaluate([("new",)])
    if backend == "mp":
        assert not {row[0] for row in first} & {child.pid for child in multiprocessing.active_children()}


def test_pool_failure_is_not_restarted_or_replayed():
    pool = make_pool("mp")
    try:
        pool.evaluate([("one",)])
        executor = pool._executor
        with pytest.raises(ExecutorFailedError):
            pool.evaluate([("crash",), ("blocked", 120)])
        with pytest.raises(ExecutorFailedError):
            pool.evaluate([("not replayed",)])
        assert pool._executor is executor
    finally:
        pool.close()


def test_unused_pool_close_does_not_launch_workers():
    pool = make_pool("mp")
    pool.close()
    assert pool._executor is None
    with pytest.raises(RuntimeError, match="closed"):
        pool.evaluate([("never",)])


class IsolationScorer(EpisodeScorer):
    def __init__(self):
        self.calls = 0

    def __call__(self, task, result):
        row = result.trajectory[-1]
        if Path(row["workspace"]).exists() or row["rules"] != task:
            raise ValueError("episode files leaked or were not cleaned")
        self.calls += 1
        return float(self.calls)


@pytest.mark.parametrize("executor_backend, workers", [("uni", 1), ("mp", 2)])
def test_backend_reuses_scorers_but_isolates_real_episode_files(tmp_path, executor_backend, workers):
    binary = tmp_path / "fake-pi"
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
    backend = CordisBackend(
        descriptor=get_adapter("pi"),
        propose=resolve_proposer(lambda *args: None),
        score_episode=IsolationScorer(),
        tasks=("one",),
        binary=str(binary),
        models=ModelBinding(base_url="http://unused", model="unused", api_key="dummy"),
        episode_workers=workers,
        worker_executor=ExecutorSettings(executor_backend),
    )
    records = RecordStore()
    trainer = Trainer(
        scenario="test",
        records=records,
        processor=DataProcessor(ProcessorContext("test")),
        training_backend=backend,
        candidate_evaluator=None,
        state=backend.initial_state(),
    )
    try:
        for cycle in range(2):
            pairings = [({"pi-agent/AGENTS.md": "candidate"}, "candidate"), ({}, "")]
            scores = backend._evaluate_pairings(pairings)
            expected = [cycle * 2 + 1, cycle * 2 + 2] if workers == 1 else [cycle + 1, cycle + 1]
            assert [row[0] for row in scores] == expected
        paths = roots.read_text().splitlines()
        assert len(paths) == len(set(paths)) == 4
        assert all(not Path(path).exists() for path in paths)
    finally:
        trainer.close()
        records.close()
    trainer.close()
    assert backend._evaluation_pool._closed
    with pytest.raises(RuntimeError, match="closed"):
        backend._evaluate_pairings([({}, "")])
