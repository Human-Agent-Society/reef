"""Concurrency protocol tests for scenario commits (issue #78, phase 1).

The version pointer (serving head + scenario_step) must move under one
concurrency protocol: per-scenario latches serialize the commit flow, and
every serving-head move is fenced with the head the writer observed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from tests.support._policy_recipe import TestPolicyRecipe
from tests.support.runtime_stubs import StubTrainingComponent, runtime_bindings

from reef.artifact import Artifact, ArtifactConflict, ArtifactRef, InMemoryRepositoryBackend, Repository
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.runtime.jobs import SavedCheckpoint, TrainingJob
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.types import TrainStepResult


class CountingComponent(StubTrainingComponent):
    """A stub training component that counts the jobs it trains, across threads."""

    def __init__(self, checkpoint_root: Path) -> None:
        super().__init__(checkpoint_root)
        self.lock = threading.Lock()
        self.train_calls = 0

    def train(self, job: TrainingJob) -> tuple[SavedCheckpoint, ...]:
        with self.lock:
            self.train_calls += 1
        return super().train(job)


def sft_inference(agent_record_id: str, runtime_load_id: str | None) -> AgentRecord:
    """An inference record from the version that served the scenario when it was created.

    Accepts race with commits, so no record can name the version that serves
    it. The tests set ``max_staleness`` to their job count instead: every
    record stays within the window of every later version.
    """
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={
            "tokens": [1, 2],
            "loss_mask": [0, 1],
            "rollout_log_probs": [-0.2],
            "runtime_load_id": runtime_load_id,
        },
        agent_record_id=agent_record_id,
    )


def sft_report(agent_record_id: str, reference: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": 1.0, "references": [reference]},
        agent_record_id=agent_record_id,
        references=(reference,),
    )


def run_workers(thread_count: int, worker) -> list[Exception]:
    errors: list[Exception] = []
    started: list[threading.Event] = [threading.Event() for _ in range(thread_count)]

    def guarded(seed: int, started_event: threading.Event) -> None:
        started_event.set()
        try:
            worker(seed)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(seed, started[seed])) for seed in range(thread_count)]
    for index, thread in enumerate(threads):
        thread.start()
        started[index].wait(timeout=0.1)
    for thread in threads:
        thread.join()
    return errors


#: The commits of one batch: the job's submission, its one checkpoint and its end.
COMMITS_PER_JOB = 3


def wait_for_jobs(dispatcher: Dispatcher, expected: int) -> None:
    """Wait until the scenario's trainer has finished ``expected`` jobs."""
    for _ in range(5000):
        if dispatcher.get_or_create_scenario("math").trainer.state["finished_jobs"] == expected:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")


@pytest.mark.unit
def test_activate_requires_the_observed_head(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend("math", initial, root=tmp_path / "repository")
    repository = Repository(backend, backend.resolve_release())
    repository.fork()
    head = repository.require_current_artifact()

    stale = ArtifactRef(content_id="artifact:stale", release_id="stale", parent_release_id=None)
    advanced = ArtifactRef(content_id="live:1", release_id="live:proc:w1:1", parent_release_id=head.release_id)
    with pytest.raises(ArtifactConflict, match="active artifact moved"):
        repository.activate(advanced, expected=stale)

    assert repository.current_artifact == head

    repository.activate(advanced, expected=head)
    assert repository.current_artifact == advanced


@pytest.mark.unit
def test_concurrent_accepts_train_and_commit_exactly_once_per_batch(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    component = CountingComponent(tmp_path / "checkpoints")
    pairs_per_thread = 4
    thread_count = 4
    dispatcher = Dispatcher(
        TestPolicyRecipe(
            **runtime_bindings(component),
            max_staleness=pairs_per_thread * thread_count,
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    base_version = dispatcher.get_or_create_scenario("math").serving_runtime_load_id()

    def worker(seed: int) -> None:
        for offset in range(pairs_per_thread):
            index = seed * pairs_per_thread + offset
            dispatcher.accept_record(sft_inference(f"i{index}", base_version))
            dispatcher.accept_record(sft_report(f"r{index}", f"i{index}"))

    errors = run_workers(thread_count, worker)

    assert errors == []
    total = pairs_per_thread * thread_count
    wait_for_jobs(dispatcher, total)
    scenario = dispatcher.get_or_create_scenario("math")
    assert component.train_calls == total
    assert len({job.job_id for job in component.submitted}) == total
    assert scenario.trainer.state["job"] is None
    assert scenario.scenario_step == COMMITS_PER_JOB * total
    # Each job published one saved checkpoint; the head is the last one.
    head = Path(dispatcher.current_artifact("math").materialize().local_path)
    assert (head / "model.txt").read_text() == f"w{total}"


@pytest.mark.unit
def test_async_worker_serializes_slow_trainer_commits(tmp_path, monkeypatch) -> None:
    """A slow commit must not let another job commit the same pending result.

    Accepts can wake the training worker while it is committing pending result
    P. The worker must finish that trainer and version-chain commit before preparing
    another batch; otherwise P could be recorded as two scenario steps.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    component = CountingComponent(tmp_path / "checkpoints")
    dispatcher = Dispatcher(
        TestPolicyRecipe(**runtime_bindings(component), max_staleness=4),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    scenario = dispatcher.get_or_create_scenario("math")
    base_version = scenario.serving_runtime_load_id()
    original_prepare_commit = scenario.trainer.prepare_commit

    def slow_prepare_commit(result):
        time.sleep(0.05)
        return original_prepare_commit(result)

    monkeypatch.setattr(scenario.trainer, "prepare_commit", slow_prepare_commit)

    def worker(seed: int) -> None:
        dispatcher.accept_record(sft_inference(f"i{seed}", base_version))
        dispatcher.accept_record(sft_report(f"r{seed}", f"i{seed}"))

    errors = run_workers(4, worker)

    assert errors == []
    wait_for_jobs(dispatcher, 4)
    assert component.train_calls == 4
    assert scenario.scenario_step == COMMITS_PER_JOB * 4


@pytest.mark.unit
def test_concurrent_checkpoint_commits_form_one_linear_chain(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = build_default_dispatcher(
        backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
        scenario_storage=SQLiteScenarioStorage(),
    )
    dispatcher.get_or_create_scenario("chat")
    commits_per_thread = 3
    thread_count = 4

    def worker(seed: int) -> None:
        for offset in range(commits_per_thread):
            index = seed * commits_per_thread + offset
            candidate = tmp_path / f"candidate-{index}"
            candidate.mkdir()
            (candidate / "model.txt").write_text(f"v{index}")
            dispatcher.registry.get("chat").commit(TrainStepResult(state=None, artifact=Artifact.local(candidate)))

    errors = run_workers(thread_count, worker)

    assert errors == []
    total = commits_per_thread * thread_count
    scenario = dispatcher.get_or_create_scenario("chat")
    assert scenario.scenario_step == total

    backend = scenario.repository.backend
    assert backend.current() == scenario.repository.current_artifact
    # Walk the durable release chain: every publish linked exactly one parent, so
    # the chain from the tip reaches the bootstrap ref in total + fork + base
    # hops, with no forks or skipped links.
    node = backend.current()
    nodes = 0
    while True:
        nodes += 1
        if node.parent_release_id is None:
            break
        node = backend.resolve_release(node.parent_release_id)
    assert nodes == total + 2
