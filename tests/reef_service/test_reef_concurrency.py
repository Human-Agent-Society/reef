"""Concurrency protocol tests for scenario commits (issue #78, phase 1).

The version pointer (serving head + scenario_step) must move under one
concurrency protocol: per-scenario latches serialize the commit flow, and
every serving-head move is fenced with the head the writer observed.
"""

from __future__ import annotations

import threading
import time

import pytest

from reef.artifact import (
    Artifact,
    ArtifactConflict,
    ArtifactRef,
    InMemoryRepositoryBackend,
    LiveWeightArtifactRef,
    Repository,
)
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.recipe import RecipeRegistry
from reef.runtime import ActivatedModel, ModelCandidate, PreparedTrainingStep, TrainingRuntime
from reef.scenario.checkpoint_strategy import EveryNVersions
from reef.train.evaluation import SelectionDecision
from reef.train.types import TrainStepResult

from ._policy_recipe import TestPolicyRecipe


class CountingRuntime(TrainingRuntime):
    """Training runtime that counts durable jobs across threads."""

    def __init__(self) -> None:
        super().__init__(base_url="http://trainer")
        self._lock = threading.Lock()
        self.train_calls = 0

    @property
    def inference_backend(self):
        return None

    def prepare_training_step(self, batch, step_preparer, algorithm_state, scenario_step):
        del batch, step_preparer
        return PreparedTrainingStep(
            action="train",
            payload={"rollout_id": scenario_step},
            next_algorithm_state={"steps": int(algorithm_state.get("steps", 0)) + 1},
            metrics={},
        )

    def train_candidate(self, payload):
        with self._lock:
            self.train_calls += 1
            version = f"w{self.train_calls}"
        job_id = f"job-{payload['rollout_id']}"
        return ModelCandidate(
            candidate_id=job_id,
            training_job_id=job_id,
            checkpoint_path="/unused",
            current_weight_version=None,
            metadata={"weight_version": version},
        )

    def activate_candidate(self, candidate):
        return ActivatedModel(candidate.candidate_id, str(candidate.metadata["weight_version"]))

    def reject_candidate(self, candidate, decision: SelectionDecision):
        del candidate, decision


def sft_inference(agent_record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.INFERENCE,
        payload={"tokens": [1, 2], "loss_mask": [0, 1], "rollout_log_probs": [-0.2]},
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


def wait_for_steps(dispatcher: Dispatcher, expected: int) -> None:
    for _ in range(5000):
        if dispatcher.get_or_create_scenario("math").scenario_step == expected:
            return
        time.sleep(0.001)
    raise AssertionError("async training did not commit")


@pytest.mark.unit
def test_advance_current_requires_the_observed_head(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    backend = InMemoryRepositoryBackend("math", initial, root=tmp_path / "repository")
    repository = Repository(backend, backend.resolve_version())
    repository.fork()
    head = repository.require_current_artifact()

    stale = ArtifactRef(artifact_id="artifact:stale", version="stale", parent_version=None)
    advanced = LiveWeightArtifactRef(
        artifact_id="live:1",
        version="live:proc:w1:1",
        parent_version=head.version,
        weight_version="w1",
    )
    with pytest.raises(ArtifactConflict, match="serving head advanced"):
        repository.advance_current(advanced, expected=stale)

    assert repository.current_artifact == head

    repository.advance_current(advanced, expected=head)
    assert repository.current_artifact == advanced


@pytest.mark.unit
def test_concurrent_accepts_train_and_commit_exactly_once_per_batch(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = CountingRuntime()
    dispatcher = Dispatcher(
        RecipeRegistry(
            recipes={"test_policy": TestPolicyRecipe(runtime, batch_size=1, checkpoint_strategy=EveryNVersions(1000))}
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
    )
    pairs_per_thread = 4
    thread_count = 4

    def worker(seed: int) -> None:
        for offset in range(pairs_per_thread):
            index = seed * pairs_per_thread + offset
            dispatcher.accept_record(sft_inference(f"i{index}"), recipe="test_policy")
            dispatcher.accept_record(sft_report(f"r{index}", f"i{index}"), recipe="test_policy")

    errors = run_workers(thread_count, worker)

    assert errors == []
    total = pairs_per_thread * thread_count
    scenario = dispatcher.get_or_create_scenario("math")
    wait_for_steps(dispatcher, total)
    assert runtime.train_calls == total
    assert scenario.trainer.state == {"steps": total}
    assert scenario.scenario_step == total
    head = scenario.repository.require_current_artifact()
    assert isinstance(head, LiveWeightArtifactRef)
    assert head.weight_version == f"w{total}"
    assert head.version.endswith(f":w{total}:{total}")


@pytest.mark.unit
def test_async_worker_serializes_slow_trainer_commits(tmp_path, monkeypatch) -> None:
    """A slow commit must not let another job commit the same pending result.

    Accepts can wake the training worker while it is committing pending result
    P. The worker must finish that trainer and version-chain commit before preparing
    another batch; otherwise P could be recorded as two scenario steps.
    """
    initial = tmp_path / "initial"
    initial.mkdir()
    runtime = CountingRuntime()
    dispatcher = Dispatcher(
        RecipeRegistry(
            recipes={"test_policy": TestPolicyRecipe(runtime, batch_size=1, checkpoint_strategy=EveryNVersions(1000))}
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        local_artifact_dir=tmp_path / "staged",
    )
    scenario = dispatcher.get_or_create_scenario("math", "test_policy")
    original_commit = scenario.trainer.commit

    def slow_commit():
        time.sleep(0.05)
        return original_commit()

    monkeypatch.setattr(scenario.trainer, "commit", slow_commit)

    def worker(seed: int) -> None:
        dispatcher.accept_record(sft_inference(f"i{seed}"), recipe="test_policy")
        dispatcher.accept_record(sft_report(f"r{seed}", f"i{seed}"), recipe="test_policy")

    errors = run_workers(4, worker)

    assert errors == []
    wait_for_steps(dispatcher, 4)
    assert runtime.train_calls == 4
    assert scenario.scenario_step == 4


@pytest.mark.unit
def test_concurrent_checkpoint_commits_form_one_linear_chain(tmp_path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = build_default_dispatcher(
        backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        checkpoint_strategy=EveryNVersions(1),
        local_artifact_dir=tmp_path / "staged",
    )
    dispatcher.get_or_create_scenario("chat", "recipe")
    commits_per_thread = 3
    thread_count = 4

    def worker(seed: int) -> None:
        for offset in range(commits_per_thread):
            index = seed * commits_per_thread + offset
            candidate = tmp_path / f"candidate-{index}"
            candidate.mkdir()
            (candidate / "model.txt").write_text(f"v{index}")
            dispatcher._commit_result("chat", TrainStepResult(state=None, artifact=Artifact.local(candidate)))

    errors = run_workers(thread_count, worker)

    assert errors == []
    total = commits_per_thread * thread_count
    scenario = dispatcher.get_or_create_scenario("chat")
    assert scenario.scenario_step == total

    backend = scenario.repository.backend
    assert backend.current() == scenario.repository.current_artifact
    # Walk the durable version chain: every publish linked exactly one parent, so
    # the chain from the tip reaches the bootstrap ref in total + fork + base
    # hops, with no forks or skipped links.
    node = backend.current()
    nodes = 0
    while True:
        nodes += 1
        if node.parent_version is None:
            break
        node = backend.resolve_version(node.parent_version)
    assert nodes == total + 2


@pytest.mark.unit
def test_saved_commit_fails_loudly_when_head_moves_mid_commit(tmp_path, monkeypatch) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = build_default_dispatcher(
        backend_factory=InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
        checkpoint_strategy=EveryNVersions(1000),
        local_artifact_dir=tmp_path / "staged",
    )
    scenario = dispatcher.get_or_create_scenario("chat", "recipe")
    repository = scenario.repository
    original_stage = repository.stage

    def stage_and_move_head(*args, **kwargs):
        staged = original_stage(*args, **kwargs)
        # A concurrent writer moving the serving head after this commit
        # observed it: the fenced advance must fail instead of overwriting.
        intruder = LiveWeightArtifactRef(
            artifact_id="live:intruder",
            version="live:other:w9:9",
            parent_version=None,
            weight_version="w9",
        )
        Repository.advance_current(repository, intruder, expected=repository.require_current_artifact())
        return staged

    monkeypatch.setattr(repository, "stage", stage_and_move_head)
    candidate = tmp_path / "candidate"
    candidate.mkdir()

    with pytest.raises(ArtifactConflict, match="serving head advanced"):
        dispatcher._commit_result("chat", TrainStepResult(state=None, artifact=Artifact.local(candidate)))

    assert scenario.scenario_step == 0
    assert repository.current_artifact.version == "live:other:w9:9"
    assert list((tmp_path / "staged").rglob("*")) == []
