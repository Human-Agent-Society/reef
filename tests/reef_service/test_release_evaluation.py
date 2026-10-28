"""Retained releases go through the real artifact, episode and scoring paths."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from threading import Event

import pytest

from reef.artifact import Artifact, ArtifactNotFound, InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.dispatcher import Dispatcher
from reef.harness.adapters import get_adapter
from reef.harness.episodes.model_binding import ModelBinding, ModelBindings
from reef.harness.episodes.run import EpisodeResult
from reef.harness.tree.mutations import Mutation
from reef.inference.http import InferenceProxyRuntime
from reef.recipe.cordis import CordisRecipe
from reef.scenario.evaluation import EvaluationConditions, EvaluationTask, RetainedHarnessEvaluation
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend.strategies import EpisodeScorer, Proposer
from reef.train.evaluation.evaluators import AlwaysSelectPluginFactory
from reef.train.types import TrainStepResult

pytestmark = pytest.mark.skipif(os.name != "posix", reason="prototype run lock requires POSIX")


class ScriptedProposal(Proposer):
    def __init__(self):
        self.index = 0

    def __call__(self, nodes, samples, models, **kwargs):
        rules = ("old", "old new", "new")[self.index]
        operation = "create" if self.index == 0 else "update"
        self.index += 1
        return Mutation(operation, "evaluation-rules", {"name": "rules", "config": {"text": rules}})


class RulesScorer(EpisodeScorer):
    def __call__(self, task: str, result: EpisodeResult) -> float:
        if task == "bad-score":
            return float("nan")
        if task == "zero":
            return 0.0
        return float(task in result.trajectory[-1]["rules"])


BINARY = """#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
prompt = sys.argv[sys.argv.index('-p') + 1]
if prompt == 'timeout':
    time.sleep(30)
agent = Path(os.environ['PI_CODING_AGENT_DIR'])
sessions = Path(os.environ['PI_CODING_AGENT_SESSION_DIR'])
sessions.mkdir(parents=True, exist_ok=True)
rules_path = agent / 'AGENTS.md'
rules = rules_path.read_text() if rules_path.exists() else ''
if prompt == 'broken-trajectory':
    (sessions / 'session.jsonl').write_text('bad-json\\nbad-json\\n')
else:
    (sessions / 'session.jsonl').write_text(json.dumps({'type': 'agent_end', 'rules': rules}) + '\\n')
"""


@pytest.fixture
def retained(tmp_path):
    binary = tmp_path / "fixture-harness"
    binary.write_text(BINARY.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    binary.chmod(0o700)
    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        CordisRecipe(
            ScriptedProposal(),
            RulesScorer(),
            ("old",),
            binary=str(binary),
            candidate_plugin=AlwaysSelectPluginFactory(),
            runtime=InferenceProxyRuntime(model_path="fixture", base_url="http://127.0.0.1:1"),
        ),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "artifacts"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "state"),
        agent_record_dir=tmp_path / "records",
    )
    scenario = dispatcher.get_or_create_scenario("comparison")
    releases = []
    for index in range(3):
        # Publish a deliberately regressing third version through the actual
        # training/commit path, using a scripted proposer and explicit selection.
        scenario.records.append_result(
            AgentRecord.create(
                scenario=scenario.name,
                request_type=RequestType.INFERENCE,
                payload={"messages": [{"role": "user", "content": "old"}]},
                agent_record_id=f"i{index}",
            )
        )
        scenario.records.append_result(
            AgentRecord.create(
                scenario=scenario.name,
                request_type=RequestType.REPORT,
                payload={"score": 0.0, "references": [f"i{index}"]},
                agent_record_id=f"r{index}",
            )
        )
        result = scenario.prepare_training_step()
        assert result is not None
        scenario.commit(result)
        releases.append(scenario.current_artifact_ref().release_id)
    conditions = EvaluationConditions("suite-v1", "rules-v1", "no-model-fixture", "fixture-v1")
    try:
        yield scenario, releases, binary, conditions
    finally:
        dispatcher.close()


def evaluator(retained, *, tasks=None, conditions=None, models=None):
    scenario, releases, binary, original = retained
    return RetainedHarnessEvaluation(
        scenario,
        releases,
        tasks or (EvaluationTask("old", "old"), EvaluationTask("new", "new")),
        conditions or original,
        descriptor=get_adapter("pi"),
        scorer=RulesScorer(),
        binary=str(binary),
        models=models,
    )


def test_retained_release_matrix_detects_regression_without_mutation(retained, tmp_path):
    scenario, releases, _, _ = retained
    head = scenario.current_artifact_ref()
    history = scenario.store.history()
    run = evaluator(retained)
    results = run.run(tmp_path / "evaluation")
    assert [r.score for r in results] == [1, 1, 0, 0, 1, 1]
    assert scenario.current_artifact_ref() == head
    assert scenario.store.history() == history
    assert {r.release_id for r in results} == set(releases)
    assert all(r.status == "scored" for r in results)
    assert "-1.0000" in (tmp_path / "evaluation/report.md").read_text()
    assert json.loads((tmp_path / "evaluation/run.json").read_text())["fingerprint"] == run.fingerprint


def test_resume_preserves_completed_records_and_budget(retained, tmp_path):
    output = tmp_path / "evaluation"
    first = evaluator(retained).run(output, max_new_episodes=2)
    assert [r.status for r in first] == ["scored", "scored", "unrun", "unrun", "unrun", "unrun"]
    saved = (output / "results/00000000.json").read_bytes()
    second = evaluator(retained).run(output)
    assert len(second) == 6 and all(r.status == "scored" for r in second)
    assert (output / "results/00000000.json").read_bytes() == saved
    assert second[:2] == first[:2]


@pytest.mark.parametrize("change", ["suite_version", "scorer_version", "model_version", "environment_version"])
def test_changed_conditions_refuse_resume(retained, tmp_path, change):
    output = tmp_path / "evaluation"
    evaluator(retained).run(output, max_new_episodes=0)
    conditions = replace(retained[3], **{change: "changed"})
    with pytest.raises(ValueError, match="conditions changed"):
        evaluator(retained, conditions=conditions).run(output)


def test_changed_prompt_and_binary_refuse_reuse(retained, tmp_path):
    output = tmp_path / "evaluation"
    run = evaluator(retained)
    run.run(output, max_new_episodes=0)
    with pytest.raises(ValueError, match="conditions changed"):
        evaluator(retained, tasks=(EvaluationTask("old", "changed"),)).run(output)
    binary = retained[2]
    binary.write_text(binary.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="binary changed"):
        run.run(output)


def test_error_invalid_score_zero_and_unrun_are_distinct(retained, tmp_path):
    tasks = tuple(EvaluationTask(prompt, prompt) for prompt in ("zero", "bad-score", "broken-trajectory", "old"))
    results = evaluator(retained, tasks=tasks).run(tmp_path / "evaluation", max_new_episodes=9)
    assert [(r.status, r.score) for r in results[::3]] == [
        ("scored", 0.0),
        ("invalid_score", None),
        ("execution_error", None),
        ("unrun", None),
    ]
    report = (tmp_path / "evaluation/report.md").read_text()
    assert "1 / 4" in report and "| 2 | 1 | 1 |" in report


def test_cancelled_run_can_resume(retained, tmp_path):
    cancel = Event()
    cancel.set()
    output = tmp_path / "evaluation"
    results = evaluator(retained).run(output, cancel=cancel)
    assert all(r.status == "unrun" for r in results)
    assert not list((output / "results").glob("*.json"))
    cancel.clear()
    assert all(r.status == "scored" for r in evaluator(retained).run(output, cancel=cancel))


def test_timeout_is_not_a_zero_score(retained, tmp_path):
    conditions = replace(retained[3], episode_timeout_seconds=0.05)
    results = evaluator(retained, tasks=(EvaluationTask("timeout", "timeout"),), conditions=conditions).run(
        tmp_path / "evaluation"
    )
    assert all(r.status == "execution_error" and r.score is None for r in results)


def test_corrupted_or_misplaced_result_is_rejected(retained, tmp_path):
    output = tmp_path / "evaluation"
    run = evaluator(retained)
    run.run(output, max_new_episodes=1)
    record = output / "results/00000000.json"
    raw = json.loads(record.read_text())
    raw["result"]["release_id"] = "other-release"
    record.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="does not match"):
        run.run(output)


def test_unknown_release_never_falls_back_to_served(retained):
    scenario, releases, binary, conditions = retained
    with pytest.raises(ArtifactNotFound):
        RetainedHarnessEvaluation(
            scenario,
            [releases[0], "missing"],
            (EvaluationTask("old", "old"),),
            conditions,
            descriptor=get_adapter("pi"),
            scorer=RulesScorer(),
            binary=str(binary),
        )


def test_model_binding_is_applied_without_exporting_credentials(retained, tmp_path):
    secret = "test-credential-must-stay-out-of-manifest"
    models = ModelBindings(served=ModelBinding(base_url="http://localhost:1", model="fixed-model", api_key=secret))
    run = evaluator(retained, models=models)
    assert any(secret in text for files in run.files for text in files.values())
    output = tmp_path / "evaluation"
    run.run(output)
    assert all(secret not in path.read_text() for path in output.rglob("*.json"))


def test_exclusive_run_lock(retained, tmp_path):
    import fcntl

    output = tmp_path / "evaluation"
    output.mkdir()
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another evaluation"):
            evaluator(retained).run(output)


def test_interrupted_result_write_resumes_only_unrecorded_work(retained, tmp_path, monkeypatch):
    import reef.scenario.evaluation as module

    output = tmp_path / "evaluation"
    original = module.atomic_json

    def interrupted_write(path, value):
        if path.name == "00000001.json":
            raise OSError("injected disk failure")
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", interrupted_write)
    with pytest.raises(OSError, match="injected"):
        evaluator(retained).run(output)
    saved = (output / "results/00000000.json").read_bytes()
    assert not (output / "results/00000001.json").exists()
    monkeypatch.setattr(module, "atomic_json", original)
    results = evaluator(retained).run(output)
    assert all(result.status == "scored" for result in results)
    assert (output / "results/00000000.json").read_bytes() == saved


def test_cancellation_during_scoring_stops_next_admission(retained, tmp_path):
    cancel = Event()

    class CancellingScorer(RulesScorer):
        def __call__(self, task, result):
            cancel.set()
            return super().__call__(task, result)

    run = evaluator(retained)
    run.worker.scorer = CancellingScorer()
    results = run.run(tmp_path / "evaluation", cancel=cancel)
    assert results[0].status == "scored"
    assert all(result.status == "unrun" for result in results[1:])


def test_pending_release_is_not_a_historical_target(retained, tmp_path):
    scenario, releases, binary, conditions = retained
    directory = tmp_path / "pending"
    directory.mkdir()
    (directory / "rules.txt").write_text("pending")
    head = scenario.current_artifact_ref()
    scenario.commit(TrainStepResult(None, artifact=Artifact.local(directory), pending=True))
    pending_id = scenario.releases()[0]["release_id"]
    assert scenario.current_artifact_ref() == head
    with pytest.raises(ValueError, match="pending publication"):
        RetainedHarnessEvaluation(
            scenario,
            [releases[0], pending_id],
            [EvaluationTask("old", "old")],
            conditions,
            descriptor=get_adapter("pi"),
            scorer=RulesScorer(),
            binary=str(binary),
        )


def test_changed_suite_can_be_evaluated_in_a_new_directory(retained, tmp_path):
    before = evaluator(retained).run(tmp_path / "original")
    after = evaluator(
        retained, tasks=[EvaluationTask("old", "old")], conditions=replace(retained[3], suite_version="suite-v2")
    ).run(tmp_path / "changed")
    assert len(before) == 6 and len(after) == 3
    assert [outcome.score for outcome in after] == [1, 1, 0]


def test_model_judge_failure_remains_an_error_without_leaking_detail(retained, tmp_path):
    from reef.harness.episodes.model_binding import ModelBindingError

    class FailedJudge(RulesScorer):
        def __call__(self, task, result):
            raise ModelBindingError("private-provider-detail", status=429)

    run = evaluator(retained)
    run.worker.scorer = FailedJudge()
    output = tmp_path / "evaluation"
    results = run.run(output)
    assert all(result.status == "execution_error" and result.failure_stage == "scorer_model" for result in results)
    assert "private-provider-detail" not in (output / "run.json").read_text()


def test_nonzero_exit_cannot_count_as_success(retained, tmp_path):
    _, _, binary, _ = retained
    binary.write_text(binary.read_text() + "\nsys.exit(7)\n")
    results = evaluator(retained).run(tmp_path / "nonzero")
    assert all(
        row.status == "execution_error" and row.score is None and row.failure_stage == "exit" for row in results
    )


def test_episode_records_are_checked_on_resume(retained, tmp_path):
    run = evaluator(retained, conditions=replace(retained[3], retain_episodes=True))
    output = tmp_path / "recorded"
    results = run.run(output, max_new_episodes=1)
    assert results[0].episode_checksum is not None
    record = output / "episodes/00000000.json"
    assert json.loads(record.read_text())["trajectory"][-1]["rules"] == "old\n"
    record.write_text("{}")
    with pytest.raises(ValueError, match="episode is missing or changed"):
        run.run(output)


def test_model_binding_preserves_retained_bytes_without_training_entries(retained, tmp_path, monkeypatch):
    from reef.harness.tree.render import render_composition
    from reef.train.cordis_backend.backend import tree_files

    scenario, releases, binary, conditions = retained
    entries = ({"id": "rules", "name": "rules", "config": {"text": "seed"}},)
    descriptor = get_adapter("native")
    retained_files = {
        **render_composition((("rules", {"text": "seed"}),), descriptor),
        **tree_files(descriptor, entries),
    }
    monkeypatch.setattr(scenario, "entries_for_version", lambda release: None)
    monkeypatch.setattr(scenario.surface.files, "read_files", lambda artifact: retained_files)
    models = ModelBindings(served=ModelBinding(base_url="http://127.0.0.1:1", model="fixture"))
    run = RetainedHarnessEvaluation(
        scenario,
        releases,
        [EvaluationTask("seed", "seed")],
        conditions,
        descriptor=descriptor,
        scorer=RulesScorer(),
        binary=str(binary),
        models=models,
    )
    assert all(json.loads(files[descriptor.tree_path]) == list(entries) for files in run.files)
    # A historical renderer can serialize tool dictionaries in a different
    # order. Even then, evaluation must execute the actual retained module.
    retained_files["native/tools/historical.py"] = "PARAMETERS = {'z': 0, 'a': 1}\n"
    second = RetainedHarnessEvaluation(
        scenario,
        releases,
        [EvaluationTask("seed", "seed")],
        conditions,
        descriptor=descriptor,
        scorer=RulesScorer(),
        binary=str(binary),
        models=models,
    )
    for files in second.files:
        assert files["native/tools/historical.py"] == retained_files["native/tools/historical.py"]
        assert {path: content for path, content in files.items() if path != "native/models.json"} == {
            path: content for path, content in retained_files.items() if path != "native/models.json"
        }


def stop_after_first_checkpoint(evaluation, output):
    import reef.scenario.evaluation as module

    write_record = module.atomic_json

    def stop_after_result(path, value):
        write_record(path, value)
        if path.parent.name == "results":
            os._exit(23)

    module.atomic_json = stop_after_result
    evaluation.run(output)


def test_process_death_releases_lock_and_preserves_completed_checkpoint(retained, tmp_path):
    import multiprocessing

    run = evaluator(retained)
    output = tmp_path / "process-death"
    process = multiprocessing.get_context("spawn").Process(target=stop_after_first_checkpoint, args=(run, output))
    process.start()
    process.join(timeout=15)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert process.exitcode == 23
    checkpoint = output / "results/00000000.json"
    before = checkpoint.read_bytes()
    results = run.run(output)
    assert checkpoint.read_bytes() == before
    assert [item.score for item in results] == [1, 1, 0, 0, 1, 1]
