from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest

from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    EvaluationCollection,
    PolicyEvaluation,
    PolicyHarness,
    PolicyModel,
    RolloutScore,
    RolloutScores,
)
from reef.storage.evaluations import EvaluationStateConflict, StoredEvaluation
from reef.storage.postgres import PostgresScenarioStorage
from reef.storage.sqlite import SQLiteScenarioStorage

REQUEST = ArtifactEvaluationRequest("r1", "weights", harness={"name": "terminus-2"}, rollouts_per_task=2)
RESULT = PolicyEvaluation(
    PolicyHarness("terminus-2"),
    PolicyModel("qwen", "tinker://run/sampler"),
    RolloutScores((("add", (RolloutScore.scored(1.0), RolloutScore.failed())),)),
)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        opened = PostgresScenarioStorage(url, tmp_path, schema=schema)
    else:
        opened = SQLiteScenarioStorage(None if request.param == "memory" else tmp_path)
    with closing(opened):
        yield opened


@pytest.fixture(params=["sqlite", "postgres"])
def open_durable_storage(request, tmp_path):
    if request.param == "postgres":
        url, schema = request.getfixturevalue("postgres_config")
        return lambda: PostgresScenarioStorage(url, tmp_path / "files", schema=schema)
    return lambda: SQLiteScenarioStorage(tmp_path / "files")


@pytest.mark.parametrize(("average", "mean"), [("task", 0.5), ("benchmark", 0.75)])
def test_an_evaluation_on_a_collection_scores_each_benchmark_and_averages_as_the_collection_says(average, mean):
    collection = EvaluationCollection(
        average=average, name="coding", benchmarks=(("lite", ("add",)), ("full", ("add", "sort", "sum")))
    )
    request = replace(REQUEST, task_names=("add", "sort", "sum"), collection=collection)
    assert ArtifactEvaluationRequest.from_json(request.to_json()) == request
    scores = RolloutScores(
        (
            ("add", (RolloutScore.scored(1.0), RolloutScore.scored(1.0))),
            ("sort", (RolloutScore.scored(0.0), RolloutScore.scored(1.0))),
            ("sum", (RolloutScore.scored(0.0), RolloutScore.scored(0.0))),
        )
    )
    running = StoredEvaluation("e1", "service-a", request, "running", 10.0)
    assert running.to_json()["benchmark_scores"] == []
    answer = replace(running, state="finished", finished_at=15.0, result=replace(RESULT, scores=scores)).to_json()
    assert answer["benchmark_scores"] == [
        {"benchmark": "lite", "task_count": 1, "mean_score": 1.0},
        {"benchmark": "full", "task_count": 3, "mean_score": 0.5},
    ]
    assert answer["mean_score"] == pytest.approx(mean)
    assert replace(running, request=REQUEST).to_json()["benchmark_scores"] == []
    # The collection's tasks are the request's tasks.
    with pytest.raises(ValueError, match="collection"):
        replace(request, task_names=("add",))


def test_an_evaluation_runs_then_ends_once_and_the_store_lists_the_newest_first(storage):
    with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
        evaluations = session.evaluations
        created = evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        assert created == StoredEvaluation("e1", "service-a", REQUEST, "running", 10.0)
        evaluations.create(evaluation_id="e2", owner="service-a", request=REQUEST, created_at=20.0)
        finished = evaluations.finish("e1", result=RESULT, finished_at=15.0)
        assert finished.state == "finished" and finished.result == RESULT and evaluations.get("e1") == finished
        failed = evaluations.fail("e2", error="the sandbox did not start", finished_at=25.0)
        assert evaluations.get("e2") == failed and failed.error == "the sandbox did not start"
        assert [item.evaluation_id for item in evaluations.newest()] == ["e2", "e1"]
        assert [item.evaluation_id for item in evaluations.newest(limit=1)] == ["e2"]
        assert evaluations.get("missing") is None and other.evaluations.newest() == ()
        with pytest.raises(EvaluationStateConflict):
            evaluations.fail("e1", error="again", finished_at=30.0)
        with pytest.raises(EvaluationStateConflict):
            evaluations.finish("missing", result=RESULT, finished_at=30.0)
        with pytest.raises(ValueError, match="stored already"):
            evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=40.0)
        queued = evaluations.create(
            evaluation_id="q1", owner="service-a", request=REQUEST, created_at=30.0, state="queued"
        )
        assert queued.state == "queued" and queued.finished_at is None
        with pytest.raises(EvaluationStateConflict):
            evaluations.finish("q1", result=RESULT, finished_at=31.0)
        assert evaluations.begin("q1").state == "running"
        with pytest.raises(EvaluationStateConflict):
            evaluations.begin("q1")
        evaluations.create(evaluation_id="q2", owner="service-a", request=REQUEST, created_at=32.0, state="queued")
        assert evaluations.fail("q2", error="the service stopped", finished_at=33.0).state == "failed"
        with pytest.raises(ValueError, match="finished_at"):
            evaluations.create(evaluation_id="e3", owner="service-a", request=REQUEST, created_at=50.0)
            evaluations.fail("e3", error="too early", finished_at=1.0)
        assert evaluations.get("e3").state == "running", "an invalid end writes nothing"
        assert finished.to_json()["mean_score"] == 1.0 and failed.to_json()["policy_evaluation"] is None


def test_an_ended_evaluation_reopens_and_ends_again(storage):
    with closing(storage.open("math")) as session:
        evaluations = session.evaluations
        evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        with pytest.raises(EvaluationStateConflict):
            evaluations.reopen("e1")
        evaluations.finish("e1", result=RESULT, finished_at=15.0)
        reopened = evaluations.reopen("e1")
        assert reopened == StoredEvaluation("e1", "service-a", REQUEST, "running", 10.0)
        assert evaluations.get("e1") == reopened
        assert evaluations.fail("e1", error="too many rollouts failed", finished_at=20.0).state == "failed"
        assert evaluations.reopen("e1").error is None
        assert evaluations.finish("e1", result=RESULT, finished_at=30.0).finished_at == 30.0
        with pytest.raises(EvaluationStateConflict):
            evaluations.reopen("missing")


def test_evaluations_survive_a_restart(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        session.evaluations.finish("e1", result=RESULT, finished_at=12.0)
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        assert session.evaluations.get("e1").result == RESULT


def test_a_request_configures_exactly_one_side_of_the_policy():
    assert ArtifactEvaluationRequest.from_json(REQUEST.to_json()) == REQUEST
    model = ArtifactEvaluationRequest("r1", "harness", model="served", task_names=("add",))
    assert ArtifactEvaluationRequest.from_json(model.to_json()) == model
    for fields, message in (
        ({}, "exactly one of harness and model"),
        ({"harness": {"name": "terminus-2"}, "model": "served"}, "exactly one of harness and model"),
        ({"harness": {"kwargs": {}}}, "name or an import_path"),
        ({"model": "served", "task_names": ("add", "add")}, "distinct"),
        ({"model": "served", "rollouts_per_task": 0}, "positive integer"),
        ({"model": "served", "environment": ""}, "environment"),
        ({"model": "served", "max_concurrent_rollouts": 0}, "max_concurrent_rollouts"),
        ({"model": "served", "max_failed_fraction": 1}, "max_failed_fraction"),
        ({"model": "served", "max_failed_fraction": True}, "max_failed_fraction"),
    ):
        with pytest.raises(ValueError, match=message):
            ArtifactEvaluationRequest("r1", "weights", **fields)
    with pytest.raises(ValueError, match="request is an object"):
        ArtifactEvaluationRequest.from_json({"artifact_id": "r1"})
    limited = replace(REQUEST, max_concurrent_rollouts=4, max_failed_fraction=0.5)
    assert ArtifactEvaluationRequest.from_json(limited.to_json()) == limited
    stored_earlier = {
        key: value
        for key, value in REQUEST.to_json().items()
        if key not in ("max_concurrent_rollouts", "max_failed_fraction")
    }
    assert ArtifactEvaluationRequest.from_json(stored_earlier) == REQUEST
