from __future__ import annotations

from contextlib import closing

import pytest

from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    PolicyEvaluation,
    PolicyHarness,
    PolicyModel,
    RolloutScore,
    RolloutScores,
)
from reef.storage.evaluations import (
    EvaluationCall,
    EvaluationCallSummary,
    EvaluationCallTotals,
    EvaluationStateConflict,
    StoredEvaluation,
)
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


def call(task_index: int, repeat: int, attempt: int, index: int, **fields: object) -> EvaluationCall:
    """A call of ``(task_index, repeat)`` that answered with 10 prompt and 2 completion tokens, unless ``fields``
    say otherwise."""
    values: dict[str, object] = {
        "path": "/v1/chat/completions",
        "status": 200,
        "started_at": 100.0 + index,
        "duration_s": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "finish_reason": "stop",
        "tool_calls": 0,
    }
    values.update(fields)
    summary = EvaluationCallSummary(task_index, repeat, attempt, index, **values)  # type: ignore[arg-type]
    return EvaluationCall(summary, {"messages": [{"role": "user", "content": "ls"}]}, {"id": f"c{index}"})


def test_an_evaluation_keeps_every_call_of_each_rollout_and_totals_them(storage):
    with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
        evaluations = session.evaluations
        # A worker can keep a call before control stores the evaluation.
        evaluations.add_call("e1", call(0, 0, 2, 0))
        evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        failed = call(0, 0, 1, 1, status=502, prompt_tokens=None, completion_tokens=None, finish_reason=None)
        failed = EvaluationCall(failed.summary, failed.request, {"error": {"message": "the engine stopped"}})
        evaluations.add_call("e1", failed)
        evaluations.add_call("e1", call(0, 0, 1, 0, tool_calls=2, finish_reason="tool_calls"))
        evaluations.add_call("e1", call(1, 0, 1, 0, completion_tokens=None))
        evaluations.add_call("e2", call(0, 0, 1, 0))

        assert [(item.attempt, item.index) for item in evaluations.call_summaries("e1", 0, 0)] == [
            (1, 0),
            (1, 1),
            (2, 0),
        ]
        assert (
            evaluations.call_summaries("e1", 0, 0)[0]
            == call(0, 0, 1, 0, tool_calls=2, finish_reason="tool_calls").summary
        )
        assert evaluations.call("e1", 0, 0, 1, 1) == failed
        assert evaluations.call("e1", 0, 0, 1, 1).to_json()["response"] == {"error": {"message": "the engine stopped"}}
        assert evaluations.call("e1", 0, 0, 3, 0) is None and evaluations.call_summaries("e1", 5, 0) == ()
        assert evaluations.call_totals("e1") == {
            (0, 0): EvaluationCallTotals(count=3, failed=1, prompt_tokens=20, completion_tokens=4),
            (1, 0): EvaluationCallTotals(count=1, failed=0, prompt_tokens=10, completion_tokens=0),
        }
        assert evaluations.call_totals("missing") == {} and other.evaluations.call_totals("e1") == {}
        with pytest.raises(ValueError, match="stored already"):
            evaluations.add_call("e1", call(0, 0, 1, 0))
        assert evaluations.call_totals("e1")[(0, 0)].count == 3, "a refused call writes nothing"


def test_archiving_a_scenario_retires_the_calls_of_its_evaluations(storage):
    with closing(storage.open("math")) as session:
        session.evaluations.add_call("e1", call(0, 0, 1, 0))
    storage.archive("math")
    with closing(storage.open("math")) as session:
        assert session.evaluations.call_totals("e1") == {} and session.evaluations.call("e1", 0, 0, 1, 0) is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"attempt": 0}, "attempt"),
        ({"status": 502, "finish_reason": "stop"}, "failed call"),
        ({"prompt_tokens": -1}, "prompt_tokens"),
        ({"duration_s": -1.0}, "duration_s"),
        ({"path": ""}, "path"),
    ],
)
def test_a_call_summary_refuses_values_outside_its_definition(fields, message):
    values: dict[str, object] = {
        "task_index": 0,
        "repeat": 0,
        "attempt": 1,
        "index": 0,
        "path": "/v1/chat/completions",
        "status": 200,
        "started_at": 1.0,
        "duration_s": 0.1,
        "prompt_tokens": None,
        "completion_tokens": None,
        "finish_reason": None,
        "tool_calls": 0,
    }
    with pytest.raises(ValueError, match=message):
        EvaluationCallSummary(**{**values, **fields})  # type: ignore[arg-type]


def test_evaluations_survive_a_restart(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        session.evaluations.finish("e1", result=RESULT, finished_at=12.0)
        session.evaluations.add_call("e1", call(0, 0, 1, 0))
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        assert session.evaluations.get("e1").result == RESULT
        assert session.evaluations.call("e1", 0, 0, 1, 0) == call(0, 0, 1, 0)


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
    ):
        with pytest.raises(ValueError, match=message):
            ArtifactEvaluationRequest("r1", "weights", **fields)
    with pytest.raises(ValueError, match="request is an object"):
        ArtifactEvaluationRequest.from_json({"artifact_id": "r1"})
