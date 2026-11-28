from __future__ import annotations

from contextlib import closing
from dataclasses import replace

import pytest
from sqlalchemy.schema import DropTable

from reef.core.errors import ScenarioDeletedError
from reef.core.evaluation import (
    ArtifactEvaluationRequest,
    EvaluationCollection,
    PolicyEvaluation,
    PolicyHarness,
    PolicyModel,
    RolloutScore,
    RolloutScores,
)
from reef.storage.scenario.evaluations import (
    EvaluationStart,
    EvaluationStartConflictError,
    EvaluationStateConflictError,
    EvaluationStore,
    StoredEvaluation,
)
from reef.storage.sql.evaluations import SQLEvaluationStore
from reef.storage.sql.postgres import PostgresScenarioStorage
from reef.storage.sql.sqlite import SQLiteScenarioStorage

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
        assert finished.state == "finished" and finished.result == RESULT and evaluations.read("e1") == finished
        failed = evaluations.fail("e2", error="the sandbox did not start", finished_at=25.0)
        assert evaluations.read("e2") == failed and failed.error == "the sandbox did not start"
        assert [item.evaluation_id for item in evaluations.newest()] == ["e2", "e1"]
        assert [item.evaluation_id for item in evaluations.newest(limit=1)] == ["e2"]
        assert evaluations.read("missing") is None and other.evaluations.newest() == ()
        with pytest.raises(EvaluationStateConflictError):
            evaluations.fail("e1", error="again", finished_at=30.0)
        with pytest.raises(EvaluationStateConflictError):
            evaluations.finish("missing", result=RESULT, finished_at=30.0)
        with pytest.raises(ValueError, match="stored already"):
            evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=40.0)
        queued = evaluations.create(
            evaluation_id="q1", owner="service-a", request=REQUEST, created_at=30.0, state="queued"
        )
        assert queued.state == "queued" and queued.finished_at is None
        with pytest.raises(EvaluationStateConflictError):
            evaluations.finish("q1", result=RESULT, finished_at=31.0)
        assert evaluations.begin("q1").state == "running"
        with pytest.raises(EvaluationStateConflictError):
            evaluations.begin("q1")
        evaluations.create(evaluation_id="q2", owner="service-a", request=REQUEST, created_at=32.0, state="queued")
        assert evaluations.fail("q2", error="the service stopped", finished_at=33.0).state == "failed"
        with pytest.raises(ValueError, match="finished_at"):
            evaluations.create(evaluation_id="e3", owner="service-a", request=REQUEST, created_at=50.0)
            evaluations.fail("e3", error="too early", finished_at=1.0)
        assert evaluations.read("e3").state == "running", "an invalid end writes nothing"
        assert finished.to_json()["mean_score"] == 1.0 and failed.to_json()["policy_evaluation"] is None


def test_the_evaluation_list_can_omit_digests_without_changing_the_stored_request(storage):
    names = tuple(f"task-{index}" for index in range(10_000))
    request = replace(
        REQUEST,
        task_names=names,
        task_digests=tuple(f"{index:064x}" for index in range(len(names))),
        harness={"name": "terminus-2", "kwargs": {"stop": "\0"}},
    )
    with closing(storage.open("math")) as session:
        evaluations = session.evaluations
        created = evaluations.create(evaluation_id="e1", owner="service-a", request=request, created_at=10.0)
        assert evaluations.newest_for_list(limit=1) == (replace(created, request=replace(request, task_digests=None)),)
        assert evaluations.newest(limit=1) == (created,)
        assert evaluations.read("e1") == created

        class LegacyStore(SQLEvaluationStore):
            newest_for_list = EvaluationStore.newest_for_list

            def newest(self, *, limit: int = 50) -> tuple[StoredEvaluation, ...]:
                return super().newest(limit=limit)

        legacy = LegacyStore(
            evaluations.database, evaluations.table, evaluations.starts, evaluations.digests, evaluations.progress
        )
        assert legacy.newest_for_list(limit=1) == (created,)


def test_an_evaluation_start_fingerprint_must_be_a_string():
    with pytest.raises(ValueError, match="fingerprint"):
        EvaluationStart("start-1", ("a",) * 64, "{}")  # type: ignore[arg-type]


def test_the_start_reservation_and_work_share_the_evaluation_transaction(storage):
    with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
        evaluations = session.evaluations
        start = EvaluationStart("start-1", "0" * 64, '{"scenario": "math"}')
        pinned = replace(REQUEST, task_names=("add",), task_digests=("1" * 64,))
        parameters = {"owner": "service-a", "request": pinned, "created_at": 10.0, "start": start}
        created = evaluations.create(evaluation_id="e1", **parameters)
        assert evaluations.read_started(start.start_id, start.fingerprint) == (created, start.first_work_json)
        with pytest.raises(EvaluationStartConflictError):
            evaluations.create(evaluation_id="e2", **parameters)
        assert evaluations.read("e2") is None
        assert other.evaluations.read_started(start.start_id, start.fingerprint) is None
        separate_start = replace(start, first_work_json='{"scenario": "other"}')
        separate = other.evaluations.create(
            evaluation_id="e1",
            **{**parameters, "start": separate_start, "request": replace(pinned, task_digests=None)},
        )
        assert other.evaluations.read("e1").request.task_digests is None
        evaluations.complete_start("e1")
        assert evaluations.pending_starts() == ()
        assert evaluations.read_started(start.start_id, start.fingerprint) == (created, None)
        assert other.evaluations.read_started(start.start_id, start.fingerprint) == (
            separate,
            separate_start.first_work_json,
        )


def test_an_ended_evaluation_reopens_and_ends_again(storage):
    with closing(storage.open("math")) as session:
        evaluations = session.evaluations
        evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        with pytest.raises(EvaluationStateConflictError):
            evaluations.reopen("e1")
        evaluations.finish("e1", result=RESULT, finished_at=15.0)
        reopened = evaluations.reopen("e1")
        assert reopened == StoredEvaluation("e1", "service-a", REQUEST, "running", 10.0)
        assert evaluations.read("e1") == reopened
        assert evaluations.fail("e1", error="too many rollouts failed", finished_at=20.0).state == "failed"
        assert evaluations.reopen("e1").error is None
        assert evaluations.finish("e1", result=RESULT, finished_at=30.0).finished_at == 30.0
        with pytest.raises(EvaluationStateConflictError):
            evaluations.reopen("missing")


def test_the_store_gives_the_kept_progress_of_the_end_that_the_evaluation_has(storage):
    with closing(storage.open("math")) as session:
        evaluations = session.evaluations
        evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        evaluations.create(evaluation_id="e2", owner="service-a", request=REQUEST, created_at=11.0)
        assert evaluations.ended_progress(["e1", "e2", "missing"]) == {}
        # An evaluation that runs, and one that the store does not hold, keep nothing.
        assert not evaluations.keep_ended_progress("e1", 15.0, {"total": 2})
        assert not evaluations.keep_ended_progress("missing", 15.0, {"total": 2})
        evaluations.finish("e1", result=RESULT, finished_at=15.0)
        evaluations.fail("e2", error="cancelled", finished_at=16.0)
        assert not evaluations.keep_ended_progress("e1", 14.0, {"total": 2})
        assert evaluations.keep_ended_progress("e1", 15.0, {"total": 2, "benchmarks": []})
        # An evaluation without rollouts has no progress, and the store keeps that too.
        assert evaluations.keep_ended_progress("e2", 16.0, None)
        assert evaluations.ended_progress(["e1", "e2"]) == {"e1": {"total": 2, "benchmarks": []}, "e2": None}
        assert evaluations.keep_ended_progress("e1", 15.0, {"total": 3})
        assert evaluations.ended_progress(["e1"]) == {"e1": {"total": 3}}
        # A new end has a new progress: the store does not give the progress of the earlier end.
        evaluations.reopen("e1")
        assert evaluations.ended_progress(["e1"]) == {}
        evaluations.finish("e1", result=RESULT, finished_at=30.0)
        assert evaluations.ended_progress(["e1"]) == {}


def test_evaluations_survive_a_restart(open_durable_storage):
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        session.evaluations.create(evaluation_id="e1", owner="service-a", request=REQUEST, created_at=10.0)
        session.evaluations.finish("e1", result=RESULT, finished_at=12.0)
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        assert session.evaluations.read("e1").result == RESULT


def test_upgrade_adds_digest_storage_and_archive_keeps_it_scoped(open_durable_storage):
    pinned = replace(REQUEST, task_names=("add",), task_digests=("1" * 64,))
    with closing(open_durable_storage()) as storage, closing(storage.open("math")) as session:
        evaluations = session.evaluations
        old = evaluations.create(evaluation_id="old", owner="service-a", request=REQUEST, created_at=10.0)
        with evaluations.database._transaction(write=True) as connection:
            connection.execute(DropTable(evaluations.digests))
    with closing(open_durable_storage()) as storage:
        with closing(storage.open("math")) as session, closing(storage.open("other")) as other:
            evaluations = session.evaluations
            assert evaluations.read("old") == old
            evaluations.create(evaluation_id="e1", owner="service-a", request=pinned, created_at=20.0)
            limited = evaluations.limit_rollouts("e1", 10)
            assert limited.request.task_digests == pinned.task_digests
            assert evaluations.finish("e1", result=RESULT, finished_at=25.0).request == limited.request
            separate = other.evaluations.create(
                evaluation_id="e1", owner="service-a", request=REQUEST, created_at=20.0
            )
        storage.archive("math")
        with pytest.raises(ScenarioDeletedError):
            storage.open("math")
        with closing(storage.open("other")) as other:
            assert other.evaluations.read("e1") == separate


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
        ({"model": "served", "log_probs": 1}, "log_probs"),
    ):
        with pytest.raises(ValueError, match=message):
            ArtifactEvaluationRequest("r1", "weights", **fields)
    with pytest.raises(ValueError, match="request is an object"):
        ArtifactEvaluationRequest.from_json({"artifact_id": "r1"})
    limited = replace(REQUEST, max_concurrent_rollouts=4, max_failed_fraction=0.5, log_probs=True)
    assert ArtifactEvaluationRequest.from_json(limited.to_json()) == limited
    stored_earlier = {
        key: value
        for key, value in REQUEST.to_json().items()
        if key not in ("max_concurrent_rollouts", "max_failed_fraction", "log_probs")
    }
    assert ArtifactEvaluationRequest.from_json(stored_earlier) == REQUEST
