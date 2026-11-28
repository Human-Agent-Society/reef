"""TrafficTaskGenerator: the failed sessions of a round go to one generation run, and its tasks come back."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from tests.recipes.continual_envs.support import MemoryGenerationService, session
from tests.support.task_service_stubs import generator_state, task_for

from recipes.beta.continual_envs import TrafficGeneratorSettings, TrafficTaskGenerator
from recipes.beta.continual_envs import generator as generator_module
from recipes.beta.continual_envs.generation_service import GenerationRun
from reef.core.plugins import PluginConfigError
from reef.recipe.generated_tasks.generator import FailedTask, HeldOutTask, RejectedTask, RolledOutTask
from reef.samples.generator.harbor import OracleResult
from reef.samples.generator.inputs import TaskGenerationRequest

SETTINGS = {"service_url": "https://console.example", "spec": "terminal-failures", "generator": "tw-ours"}


def traffic_generator(tmp_path: Path, service: MemoryGenerationService, **changes: object) -> TrafficTaskGenerator:
    settings = TrafficGeneratorSettings(**{**SETTINGS, "poll_seconds": 0.001, "oracle_check": False, **changes})
    return TrafficTaskGenerator(service, settings, generator_state(tmp_path / "state"))


def request(*records, generation: int = 0) -> TaskGenerationRequest:
    return TaskGenerationRequest(tuple(records), "terminal work", generation=generation)


def test_the_failed_sessions_go_to_one_run_and_its_tasks_carry_their_sessions(tmp_path: Path) -> None:
    service = MemoryGenerationService(polls_to_end=3)
    generator = traffic_generator(tmp_path, service, params={"per_failure": 2})
    records = (*session("a", 0.0), *session("b", 1.0), *session("c", 0.5), *session("d", None))
    tasks = asyncio.run(generator.generate(request(*records, generation=4)))
    (started,) = service.started
    assert (started["spec"], started["generator"], started["params"]) == (
        "terminal-failures",
        "tw-ours",
        {"per_failure": 2},
    )
    (source,) = started["sources"]
    # A session is a failure at a score of at most 0.5; one with no score is not a failure.
    assert (source["name"], source["kind"], source["id"]) == ("traffic", "records", "generation-00004")
    assert sorted(source["files"]) == ["a-report/session.json", "c-report/session.json"]
    assert [task.source_agent_record_ids for task in tasks] == [
        ("a-call-0", "a-call-1", "a-report"),
        ("c-call-0", "c-call-1", "c-report"),
    ]
    assert service.reads["run-1"] == 3 and generator.state.read("run/00004") == json.dumps({"run_id": "run-1"})


def test_a_round_with_no_failure_starts_no_run(tmp_path: Path) -> None:
    service = MemoryGenerationService()
    assert asyncio.run(traffic_generator(tmp_path, service).generate(request(*session("a", 1.0)))) == ()
    assert service.started == []


def test_a_restart_waits_for_the_run_of_its_generation(tmp_path: Path) -> None:
    service = MemoryGenerationService(polls_to_end=2)
    first = traffic_generator(tmp_path, service)
    asyncio.run(first.generate(request(*session("a", 0.0), generation=1)))
    # The next process reads the same state: the round of generation 1 again starts no second run.
    again = traffic_generator(tmp_path, service)
    tasks = asyncio.run(again.generate(request(*session("a", 0.0), generation=1)))
    assert len(service.started) == 1 and len(tasks) == 1


def test_a_start_whose_answer_was_lost_takes_the_run_that_it_queued(tmp_path: Path, monkeypatch) -> None:
    service = MemoryGenerationService()
    generator = traffic_generator(tmp_path, service)
    generator.state.write("starting/00002", json.dumps({"requested_at": 1000.0}))
    generator.state.write("run/00001", json.dumps({"run_id": "run-old"}))
    asyncio.run(
        service.start(
            spec="s", generator="g", sources=[{"files": {"a-report/session.json": "{}"}}], params={}, targets={}
        )
    )
    service.listed = [GenerationRun("run-old", "finished", "", 1100.0), GenerationRun("run-1", "running", "", 990.0)]
    tasks = asyncio.run(generator.generate(request(*session("a", 0.0), generation=2)))
    # The run of another generation is not taken; one queued a little before the start is, for the clocks differ.
    assert len(service.started) == 1 and generator.state.read("run/00002") == json.dumps({"run_id": "run-1"})
    assert len(tasks) == 1


@pytest.mark.parametrize(
    ("end_state", "message"), [("failed", "ended failed: a stage failed"), ("cancelled", "ended cancelled")]
)
def test_a_run_that_did_not_finish_stops_the_round(tmp_path: Path, end_state: str, message: str) -> None:
    generator = traffic_generator(tmp_path, MemoryGenerationService(end_state=end_state))
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(generator.generate(request(*session("a", 0.0))))


def test_the_next_attempt_of_a_generation_starts_a_new_run_after_one_that_failed(tmp_path: Path) -> None:
    service = MemoryGenerationService(end_state="failed")
    generator = traffic_generator(tmp_path, service)
    with pytest.raises(RuntimeError, match="ended failed"):
        asyncio.run(generator.generate(request(*session("a", 0.0), generation=5)))
    service.end_state = "finished"
    tasks = asyncio.run(generator.generate(request(*session("a", 0.0), generation=5)))
    assert len(service.started) == 2 and generator.state.read("run/00005") == json.dumps({"run_id": "run-2"})
    assert len(tasks) == 1


def test_a_run_past_its_timeout_is_cancelled(tmp_path: Path) -> None:
    service = MemoryGenerationService(polls_to_end=10**9)
    generator = traffic_generator(tmp_path, service, run_timeout_seconds=0.01)
    with pytest.raises(TimeoutError, match="ran past"):
        asyncio.run(generator.generate(request(*session("a", 0.0))))
    assert service.cancelled == ["run-1"]


def test_validation_runs_the_oracle_check_on_this_host(tmp_path: Path, monkeypatch) -> None:
    checked = []

    def oracle(path: Path, *, environment: str) -> OracleResult:
        checked.append((path.name, environment))
        return OracleResult(False, "the image did not build")

    monkeypatch.setattr(generator_module, "oracle_check", oracle)
    generator = traffic_generator(tmp_path, MemoryGenerationService(), oracle_check=True)
    result = asyncio.run(generator.validate(task_for("fail-000000000000__task", "a-report", 9001)))
    assert result.errors == ("the image did not build",) and checked == [("fail-000000000000__task", "docker")]
    unchecked = traffic_generator(tmp_path, MemoryGenerationService())
    assert asyncio.run(unchecked.validate(task_for("fail-000000000000__task", "a-report", 9001))).is_valid


def test_the_results_of_a_generation_are_kept_for_analysis(tmp_path: Path) -> None:
    generator = traffic_generator(tmp_path, MemoryGenerationService())
    task = task_for("fail-000000000000__task", "a-report", 9001)
    results = (
        RolledOutTask(task, (1.0, 0.0)),
        RejectedTask(task, "a duplicate"),
        HeldOutTask(task),
        FailedTask(task, "lost", True),
    )
    asyncio.run(generator.record_results(results, generation=3))
    assert [row["result"] for row in json.loads(generator.state.read("results/00003"))] == [
        "rolled_out",
        "rejected",
        "held_out",
        "failed",
    ]


def test_the_config_is_checked_and_the_token_comes_from_the_environment(tmp_path: Path) -> None:
    state = generator_state(tmp_path / "state")
    built = TrafficTaskGenerator.from_config(
        SETTINGS, scenario="s", environ={"REEF_ACCESS_TOKEN": "rft_1"}, state=state
    )
    assert built.service.headers == {"Authorization": "Bearer rft_1"}
    for config, message in (
        ({**SETTINGS, "unknown": 1}, "does not take unknown"),
        ({"spec": "s"}, "needs service_url, spec and generator"),
        ({**SETTINGS, "max_score": 2}, "max_score must be from 0 to 1"),
        ({**SETTINGS, "poll_seconds": 0}, "must be above 0"),
        (SETTINGS, "needs a workspace access token in REEF_ACCESS_TOKEN"),
    ):
        with pytest.raises(PluginConfigError, match=message):
            TrafficTaskGenerator.from_config(config, scenario="s", environ={}, state=state)
