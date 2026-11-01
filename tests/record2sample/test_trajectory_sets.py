"""Trajectory record sets (RFC #1054): reports score trajectory ends, and trajectories end after an idle time."""

from __future__ import annotations

from dataclasses import replace

import pytest
from tests.support.conversations import conversation, turn

from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.record2sample import ReportSamples, ReportSet, TrajectorySamples
from reef.train import PipelineContext


def report(record_id: str, *references: str, score: float = 1.0) -> AgentRecord:
    return AgentRecord.create(
        agent_record_id=record_id,
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": list(references)},
    )


def report_stage() -> ReportSamples:
    return ReportSamples(PipelineContext(scenario="math"))


def test_a_report_on_the_last_call_trains_the_whole_trajectory() -> None:
    stage = report_stage()
    for turns in (1, 2, 3):
        stage.push(conversation(f"call-{turns}", turns))
    stage.push(conversation("other", 1, tools=("bash",)))
    stage.push(report("r1", "call-3", "other", score=0.5))
    (unit,) = stage.take_ready_units()
    assert isinstance(unit, ReportSet)
    (entry,) = unit.entries
    assert entry.context.references == ("call-1", "call-2", "call-3", "other")
    assert entry.context.score == 0.5
    ((_, (sample,)),) = stage.unit_samples(unit)
    assert sample.source_agent_record_ids == ("call-1", "call-2", "call-3", "other", "r1")
    stage.consume(unit)
    assert {"call-1", "call-2", "call-3", "other", "r1"} <= stage.releasable_record_ids()


def test_a_report_that_lists_calls_of_one_trajectory_ends_without_training(caplog) -> None:
    stage = report_stage()
    for turns in (1, 2):
        stage.push(conversation(f"call-{turns}", turns))
    stage.push(report("r1", "call-1", "call-2"))
    assert stage.take_ready_units() == ()
    assert "inside the trajectory" in caplog.text
    assert "r1" in stage.releasable_record_ids()


def test_a_report_whose_trajectory_holds_another_candidates_call_ends() -> None:
    stage = report_stage()
    for turns in (1, 2, 3):
        stage.push(conversation(f"call-{turns}", turns))
    stage.push(report("r1", "call-2"))
    stage.push(report("r2", "call-3"))
    (unit,) = stage.take_ready_units()
    assert unit.set_ids() == ("r1",)
    assert "r2" in stage.releasable_record_ids()


def trajectory_stage(idle: float = 10.0) -> TrajectorySamples:
    return TrajectorySamples(PipelineContext(scenario="math"), trajectory_idle_seconds=idle)


def test_a_trajectory_forms_its_set_after_its_last_call_was_idle(monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("reef.record2sample.trajectories.time.monotonic", lambda: clock[0])
    stage = trajectory_stage()
    stage.push(conversation("call-1", 1))
    stage.push(report("r1", "call-1"))
    clock[0] = 105.0
    stage.push(conversation("call-2", 2))
    stage.advance(112.0)
    # call-1 is idle but continued; call-2 is the end, and it is not idle yet.
    assert stage.take_ready_units() == ()
    stage.advance(115.0)
    (unit,) = stage.take_ready_units()
    assert unit.set_ids() == (("trajectory", "call-2"),)
    assert unit.consumed_ids() == {"call-1", "call-2"}
    assert "r1" in stage.releasable_record_ids()
    assert stage.operational_metrics() == {"waiting_calls": 0}


def test_a_retried_call_ends_with_the_trajectory_that_replaced_it(monkeypatch) -> None:
    monkeypatch.setattr("reef.record2sample.trajectories.time.monotonic", lambda: 0.0)
    stage = trajectory_stage()
    first = conversation("first", 1)
    stage.push(first)
    stage.push(replace(first, agent_record_id="retry", payload={**first.payload, "response": {"output": [turn(9)]}}))
    later = conversation("later", 1)
    later.payload["messages"] += [turn(9), turn(2)]
    stage.push(later)
    stage.advance(60.0)
    (unit,) = stage.take_ready_units()
    assert unit.consumed_ids() == {"retry", "later"}
    assert "first" in stage.releasable_record_ids()


def test_a_branch_that_shares_calls_with_an_earlier_set_ends_without_a_set(monkeypatch, caplog) -> None:
    clock = [0.0]
    monkeypatch.setattr("reef.record2sample.trajectories.time.monotonic", lambda: clock[0])
    stage = trajectory_stage()
    stage.push(conversation("call-1", 1))
    stage.push(conversation("call-2", 2))
    clock[0] = 5.0
    branch = conversation("branch", 2)
    branch.payload["messages"][-1] = {"role": "assistant", "content": "another answer"}
    stage.push(branch)
    stage.advance(12.0)
    (unit,) = stage.take_ready_units()
    assert unit.consumed_ids() == {"call-1", "call-2"}
    stage.advance(20.0)
    assert stage.take_ready_units() == ()
    assert "branch" in stage.releasable_record_ids()
    assert "does not train" in caplog.text


def test_the_idle_time_must_be_positive() -> None:
    with pytest.raises(PluginConfigError, match="trajectory_idle_seconds"):
        TrajectorySamples.from_config(
            {"trajectory_idle_seconds": 0}, context=PipelineContext(scenario="math"), environ={}
        )
    stage = TrajectorySamples.from_config(
        {"trajectory_idle_seconds": 30}, context=PipelineContext(scenario="math"), environ={}
    )
    assert stage.trajectory_idle_seconds == 30
