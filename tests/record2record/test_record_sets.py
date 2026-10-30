"""Record sets and record set units: the type definitions, the one-set-per-record rule, ready units, and staleness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.artifact_ref import RuntimeLoadSpan
from reef.core.job_data import TaskSample
from reef.core.plugins import PluginConfigError
from reef.record2record import (
    InferenceRecord2Record,
    Record2RecordSettings,
    RecordSet,
    RecordSetUnit,
    ReportGrouping,
    ReportRecord2Record,
    SampleStatus,
    SingleRecordSet,
    SingleSetUnit,
)
from reef.record2record.staleness import is_stale, producing_runtime_load_ids
from reef.record2sample import SampleResult
from reef.train import PipelineContext


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(scenario="s", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id)


def report(record_id: str, *references: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.REPORT, payload={}, agent_record_id=record_id, references=references
    )


@dataclass(frozen=True, kw_only=True)
class RecordsSet(RecordSet):
    held: tuple[AgentRecord, ...]

    def records(self) -> tuple[AgentRecord, ...]:
        return self.held


@dataclass(frozen=True, kw_only=True)
class ManySetUnit(RecordSetUnit):
    held: tuple[RecordSet, ...]

    def record_sets(self) -> tuple[RecordSet, ...]:
        return self.held


def single(record_id: str) -> SingleRecordSet:
    return SingleRecordSet(set_id=record_id, record=inference(record_id))


def test_a_record_set_holds_one_or_more_records_and_every_record_they_reference() -> None:
    assert RecordsSet(set_id="r", held=(inference("i"), report("r", "i"))).record_ids() == ("i", "r")
    with pytest.raises(ValueError, match="one or more records"):
        RecordsSet(set_id="empty", held=())
    with pytest.raises(ValueError, match=r"does not hold the referenced records \['i'\]"):
        RecordsSet(set_id="r", held=(report("r", "i"),))


def test_a_record_set_unit_holds_one_or_more_record_sets_and_consumes_all_their_records() -> None:
    unit = ManySetUnit(unit_id="u", held=(single("a"), single("b")))
    assert unit.set_ids() == ("a", "b")
    assert unit.consumed_ids() == frozenset({"a", "b"})
    record_set = single("c")
    assert SingleSetUnit.of(record_set) == SingleSetUnit(unit_id="c", record_set=record_set)
    with pytest.raises(ValueError, match="one or more record sets"):
        ManySetUnit(unit_id="empty", held=())


def test_a_stage_puts_a_record_in_one_record_set_only() -> None:
    stage = InferenceRecord2Record()
    stage.form(RecordsSet(set_id="first", held=(inference("i"), report("r1", "i"))))
    # A retry of a set takes its records again.
    stage.form(RecordsSet(set_id="first", held=(inference("i"), report("r1", "i"))))
    with pytest.raises(ValueError, match=r"records \['i'\] are in another record set"):
        stage.form(RecordsSet(set_id="second", held=(inference("i"), report("r2", "i"))))
    stage.release_records(frozenset({"i", "r1"}))
    stage.form(RecordsSet(set_id="second", held=(inference("i"), report("r2", "i"))))


def test_a_ready_result_holds_one_or_more_samples_and_a_skipped_result_none() -> None:
    sample = TaskSample(Path("task"))
    assert SampleResult.ready((sample,)).samples == (sample,)
    for status, samples in ((SampleStatus.READY, ()), (SampleStatus.SKIPPED, (sample,))):
        with pytest.raises(ValueError, match="READY result holds one or more samples"):
            SampleResult(status, samples)


def test_a_stage_hands_out_each_ready_unit_once_in_the_order_it_first_became_ready() -> None:
    stage = InferenceRecord2Record()
    for name in ("a", "b", "a"):
        stage.settle(single(name), SampleStatus.READY)
    assert [unit.unit_id for unit in stage.take_ready_units()] == ["a", "b"]
    assert stage.take_ready_units() == ()


def test_staleness_check_measures_lag_in_versions_the_scenario_served() -> None:
    versions = {
        "current": frozenset({"e:5"}),
        "one-behind": frozenset({"e:4"}),
        "two-behind": frozenset({"e:3"}),
        "never-served": frozenset({"e:6"}),
        "unknown": frozenset({None}),
        "mixed": frozenset({"e:5", "e:3"}),
        "rolled-out": frozenset(),
    }
    # The engine sequence skips e:2: lag counts the scenario's own versions, not sequence numbers.
    served = ("e:1", "e:3", "e:4", "e:5")
    stale = {"two-behind", "never-served", "unknown", "mixed"}
    assert {name for name, produced in versions.items() if is_stale(produced, served, 1)} == stale
    # Window 0 keeps only data of the current version, and data that the algorithm rolls out.
    assert {name for name, produced in versions.items() if is_stale(produced, served, 0)} == stale | {"one-behind"}
    # An unknown serving version checks nothing.
    assert not any(is_stale(produced, (), 0) for produced in versions.values())


def test_producing_runtime_load_ids_come_from_every_sample_and_token_span() -> None:
    spans = (
        RuntimeLoadSpan(start=0, end=1, runtime_load_id="e:1"),
        RuntimeLoadSpan(start=1, end=2, runtime_load_id="e:2"),
    )
    tasks = (
        policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0, runtime_load_id="e:3"),
        policy_trajectory("b", (1, 2, 3), (1, 1), (-0.1, -0.2), 1.0, runtime_load_spans=spans),
        policy_trajectory("c", (1, 2), (1,), (-0.1,), 1.0),
        TaskSample(Path("task")),
    )
    assert producing_runtime_load_ids(tasks) == frozenset({"e:1", "e:2", "e:3", None})


def test_the_stage_finds_samples_behind_the_serving_version_stale() -> None:
    stage = InferenceRecord2Record()
    old = policy_trajectory("old", (1, 2), (1,), (-0.1,), 1.0, runtime_load_id="e:1")
    new = policy_trajectory("new", (1, 2), (1,), (-0.1,), 1.0, runtime_load_id="e:2")
    unversioned = policy_trajectory("unversioned", (1, 2), (1,), (-0.1,), 1.0)
    # Before the serving version is known, no sample is stale.
    assert not stage.is_stale((old, unversioned))
    stage.observe_serving_runtime_load_id("e:1")
    stage.observe_serving_runtime_load_id("e:2")
    assert stage.is_stale((old,)) and not stage.is_stale((new,))
    assert stage.is_stale((new, old)), "one stale sample makes the samples stale"
    assert stage.is_stale((unversioned,)), "a sample without a producing version is stale"
    assert not stage.is_stale((TaskSample(Path("task")),)), "a task rolls out with the weights it trains"


def test_record_stage_settings_default_to_zero_and_refuse_a_negative_limit() -> None:
    assert Record2RecordSettings().max_staleness == 0
    assert InferenceRecord2Record.from_config({}, context=PipelineContext("s"), environ={}).max_staleness == 0
    with pytest.raises(ValueError, match="max_staleness must be a non-negative int"):
        Record2RecordSettings(max_staleness=-1)
    with pytest.raises(PluginConfigError, match="max_staleness must be a non-negative int"):
        InferenceRecord2Record.from_config({"max_staleness": -1}, context=PipelineContext("s"), environ={})


def test_a_stage_finds_only_the_samples_more_than_its_configured_limit_behind_stale() -> None:
    stage = InferenceRecord2Record.from_config({"max-staleness": 1}, context=PipelineContext("s"), environ={})
    for version in ("e:1", "e:2", "e:3"):
        stage.observe_serving_runtime_load_id(version)
    stale = {
        name: stage.is_stale((policy_trajectory(name, (1, 2), (1,), (-0.1,), 1.0, runtime_load_id=version),))
        for name, version in (("two-behind", "e:1"), ("one-behind", "e:2"), ("current", "e:3"))
    }
    assert stale == {"two-behind": True, "one-behind": False, "current": False}


def test_a_serving_version_of_another_incarnation_restarts_the_served_versions() -> None:
    stage = InferenceRecord2Record()
    for version in ("e:1", "e:2", "e:2", "f:0"):
        stage.observe_serving_runtime_load_id(version)
    assert stage.served_versions.versions == ["f:0"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "response",
    [
        {"finish_reason": "abort", "training": {"tokens": [1, 2]}},
        {"meta_info": {"finish_reason": {"type": "abort"}}},
        {"choices": [{"finish_reason": "abort"}]},
        {"choices": [{"finish_reason": "tool_calls", "meta_info": {"finish_reason": "abort"}}]},
        {"stop_reason": "end_turn", "training": {"finish_reason": "abort", "tokens": [1, 2]}},
        {
            "stop_reason": "end_turn",
            "stream_delivery": {"complete": False, "error": "upstream engine aborted the generation"},
        },
    ],
    ids=["native", "sglang", "openai", "tool-call", "anthropic", "legacy-stripped-anthropic"],
)
def test_aborted_inference_ends_before_task_creation(response: dict[str, object]) -> None:
    stage = InferenceRecord2Record()
    aborted = AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload={"response": response}, agent_record_id="aborted"
    )
    stage.push(aborted)
    stage.push(inference("valid-without-tensors"))
    assert [record_set.record_ids() for record_set in stage.pull()] == [("valid-without-tensors",)]
    assert stage.releasable_record_ids() == {"aborted"}
    assert aborted.payload["response"] == response
    stage.release_records(frozenset({"aborted"}))
    assert not stage.releasable_record_ids()


@pytest.mark.unit
@pytest.mark.parametrize("report_stage", [False, True], ids=["inference", "report"])
def test_a_rejected_set_formed_twice_frees_its_records_on_release_for_a_retry(report_stage: bool) -> None:
    aborted = AgentRecord.create(
        scenario="s",
        request_type=RequestType.INFERENCE,
        payload={"response": {"choices": [{"finish_reason": "abort"}]}},
        agent_record_id="aborted",
    )
    stage: InferenceRecord2Record | ReportRecord2Record
    if report_stage:
        stage = ReportRecord2Record(PipelineContext("s"), ReportGrouping())
        stage.push(aborted)
        rejected, retry = report("set", "aborted"), report("set", "clean")
    else:
        stage = InferenceRecord2Record()
        rejected, retry = aborted, inference("aborted")
    set_id = rejected.agent_record_id
    record_ids = {"aborted", set_id}
    stage.push(rejected)
    stage.push(rejected)
    assert stage.pull() == ()
    # The rejected set keeps its records until release, so another set cannot take them first.
    assert stage.set_of_record == dict.fromkeys(record_ids, set_id)
    assert stage.releasable_record_ids() == record_ids
    assert stage.take_ended() == {set_id}
    stage.release_records(stage.releasable_record_ids())
    assert stage.set_of_record == {}
    assert not stage.releasable_record_ids()
    if report_stage:
        stage.push(inference("clean"))
    stage.push(retry)
    assert [record_set.set_id for record_set in stage.pull()] == [set_id]
