"""Record sets and record set units: the type definitions, the one-set-per-record rule, ready units, and staleness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from tests.support._trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.job_data import TaskSample
from reef.core.plugins import PluginConfigError
from reef.record2sample import (
    InferenceRecordSamples,
    Record2SampleSettings,
    RecordSet,
    RecordSetUnit,
    ReportSamples,
    SampleResult,
    SampleStatus,
    SingleRecordSet,
    SingleSetUnit,
)
from reef.record2sample.staleness import is_stale, producing_artifact_ids
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


class EachInference(InferenceRecordSamples):
    """Each inference record as one set with one task sample."""

    @classmethod
    def from_config(cls, config, *, context, environ):
        return cls(max_staleness=Record2SampleSettings.parse(config, plugin_class=cls).max_staleness)

    def make_samples(self, record_set: RecordSet) -> SampleResult:
        return SampleResult.ready((TaskSample(Path(str(record_set.set_id))),))


def ready_set_ids(stage) -> list[object]:
    return [set_id for unit in stage.take_ready_units() for set_id in unit.set_ids()]


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
    stage = EachInference()
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
    stage = EachInference()
    for name in ("a", "b", "a"):
        stage.settle(single(name), SampleResult.ready((TaskSample(Path(name)),)))
    units = stage.take_ready_units()
    assert [unit.unit_id for unit in units] == ["a", "b"]
    assert stage.unit_samples(units[0]) == (("a", (TaskSample(Path("a")),)),)
    assert stage.take_ready_units() == ()


def test_staleness_check_measures_lag_in_artifacts_the_scenario_served() -> None:
    artifacts = {
        "current": frozenset({"a5"}),
        "one-behind": frozenset({"a4"}),
        "two-behind": frozenset({"a3"}),
        "never-served": frozenset({"a6"}),
        "unknown": frozenset({None}),
        "mixed": frozenset({"a5", "a3"}),
        "rolled-out": frozenset(),
    }
    # The scenario never served a2: lag counts the artifacts that it served, not the artifacts that it made.
    served = ("a1", "a3", "a4", "a5")
    stale = {"two-behind", "never-served", "unknown", "mixed"}
    assert {name for name, produced in artifacts.items() if is_stale(produced, served, 1)} == stale
    # Window 0 keeps only data of the served artifact, and data that the algorithm rolls out.
    assert {name for name, produced in artifacts.items() if is_stale(produced, served, 0)} == stale | {"one-behind"}
    # An unknown served artifact checks nothing.
    assert not any(is_stale(produced, (), 0) for produced in artifacts.values())


def test_producing_artifact_ids_come_from_every_sample() -> None:
    samples = (
        policy_trajectory("a", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a3"),
        policy_trajectory("b", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a4"),
        policy_trajectory("c", (1, 2), (1,), (-0.1,), 1.0),
        TaskSample(Path("task")),
    )
    assert producing_artifact_ids(samples) == frozenset({"a3", "a4", None})


def test_the_stage_finds_samples_behind_the_served_artifact_stale() -> None:
    stage = EachInference()
    old = policy_trajectory("old", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a1")
    new = policy_trajectory("new", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a2")
    unnamed = policy_trajectory("unnamed", (1, 2), (1,), (-0.1,), 1.0)
    # Before the served artifact is known, no sample is stale.
    assert not stage.is_stale((old, unnamed))
    stage.observe_serving_artifact_id("a1")
    stage.observe_serving_artifact_id("a2")
    assert stage.is_stale((old,)) and not stage.is_stale((new,))
    assert stage.is_stale((new, old)), "one stale sample makes the samples stale"
    assert stage.is_stale((unnamed,)), "a sample without a producing artifact is stale"
    assert not stage.is_stale((TaskSample(Path("task")),)), "a task rolls out with the weights it trains"


def test_stage_settings_default_to_zero_and_refuse_a_negative_limit() -> None:
    assert Record2SampleSettings().max_staleness == 0
    assert EachInference.from_config({}, context=PipelineContext("s"), environ={}).max_staleness == 0
    with pytest.raises(ValueError, match="max_staleness must be a non-negative int"):
        Record2SampleSettings(max_staleness=-1)
    with pytest.raises(PluginConfigError, match="max_staleness must be a non-negative int"):
        EachInference.from_config({"max_staleness": -1}, context=PipelineContext("s"), environ={})


def test_a_stage_finds_only_the_samples_more_than_its_configured_limit_behind_stale() -> None:
    stage = EachInference.from_config({"max-staleness": 1}, context=PipelineContext("s"), environ={})
    for artifact_id in ("a1", "a2", "a3"):
        stage.observe_serving_artifact_id(artifact_id)
    stale = {
        name: stage.is_stale((policy_trajectory(name, (1, 2), (1,), (-0.1,), 1.0, artifact_id=artifact_id),))
        for name, artifact_id in (("two-behind", "a1"), ("one-behind", "a2"), ("current", "a3"))
    }
    assert stale == {"two-behind": True, "one-behind": False, "current": False}


def test_an_artifact_that_serves_again_is_the_newest_and_its_data_is_fresh() -> None:
    stage = EachInference()
    for artifact_id in ("a1", "a2", "a2", "a1"):
        stage.observe_serving_artifact_id(artifact_id)
    assert stage.served_artifacts.artifact_ids == ["a1", "a2", "a1"]
    assert not stage.is_stale((policy_trajectory("again", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a1"),))
    assert stage.is_stale((policy_trajectory("before", (1, 2), (1,), (-0.1,), 1.0, artifact_id="a2"),))


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
def test_aborted_inference_ends_before_sample_creation(response: dict[str, object]) -> None:
    stage = EachInference()
    aborted = AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload={"response": response}, agent_record_id="aborted"
    )
    stage.push(aborted)
    stage.push(inference("valid-without-tensors"))
    assert ready_set_ids(stage) == ["valid-without-tensors"]
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
    stage: EachInference | ReportSamples
    if report_stage:
        stage = ReportSamples(PipelineContext("s"))
        stage.push(aborted)
        rejected, retry = report("set", "aborted"), report("set", "clean")
    else:
        stage = EachInference()
        rejected, retry = aborted, inference("aborted")
    set_id = rejected.agent_record_id
    record_ids = {"aborted", set_id}
    stage.push(rejected)
    stage.push(rejected)
    assert stage.take_ready_units() == ()
    # The rejected set keeps its records until release, so another set cannot take them first.
    assert stage.set_of_record == dict.fromkeys(record_ids, set_id)
    assert stage.releasable_record_ids() == record_ids
    assert set_id not in stage.set_samples
    stage.release_records(stage.releasable_record_ids())
    assert stage.set_of_record == {}
    assert not stage.releasable_record_ids()
    if report_stage:
        stage.push(inference("clean"))
    stage.push(retry)
    assert ready_set_ids(stage) == [set_id]
