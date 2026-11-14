"""Record sets and record set units: the type definitions, the one-set-per-record rule, ready units, and staleness."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pytest
from tests.support import records
from tests.support.trajectories import policy_trajectory

from reef.core import AgentRecord, RequestType
from reef.core.job_data import TaskSample
from reef.core.plugins import NoSettings, PluginConfigError
from reef.record2sample import (
    InferenceRecordSamples,
    RecordSet,
    RecordSetUnit,
    ReportSamples,
    SampleResult,
    SampleStatus,
    SingleRecordSet,
    SingleSetUnit,
)
from reef.record2sample.staleness import ServedArtifacts, lags_behind
from reef.train import PipelineContext

inference = partial(records.inference, scenario="s")
report = partial(records.report, scenario="s")


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
        NoSettings.parse(config, plugin_class=cls)
        return cls()

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


def trajectory(name: str, artifact_id: str | None = None):
    return policy_trajectory(name, (1, 2), (1,), (-0.1,), 1.0, artifact_id=artifact_id)


def test_lag_counts_the_artifacts_the_scenario_served() -> None:
    samples = {
        "current": (trajectory("current", "a5"),),
        "one-behind": (trajectory("one", "a4"),),
        "two-behind": (trajectory("two", "a3"),),
        "never-served": (trajectory("never", "a6"),),
        "unknown": (trajectory("unknown"),),
        "mixed": (trajectory("five", "a5"), trajectory("three", "a3")),
        "rolled-out": (TaskSample(Path("task")),),
    }
    # The scenario never served a2: lag counts the artifacts that it served, not the artifacts that it made.
    served = ServedArtifacts(("a1", "a3", "a4", "a5"))
    behind = {"two-behind", "never-served", "unknown", "mixed"}
    assert {name for name, group in samples.items() if lags_behind(group, served, 1)} == behind
    # A bound of 0 keeps only data of the served artifact, and data that the algorithm rolls out.
    assert {name for name, group in samples.items() if lags_behind(group, served, 0)} == behind | {"one-behind"}
    # A scenario that loads no weights serves no artifact, so nothing lags.
    assert not any(lags_behind(group, ServedArtifacts(), 0) for group in samples.values())


def test_served_artifacts_have_no_equal_neighbors_and_a_non_negative_commit_number() -> None:
    with pytest.raises(ValueError, match="no two neighbors equal"):
        ServedArtifacts(("a1", "a1"))
    for commit_number in (-1, True, 1.0):
        with pytest.raises(ValueError, match="commit_number must be a non-negative int"):
            ServedArtifacts(("a1",), commit_number=commit_number)  # type: ignore[arg-type]


def test_the_default_stage_is_on_policy() -> None:
    stage = EachInference()
    old, new, unnamed = trajectory("old", "a1"), trajectory("new", "a2"), trajectory("unnamed")
    assert not stage.is_stale((old, unnamed), ServedArtifacts())
    served = ServedArtifacts(("a1", "a2"))
    assert stage.is_stale((old,), served) and not stage.is_stale((new,), served)
    assert stage.is_stale((new, old), served), "one stale sample makes the group stale"
    assert stage.is_stale((unnamed,), served), "a sample without a producing artifact is stale"
    assert not stage.is_stale((TaskSample(Path("task")),), served), "a task rolls out with the weights it trains"
    # An artifact served beside the active selection answers requests now, so its data is not behind.
    assert not stage.is_stale((old,), ServedArtifacts(("a1", "a2"), frozenset({"a1"})))
    # An artifact that serves again is the newest, and its data is fresh.
    again = ServedArtifacts(("a1", "a2", "a1"))
    assert not stage.is_stale((old,), again) and stage.is_stale((new,), again)


def test_a_stage_takes_no_staleness_key() -> None:
    with pytest.raises(PluginConfigError, match="takes no config, got max_staleness"):
        EachInference.from_config({"max_staleness": 0}, context=PipelineContext("s"), environ={})


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
        {"choices": [{"finish_reason": "stop"}], "stream_delivery": {"complete": False}},
        {"stream": True, "complete": False, "body": "partial"},
    ],
    ids=["native", "sglang", "openai", "tool-call", "anthropic", "legacy-stripped-anthropic", "capture", "raw"],
)
def test_incomplete_inference_ends_before_sample_creation(response: dict[str, object]) -> None:
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
