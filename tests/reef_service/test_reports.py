"""Report schemas: contracts, violations, and their wiring (#266)."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, make_dataclass
from pathlib import Path
from typing import Any

import pytest

from recipes.tttd import TTTDGroupedRolloutReport, TTTDProcessor
from reef.artifact.memory import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport, TeacherContextReport
from reef.dispatcher import Dispatcher
from reef.recipe.base import Recipe
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train import ProcessorContext


@dataclass(frozen=True)
class TaskOutcome(ReportBase):
    """A benchmark-style outcome with typed method metadata."""

    score: float
    task_id: str | None = None
    resolved: bool | None = None
    failure_mode: str | None = None
    parser_results: Mapping[str, str] | None = None
    trajectory: str | None = None


# ------------------------------------------------------------------ round trips


def test_fields_need_no_wire_location_by_default() -> None:
    # score rides the score channel; every other bare field lands at
    # metadata.<name>.
    from dataclasses import dataclass

    from reef.core.reports import ReportBase

    @dataclass(frozen=True)
    class _Outcome(ReportBase):
        score: float
        passed: bool = False

    body = _Outcome(score=1.0, passed=True).to_dict()
    assert body == {"score": 1.0, "metadata": {"passed": True}}
    assert _Outcome.from_dict(body) == _Outcome(score=1.0, passed=True)


def test_scored_rollout_round_trip() -> None:
    schema = ScoredRolloutReport(score=0.83)
    body = schema.to_dict(references=["receipt-1"])
    assert body == {"score": 0.83, "references": ["receipt-1"]}
    assert ScoredRolloutReport.from_dict(body) == schema


def test_teacher_context_round_trip() -> None:
    # The distilling recipes' contract: the teacher context rides metadata, the score is optional.
    schema = TeacherContextReport(teacher_context="100 degrees Celsius.")
    body = schema.to_dict(references=["receipt-1"])
    assert body == {"metadata": {"teacher_context": "100 degrees Celsius."}, "references": ["receipt-1"]}
    assert TeacherContextReport.from_dict(body) == schema
    scored = TeacherContextReport.from_dict({"score": 0.5, "metadata": {"teacher_context": "demo"}})
    assert scored == TeacherContextReport(teacher_context="demo", score=0.5)
    # A teacher that reads no privileged text (on-policy distillation) leaves the teacher context empty,
    # and a field at its default is not serialized.
    assert TeacherContextReport.from_dict({"metadata": {}}) == TeacherContextReport()
    assert TeacherContextReport().to_dict(references=["receipt-1"]) == {"references": ["receipt-1"]}


def test_grouped_rollout_round_trip() -> None:
    schema = TTTDGroupedRolloutReport(score=0.5, step=4, group=1, rollout=17, groups_per_step=8, rollouts_per_group=64)
    body = schema.to_dict()
    assert isinstance(schema, ScoredRolloutReport)
    assert body["metadata"]["comparison_set"] == "tttd-step-4-group-1"
    assert TTTDGroupedRolloutReport.from_dict(body) == schema


def test_task_outcome_round_trip() -> None:
    schema = TaskOutcome(
        score=1.0,
        task_id="fix-git",
        resolved=True,
        failure_mode="none",
        parser_results={"test_push": "passed"},
        trajectory="final transcript tail",
    )
    body = schema.to_dict(references=["receipt-1"])
    assert body["metadata"]["task_id"] == "fix-git"
    assert body["metadata"]["trajectory"] == "final transcript tail"
    assert TaskOutcome.from_dict(body) == schema


def test_minimal_score_only_report_is_a_valid_task_outcome() -> None:
    # Back-compat floor: plain scored reports parse with all context absent.
    assert TaskOutcome.from_dict({"score": 0.0}) == TaskOutcome(score=0.0)


# ------------------------------------------------------------------- violations


@pytest.mark.parametrize(
    ("report_type", "payload", "fragment"),
    [
        (ScoredRolloutReport, {}, "score is required"),
        (ScoredRolloutReport, {"score": True}, "score must be a number"),
        (ScoredRolloutReport, {"score": float("nan")}, "score must be finite"),
        (TTTDGroupedRolloutReport, {"score": 1.0}, "metadata.step is required"),
        (
            TTTDGroupedRolloutReport,
            {
                "score": 1.0,
                "metadata": {
                    "algorithm": "grpo",
                    "step": 0,
                    "group": 0,
                    "rollout": 0,
                    "groups_per_step": 2,
                    "rollouts_per_group": 3,
                },
            },
            "metadata.algorithm",
        ),
        (TaskOutcome, {"score": 1.0, "metadata": {"resolved": "yes"}}, "resolved must be a boolean"),
        (TeacherContextReport, {"metadata": {"teacher_context": 3}}, "metadata.teacher_context must be a string"),
    ],
)
def test_violations_name_the_broken_field(report_type: type[ReportBase], payload: dict, fragment: str) -> None:
    with pytest.raises(ReportValidationError) as excinfo:
        report_type.from_dict(payload)
    assert fragment in str(excinfo.value)


def test_tttd_coordinates_must_sit_inside_their_announced_grid() -> None:
    body = TTTDGroupedRolloutReport(
        score=1.0, step=0, group=0, rollout=0, groups_per_step=2, rollouts_per_group=3
    ).to_dict()
    body["metadata"]["group"] = 2  # == groups_per_step
    body["metadata"]["comparison_set"] = "tttd-step-0-group-2"
    with pytest.raises(ReportValidationError, match="group"):
        TTTDGroupedRolloutReport.from_dict(body)


def test_tttd_comparison_set_must_echo_coordinates() -> None:
    body = TTTDGroupedRolloutReport(
        score=1.0, step=0, group=0, rollout=0, groups_per_step=2, rollouts_per_group=3
    ).to_dict()
    body["metadata"]["comparison_set"] = "tttd-step-9-group-9"
    with pytest.raises(ReportValidationError, match="comparison_set"):
        TTTDGroupedRolloutReport.from_dict(body)


# --------------------------------------------------------- floor, not a ceiling


def test_extra_keys_pass_through_untouched() -> None:
    # The schema only owns its fields: producers add extras on the returned
    # dict, and from_dict ignores them — untyped keys ride along freely.
    body = ScoredRolloutReport(score=0.5).to_dict(references=["receipt-1"])
    body["metadata"] = {"run_id": "run-7"}
    body["feedback"] = {"notes": "next state unchanged"}
    assert ScoredRolloutReport.from_dict(body).score == 0.5


# ------------------------------------------------------------------ declarations


@pytest.mark.parametrize(
    "annotation",
    [list[str], int | str, Mapping[str, int]],
)
def test_report_declarations_reject_unsupported_annotations(annotation: Any) -> None:
    UnsupportedReport = make_dataclass("UnsupportedReport", [("value", annotation)], bases=(ReportBase,), frozen=True)

    with pytest.raises(TypeError, match=r"UnsupportedReport\.value has unsupported report annotation"):
        UnsupportedReport.from_dict({"metadata": {"value": "anything"}})


def test_report_spec_cache_does_not_mutate_the_report_class() -> None:
    ScoredRolloutReport.from_dict({"score": 1.0})
    assert "_wire_specs_cache" not in ScoredRolloutReport.__dict__


def test_report_type_is_not_recipe_configuration() -> None:
    assert "report_type" not in Recipe.__dataclass_fields__
    assert Recipe().report_type is None


# ------------------------------------------------------------ ingress (mandatory)


def _dispatcher(recipe: Recipe, name: str) -> Dispatcher:
    root = Path(tempfile.mkdtemp(prefix="reef-artifacts-"))
    initial = root / "initial"
    initial.mkdir()
    return Dispatcher(
        recipe,
        InMemoryRepositoryBackend.factory(initial, root=root / "repository"),
        scenario_storage=SQLiteScenarioStorage(),
    )


def _report_record(agent_record_id: str, payload: dict) -> AgentRecord:
    return AgentRecord.create(
        scenario="workload",
        request_type=RequestType.REPORT,
        agent_record_id=agent_record_id,
        payload=payload,
    )


def test_declared_schema_is_enforced_at_ingress() -> None:
    class _ScoredRecipe(Recipe):
        @property
        def report_type(self) -> type[ReportBase]:
            return ScoredRolloutReport

    dispatcher = _dispatcher(_ScoredRecipe(name="scored"), "scored")
    with pytest.raises(ReportValidationError, match="score must be a number"):
        dispatcher.accept_record(_report_record("bad", {"score": "high"}))
    # A compliant report on the same scenario still lands.
    stored = dispatcher.accept_record(_report_record("good", {"score": 1.0}))
    assert stored.agent_record_id == "good"
    scenario = dispatcher.get_or_create_scenario("workload")
    assert scenario is not None
    assert scenario.trainer.processor.context.report_type is ScoredRolloutReport


def test_undeclared_recipe_keeps_open_ingress_via_anyreport() -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    stored = dispatcher.accept_record(_report_record("open", {"feedback": "no score at all"}))
    assert stored.agent_record_id == "open"


# ------------------------------------------------- processors parse and name


# openclawrl is absent: its processor consumes the (already
# ingress-validated) hint text directly and derives every verdict itself, so
# there is no report parse for it to name a violation against.


def test_tttd_raises_for_grid_mismatch_and_schema_violations() -> None:
    processor = TTTDProcessor(
        ProcessorContext(
            "discovery",
            {"groups_per_step": 2, "rollouts_per_group": 3},
            report_type=TTTDGroupedRolloutReport,
        )
    )
    processor.ingest(
        AgentRecord.create(
            scenario="discovery",
            request_type=RequestType.INFERENCE,
            agent_record_id="inference-a",
            payload={},
        )
    )
    mismatched = TTTDGroupedRolloutReport(
        score=1.0,
        step=0,
        group=0,
        rollout=0,
        groups_per_step=8,
        rollouts_per_group=64,
    ).to_dict(references=["inference-a"])
    with pytest.raises(ValueError, match="8x64"):
        processor.ingest(
            AgentRecord.create(
                scenario="discovery",
                request_type=RequestType.REPORT,
                agent_record_id="mismatch",
                payload=mismatched,
            )
        )
    with pytest.raises(ReportValidationError, match="metadata"):
        processor.ingest(
            AgentRecord.create(
                scenario="discovery",
                request_type=RequestType.REPORT,
                agent_record_id="uncoordinated",
                payload={"score": 1.0, "references": ["inference-a"]},
            )
        )


def test_report_references_are_checked_against_storage_not_processor_cache() -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    try:
        scenario = dispatcher.get_or_create_scenario("workload")
        source = AgentRecord.create(
            scenario="workload",
            request_type=RequestType.INFERENCE,
            agent_record_id="source",
            payload={},
        )
        scenario.records.append(source)  # Stored, but deliberately not ingested by the processor.
        stored = dispatcher.accept_record(_report_record("feedback", {"references": ["source"], "score": 0.0}))
        assert stored.references == ("source",)
    finally:
        dispatcher.close()


@pytest.mark.parametrize("references", [["missing"], ["foreign"], ["report-source"], ["source", "source"]])
def test_invalid_report_references_are_not_persisted(references: list[str]) -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    try:
        for scenario_name, record_id, request_type in (
            ("workload", "source", RequestType.INFERENCE),
            ("other", "foreign", RequestType.INFERENCE),
            ("workload", "report-source", RequestType.REPORT),
        ):
            dispatcher.accept_record(
                AgentRecord.create(
                    scenario=scenario_name,
                    request_type=request_type,
                    agent_record_id=record_id,
                    payload={},
                )
            )
        scenario = dispatcher.get_or_create_scenario("workload")
        with pytest.raises(ReportValidationError, match="reference"):
            dispatcher.accept_record(_report_record("invalid", {"references": references, "score": 1.0}))
        assert scenario.records.get_for_audit("workload", "invalid") is None
    finally:
        dispatcher.close()


def test_missing_reference_rejection_does_not_queue_or_reserve_the_report_id() -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    try:
        record = _report_record("feedback", {"references": ["source"], "score": 1.0})
        with pytest.raises(ReportValidationError):
            dispatcher.accept_record(record)
        dispatcher.accept_record(
            AgentRecord.create(
                scenario="workload",
                request_type=RequestType.INFERENCE,
                agent_record_id="source",
                payload={},
            )
        )
        scenario = dispatcher.get_or_create_scenario("workload")
        assert scenario.records.get("workload", "feedback") is None
        assert dispatcher.accept_record(record).agent_record_id == "feedback"
    finally:
        dispatcher.close()


def test_report_retry_after_source_purge_remains_idempotent_and_conflicts_fail() -> None:
    from dataclasses import replace
    from reef.storage.records import RecordConflict

    dispatcher = _dispatcher(Recipe(), "recipe")
    try:
        dispatcher.accept_record(
            AgentRecord.create(
                scenario="workload",
                request_type=RequestType.INFERENCE,
                agent_record_id="source",
                payload={},
            )
        )
        record = _report_record("feedback", {"references": ["source"], "score": 1.0})
        dispatcher.accept_record(record)
        scenario = dispatcher.get_or_create_scenario("workload")
        scenario.records.compact("workload", frozenset({"source", "feedback"}))
        scenario.records.purge_compacted("workload", before=1e20)
        assert dispatcher.accept_record(record).agent_record_id == "feedback"
        with pytest.raises(RecordConflict):
            dispatcher.accept_record(replace(record, payload={**record.payload, "score": 0.0}))
        assert scenario.records.count("workload") == 0
    finally:
        dispatcher.close()


@pytest.mark.parametrize("eligible", [True, False])
def test_report_eligibility_flag_is_rejected_at_ingress(eligible: bool) -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    try:
        with pytest.raises(ReportValidationError, match="eligible"):
            dispatcher.accept_record(_report_record("invalid", {"metadata": {"training": {"eligible": eligible}}}))
        scenario = dispatcher.get_or_create_scenario("workload")
        assert scenario.records.get_for_audit("workload", "invalid") is None
    finally:
        dispatcher.close()
