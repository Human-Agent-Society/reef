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
from reef.core.reports import ReportBase, ReportValidationError, ScoredRolloutReport
from reef.dispatcher import Dispatcher
from reef.recipe.base import Recipe
from reef.recipe.registry import RecipeRegistry
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
        RecipeRegistry({name: recipe}),
        InMemoryRepositoryBackend.factory(initial, root=root / "repository"),
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
        dispatcher.accept_record(_report_record("bad", {"score": "high"}), recipe="scored")
    # A compliant report on the same scenario still lands.
    stored = dispatcher.accept_record(_report_record("good", {"score": 1.0}), recipe="scored")
    assert stored.agent_record_id == "good"
    scenario = dispatcher.get_or_create_scenario("workload", "scored")
    assert scenario is not None
    assert scenario.trainer.processor.context.report_type is ScoredRolloutReport


def test_undeclared_recipe_keeps_open_ingress_via_anyreport() -> None:
    dispatcher = _dispatcher(Recipe(), "recipe")
    stored = dispatcher.accept_record(_report_record("open", {"feedback": "no score at all"}), recipe="recipe")
    assert stored.agent_record_id == "open"


# ------------------------------------------------- processors parse and name


# openclawrl is absent: its processor consumes the (already
# ingress-validated) hint text directly and derives every verdict itself, so
# there is no report parse for it to name a violation against.


def test_tttd_names_grid_mismatch_and_schema_violations() -> None:
    processor = TTTDProcessor(
        ProcessorContext(
            "discovery",
            {"groups_per_step": 2, "rollouts_per_group": 3},
            report_type=TTTDGroupedRolloutReport,
        )
    )
    # Announces an 8x64 grid at a 2x3 scenario: a config-relative rejection
    # from the judge, named.
    mismatched = TTTDGroupedRolloutReport(
        score=1.0, step=0, group=0, rollout=0, groups_per_step=8, rollouts_per_group=64
    ).to_dict(references=["inference-a"])
    processor.ingest(
        AgentRecord.create(
            scenario="discovery",
            request_type=RequestType.REPORT,
            agent_record_id="mismatch",
            references=("inference-a",),
            payload=mismatched,
        )
    )
    # Structurally malformed: no coordinates at all — the judge's schema
    # parse rejects it, named.
    processor.ingest(
        AgentRecord.create(
            scenario="discovery",
            request_type=RequestType.REPORT,
            agent_record_id="uncoordinated",
            references=("inference-b",),
            payload={"score": 1.0, "references": ["inference-b"]},
        )
    )
    reasons = processor.never_reasons
    assert any("8x64" in reason for reason in reasons)
    assert any("metadata" in reason for reason in reasons)
