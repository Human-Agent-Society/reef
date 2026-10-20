"""Human-readable check plans, task results, comparisons and failures."""

import html

import pytest

from reef.service.check_page import checks_html, evaluation_html


def report() -> dict[str, object]:
    return {
        "plan": {
            "prompt": "fix the bug in adder.py",
            "checks": ["See the test fail first", "Have another agent review"],
        },
        "checks": [
            {"id": "request-plan", "group": "request", "status": "pass", "observed": "a valid plan"},
            {
                "id": "request-behavior",
                "group": "request",
                "status": "pass",
                "expected": "joined requirements",
                "observed": "The tests and the second-agent review completed successfully.",
            },
            {
                "id": "regression-0",
                "group": "regression",
                "status": "pass",
                "expected": "Write and read a note",
                "current_score": 1.0,
                "score": 1.0,
            },
            {"id": "change-review", "group": "review", "status": "pass", "observed": "The changes match the request"},
        ],
        "reviewer_model": "local-model",
        "same_model": True,
        "current_release_id": "serving-release",
    }


def test_requested_behavior_lists_generated_requirements_with_one_task_result() -> None:
    evaluation = report()
    page = checks_html(evaluation["checks"], evaluation, result="pending")
    assert "fix the bug in adder.py" in page
    assert "<li>See the test fail first</li>" in page
    assert "<li>Have another agent review</li>" in page
    assert "result for the whole task" in page and "does not contain separate results" in page
    assert "Test plan: " in page
    assert "waiting for your review" in page
    assert page.index('id="evaluation-request"') < page.index('id="evaluation-regression"')
    assert 'id="evaluation-limit"' not in page and "Not possible on this harness" not in page
    assert '<details class="evaluation-group" id="evaluation-regression">' in page
    assert "Current release" in page and "Candidate" in page


def test_limits_show_as_their_own_group_and_explain_the_review() -> None:
    evaluation = report()
    evaluation["checks"].insert(
        1,
        {
            "id": "limit-0",
            "group": "limit",
            "status": "unsupported",
            "expected": "Refuse every tool except web search",
            "reason": "A command cannot take a tool away on this harness",
        },
    )
    page = checks_html(evaluation["checks"], evaluation, result="pending")
    assert '<details class="evaluation-group" id="evaluation-limit" open>' in page
    assert "Refuse every tool except web search</h3>" in page
    assert '<span class="evaluation-label">What the request asks</span>' in page
    assert '<span class="evaluation-label">Why</span>' in page
    assert "A command cannot take a tool away on this harness" in page
    assert 'data-status="unsupported">Not possible here' in page
    assert "part of the request is not possible on this harness" in page
    assert "1 not possible here" in page and "need attention" not in page


@pytest.mark.parametrize("status", ["fail", "invalid", "not_run", "running", "pending"])
def test_incomplete_or_failed_groups_expand_with_the_real_status(status: str) -> None:
    checks = [{"id": "regression-0", "group": "regression", "status": status, "reason": "Recorded result"}]
    page = checks_html(checks)
    assert '<details class="evaluation-group" id="evaluation-regression" open>' in page
    assert f'data-status="{status}"' in page
    assert "Recorded result" in page
    if status == "running":
        assert "evaluation-spinner" in page
    assert "Not scored" in page


def test_a_ready_plan_does_not_mark_the_pending_application_task_passed() -> None:
    checks = [
        {"id": "request-plan", "group": "request", "status": "pass", "observed": "Read the file; Run its test"},
        {"id": "request-behavior", "group": "request", "status": "pending"},
    ]
    page = checks_html(checks)
    assert "Test plan: " in page
    assert "Read the file;\nRun its test" in page
    assert 'Requested behavior</h3><span class="evaluation-badge pending"' in page
    assert "Recorded checks passed" not in page


def test_protected_comparison_keeps_a_regression_and_the_rejection_reason_visible() -> None:
    checks = [
        {
            "id": "regression-0",
            "group": "regression",
            "status": "fail",
            "current_score": 1.0,
            "score": 0.0,
            "reason": "The candidate could not read the note",
        }
    ]
    page = checks_html(checks, reason="protected task regressed", result="rejected")
    assert "Current release</span><strong>Passed" in page
    assert "Candidate</span><strong>Failed" in page
    assert "The candidate could not read the note" in page and "protected task regressed" in page
    assert "current release stays in use" in page


def test_plan_task_results_and_metadata_escape_untrusted_html() -> None:
    text = '<script>alert("model text")</script>'
    evaluation = {
        "plan": {"prompt": text, "checks": [text]},
        "reviewer_model": text,
        "current_release_id": text,
        "checks": [{"id": "request-behavior", "group": "request", "status": "pass", "observed": text}],
    }
    page = checks_html(evaluation["checks"], evaluation, text)
    assert f"<li>{html.escape(text)}</li>" in page and "<script>" not in page


def test_records_without_a_plan_keep_the_expected_task_and_result() -> None:
    checks = [
        {
            "id": "request-behavior",
            "group": "request",
            "status": "fail",
            "expected": "Read the downloaded paper",
            "reason": "No paper was read",
        }
    ]
    page = evaluation_html({"evaluation_report": {"checks": checks}}, result="rejected")
    assert "Read the downloaded paper" in page and "No paper was read" in page
    assert "Test plan: " not in page


def test_pages_without_independent_evaluation_keep_their_existing_content() -> None:
    assert evaluation_html({"candidate_score": 1}) == ""
    assert checks_html([]) == ""


def test_a_change_review_concern_is_shown_for_the_persons_review() -> None:
    evaluation = report()
    review = next(check for check in evaluation["checks"] if check["group"] == "review")
    review.update(status="concern", reason="The note adds an exception the request lacks.")
    page = checks_html(evaluation["checks"], evaluation, result="pending")
    assert 'data-status="concern">Needs your review' in page
    assert "the change review found something for you to judge" in page
    assert "1 for your review" in page and "need attention" not in page
