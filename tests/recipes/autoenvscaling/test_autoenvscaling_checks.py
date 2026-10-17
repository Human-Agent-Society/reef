"""AutoEnvScaling's task checks: the test review, the agent's check and the report."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    NAMES,
    REQUEST,
    NoChecks,
    StandInAuthor,
    agent_files,
    autoenvscaling,
    fenced,
    generating_config,
    serving_reef,
    stand_in_e2b_package,
)
from tests.support.task_service_stubs import StandInGenerator, task_for

from recipes.beta.autoenvscaling import AutoEnvScalingGenerator, checks
from recipes.beta.autoenvscaling import generator as generator_module
from recipes.beta.autoenvscaling.author import REVIEW_TEXT
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, TaskPlayResult


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


FINDING = {
    "test_file": "tests/test.sh",
    "kind": "wording",
    "requirement": "the report says 'OpenID Connect Discovery gives all three fields defaults'",
    "reason": "the instruction asks for a report and gives none of its wording",
}
CLOSED_REEF = "http://127.0.0.1:9"


def reviewer(reef_url: str, *, is_refusing: bool = True) -> checks.ReviewCheck:
    return checks.ReviewCheck(
        reef_url=reef_url,
        token="token",
        scenario="autoenvscaling",
        model="reviewer",
        api="openai",
        is_refusing=is_refusing,
    )


def reef_url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


def validated(generator: AutoEnvScalingGenerator) -> tuple[HarborTask, tuple[str, ...]]:
    """Generate one task and validate it: the task and its validation errors."""
    (item,) = asyncio.run(generator.generate(REQUEST))
    return item.task, asyncio.run(generator.validate(item.task)).errors


def closed_with(generator: AutoEnvScalingGenerator, task: HarborTask, errors: tuple[str, ...]) -> dict[str, Any]:
    """Close the generation as the recipe would after its plays, and read its report."""
    if errors:
        played = TaskPlayResult(task, error="; ".join(errors))
    else:
        # The recipe writes a valid task again to play it.
        asyncio.run(generator.service.write_task(task))
        played = TaskPlayResult(task, (0.0, 1.0))
    asyncio.run(generator.record_results((played,), generation=0))
    return json.loads((generator.state_dir / f"generation-{generator.generation:05d}.json").read_text())


@pytest.mark.parametrize("is_refusing", [True, False])
def test_the_review_lists_unstated_requirements_and_refuses_or_measures(tmp_path: Path, is_refusing: bool) -> None:
    reef, calls = serving_reef(json.dumps({"findings": [FINDING]}))
    service = StandInGenerator(tmp_path / "tasks")
    try:
        check = reviewer(reef_url(reef), is_refusing=is_refusing)
        generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1, task_checks=(check,))
        task, errors = validated(generator)
    finally:
        reef.shutdown()
        reef.server_close()
    # A refusing review also runs in the agent's own check, before validate reviews the task again.
    assert len(calls) == (2 if is_refusing else 1)
    call = calls[-1]
    assert call["model"] == "reviewer" and call["messages"][0]["content"] == checks.REVIEW_SYSTEM_PROMPT
    assert call["headers"]["x-reef-scenario"] == "autoenvscaling"
    assert call["headers"]["x-reef-tag-role"] == "test-review" and call["headers"]["x-reef-tag-task"] == NAMES[0]
    prompt = call["messages"][1]["content"]
    assert "[BEGIN instruction.md" in prompt and task.instruction.strip() in prompt
    assert "[BEGIN tests/test.sh" in prompt and "[BEGIN solution/solve.sh" in prompt
    expected = (f"wording in tests/test.sh: {FINDING['requirement']}",)
    assert errors == (expected if is_refusing else ())

    document = closed_with(generator, task, errors)
    [proposal] = document["proposals"]
    assert proposal["stage"] == ("check" if is_refusing else "")
    oracle, review = proposal["checks"]
    assert oracle["check"] == "oracle" and oracle["details"] == {"oracle_reward": 1.0, "nop_reward": 0.0}
    assert review == {
        "check": "review",
        "errors": list(expected),
        "could_not_run": "",
        "details": {"record_id": f"rec-{len(calls)}", "findings": [FINDING]},
    }
    if not is_refusing:
        assert document["tasks"][0]["checks"] == proposal["checks"]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ("The tests look fine to me.", "is not the JSON asked for"),
        (json.dumps({"issues": []}), "holds no findings list"),
        (json.dumps({"findings": [{**FINDING, "kind": "style"}]}), "kind must be one of"),
        (json.dumps({"findings": [{**FINDING, "reason": ""}]}), "non-empty text"),
    ],
)
def test_a_review_reply_that_is_not_the_json_asked_for_means_the_review_could_not_run(
    tmp_path: Path, reply: str, message: str
) -> None:
    reef, _ = serving_reef(reply)
    service = StandInGenerator(tmp_path / "tasks")
    try:
        generator = autoenvscaling(
            tmp_path, StandInAuthor(), service, count=1, task_checks=(reviewer(reef_url(reef)),)
        )
        task, errors = validated(generator)
    finally:
        reef.shutdown()
        reef.server_close()
    assert len(errors) == 1 and errors[0].startswith("the review check could not run: ") and message in errors[0]
    document = closed_with(generator, task, errors)
    [proposal] = document["proposals"]
    assert proposal["stage"] == "unchecked"
    review = proposal["checks"][1]
    assert review["errors"] == [] and message in review["could_not_run"]


def test_a_review_that_cannot_reach_its_model_could_not_run_and_only_measures_in_measure_mode(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(
        tmp_path, StandInAuthor(), service, count=1, task_checks=(reviewer(CLOSED_REEF, is_refusing=False),)
    )
    task, errors = validated(generator)
    assert errors == (), "a measuring review keeps no task out"
    document = closed_with(generator, task, errors)
    review = document["proposals"][0]["checks"][1]
    assert review["could_not_run"].startswith("the review's call failed: the designer call did not complete")
    assert document["proposals"][0]["stage"] == "" and document["tasks"][0]["checks"][1] == review


def test_the_agents_rules_say_how_to_read_the_review_only_when_its_checks_review(tmp_path: Path) -> None:
    reviewing = NoChecks(1, is_reviewing=True)
    assert REVIEW_TEXT.strip() in agent_files(tmp_path / "on", has_behaviour_rule=True, checks=reviewing)
    assert "reviews the tests" not in agent_files(tmp_path / "off", has_behaviour_rule=True, checks=NoChecks(1))


@pytest.mark.parametrize(
    ("reply", "passed", "review"),
    [
        (json.dumps({"findings": [FINDING]}), False, {"ran": True, "findings": [FINDING]}),
        ('```json\n{"findings": []}\n```', True, {"ran": True, "findings": []}),
        ("not json", True, None),
    ],
)
def test_the_agents_check_returns_the_reviews_findings(
    tmp_path: Path, reply: str, passed: bool, review: dict[str, object] | None
) -> None:
    reef, calls = serving_reef(reply)
    author = StandInAuthor()
    try:
        generator = autoenvscaling(tmp_path, author, count=1, task_checks=(reviewer(reef_url(reef)),))
        asyncio.run(generator.generate(REQUEST))
    finally:
        reef.shutdown()
        reef.server_close()
    check = author.calls[0]["check"]
    assert check["passed"] is passed and check["oracle_reward"] == 1.0
    if review is None:
        assert check["review"]["ran"] is False and "not the JSON asked for" in check["review"]["error"]
    else:
        assert check["review"] == review and check["stage"] == "review"
    assert calls[0]["headers"]["x-reef-tag-check"] == "1" and calls[0]["headers"]["x-reef-tag-index"] == "0"


# YAML reads a bare off as false.
@pytest.mark.parametrize(
    "fields", [{}, {"review": "measure", "review_model": "reviewer"}, yaml.safe_load("review: off")]
)
def test_the_configured_review_refuses_by_default_and_only_a_refusing_one_reaches_the_agents_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, object]
) -> None:
    reef, calls = serving_reef(json.dumps({"findings": [FINDING]}))
    service = StandInGenerator(tmp_path / "tasks")
    monkeypatch.setattr(generator_module, "HttpGenerator", lambda url: service)
    try:
        generator = AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, count=1, agent_reef_url=reef_url(reef), **fields),
            scenario="autoenvscaling",
            environ={"E2B_API_KEY": "e2b-test"},
        )
        # The agent and the generator service are stand-ins; the review is the configured one.
        author = StandInAuthor()
        monkeypatch.setattr(generator, "author", author)
        task, errors = validated(generator)
    finally:
        reef.shutdown()
        reef.server_close()
    check = author.calls[0]["check"]
    finding = f"wording in tests/test.sh: {FINDING['requirement']}"
    if not fields:
        # The agent's check reviews, then validate reviews again, both on the agent's model.
        assert check["passed"] is False and check["review"] == {"ran": True, "findings": [FINDING]}
        assert errors == (finding,) and [call["model"] for call in calls] == ["gpt-6-sol"] * 2
    elif fields["review"] == "measure":
        # A measuring review records its result in validate only, on its own model.
        assert check["passed"] is True and "review" not in check
        assert errors == () and [call["model"] for call in calls] == ["reviewer"]
        assert closed_with(generator, task, errors)["proposals"][0]["checks"][1]["errors"] == [finding]
    else:
        assert check["passed"] is True and "review" not in check and errors == () and calls == []


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"review": "maybe"}, "review must be one of off, measure, refuse"),
        ({"agent_api": "anthropic"}, "asks through agent_api openai or responses"),
    ],
)
def test_from_config_refuses_check_settings_that_cannot_run(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    with pytest.raises(PluginConfigError, match=message):
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **fields), scenario="one", environ={"E2B_API_KEY": "e2b-test"}
        )
    if fields == {"agent_api": "anthropic"}:
        # Without the review, the agent may ask in any dialect.
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, review=False, **fields), scenario="one", environ={"E2B_API_KEY": "e2b-test"}
        )


def test_the_review_reads_capped_files_and_takes_a_fenced_reply() -> None:
    fenced_reply = "```json\n" + json.dumps({"findings": [FINDING]}) + "\n```"
    assert checks.review_findings(fenced_reply) == (checks.ReviewFinding(**FINDING),)
    assert checks.review_findings('{"findings": []}') == ()
    with pytest.raises(ValueError):
        checks.review_findings(fenced({"findings": []}))  # text around the JSON
    task = task_for("harbor-00000-000", "r", 8471)
    prompt = checks.review_prompt(
        replace(task, tests={**task.tests, "data.txt": "x" * (checks.REVIEW_FILE_CHARS + 10)})
    )
    assert "[... 10 characters cut]" in prompt and "x" * (checks.REVIEW_FILE_CHARS + 1) not in prompt
