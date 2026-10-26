"""Adaptive AutoEnvScaling: tasks outside the band become parents, and later generations vary them."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    NAMES,
    REQUEST,
    NoChecks,
    StandInAuthor,
    autoenvscaling,
    cursor,
    generating_config,
    report,
    run_tasks,
    stand_in_e2b_package,
)
from recipes.beta.autoenvscaling import AgentAuthor, AutoEnvScalingGenerator, TaskVariant
from recipes.beta.autoenvscaling import author as author_module
from recipes.beta.autoenvscaling import generator as generator_module
from recipes.beta.autoenvscaling.sources import Repository, SessionSource
from reef.core.plugins import PluginConfigError
from reef.recipe.generated_tasks.generator import FailedTask, HeldOutTask


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


def parents(tmp_path: Path) -> list[dict[str, Any]]:
    return json.loads((tmp_path / "state" / "adaptive.json").read_text())


def test_without_adaptive_the_generator_keeps_no_parents_and_its_report_is_unchanged(tmp_path: Path) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author, count=2)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0)), generation=0))
    assert not (tmp_path / "state" / "adaptive.json").exists()
    assert not (tmp_path / "state" / "parents").exists()
    assert all(set(proposal) >= {"index", "position"} for proposal in report(tmp_path)["proposals"])
    assert not any("parent_task" in proposal for proposal in report(tmp_path)["proposals"])
    asyncio.run(generator.generate(REQUEST))
    assert [call["variant"] for call in author.calls] == [None] * 4


@pytest.mark.parametrize(("training", "direction"), [((1.0, 1.0), "harder"), ((0.0, 0.0), "simpler")])
def test_a_task_outside_the_band_gets_a_variant_from_its_own_session(
    tmp_path: Path, training: tuple[float, ...], direction: str
) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(generator.record_results(run_tasks(generator, training, (0.5, 0.5)), generation=0))
    [parent] = parents(tmp_path)
    assert (parent["name"], parent["direction"], parent["pass_rate"], parent["episodes"]) == (
        NAMES[0],
        direction,
        sum(training) / 2,
        2,
    )
    assert (parent["source_record_id"], parent["varied"]) == ("session-1", False)
    assert (tmp_path / "state" / "parents" / NAMES[0] / "instruction.md").is_file()

    tasks = asyncio.run(generator.generate(REQUEST))
    variant = author.calls[2]["variant"]
    assert isinstance(variant, TaskVariant)
    assert (variant.parent, variant.direction, variant.episodes) == (NAMES[0], direction, 2)
    assert variant.parent_dir == tmp_path / "state" / "parents" / NAMES[0]
    assert author.calls[2]["source"] == "session-1", "a variant comes from its parent's session"
    assert author.calls[3]["variant"] is None, "the other slot rebuilds the next session"
    # The variant keeps its parent's session as its source, so the split places it with its parent.
    assert tasks[0].source_agent_record_ids[1] == "session-1"
    assert (tasks[0].metadata["parent_task"], tasks[0].metadata["variant"]) == (NAMES[0], direction)
    assert "parent_task" not in tasks[1].metadata
    assert cursor(tmp_path)["next_position"] == 3, "a variant takes no session from the walk"
    assert parents(tmp_path)[0]["varied"] is True


def test_the_report_names_each_variant_with_its_parent(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    asyncio.run(generator.record_results(run_tasks(generator, (0.5, 0.5)), generation=0))
    first, second = report(tmp_path, 1)["proposals"]
    assert (first["position"], first["parent_task"], first["direction"]) == (-1, NAMES[0], "harder")
    assert "parent_task" not in second and second["position"] == 2


def test_eval_tasks_and_outages_never_become_parents(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    first, second = asyncio.run(generator.generate(REQUEST))
    for task in (first, second):
        assert asyncio.run(generator.validate(task)).is_valid
    outage = FailedTask(second, "the sandbox did not start", infrastructure_failure=True)
    asyncio.run(generator.record_results((HeldOutTask(first), outage), generation=0))
    assert parents(tmp_path) == []
    assert not (tmp_path / "state" / "parents").exists()


def test_a_replayed_result_adds_its_parent_once_and_a_parent_is_varied_once_across_restarts(tmp_path: Path) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    results = run_tasks(generator, (1.0, 1.0), (0.5, 0.5))
    asyncio.run(generator.record_results(results, generation=0))
    asyncio.run(generator.record_results(results, generation=0))
    assert [parent["name"] for parent in parents(tmp_path)] == [NAMES[0]]
    asyncio.run(generator.generate(REQUEST))
    restarted = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(restarted.generate(REQUEST))
    assert [call["variant"] is not None for call in author.calls] == [False, False, True, False, False, False]


def test_the_oldest_parents_and_their_files_go_beyond_the_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generator_module, "MAX_PARENTS", 1)
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.0, 0.0)), generation=0))
    assert [parent["name"] for parent in parents(tmp_path)] == [NAMES[1]]
    kept = sorted(path.name for path in (tmp_path / "state" / "parents").iterdir() if not path.name.startswith("."))
    assert kept == [NAMES[1]]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"adaptive": "yes"}, "adaptive must be true or false"),
        ({"adaptive": True, "author": "prompt"}, "adaptive generation needs author: agent"),
        ({"adaptive": True, "variant_share": 0}, r"variant_share must be in \(0, 1\]"),
    ],
)
def test_from_config_refuses_an_adaptive_config_that_cannot_run(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    with pytest.raises(PluginConfigError, match=message):
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **fields), scenario="autoenvscaling", environ={"E2B_API_KEY": "key"}
        )


def test_the_agent_gets_the_parent_and_its_pass_rate_and_the_variant_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_dir = tmp_path / "parent"
    (parent_dir / "tests").mkdir(parents=True)
    (parent_dir / "instruction.md").write_text("Add a --dry-run flag.\n")
    (parent_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    repository = tmp_path / "repository"
    repository.mkdir()
    monkeypatch.setattr(author_module, "session_text", lambda reef, source: "the session")
    monkeypatch.setattr(author_module, "repository_snapshot", lambda source, cache_dir: repository)
    seen: dict[str, object] = {}

    def run(
        self: AgentAuthor, work_dir: Path, gateway: object, tools: object, calls: object, stop: object, **kw: object
    ) -> None:
        seen.update(kw)
        seen["inputs"] = sorted(str(path.relative_to(work_dir)) for path in (work_dir / "inputs").rglob("*"))
        seen["variant.md"] = (work_dir / "inputs" / "variant.md").read_text()

    monkeypatch.setattr(AgentAuthor, "run", run)
    agent = AgentAuthor(
        reef_url="http://127.0.0.1:9",
        token=None,
        scenario="autoenvscaling",
        model="designer",
        api="openai",
        executor=object(),  # type: ignore[arg-type]
        cache_dir=tmp_path / "cache",
        timeout_seconds=60.0,
        max_model_calls=5,
    )
    variant = TaskVariant(parent="t-0", parent_dir=parent_dir, direction="simpler", pass_rate=0.125, episodes=8)
    source = SessionSource("session-1", "s", Repository("https://example.com/r.git", "abc1234", "/app"))
    agent.author(source, tmp_path / "run", NoChecks(1), tags={}, stop=threading.Event(), variant=variant)
    assert seen["variant"] == variant
    assert {"inputs/parent/instruction.md", "inputs/parent/tests/test.sh", "inputs/variant.md"} <= set(
        seen["inputs"]  # type: ignore[arg-type]
    )
    assert seen["variant.md"] == (
        "The solver passed the task in ../inputs/parent/ (t-0) in 12% of 8 episodes.\n"
        "Write a simpler task: the solver passes the parent too rarely to learn from it.\n"
    )
    rules_with = author_module.AGENT_RULES.format(
        system_prompt="", rules="", max_checks=1, review="", variant=author_module.VARIANT_INPUTS_TEXT
    )
    rules_without = author_module.AGENT_RULES.format(system_prompt="", rules="", max_checks=1, review="", variant="")
    assert "THIS RUN WRITES A VARIANT" in rules_with and "../inputs/parent/" in rules_with
    assert "../inputs/parent/" not in rules_without and "VARIANT" not in rules_without


@pytest.mark.parametrize(
    "fields",
    [
        {"direction": "sideways"},
        {"pass_rate": 1.5},
        {"episodes": 0},
        {"parent": ""},
    ],
)
def test_a_variant_refuses_fields_outside_its_definition(tmp_path: Path, fields: dict[str, object]) -> None:
    values: dict[str, Any] = {
        "parent": "t-0",
        "parent_dir": tmp_path,
        "direction": "harder",
        "pass_rate": 1.0,
        "episodes": 2,
        **fields,
    }
    with pytest.raises(ValueError, match="a variant's"):
        TaskVariant(**values)


def test_the_prompt_designer_refuses_a_variant(tmp_path: Path) -> None:
    designer = author_module.PromptAuthor(
        reef_url="http://127.0.0.1:9", token=None, scenario="autoenvscaling", model="designer", api="openai"
    )
    variant = TaskVariant(parent="t-0", parent_dir=tmp_path, direction="harder", pass_rate=1.0, episodes=2)
    source = SessionSource("session-1", "s", Repository("https://example.com/r.git", "abc1234", "/app"))
    with pytest.raises(ValueError, match="adaptive generation needs author: agent"):
        designer.author(source, tmp_path / "run", NoChecks(1), tags={}, stop=threading.Event(), variant=variant)
