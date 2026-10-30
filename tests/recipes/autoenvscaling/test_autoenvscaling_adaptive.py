"""Adaptive AutoEnvScaling: tasks outside the band become parents, and later generations vary them."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    NAMES,
    REQUEST,
    NoChecks,
    ScriptedRollouts,
    ScriptedSession,
    StandInAuthor,
    agent_files,
    autoenvscaling,
    cursor,
    generating_config,
    report,
    run_tasks,
    stand_in_e2b_package,
)

from recipes.beta.autoenvscaling import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    AutoEnvScalingGenerator,
    CalibrationCheck,
    CandidateChecks,
    TaskVariant,
    VariantAuthor,
)
from recipes.beta.autoenvscaling import author as author_module
from recipes.beta.autoenvscaling.author_credentials import CredentialHeld
from recipes.beta.autoenvscaling import generator as generator_module
from recipes.beta.autoenvscaling.checks import PassRateBand
from recipes.beta.autoenvscaling.generator import ParentTask
from recipes.beta.autoenvscaling.sources import Repository, SessionSource
from reef.core.plugins import PluginConfigError
from reef.recipe.generated_tasks.generator import FailedTask, HeldOutTask, RejectedTask


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


def parents(tmp_path: Path) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = json.loads((tmp_path / "state" / "adaptive.json").read_text())
    return kept


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
    assert (parent["name"], parent["direction"], parent["pass_rate"], parent["rollouts"]) == (
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
    assert (variant.parent, variant.direction, variant.rollouts) == (NAMES[0], direction, 2)
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


def test_a_generation_closed_again_after_a_crash_adds_its_parent_once(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    results = run_tasks(generator, (1.0, 1.0), (0.5, 0.5))
    open_file = generator.open_path(0).read_bytes()
    asyncio.run(generator.record_results(results, generation=0))
    # A crash after the parents were saved and before the open generation was removed: a restart closes it again.
    generator.open_path(0).write_bytes(open_file)
    restarted = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    asyncio.run(restarted.record_results(results, generation=0))
    assert [parent["name"] for parent in parents(tmp_path)] == [NAMES[0]]
    assert not restarted.open_path(0).exists()


class HoldingAuthor(StandInAuthor):
    """The stand-in author, whose variants hold the run's Claude Code token."""

    def author(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant | None = None,
    ) -> AuthoredTask:
        if variant is not None:
            raise CredentialHeld("the agent's task held the run's token")
        return super().author(source, work_dir, checks, tags=tags, stop=stop, variant=variant)


@pytest.mark.parametrize(
    ("failing", "is_varied"),
    [(StandInAuthor(unreachable=("session-1",)), False), (HoldingAuthor(), True)],
    ids=["outage", "credential"],
)
def test_an_outage_leaves_the_parent_its_slot_and_a_held_token_takes_it(
    tmp_path: Path, failing: StandInAuthor, is_varied: bool
) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    failed = autoenvscaling(tmp_path, failing, count=2, is_adaptive=True, variant_share=0.5)
    assert asyncio.run(failed.generate(REQUEST)) == ()
    assert parents(tmp_path)[0]["varied"] is is_varied
    author = StandInAuthor()
    again = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(again.generate(REQUEST))
    varied = [call["variant"].parent for call in author.calls if call["variant"] is not None]
    assert varied == ([] if is_varied else [NAMES[0]]), "the next generation varies a parent the outage left"


def test_a_restart_keeps_the_parents_and_a_parent_is_varied_once(tmp_path: Path) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    # The parent was saved by the first generator; the restarted one reads it and varies it.
    restarted = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    # The variant, the first task of generation 1, falls below the band and becomes a parent too.
    asyncio.run(restarted.record_results(run_tasks(restarted, (0.0, 0.0), (0.5, 0.5)), generation=0))
    variant_name = "harbor-00001-000-autoenvscaling"
    assert [parent["name"] for parent in parents(tmp_path)] == [NAMES[0], variant_name]
    again = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(again.generate(REQUEST))
    varied = [call["variant"].parent for call in author.calls if call["variant"] is not None]
    assert varied == [NAMES[0], variant_name], "each parent is varied once, after a restart too"


def test_a_generation_varies_the_newest_unvaried_parents_that_still_have_a_session_and_files(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=4, is_adaptive=True, variant_share=0.5)
    generator.parents = [ParentTask(f"t-{number}", "session-1", number, 1.0, 2, "harder") for number in range(6)]
    generator.parents[4] = replace(generator.parents[4], varied=True)
    generator.parents[3] = replace(generator.parents[3], source_record_id="session-9")
    for parent in generator.parents[:5]:
        (tmp_path / "state" / "parents" / parent.name).mkdir(parents=True)
    # t-5 has no files, t-4 was varied and t-3's session left the sources: two slots go to t-2 and t-1.
    assert [parent.name for parent, _ in generator.variants()] == ["t-2", "t-1"]


@pytest.mark.parametrize(("count", "share", "slots"), [(2, 0.5, 1), (3, 0.5, 1), (100, 0.29, 29), (4, 1, 4)])
def test_variant_slots_are_the_share_of_count_rounded_down(
    tmp_path: Path, count: int, share: float, slots: int
) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=count, is_adaptive=True, variant_share=share)
    assert generator.variant_slots == slots


def test_a_task_whose_mean_reward_is_outside_zero_to_one_becomes_no_parent_and_the_generation_closes(
    tmp_path: Path,
) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    asyncio.run(generator.record_results(run_tasks(generator, (2.0, 2.0), (-1.0, -1.0)), generation=0))
    assert parents(tmp_path) == []
    assert not (tmp_path / "state" / "parents").exists()
    assert report(tmp_path)["tasks"] and not generator.open_path(0).exists()


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
        ({"adaptive": True, "variant_share": True}, r"variant_share must be in \(0, 1\]"),
        ({"adaptive": True, "variant_share": "0.5"}, r"variant_share must be in \(0, 1\]"),
        ({"adaptive": True, "count": 1}, "variant_share 0.5 of count 1 gives no variant slot"),
        ({"variant_share": 0.5}, "variant_share needs adaptive: true"),
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
    inputs: set[str] = set()

    def run(
        self: AgentAuthor, work_dir: Path, gateway: object, tools: object, calls: object, stop: object, **kw: object
    ) -> None:
        seen.update(kw)
        inputs.update(str(path.relative_to(work_dir)) for path in (work_dir / "inputs").rglob("*"))
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
    variant = TaskVariant(parent="t-0", parent_dir=parent_dir, direction="simpler", pass_rate=0.125, rollouts=8)
    source = SessionSource("session-1", "s", Repository("https://example.com/r.git", "abc1234", "/app"))
    agent.vary(source, tmp_path / "run", NoChecks(1), tags={}, stop=threading.Event(), variant=variant)
    assert seen["prompt"] == author_module.VARIANT_PROMPT.format(direction="simpler")
    assert {"inputs/parent/instruction.md", "inputs/parent/tests/test.sh", "inputs/variant.md"} <= inputs
    assert seen["variant.md"] == (
        "The solver's mean reward on the task in ../inputs/parent/ (t-0) is 0.12 over 8 rollouts; a rollout that "
        "passes scores 1.\n"
        "Write a simpler task: the solver passes the parent too rarely to learn from it.\n"
    )


def test_a_variant_run_gives_the_agent_the_variant_rules_and_prompt(tmp_path: Path) -> None:
    variant = TaskVariant(parent="t-0", parent_dir=tmp_path, direction="harder", pass_rate=1.0, rollouts=2)
    sessions = {"variant": ScriptedSession([]), "rebuild": ScriptedSession([])}
    with_variant = agent_files(
        tmp_path / "variant", has_behaviour_rule=True, checks=NoChecks(1), session=sessions["variant"], variant=variant
    )
    without = agent_files(
        tmp_path / "rebuild", has_behaviour_rule=True, checks=NoChecks(1), session=sessions["rebuild"]
    )
    assert "THIS RUN WRITES A VARIANT" in with_variant and "../inputs/parent/" in with_variant
    assert "THIS RUN WRITES A VARIANT" not in without and "../inputs/parent/" not in without
    # pi takes its prompt as the argument after -p.
    prompts = {name: session.argv[session.argv.index("-p") + 1] for name, session in sessions.items()}
    assert prompts == {
        "variant": author_module.VARIANT_PROMPT.format(direction="harder"),
        "rebuild": author_module.AGENT_PROMPT,
    }


@pytest.mark.parametrize(
    ("parent", "direction", "pass_rate", "rollouts"),
    [("t-0", "sideways", 1.0, 2), ("t-0", "harder", 1.5, 2), ("t-0", "harder", 1.0, 0), ("", "harder", 1.0, 2)],
)
def test_a_variant_refuses_fields_outside_its_definition(
    tmp_path: Path, parent: str, direction: str, pass_rate: float, rollouts: int
) -> None:
    with pytest.raises(ValueError, match="a variant's"):
        TaskVariant(parent=parent, parent_dir=tmp_path, direction=direction, pass_rate=pass_rate, rollouts=rollouts)


@pytest.mark.parametrize("name", ["", "../escape", "Upper", "a/b"])
def test_a_parent_refuses_a_name_that_is_not_a_task_name(name: str) -> None:
    with pytest.raises(ValueError, match="a parent's name"):
        ParentTask(name, "session-1", 0, 1.0, 2, "harder")


def test_adaptive_requires_an_author_that_implements_variants(tmp_path: Path) -> None:
    designer = author_module.PromptAuthor(
        reef_url="http://127.0.0.1:9", token=None, scenario="autoenvscaling", model="designer", api="openai"
    )
    with pytest.raises(ValueError, match="adaptive generation needs a VariantAuthor"):
        autoenvscaling(tmp_path, designer, count=2, is_adaptive=True)


class BlockingVariantAuthor(VariantAuthor):
    """An author whose run lasts until it is stopped; it keeps the parent of each variant it was asked for."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.varied: list[str] = []

    def vary(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant,
    ) -> AuthoredTask:
        return self.author(source, work_dir, checks, tags=tags, stop=stop, variant=variant)

    def author(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant | None = None,
    ) -> AuthoredTask:
        if variant is not None:
            self.varied.append(variant.parent)
        self.started.set()
        if stop.wait(10):
            self.stopped.set()
        raise AuthorError("the agent's run was stopped")


def test_a_cancelled_variant_leaves_its_parent_its_slot(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True, variant_share=0.5)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    author = BlockingVariantAuthor()
    blocked = autoenvscaling(tmp_path, author, count=2, is_adaptive=True, variant_share=0.5)

    async def cancelled() -> None:
        generation = asyncio.ensure_future(blocked.generate(REQUEST))
        await asyncio.to_thread(author.started.wait, 10)
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation
        await asyncio.to_thread(author.stopped.wait, 10)

    asyncio.run(cancelled())
    assert author.varied == [NAMES[0]], "the cancelled run was the parent's variant"
    assert parents(tmp_path)[0]["varied"] is False, "a cancelled run takes no slot, as it takes no session"


def test_with_a_refusing_calibration_a_task_outside_the_band_is_refused_and_becomes_no_parent(
    tmp_path: Path,
) -> None:
    service = ScriptedRollouts(tmp_path / "tasks", [1.0] * 8)
    calibration = CalibrationCheck(
        service,
        scenario="autoenvscaling",
        rollouts=4,
        max_rollouts=8,
        margin=0.15,
        min_std=0.1,
        band=PassRateBand(0.25, 0.75),
    )
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2, is_adaptive=True, calibration=calibration)
    results = run_tasks(generator, (1.0, 1.0))
    assert len(results) == 2 and all(isinstance(result, RejectedTask) for result in results)
    asyncio.run(generator.record_results(results, generation=0))
    assert [proposal["stage"] for proposal in report(tmp_path)["proposals"]] == ["calibration", "calibration"]
    assert parents(tmp_path) == [], "only a task that trained can become a parent"


@pytest.mark.parametrize(
    "changes",
    [
        {"position": -1},
        {"parent_task": "parent"},
        {"direction": "harder"},
        {"position": -1, "parent_task": "parent", "direction": "sideways"},
        {"position": -2},
        {"source_record_id": ""},
    ],
)
def test_proposal_rejects_inconsistent_variant_fields(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        generator_module.SessionProposal(**{"index": 0, "position": 0, "source_record_id": "session", **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"source_record_id": ""},
        {"generation": -1},
        {"generation": True},
        {"rollouts": 1.5},
        {"pass_rate": True},
        {"pass_rate": float("nan")},
        {"direction": "sideways"},
        {"varied": "yes"},
    ],
)
def test_parent_rejects_invalid_metadata(changes: dict[str, object]) -> None:
    fields = asdict(ParentTask("parent", "session", 0, 1.0, 2, "harder"))
    with pytest.raises(ValueError):
        ParentTask(**{**fields, **changes})


@pytest.mark.parametrize("duplicate", [True, False])
def test_restart_rejects_duplicate_or_oversized_parent_files(tmp_path: Path, duplicate: bool) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    entry = asdict(ParentTask("parent", "session", 0, 1.0, 2, "harder"))
    entries = (
        [entry, entry]
        if duplicate
        else [{**entry, "name": f"parent-{index}"} for index in range(generator_module.MAX_PARENTS + 1)]
    )
    generator.state_dir.mkdir(parents=True, exist_ok=True)
    (generator.state_dir / "adaptive.json").write_text(json.dumps(entries))
    with pytest.raises(ValueError, match="parents with unique names"):
        autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)


def test_restart_recovers_a_parent_written_before_its_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    results = run_tasks(generator, (1.0, 1.0), (0.5, 0.5))

    def interrupted_save() -> None:
        raise OSError("interrupted before metadata was saved")

    monkeypatch.setattr(generator, "save_parents", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        asyncio.run(generator.record_results(results, generation=0))
    parent_dir = generator.state_dir / "parents" / NAMES[0]
    assert parent_dir.is_dir() and not (generator.state_dir / "adaptive.json").exists()
    restarted = autoenvscaling(tmp_path, StandInAuthor(), count=2, is_adaptive=True)
    asyncio.run(restarted.record_results(results, generation=0))
    assert [parent["name"] for parent in parents(tmp_path)] == [NAMES[0]]
    assert parent_dir.is_dir() and not restarted.open_path(0).exists()
