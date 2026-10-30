"""Seed tasks in adaptive AutoEnvScaling: their slots, checks and split, and the variants of a seeded parent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    NAMES,
    REQUEST,
    NoChecks,
    ScriptedExecutor,
    ScriptedRollouts,
    ScriptedSession,
    StandInAuthor,
    cursor,
    generating_config,
    report,
    run_tasks,
    serving_reef,
    sources_file,
    stand_in_e2b_package,
    write_workspace_task,
)
from tests.recipes.autoenvscaling.test_autoenvscaling_claude_author import TOKEN, ClaudeExecutor, ScriptedClaudeSession
from tests.support.task_service_stubs import DOCUMENT, StandInGenerator

from recipes.beta.autoenvscaling import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    AutoEnvScalingGenerator,
    CalibrationCheck,
    CandidateChecks,
    ClaudeCodeAuthor,
    ContaminationCheck,
    ReviewCheck,
    SeedVariantAuthor,
    SessionSource,
    TaskAuthor,
    TaskVariant,
    admitted_task,
)
from recipes.beta.autoenvscaling import author as author_module
from recipes.beta.autoenvscaling import claude_author as claude_author_module
from recipes.beta.autoenvscaling import load_sources, workspace_reply
from recipes.beta.autoenvscaling.author import SEED_VARIANT_PROMPT, VARIANT_INPUTS_TEXT, agent_prompt, agent_rules
from recipes.beta.autoenvscaling.checks import PassRateBand
from recipes.beta.autoenvscaling.claude_author import CLAUDE_CHECK_TEXT
from recipes.beta.autoenvscaling.generator import (
    GenerationCursor,
    ParentTask,
    SessionProposal,
    seed_record_id,
    seed_tasks,
)
from recipes.beta.autoenvscaling.sources import Repository
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, read_harbor_task, write_harbor_task
from reef.core.tasks.split import TaskSplit, extend_split_by_source
from reef.execution.sandbox.process import ProcessOutcome
from reef.recipe.generated_tasks.generator import HeldOutTask, RejectedTask, RolledOutTask
from reef.recipe.generated_tasks.processor import split_source_ids
from reef.record2sample.generator import OracleResult

# The session rules and prompts, rendered before seed tasks existed: a seeded run must not change them.
SESSION_RULES_SHA256 = "98840322e47c9f267ef5fe41b23439225ed2565f58b58e46979881f20aed4047"
SESSION_PROMPTS_SHA256 = "e52fc63542fedbb4958efd3f87824252ee549820f6364887549c7806e205ee2d"
HISTORY_MARK = "keeps its history up to the checked-out commit"
# A seed's source record id: "seed-" and 16 hex digits of its digest.
SEED_ID = "seed-0123456789abcdef"


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


def seed_task(number: int) -> HarborTask:
    """A seed as the TMax converter writes one: its own image and fixture, and a solution that does nothing."""
    document = json.loads(json.dumps(DOCUMENT).replace("8471", str(9100 + number)))
    return HarborTask(
        name=f"tmax-{number}",
        instruction=document["instruction"],
        tests=document["tests"],
        environment={**document["environment"], "fixtures/data.txt": f"seed {number}\n"},
        solution={"solve.sh": "#!/bin/bash\nexit 0\n"},
        metadata={"tmax_task": f"TMax_{number}"},
    )


def write_seeds(root: Path, count: int) -> Path:
    for number in range(count):
        write_harbor_task(seed_task(number), root)
    return root


class SeedAuthor(SeedVariantAuthor):
    """Writes a task as the agent would, from a session or, without one, from the parent alone; it checks the task
    once through the generator's checks and returns two receipts. ``sources``, ``variants`` and ``checks`` keep
    what each run got and what its check answered."""

    def __init__(self) -> None:
        self.sources: list[str | None] = []
        self.variants: list[TaskVariant | None] = []
        self.checks: list[dict[str, object]] = []

    def author(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
    ) -> AuthoredTask:
        return self.write(source, work_dir, checks, tags=tags, variant=None)

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
        return self.write(source, work_dir, checks, tags=tags, variant=variant)

    def vary_seed(
        self,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant,
    ) -> AuthoredTask:
        return self.write(None, work_dir, checks, tags=tags, variant=variant)

    def write(
        self,
        source: SessionSource | None,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        variant: TaskVariant | None,
    ) -> AuthoredTask:
        self.sources.append(None if source is None else source.record_id)
        self.variants.append(variant)
        port = 8000 + 100 * int(tags["generation"]) + int(tags["index"])
        task_dir = write_workspace_task(work_dir / "workspace" / "task", port)
        self.checks.append(checks.check(task_dir))
        return AuthoredTask(workspace_reply(task_dir), ("agent-1", "agent-2"))


class ScriptedOracle(StandInGenerator):
    """A generator service whose oracle and nop check answers ``result``."""

    def __init__(self, root: Path, result: OracleResult) -> None:
        super().__init__(root)
        self.result = result

    async def check(self, task_path: Path) -> OracleResult:
        self.checks.append(task_path)
        return self.result


def seeded(
    tmp_path: Path,
    author: TaskAuthor,
    service: StandInGenerator | None = None,
    *,
    seed_count: int = 2,
    source_count: int = 0,
    count: int = 2,
    task_checks: Sequence[ContaminationCheck | ReviewCheck] = (),
    calibration: CalibrationCheck | None = None,
) -> AutoEnvScalingGenerator:
    """An adaptive generator with ``seed_count`` seeds and ``source_count`` sessions, half its slots for variants."""
    return AutoEnvScalingGenerator(
        scenario="autoenvscaling",
        service=service if service is not None else StandInGenerator(tmp_path / "tasks"),
        author=author,
        sources=load_sources(sources_file(tmp_path, source_count)) if source_count else (),
        state_dir=tmp_path / "state",
        count=count,
        task_checks=tuple(task_checks),
        calibration=calibration,
        is_adaptive=True,
        seeds=seed_tasks(write_seeds(tmp_path / "seeds", seed_count)),
    )


def parents(tmp_path: Path) -> list[dict[str, object]]:
    kept: list[dict[str, object]] = json.loads((tmp_path / "state" / "adaptive.json").read_text())
    return kept


def test_seeds_fill_the_slots_after_variants_and_before_sessions_once_each(tmp_path: Path) -> None:
    author = SeedAuthor()
    generator = seeded(tmp_path, author, source_count=2, count=3)
    seeds = generator.seeds
    first = run_tasks(generator, (1.0, 1.0), (0.5, 0.5), (0.5, 0.5))
    # The seeds take their slots as they are; only the session's slot runs the author.
    assert author.sources == ["session-1"]
    assert [result.task.metadata.get("seed_task") for result in first] == [seeds[0].name, seeds[1].name, None]
    assert first[0].task.name == NAMES[0] and first[0].task.instruction == seeds[0].instruction
    assert first[0].task.source_agent_record_ids == (seed_record_id(seeds[0]),)
    assert seed_record_id(seeds[0]) == f"seed-{seeds[0].digest[:16]}"
    asyncio.run(generator.record_results(first, generation=0))
    document = report(tmp_path)
    proposal = document["proposals"][0]
    assert (proposal["position"], proposal["seed_task"], proposal["source_record_id"]) == (
        -1,
        seeds[0].name,
        seed_record_id(seeds[0]),
    )
    assert "seed_task" not in document["proposals"][2] and document["proposals"][2]["position"] == 0
    assert document["next_seed"] == 2
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 1, "next_seed": 2}

    # The seed above the band is varied first; no seed is taken again, so the sessions fill the rest.
    second = asyncio.run(generator.generate(REQUEST))
    assert author.sources == ["session-1", None, "session-2", "session-1"]
    variant = author.variants[1]
    assert variant is not None and (variant.parent, variant.seed_task) == (NAMES[0], seeds[0].name)
    assert [task.metadata.get("seed_task") for task in second] == [seeds[0].name, None, None]


@pytest.mark.parametrize(
    "result",
    [
        OracleResult(is_solvable=False, reason="the reference solution scored 0.25, not 1", oracle_reward=0.25),
        OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0),
    ],
    ids=("no-op-solution-fails", "solution-passes"),
)
def test_a_seed_is_admitted_by_its_build_and_its_no_op_solution_failing(tmp_path: Path, result: OracleResult) -> None:
    service = ScriptedOracle(tmp_path / "tasks", result)
    generator = seeded(tmp_path, SeedAuthor(), service, seed_count=1)
    [task] = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(task)).is_valid
    asyncio.run(generator.record_results((RolledOutTask(task, (0.5, 0.5)),), generation=0))
    [proposal] = report(tmp_path)["proposals"]
    assert proposal["checks"] == [
        {
            "check": "seed",
            "errors": [],
            "could_not_run": "",
            "details": {"oracle_reward": result.oracle_reward, "nop_reward": result.nop_reward},
        }
    ]
    assert proposal["stage"] == "" and len(service.checks) == 1


@pytest.mark.parametrize(
    ("result", "error"),
    [
        (
            OracleResult(False, "the reference solution scored None, not 1; the trial ended with BuildError"),
            "the environment did not build or its verifier wrote no reward",
        ),
        (
            OracleResult(False, "doing nothing scored 1.0, not below 1", oracle_reward=1.0, nop_reward=1.0),
            "doing nothing passes the seed",
        ),
    ],
    ids=("no-build", "nop-passes"),
)
def test_a_seed_whose_environment_does_not_build_is_refused(tmp_path: Path, result: OracleResult, error: str) -> None:
    generator = seeded(tmp_path, SeedAuthor(), ScriptedOracle(tmp_path / "tasks", result), seed_count=1)
    [task] = asyncio.run(generator.generate(REQUEST))
    validation = asyncio.run(generator.validate(task))
    assert not validation.is_valid and error in validation.errors[0]
    asyncio.run(generator.record_results((RejectedTask(task, validation.errors[0]),), generation=0))
    [proposal] = report(tmp_path)["proposals"]
    assert proposal["stage"] == "check" and proposal["checks"][0]["details"]["oracle_reward"] == result.oracle_reward


def test_a_seed_skips_the_review_and_the_calibration(tmp_path: Path) -> None:
    server, review_calls = serving_reef('{"findings": []}')
    try:
        reef_url = f"http://127.0.0.1:{server.server_address[1]}"
        review = ReviewCheck(reef_url=reef_url, token=None, scenario="autoenvscaling", model="reviewer", api="openai")
        # No rollout outcome is scripted: a calibration rollout would fail the test.
        service = ScriptedRollouts(tmp_path / "tasks", [])
        calibration = CalibrationCheck(
            service,
            scenario="autoenvscaling",
            rollouts=2,
            max_rollouts=2,
            margin=0.1,
            min_std=0.0,
            band=PassRateBand(0.25, 0.75),
        )
        contamination = ContaminationCheck(
            {"reference": "a reference task about something else"}, n=5, tasks_path="refs"
        )
        generator = seeded(
            tmp_path, SeedAuthor(), service, seed_count=1, task_checks=(contamination, review), calibration=calibration
        )
        [task] = asyncio.run(generator.generate(REQUEST))
        assert asyncio.run(generator.validate(task)).is_valid
    finally:
        server.shutdown()
        server.server_close()
    assert review_calls == [] and service.rollout_requests == []
    asyncio.run(generator.record_results((RolledOutTask(task, (0.5, 0.5)),), generation=0))
    assert [check["check"] for check in report(tmp_path)["proposals"][0]["checks"]] == ["seed", "contamination"]


def test_a_seed_outside_the_band_becomes_a_seeded_parent(tmp_path: Path) -> None:
    author = SeedAuthor()
    generator = seeded(tmp_path, author)
    seeds = generator.seeds
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    [parent] = parents(tmp_path)
    assert (parent["name"], parent["seed_task"], parent["source_record_id"], parent["direction"]) == (
        NAMES[0],
        seeds[0].name,
        seed_record_id(seeds[0]),
        "harder",
    )

    # The variant has no session; it falls below the band, so it becomes a parent of the same seed.
    second = run_tasks(generator, (0.0, 0.0))
    assert author.sources == [None]
    [variant] = author.variants
    assert variant is not None and (variant.parent, variant.seed_task) == (NAMES[0], seeds[0].name)
    [result] = second
    assert (result.task.metadata["parent_task"], result.task.metadata["seed_task"]) == (NAMES[0], seeds[0].name)
    assert result.task.source_agent_record_ids == ("agent-2", seed_record_id(seeds[0]))
    asyncio.run(generator.record_results(second, generation=0))
    [proposal] = report(tmp_path, 1)["proposals"]
    assert (proposal["position"], proposal["parent_task"], proposal["seed_task"], proposal["source_record_id"]) == (
        -1,
        NAMES[0],
        seeds[0].name,
        seed_record_id(seeds[0]),
    )
    assert [kept["seed_task"] for kept in parents(tmp_path)] == [seeds[0].name, seeds[0].name]


def test_the_split_puts_a_seed_variant_with_its_seed(tmp_path: Path) -> None:
    generator = seeded(tmp_path, SeedAuthor())
    first = run_tasks(generator, (1.0, 1.0), (0.5, 0.5))
    asyncio.run(generator.record_results(first, generation=0))
    [variant] = asyncio.run(generator.generate(REQUEST))
    seed, other = first[0].task, first[1].task
    # Each seed is a split group of its own, and its variant joins it.
    assert split_source_ids(seed) == (seed_record_id(generator.seeds[0]),)
    assert split_source_ids(variant) == split_source_ids(seed) != split_source_ids(other)
    for split in ("train", "eval"):
        previous = TaskSplit(
            train=(seed.name,) if split == "train" else (other.name,),
            eval=(seed.name,) if split == "eval" else (other.name,),
            seed=0,
            eval_fraction=0.5,
        )
        assigned = extend_split_by_source(
            previous=previous,
            previous_sources={task.name: split_source_ids(task) for task in (seed, other)},
            sources={variant.name: split_source_ids(variant)},
        )
        assert variant.name in (assigned.train if split == "train" else assigned.eval)


def test_a_seed_from_an_earlier_runs_session_task_keeps_its_session_and_is_a_seed_here(tmp_path: Path) -> None:
    # A task an earlier run varied from session-1, used as a seed; this run's sources hold session-1 too.
    earlier = replace(
        seed_task(0),
        source_agent_record_ids=("designer-x0", "session-1"),
        metadata={
            "designer_record_id": "designer-x0",
            "parent_task": "harbor-00000-000-autoenvscaling",
            "variant": "harder",
        },
    )
    write_harbor_task(earlier, tmp_path / "earlier-run")
    # Its reference solution scores 0.25: the seed check admits it, the oracle check would not.
    service = ScriptedOracle(tmp_path / "tasks", OracleResult(False, "the reference solution scored 0.25", 0.25))
    generator = seeded(tmp_path, SeedAuthor(), service, source_count=1)
    generator.seeds = seed_tasks(tmp_path / "earlier-run")
    seed, session_task = asyncio.run(generator.generate(REQUEST))
    seed_id = seed_record_id(generator.seeds[0])
    assert seed.source_agent_record_ids == ("designer-x0", "session-1", seed_id)
    assert split_source_ids(seed) == ("session-1", seed_id) and split_source_ids(session_task) == ("session-1",)
    previous = TaskSplit(train=(), eval=(seed.name,), seed=0, eval_fraction=0.5)
    assigned = extend_split_by_source(
        previous=previous,
        previous_sources={seed.name: split_source_ids(seed)},
        sources={session_task.name: split_source_ids(session_task)},
    )
    assert session_task.name in assigned.eval, "the session's rebuild joins the eval seed of the same session"
    # The earlier run's lineage names that run's tasks; here the task is a seed and runs the seed check.
    assert "parent_task" not in seed.metadata and "variant" not in seed.metadata
    assert asyncio.run(generator.validate(seed)).is_valid


@pytest.mark.parametrize("is_cancelled", [False, True], ids=("raises", "cancelled"))
def test_an_attempt_that_raises_or_is_cancelled_takes_no_seed(tmp_path: Path, is_cancelled: bool) -> None:
    class FailingSessionAuthor(SeedAuthor):
        """Its session slot raises an error the generator does not catch, or lasts until it is cancelled."""

        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()

        def author(
            self,
            source: SessionSource,
            work_dir: Path,
            checks: CandidateChecks,
            *,
            tags: Mapping[str, str],
            stop: threading.Event,
        ) -> AuthoredTask:
            self.started.set()
            if not is_cancelled:
                raise OSError("disk full while copying the snapshot")
            stop.wait(10)
            raise AuthorError("the agent's run was stopped")

    failing = FailingSessionAuthor()
    generator = seeded(tmp_path, failing, seed_count=1, source_count=1)

    async def attempt() -> None:
        generation = asyncio.ensure_future(generator.generate(REQUEST))
        if is_cancelled:
            await asyncio.to_thread(failing.started.wait, 10)
            generation.cancel()
        with pytest.raises(asyncio.CancelledError if is_cancelled else OSError):
            await generation

    asyncio.run(attempt())
    assert failing.started.is_set(), "the seed's slot came first, then the session's failed"
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 0, "next_seed": 0}
    # The recipe's next attempt takes the seed again.
    retried = asyncio.run(seeded(tmp_path, SeedAuthor(), seed_count=1, source_count=1).generate(REQUEST))
    assert [task.metadata.get("seed_task") for task in retried] == ["tmax-0", None]
    assert cursor(tmp_path) == {"next_generation": 2, "next_position": 1, "next_seed": 1}


def test_a_seed_variant_runs_the_task_checks_not_the_seed_check(tmp_path: Path) -> None:
    service = ScriptedOracle(tmp_path / "tasks", OracleResult(True, "", oracle_reward=1.0, nop_reward=0.0))
    generator = seeded(tmp_path, SeedAuthor(), service)
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    [variant] = asyncio.run(generator.generate(REQUEST))
    assert variant.metadata["seed_task"] == generator.seeds[0].name
    # The variant's reference solution scores 0.25: the seed check would admit it, the oracle check refuses it.
    service.result = OracleResult(False, "the reference solution scored 0.25, not 1", oracle_reward=0.25)
    validation = asyncio.run(generator.validate(variant))
    assert validation.errors == ("oracle check refused: the reference solution scored 0.25, not 1",)
    asyncio.run(generator.record_results((RejectedTask(variant, validation.errors[0]),), generation=0))
    [proposal] = report(tmp_path, 1)["proposals"]
    assert [check["check"] for check in proposal["checks"]] == ["oracle"] and proposal["stage"] == "check"


def test_an_eval_seed_is_held_out_and_never_varied(tmp_path: Path) -> None:
    author = SeedAuthor()
    generator = seeded(tmp_path, author)
    held_out, trained = asyncio.run(generator.generate(REQUEST))
    for task in (held_out, trained):
        assert asyncio.run(generator.validate(task)).is_valid
    # Both seeds are outside the band, but the recipe put the first in the eval split: it ran no rollout.
    results = (HeldOutTask(held_out), RolledOutTask(trained, (1.0, 1.0)))
    asyncio.run(generator.record_results(results, generation=0))
    assert report(tmp_path)["proposals"][0]["stage"] == "held_out"
    assert [parent["name"] for parent in parents(tmp_path)] == [trained.name]
    asyncio.run(generator.generate(REQUEST))
    assert [variant.parent for variant in author.variants if variant is not None] == [trained.name]


def files_under(root: Path, *folders: str) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text()
        for folder in folders
        for path in sorted((root / folder).rglob("*"))
        if path.is_file()
    }


class SnapshotSession(ScriptedSession):
    """The scripted pi sandbox; it keeps the inputs and the workspace as the agent finds them."""

    seen: dict[str, str] = {}

    def launch(self, argv: Sequence[str], *, root: Path, workspace: Path, **options: object) -> ProcessOutcome:
        self.seen = files_under(root, "inputs", "workspace")
        return super().launch(argv, root=root, workspace=workspace, **options)


class SnapshotClaudeSession(ScriptedClaudeSession):
    """The scripted Claude Code sandbox; it keeps the inputs and the workspace as the agent finds them."""

    seen: dict[str, str] = {}

    def launch(self, argv: Sequence[str], *, root: Path, workspace: Path, **options: object) -> ProcessOutcome:
        self.seen = files_under(root, "inputs", "workspace")
        return super().launch(argv, root=root, workspace=workspace, **options)


@pytest.mark.parametrize("harness", ["pi", "claude-code"])
def test_a_seed_variant_gets_only_the_parent_and_variant_md_and_the_parents_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: str
) -> None:
    def refused(*arguments: object) -> str:
        raise AssertionError("a seeded run reads no session and takes no repository snapshot")

    monkeypatch.setattr(author_module, "session_text", refused)
    monkeypatch.setattr(author_module, "repository_snapshot", refused)
    monkeypatch.setattr(claude_author_module, "repository_snapshot", refused)
    parent = read_harbor_task(write_harbor_task(seed_task(0), tmp_path / "parents"))
    variant = TaskVariant(
        parent=parent.name,
        parent_dir=tmp_path / "parents" / parent.name,
        direction="simpler",
        pass_rate=0.0,
        rollouts=2,
        seed_task=parent.name,
    )
    session: SnapshotSession | SnapshotClaudeSession
    author: AgentAuthor | ClaudeCodeAuthor
    if harness == "pi":
        session = SnapshotSession([])
        author = AgentAuthor(
            reef_url="http://127.0.0.1:9",
            token=None,
            scenario="autoenvscaling",
            model="designer",
            api="openai",
            executor=ScriptedExecutor(session=session),
            cache_dir=tmp_path / "cache",
            timeout_seconds=60.0,
            max_model_calls=5,
        )
    else:
        session = SnapshotClaudeSession()
        author = ClaudeCodeAuthor(
            reef_url="http://127.0.0.1:9",
            token=None,
            claude_token=TOKEN,
            model="opus",
            executor=ClaudeExecutor(session=session),
            cache_dir=tmp_path / "cache",
            timeout_seconds=60.0,
            max_turns=40,
        )
    run = tmp_path / "run"
    checks = NoChecks(1)
    author.vary_seed(run, checks, tags={}, stop=threading.Event(), variant=variant)

    assert {name for name in session.seen if name.startswith("inputs/")} == {
        "inputs/variant.md",
        *(f"inputs/parent/{name}" for name in parent.files),
    }
    assert session.seen["inputs/variant.md"].startswith(
        "The solver's mean reward on the task in ../inputs/parent/ (tmax-0) is 0.00 over 2 rollouts;"
    )
    assert {name: text for name, text in session.seen.items() if name.startswith("workspace/")} == {
        f"workspace/task/environment/{name}": text for name, text in parent.environment.items()
    }
    rules = agent_rules(checks, True, "", True)
    assert "session" not in rules.lower() and "../inputs/repo" not in rules and "history" not in rules
    written = "\n".join(text for name, text in files_under(run, ".").items() if not name.startswith("inputs/"))
    assert rules in written
    if harness == "claude-code":
        assert CLAUDE_CHECK_TEXT in (run / "claude" / "CLAUDE.md").read_text()
    assert session.argv[session.argv.index("-p") + 1] == SEED_VARIANT_PROMPT.format(direction="simpler")
    with pytest.raises(ValueError, match="needs a variant of a seeded parent"):
        author.vary_seed(
            tmp_path / "other", checks, tags={}, stop=threading.Event(), variant=replace(variant, seed_task="")
        )
    source = SessionSource("session-1", "s", Repository("https://example.com/r.git", "abc1234", "/app"))
    with pytest.raises(ValueError, match="a variant of a seeded parent takes no session"):
        author.vary(source, tmp_path / "third", checks, tags={}, stop=threading.Event(), variant=variant)


@pytest.mark.parametrize("is_reviewing", [True, False])
@pytest.mark.parametrize("has_behaviour_rule", [True, False])
def test_the_seed_rules_name_no_session_with_the_review_on_or_off(
    is_reviewing: bool, has_behaviour_rule: bool
) -> None:
    rules = agent_rules(NoChecks(6, is_reviewing=is_reviewing), has_behaviour_rule, "", True)
    assert "session" not in rules.lower() and "../inputs/repo" not in rules and "history" not in rules
    assert ("the check also reviews the tests" in rules) is is_reviewing
    assert "write a real one" in rules and "at most 6 checks" in rules


def test_a_seed_variant_task_has_no_history_step(tmp_path: Path) -> None:
    author = SeedAuthor()
    generator = seeded(tmp_path, author, source_count=1)
    parent = read_harbor_task(write_harbor_task(seed_task(0), tmp_path / "parents"))
    variant = TaskVariant(
        parent=parent.name,
        parent_dir=tmp_path / "parents" / parent.name,
        direction="harder",
        pass_rate=1.0,
        rollouts=2,
        seed_task=parent.name,
    )
    seed_id = seed_record_id(parent)
    seeded_variant = asyncio.run(generator.rebuilt(None, 0, variant=variant, source_record_id=seed_id))
    assert HISTORY_MARK not in seeded_variant.environment["Dockerfile"]
    assert author.checks[0]["passed"] is True, "the agent's own check admits the task without the step"
    session_task = asyncio.run(generator.rebuilt(generator.sources[0], 1))
    assert HISTORY_MARK in session_task.environment["Dockerfile"]
    with pytest.raises(ValueError, match="variant of a seeded parent"):
        asyncio.run(generator.rebuilt(None, 2, variant=variant))
    reply = workspace_reply(write_workspace_task(tmp_path / "task", 8800))
    task = admitted_task(reply, generation=0, index=0, record_id=seed_id, checkout_path=None)
    assert task.environment["Dockerfile"] == (tmp_path / "task" / "environment" / "Dockerfile").read_text()


def test_the_session_rules_and_prompts_are_unchanged(tmp_path: Path) -> None:
    variant = TaskVariant(parent="t-0", parent_dir=tmp_path, direction="harder", pass_rate=1.0, rollouts=2)
    rules = [
        agent_rules(
            NoChecks(6, is_reviewing=is_reviewing),
            has_behaviour_rule,
            VARIANT_INPUTS_TEXT if has_variant else "",
            False,
        )
        for has_behaviour_rule in (True, False)
        for is_reviewing in (True, False)
        for has_variant in (False, True)
    ]
    prompts = [
        agent_prompt(None, False),
        agent_prompt(variant, False),
        agent_prompt(replace(variant, direction="simpler"), False),
    ]
    assert hashlib.sha256("\0".join(rules).encode()).hexdigest() == SESSION_RULES_SHA256
    assert hashlib.sha256("\0".join(prompts).encode()).hexdigest() == SESSION_PROMPTS_SHA256


def test_a_seeds_only_generation_ends_when_seeds_and_parents_run_out(tmp_path: Path) -> None:
    author = SeedAuthor()
    generator = seeded(tmp_path, author, count=4)
    first = run_tasks(generator, (1.0, 1.0), (0.5, 0.5))
    assert len(first) == 2, "two seeds and no session: the generation ends after the seeds"
    asyncio.run(generator.record_results(first, generation=0))
    second = run_tasks(generator, (0.5, 0.5))
    assert len(second) == 1, "one parent and no seed left"
    asyncio.run(generator.record_results(second, generation=0))
    assert asyncio.run(generator.generate(REQUEST)) == ()
    asyncio.run(generator.record_results((), generation=0))
    assert report(tmp_path, 2)["proposals"] == [] and cursor(tmp_path)["next_seed"] == 2


def test_a_restart_continues_with_the_next_seed(tmp_path: Path) -> None:
    generator = seeded(tmp_path, SeedAuthor(), seed_count=3, source_count=1)
    first = asyncio.run(generator.generate(REQUEST))
    assert [task.metadata["seed_task"] for task in first] == ["tmax-0", "tmax-1"]
    restarted = seeded(tmp_path, SeedAuthor(), seed_count=3, source_count=1)
    second = asyncio.run(restarted.generate(REQUEST))
    assert second[0].metadata["seed_task"] == "tmax-2" and "seed_task" not in second[1].metadata
    assert cursor(tmp_path) == {"next_generation": 2, "next_position": 1, "next_seed": 3}


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"seed_tasks_path": "seeds"}, "seed_tasks_path needs adaptive: true"),
        ({"adaptive": True, "seed_tasks_path": "mixed"}, r"the seed task plain cannot be read: .*metadata\.reef"),
        ({"adaptive": True, "seed_tasks_path": "missing"}, "is not a directory"),
        ({"adaptive": True, "seed_tasks_path": "empty"}, "holds no seed task"),
        ({"adaptive": True, "seed_tasks_path": 7}, "seed_tasks_path must be a directory path or null"),
        ({"adaptive": True, "sources_path": None}, "needs a state_dir and a sources_path or a seed_tasks_path"),
    ],
)
def test_from_config_refuses_seeds_without_adaptive_or_with_an_unreadable_seed(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    write_seeds(tmp_path / "seeds", 2)
    write_seeds(tmp_path / "mixed", 1)
    # A Harbor task that reef did not write: it has no [metadata.reef] table.
    (tmp_path / "mixed" / "plain" / "tests").mkdir(parents=True)
    (tmp_path / "mixed" / "plain" / "task.toml").write_text('version = "1.0"\n')
    (tmp_path / "empty").mkdir()
    if isinstance(fields.get("seed_tasks_path"), str):
        fields = {**fields, "seed_tasks_path": str(tmp_path / str(fields["seed_tasks_path"]))}
    environ = {"E2B_API_KEY": "e2b-test"}
    with pytest.raises(PluginConfigError, match=message):
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **fields), scenario="autoenvscaling", environ=environ
        )
    # Seeds alone, without a sources file, make a generator.
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, adaptive=True, sources_path=None, seed_tasks_path=str(tmp_path / "seeds")),
        scenario="autoenvscaling",
        environ=environ,
    )
    assert generator.sources == () and [seed.name for seed in generator.seeds] == ["tmax-0", "tmax-1"]


@pytest.mark.parametrize(
    "fields",
    [
        {"seed_task": "tmax-0"},
        {"seed_task": "tmax-0", "parent_task": "harbor-00000-000-autoenvscaling", "direction": "harder"},
    ],
    ids=("seed", "variant-of-a-seed"),
)
def test_a_proposal_at_position_minus_one_is_a_seed_or_a_variant(fields: dict[str, str]) -> None:
    proposal = SessionProposal(0, -1, SEED_ID, **fields)
    assert (proposal.position, proposal.seed_task) == (-1, "tmax-0")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_record_id": "session-1"}, "its source_record_id the seed's id"),
        ({"seed_task": "../tmax-0"}, "seed_task must be a task name"),
        ({"seed_task": 7}, "seed and stage fields must be text"),
        ({"seed_task": ""}, "needs a parent task and a direction, or a seed_task"),
        ({"position": 0}, "no variant or seed fields"),
        ({"direction": "harder"}, "a variant proposal needs a parent task name"),
        ({"parent_task": "harbor-00000-000-autoenvscaling"}, "a variant proposal needs direction"),
    ],
)
def test_a_seed_proposal_is_neither_a_variant_nor_a_session_proposal(changes: dict[str, object], message: str) -> None:
    fields: dict[str, object] = {"index": 0, "position": -1, "source_record_id": SEED_ID, "seed_task": "tmax-0"}
    with pytest.raises(ValueError, match=message):
        SessionProposal(**{**fields, **changes})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source_record_id", "seed_task"),
    [("session-1", "tmax-0"), (SEED_ID, "Upper"), (SEED_ID, "../tmax-0"), (SEED_ID, 7)],
)
def test_a_seeded_parent_names_a_seed_task_and_has_the_seeds_id(source_record_id: str, seed_task: object) -> None:
    with pytest.raises(ValueError, match="a parent's seed_task"):
        ParentTask("t-0", source_record_id, 0, 1.0, 2, "harder", seed_task=seed_task)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed_task", ["Upper", "../tmax-0", 7])
def test_a_variants_seed_task_is_empty_or_a_task_name(seed_task: object) -> None:
    with pytest.raises(ValueError, match="a variant's seed_task"):
        TaskVariant(
            parent="t-0",
            parent_dir=Path("parents"),
            direction="harder",
            pass_rate=1.0,
            rollouts=2,
            seed_task=seed_task,  # type: ignore[arg-type]
        )


def test_seeds_and_seeded_parents_need_an_author_that_varies_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed tasks need a SeedVariantAuthor"):
        seeded(tmp_path, StandInAuthor())
    generator = seeded(tmp_path, SeedAuthor())
    asyncio.run(generator.record_results(run_tasks(generator, (1.0, 1.0), (0.5, 0.5)), generation=0))
    assert [parent["seed_task"] for parent in parents(tmp_path)] == [generator.seeds[0].name]
    # Without seeds, the state's seeded parents still need an author that can vary them.
    with pytest.raises(ValueError, match="seeded parents, which need a SeedVariantAuthor"):
        AutoEnvScalingGenerator(
            scenario="autoenvscaling",
            service=StandInGenerator(tmp_path / "tasks"),
            author=StandInAuthor(),
            sources=load_sources(sources_file(tmp_path, 1)),
            state_dir=tmp_path / "state",
            count=2,
            is_adaptive=True,
        )


@pytest.mark.parametrize(
    "fields",
    [{"next_seed": -1}, {"next_seed": "1"}, {"next_seed": True}, {"next_position": -2}, {"next_generation": 1.0}],
)
def test_a_cursor_holds_non_negative_ints(fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be a non-negative int"):
        GenerationCursor(**fields)  # type: ignore[arg-type]
