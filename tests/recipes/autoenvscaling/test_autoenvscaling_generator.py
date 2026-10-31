"""AutoEnvScaling as a task generator: the session cursor, admission, validation, the band and the reports."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    DESCRIPTION,
    NAMES,
    REQUEST,
    RefusingGenerator,
    ScriptedRollouts,
    StandInAuthor,
    autoenvscaling,
    cursor,
    fenced,
    generating_config,
    report,
    run_tasks,
    serving_reef,
    sources_file,
    stand_in_e2b_package,
    write_workspace_task,
)
from tests.support.task_service_stubs import DOCUMENT, StandInGenerator

from recipes.beta.autoenvscaling import (
    AgentAuthor,
    AuthoredTask,
    AuthorError,
    AutoEnvScalingGenerator,
    CalibrationCheck,
    CandidateChecks,
    GeneratorChecks,
    PromptAuthor,
    SessionSource,
    TaskAuthor,
    TaskCheck,
    admitted_task,
)
from recipes.beta.autoenvscaling import generator as generator_module
from recipes.beta.autoenvscaling import workspace_reply
from recipes.beta.autoenvscaling.author import AgentTools, TaskVariant
from recipes.beta.autoenvscaling.checks import CheckResult, PassRateBand
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, read_harbor_task
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskRecord2Record, GeneratedTaskSamples
from reef.recipe.generated_tasks.generator import FailedTask, HeldOutTask, RejectedTask, RolledOutTask
from reef.recipe.generated_tasks.processor import (
    GenerationJob,
    RoundSettings,
    TaskGroupSettings,
    complete_generation_count,
)
from reef.recipe.generated_tasks.results import TaskRolloutResult, rollout_results
from reef.record2sample.generator import DuplicateTask, GeneratorError, WrittenTask
from reef.record2sample.generator.harbor import ORACLE_TIMEOUT_S
from reef.record2sample.generator.inputs import TaskGenerationRequest
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping
from reef.storage.samples import TrainerSamples
from reef.storage.sqlite import SQLiteScenarioStorage


@pytest.fixture(autouse=True)
def e2b_package(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_e2b_package(monkeypatch)


def test_a_generation_rebuilds_the_next_sessions_and_admits_each_task(tmp_path: Path) -> None:
    author = StandInAuthor()
    service = StandInGenerator(tmp_path / "tasks")
    tasks = asyncio.run(autoenvscaling(tmp_path, author, service).generate(REQUEST))

    assert [call["source"] for call in author.calls] == ["session-1", "session-2", "session-1"]
    assert author.calls[0]["tags"] == {"role": "designer", "generation": "0", "index": "0"}
    assert author.calls[0]["work_dir"] == tmp_path / "state" / "runs" / "00000-000"
    assert author.calls[0]["check"] == {
        "check": 1,
        "ran": True,
        "passed": True,
        "stage": "oracle",
        "oracle_reward": 1.0,
        "nop_reward": 0.0,
        "reason": "",
        "checks_left": 5,
    }
    # The agent's candidates are written, checked and deleted; the tasks themselves are not written yet.
    assert [path.name for path in service.checks] == [f"{name}-check-1" for name in NAMES]
    assert service.deleted == [f"{name}-check-1" for name in NAMES]
    assert [task.name for task in tasks] == NAMES
    task = tasks[0]
    assert task.source_agent_record_ids == ("agent-2", "session-1")
    assert "RUN set -e; cd /app; if [ -d .git ]" in task.environment["Dockerfile"], "the source's history step"
    assert task.solution["hint.txt"]
    # A slow build or verifier fails the task's oracle trial, inside the service's bound on that trial; a trial past
    # the bound would stop the round instead of refusing the task.
    build_seconds = task.config["environment"]["build_timeout_sec"]
    assert build_seconds == 1200 and build_seconds + task.config["verifier"]["timeout_sec"] < ORACLE_TIMEOUT_S
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 3}


def test_a_restarted_generator_continues_with_the_next_session_and_generation(tmp_path: Path) -> None:
    author = StandInAuthor()
    first = autoenvscaling(tmp_path, author, source_count=3, count=2)
    asyncio.run(first.generate(REQUEST))
    asyncio.run(first.record_results((), generation=0))
    again = autoenvscaling(tmp_path, author, source_count=3, count=2)
    tasks = asyncio.run(again.generate(REQUEST))
    # The walk cycles: positions 2 and 3 are the third session and the first.
    assert [call["source"] for call in author.calls] == ["session-1", "session-2", "session-3", "session-1"]
    assert [task.name for task in tasks] == [
        "harbor-00001-000-autoenvscaling",
        "harbor-00001-001-autoenvscaling",
    ]
    assert cursor(tmp_path) == {"next_generation": 2, "next_position": 4}


class BlockingAuthor(TaskAuthor):
    """An author whose run lasts until it is stopped."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.work_dirs: list[Path] = []

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
        self.work_dirs.append(work_dir)
        self.started.set()
        if stop.wait(10):
            self.stopped.set()
        raise AuthorError("the agent's run was stopped")


def test_a_cancelled_generation_stops_its_agent_and_takes_no_session(tmp_path: Path) -> None:
    author = BlockingAuthor()
    generator = autoenvscaling(tmp_path, author)

    async def cancelled() -> None:
        generation = asyncio.ensure_future(generator.generate(REQUEST))
        await asyncio.to_thread(author.started.wait, 10)
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation
        await asyncio.to_thread(author.stopped.wait, 10)

    for _ in range(2):
        author.started.clear()
        author.stopped.clear()
        asyncio.run(cancelled())
        assert author.stopped.is_set(), "the cancellation reached the agent's thread"
    assert [path.name for path in author.work_dirs] == ["00000-000", "00001-000"], "a rerun is a new generation"
    assert cursor(tmp_path) == {"next_generation": 2, "next_position": 0}, "the session is taken again"


class GatheringStandIn(StandInAuthor):
    """The stand-in, which writes its task only once ``limit`` tasks are being written at once, and keeps the most
    written at once."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.lock = threading.Lock()
        self.running = 0
        self.most = 0
        self.full = threading.Event()

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
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
            if self.running == self.limit:
                self.full.set()
        try:
            self.full.wait(10)
            return super().author(source, work_dir, checks, tags=tags, stop=stop, variant=variant)
        finally:
            with self.lock:
                self.running -= 1


def test_tasks_are_written_at_once_up_to_the_concurrency_and_keep_their_order(tmp_path: Path) -> None:
    stand_in = GatheringStandIn(limit=3)
    generator = autoenvscaling(tmp_path, stand_in, source_count=5, count=5, concurrency=3)
    tasks = asyncio.run(generator.generate(REQUEST))
    assert stand_in.full.is_set() and stand_in.most == 3, "three tasks were written at once, and never more"
    assert [task.name for task in tasks] == [f"harbor-00000-{index:03d}-autoenvscaling" for index in range(5)]
    assert [task.source_agent_record_ids[1] for task in tasks] == [f"session-{number}" for number in range(1, 6)]
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 5}


class OutageStandIn(TaskAuthor):
    """The first session's task is an outage once the second session's task is being written; that one runs until
    it is stopped."""

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.second = threading.Event()
        self.stopped = threading.Event()

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
        self.sources.append(source.record_id)
        if source.record_id == "session-1":
            self.second.wait(10)
            raise AuthorError("the agent could not run: the sandbox did not start")
        self.second.set()
        if stop.wait(10):
            self.stopped.set()
        raise AuthorError("the agent's run was stopped")


def test_an_outage_starts_no_further_task_and_stops_the_running_ones_after_it(tmp_path: Path) -> None:
    stand_in = OutageStandIn()
    generator = autoenvscaling(tmp_path, stand_in, source_count=4, count=4, concurrency=2)
    with pytest.raises(AuthorError, match="the author could not rebuild session-1"):
        asyncio.run(generator.generate(REQUEST))
    assert stand_in.stopped.wait(10), "the task being written after the outage was stopped"
    assert sorted(stand_in.sources) == ["session-1", "session-2"], "no task started after the outage"
    asyncio.run(generator.record_results((), generation=0))
    document = report(tmp_path)
    assert document["error"].startswith("the author could not rebuild session-1")
    assert [(entry["position"], entry["stage"]) for entry in document["proposals"]] == [(0, "outage")]
    assert cursor(tmp_path)["next_position"] == 1, "the stopped task's session is taken again"


def test_a_task_the_authoring_checks_refuse_is_refused_and_names_the_last_call(tmp_path: Path) -> None:
    author = StandInAuthor(has_tmux=False)
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, author, service, count=1)
    assert run_tasks(generator) == ()
    check = author.calls[0]["check"]
    assert check["stage"] == "authoring" and "does not install tmux" in check["errors"]
    assert service.checks == [], "a candidate the authoring checks refuse never reaches Harbor"
    asyncio.run(generator.record_results((), generation=0))
    [proposal] = report(tmp_path)["proposals"]
    assert proposal["stage"] == "authoring" and proposal["designer_record_id"] == "agent-2"
    assert proposal["refusal"].startswith("the Designer's task is refused") and "tmux" in proposal["refusal"]


def test_sessions_without_a_task_are_recorded_by_stage_and_an_outage_ends_the_generation(tmp_path: Path) -> None:
    author = StandInAuthor(unusable=("session-1",), unreachable=("session-3",))
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, author, service, source_count=4, count=4)
    results = run_tasks(generator)
    assert [result.task.name for result in results] == ["harbor-00000-001-autoenvscaling"]
    assert [call["source"] for call in author.calls] == ["session-1", "session-2", "session-3"]
    asyncio.run(generator.record_results(results, generation=0))

    document = report(tmp_path)
    assert (
        document["error"]
        == "the author could not rebuild session-3: the agent could not run: the sandbox did not start"
    )
    assert [(entry["position"], entry["stage"]) for entry in document["proposals"]] == [
        (0, "source"),
        (1, ""),
        (2, "outage"),
    ]
    assert document["proposals"][0]["refusal"] == (
        "the source session-1 cannot be rebuilt: the record session-1 holds no session messages"
    )
    assert [entry["designer_record_id"] for entry in document["proposals"]] == ["", "agent-2", ""]
    assert document["next_position"] == 3 and cursor(tmp_path)["next_position"] == 3, "the outage's session is taken"

    silent = StandInGenerator(tmp_path / "silent")
    generator = autoenvscaling(tmp_path / "silent", StandInAuthor(receipts=()), silent, count=1)
    assert run_tasks(generator) == ()
    asyncio.run(generator.record_results((), generation=0))
    [proposal] = report(tmp_path / "silent")["proposals"]
    assert proposal["refusal"] == "the Designer made no model call that Reef recorded"
    assert proposal["stage"] == "authoring" and proposal["designer_record_id"] == ""


class RefusingCheck(TaskCheck):
    """A further task check that finds one defect and counts its runs."""

    def __init__(self) -> None:
        super().__init__("wording")
        self.runs: list[str] = []

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        self.runs.append(written.name)
        return CheckResult(self.name, ("the verifier checks the wording of the change",))


def test_validation_runs_the_checks_in_order_on_a_written_copy_and_removes_it(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item)).is_valid
    assert service.checks[-1] == tmp_path / "tasks" / NAMES[0]
    assert service.deleted[-1] == NAMES[0] and not (tmp_path / "tasks" / NAMES[0]).exists()

    further = RefusingCheck()
    generator = autoenvscaling(tmp_path / "further", StandInAuthor(), task_checks=(further,), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item)).errors == ("the verifier checks the wording of the change",)
    assert further.runs == [NAMES[0]]

    refusing = StandInGenerator(tmp_path / "refusing", is_solvable=False)
    further = RefusingCheck()
    generator = autoenvscaling(tmp_path / "refusing", StandInAuthor(), refusing, task_checks=(further,), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item)).errors == ("oracle check refused: the oracle scored 0",)
    assert further.runs == [], "a check that found a defect ends the validation"
    assert refusing.deleted[-1] == NAMES[0]


def test_a_staged_copy_of_the_same_task_passes_validation_and_stays_staged(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    staged = asyncio.run(service.write_task(item))
    assert asyncio.run(generator.validate(item)).is_valid
    assert staged.path.is_dir() and item.name not in service.deleted, "a rollout may be using the staged copy"


def test_a_duplicate_is_invalid(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    asyncio.run(service.write_task(replace(item, name="harbor-copy")))
    assert asyncio.run(generator.validate(item)).errors == ("refused: duplicate of a task already under the root",)


def test_a_check_that_cannot_run_raises_and_the_report_names_it(tmp_path: Path) -> None:
    # Two candidate checks, then the first task's check; the second task's check cannot run.
    service = StandInGenerator(tmp_path / "tasks", check_fail_after=3)
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2)
    first, second = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(first)).is_valid
    asyncio.run(service.write_task(first))
    with pytest.raises(GeneratorError, match="the harbor command line is not installed"):
        asyncio.run(generator.validate(second))
    assert not (tmp_path / "tasks" / NAMES[1]).exists(), "no copy is left"
    asyncio.run(generator.record_results((RolledOutTask(first, (0.0, 1.0)),), generation=0))
    document = report(tmp_path)
    assert document["error"].startswith(f"the check of {NAMES[1]} could not run")
    assert [entry["stage"] for entry in document["proposals"]] == ["", "unreached"]


@pytest.mark.parametrize(
    ("training", "band", "position"),
    [
        ((0.0, 1.0), {}, "inside"),
        ((0.0, 0.0, 0.0, 1.0), {}, "inside"),
        ((1.0, 1.0), {}, "above"),
        ((0.0, 0.0), {}, "below"),
        ((1.0, 1.0), {"band_low": None, "band_high": None}, "inside"),
    ],
)
def test_the_band_keeps_a_task_that_ran_only_when_its_pass_rate_is_inside_it(
    tmp_path: Path, training: tuple[float, ...], band: dict[str, float | None], position: str
) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1, **band)
    asyncio.run(generator.record_results(run_tasks(generator, training), generation=0))
    document = report(tmp_path)
    pass_rate = sum(training) / len(training)
    [task] = document["tasks"]
    assert (task["name"], task["pass_rate"], task["band"]) == (NAMES[0], pass_rate, position)
    # Inside the band or not, the task stays under the tasks root, and the report names where.
    assert task["path"] == str(tmp_path / "tasks" / NAMES[0]) and (tmp_path / "tasks" / NAMES[0]).is_dir()
    assert task["rollout_rewards"] == list(training) and task["source_record_id"] == "session-1"
    [proposal] = document["proposals"]
    if position == "inside":
        assert proposal["stage"] == ""
    else:
        assert (
            proposal["stage"] == "band"
            and proposal["refusal"] == f"outside the solver band: pass rate {pass_rate:.2f}"
        )
        assert proposal["task_name"] == NAMES[0]
        assert service.deleted == [f"{NAMES[0]}-check-1", NAMES[0]], "the candidate and the validated copy only"


@pytest.mark.unit
def test_eval_tasks_are_held_out_without_a_band_measurement_or_failure(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (generated,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(generated)).is_valid
    asyncio.run(generator.record_results((HeldOutTask(generated),), generation=0))
    document = report(tmp_path)
    assert document["tasks"] == []
    assert document["proposals"][0]["stage"] == "held_out"
    assert document["proposals"][0]["refusal"] == ""
    assert service.reports == []


def test_the_results_name_where_each_refused_task_stopped(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service)
    first, second, third = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(first)).is_valid
    assert asyncio.run(generator.validate(third)).is_valid
    asyncio.run(service.write_task(first))
    service.is_solvable = False
    errors = asyncio.run(generator.validate(second)).errors
    results = (
        RolledOutTask(first, (0.0, 1.0)),
        RejectedTask(second, "; ".join(errors)),
        FailedTask(third, "the agent could not run the task: no sandbox", False),
    )
    asyncio.run(generator.record_results(results, generation=0))

    document = report(tmp_path)
    assert [(entry["stage"], entry["refusal"]) for entry in document["proposals"]] == [
        ("", ""),
        ("check", "oracle check refused: the oracle scored 0"),
        ("run", "the agent could not run the task: no sandbox"),
    ]
    assert service.reports == [], "the generator sends no Designer report"


def test_a_request_with_source_records_is_refused_and_named_in_the_report(tmp_path: Path) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author)
    record = AgentRecord.create(scenario="autoenvscaling", request_type=RequestType.INFERENCE, payload={})
    with pytest.raises(ValueError, match="takes its sessions from its sources file"):
        asyncio.run(generator.generate(TaskGenerationRequest((record,), DESCRIPTION)))
    asyncio.run(generator.record_results((), generation=0))
    assert author.calls == [] and "not from source records" in report(tmp_path)["error"]
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 0}


def polled(tools: AgentTools) -> dict[str, Any]:
    """What the agent's task_check does: start a check with /check, then poll /check with its id until it is done."""
    check_id = tools.call({})["check_id"]
    while (result := tools.call({"check_id": check_id}))["state"] == "running":
        time.sleep(0.01)
    return result


def test_the_check_tool_writes_checks_and_deletes_a_candidate_through_the_generator(tmp_path: Path) -> None:
    write_workspace_task(tmp_path / "run" / "workspace" / "task", 9000)
    generator = StandInGenerator(tmp_path / "tasks")

    def checks_on(service: StandInGenerator) -> GeneratorChecks:
        return GeneratorChecks(
            service,
            asyncio.get_running_loop(),
            generation=3,
            index=1,
            source_record_id="session-1",
            checkout_path="/app",
            max_checks=2,
        )

    async def checked() -> list[dict[str, Any]]:
        checks = checks_on(generator)
        tools = AgentTools(tmp_path / "run", checks)
        results = [await asyncio.to_thread(polled, tools)]
        generator.check_fail_after = 1
        results.append(await asyncio.to_thread(polled, tools))
        generator.check_fail_after = None
        results.extend([await asyncio.to_thread(polled, tools), await asyncio.to_thread(polled, tools)])
        await asyncio.to_thread(checks.close)
        results.append(await asyncio.to_thread(polled, tools))
        for error in (DuplicateTask("duplicate of a task already under the root"), GeneratorError("refused (500)")):
            refusing = AgentTools(tmp_path / "run", checks_on(RefusingGenerator(tmp_path / "other", error)))
            results.append(await asyncio.to_thread(polled, refusing))
        return results

    passed, unavailable, passed_again, spent, closed, duplicate, refused = asyncio.run(checked())
    assert [result["check_id"] for result in (passed, unavailable, passed_again, spent, closed)] == [1, 2, 3, 4, 5]
    assert passed["state"] == "done" and passed["passed"] is True and passed["check"] == 1
    assert unavailable["ran"] is False and "OracleUnavailable" in unavailable["error"]
    assert unavailable["checks_left"] == 1, "a check that could not run uses up none"
    assert passed_again["check"] == 2 and passed_again["checks_left"] == 0
    assert spent["error"] == "all 2 checks are used; finish with the task as it stands"
    assert closed["error"] == "the agent's run is over; no more checks"
    assert duplicate["ran"] is True and duplicate["passed"] is False and duplicate["checks_left"] == 1
    assert "the same as one the generator holds already" in duplicate["errors"]
    assert refused["ran"] is False and refused["error"] == "refused (500)"
    # The check that could not run gave its number back: the next candidate takes it again.
    candidates = [f"harbor-00003-001-autoenvscaling-check-{number}" for number in (1, 2, 2)]
    assert [path.name for path in generator.checks] == candidates
    assert generator.deleted == candidates and not any((tmp_path / "tasks").glob("harbor-*"))


def test_admission_appends_a_step_that_keeps_only_the_base_commits_history(tmp_path: Path) -> None:
    task_dir = write_workspace_task(tmp_path / "task", 9000)
    dockerfile = task_dir / "environment" / "Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + "USER agent\n")
    task = admitted_task(
        workspace_reply(task_dir),
        generation=0,
        index=0,
        record_id="r",
        checkout_path="/srv/my repo",
    )
    step = task.environment["Dockerfile"].split("USER agent\n", 1)[1]
    assert step.startswith("USER root\n") and step.endswith("USER agent\n")
    assert "cd '/srv/my repo';" in step


def path_with_gnu_stat(tmp_path: Path) -> str | None:
    """A PATH whose stat takes -c, as the history step's image has: the host's own, or GNU coreutils' gstat."""
    path = os.environ["PATH"]
    if subprocess.run(["stat", "-c", "%u", "."], capture_output=True).returncode == 0:
        return path
    gstat = shutil.which("gstat")
    if gstat is None:
        return None
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "stat").symlink_to(gstat)
    return f"{tmp_path / 'bin'}{os.pathsep}{path}"


@pytest.mark.parametrize("is_on_branch", [True, False])
def test_the_history_step_leaves_only_the_base_commit_in_the_repository(tmp_path: Path, is_on_branch: bool) -> None:
    path = path_with_gnu_stat(tmp_path)
    if path is None:
        pytest.skip("the history step runs stat -c, which this host lacks")
    environ = {**os.environ, "PATH": path, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}

    def git(*arguments: str, cwd: Path) -> str:
        command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments]
        return subprocess.run(command, cwd=cwd, env=environ, check=True, capture_output=True, text=True).stdout.strip()

    # Upstream has the base commit, the later commit the task asks for, and a tag on it.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git("init", "-q", "-b", "main", cwd=upstream)
    for name in ("base", "later"):
        (upstream / f"{name}.txt").write_text(name)
        git("add", f"{name}.txt", cwd=upstream)
        git("commit", "-q", "-m", name, cwd=upstream)
    git("tag", "v1", cwd=upstream)
    base, later = git("rev-parse", "HEAD~1", "HEAD", cwd=upstream).split()
    # The task's checkout: a clone (a remote, remote-tracking refs, the tag), fetched again, at the base commit.
    checkout = tmp_path / "checkout"
    git("clone", "-q", str(upstream), str(checkout), cwd=tmp_path)
    git("fetch", "-q", "origin", cwd=checkout)
    if is_on_branch:
        git("checkout", "-q", "-B", "work", base, cwd=checkout)
    else:
        git("checkout", "-q", base, cwd=checkout)

    task = admitted_task(
        workspace_reply(write_workspace_task(tmp_path / "task", 9000)),
        generation=0,
        index=0,
        record_id="r",
        checkout_path=str(checkout),
    )
    dockerfile = task.environment["Dockerfile"]
    run = dockerfile[dockerfile.index("RUN set -e;") :].removeprefix("RUN ")
    subprocess.run(["sh", "-c", run], env=environ, check=True, capture_output=True)

    assert git("rev-list", "--all", cwd=checkout).split() == [base]
    assert git("remote", cwd=checkout) == "" and git("tag", cwd=checkout) == ""
    assert git("rev-parse", "HEAD", cwd=checkout) == base
    assert subprocess.run(["git", "cat-file", "-e", later], cwd=checkout, env=environ).returncode != 0, "pruned"
    assert not (checkout / ".git" / "FETCH_HEAD").exists()
    # Only HEAD's branch, if any, is left; git gc packs it again.
    assert git("for-each-ref", "--format=%(refname)", cwd=checkout) == ("refs/heads/work" if is_on_branch else "")


def test_a_prompt_designer_generation_admits_its_reply_and_reports_against_its_call(tmp_path: Path) -> None:
    reef, calls = serving_reef(fenced(DOCUMENT))
    service = StandInGenerator(tmp_path / "tasks")
    try:
        author = PromptAuthor(
            reef_url=f"http://127.0.0.1:{reef.server_address[1]}",
            token="token",
            scenario="autoenvscaling",
            model="designer",
            api="openai",
        )
        generator = autoenvscaling(tmp_path, author, service, count=1, band_low=None, band_high=None)
        results = run_tasks(generator, (1.0, 1.0))
        asyncio.run(generator.record_results(results, generation=0))
    finally:
        reef.shutdown()
        reef.server_close()
    assert len(calls) == 1 and calls[0]["headers"]["x-reef-tag-index"] == "0"
    assert service.checks == [tmp_path / "tasks" / NAMES[0]], "the prompt Designer checks no candidate"
    task = read_harbor_task(tmp_path / "tasks" / NAMES[0])
    assert task.source_agent_record_ids == ("rec-1", "session-1")
    assert task.config["environment"]["build_timeout_sec"] == 1200
    assert task.instruction == DOCUMENT["instruction"] + "\n"
    # Every rollout passed: outside the default band, inside with the band off.
    assert [entry["band"] for entry in report(tmp_path)["tasks"]] == ["inside"]
    assert report(tmp_path)["proposals"][0]["designer_record_id"] == "rec-1"


def test_from_config_builds_the_author_and_keeps_one_state_directory_for_each_scenario(tmp_path: Path) -> None:
    environ = {"E2B_API_KEY": "e2b-test", "DESIGNER_TOKEN": "designer-token"}
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, agent_token_env="DESIGNER_TOKEN", count=2, band_high=None),
        scenario="one",
        environ=environ,
    )
    assert isinstance(generator.author, AgentAuthor) and generator.author.upstream.api_key == "designer-token"
    assert generator.author.cache_dir == tmp_path / "state" / "one" / "repositories"
    assert generator.state_dir == tmp_path / "state" / "one" and generator.count == 2
    assert (generator.band.low, generator.band.high) == (0.25, None)
    assert [source.record_id for source in generator.sources] == ["session-1", "session-2"]

    prompt = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, author="prompt", agent_api="responses", behaviour_rule=False),
        scenario="two",
        environ={},
    )
    assert isinstance(prompt.author, PromptAuthor) and prompt.author.replies.api == "responses"
    assert generator.author.has_behaviour_rule and not prompt.author.has_behaviour_rule
    with pytest.raises(PluginConfigError, match="an E2B sandbox needs an API key"):
        AutoEnvScalingGenerator.from_config(generating_config(tmp_path), scenario="three", environ={})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"rollouts": 2}, "does not take rollouts"),
        ({"generator_url": "${endpoints.generator}"}, "generator_url"),
        ({"state_dir": ""}, "state_dir and a sources_path"),
        ({"sources_path": "/no/such/sources.json"}, "No such file"),
        ({"author": "human"}, "author must be one of"),
        ({"agent_reef_url": "${endpoints.designer}"}, "agent_model and agent_reef_url"),
        ({"agent_api": "grpc"}, "api must be one of"),
        ({"author": "prompt", "agent_api": "anthropic"}, "agent_api openai or responses"),
        ({"max_model_calls": 0}, "max_model_calls must be positive"),
        ({"count": 0}, "count and max_checks must be positive"),
        ({"concurrency": 0}, "concurrency must be a positive integer"),
        ({"concurrency": True}, "concurrency must be a positive integer"),
        ({"concurrency": "2"}, "concurrency must be a positive integer"),
        ({"band_low": 0.8}, "band_low must not exceed band_high"),
        ({"band_high": 1.5}, r"band_high must be in \[0, 1\]"),
        ({"band_low": "low"}, "band_low must be a number"),
        ({"eval_fraction": 1.0}, "eval_fraction"),
    ],
)
def test_from_config_refuses_a_config_that_cannot_run(tmp_path: Path, fields: dict[str, object], message: str) -> None:
    with pytest.raises(PluginConfigError, match=message):
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **fields), scenario="one", environ={"E2B_API_KEY": "e2b-test"}
        )


def configured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: ScriptedRollouts,
    *training: tuple[float, ...],
    **fields: object,
) -> tuple[AutoEnvScalingGenerator, dict[str, Any]]:
    """A generator from the config on ``service``, without the test review; one generation run and closed."""
    monkeypatch.setattr(generator_module, "HttpGenerator", lambda url: service)
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, count=1, review=False, **fields),
        scenario="autoenvscaling",
        environ={"E2B_API_KEY": "e2b-test"},
    )
    monkeypatch.setattr(generator, "author", StandInAuthor())
    asyncio.run(generator.record_results(run_tasks(generator, *training), generation=0))
    return generator, json.loads((generator.state_dir / "generation-00000.json").read_text())


# YAML reads a bare off as false.
@pytest.mark.parametrize("fields", [{}, {"calibration": "off"}, yaml.safe_load("calibration: off")])
def test_calibration_off_runs_no_rollout_and_leaves_the_report_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fields: dict[str, object]
) -> None:
    service = ScriptedRollouts(tmp_path / "tasks", [0.0, 1.0, 1.0, 0.0])
    generator, document = configured_report(tmp_path, monkeypatch, service, **fields)
    assert generator.calibration is None and service.rollout_requests == []
    assert list(document) == ["generation", "error", "band", "next_position", "proposals", "tasks"]
    assert [check["check"] for check in document["tasks"][0]["checks"]] == ["oracle"]


def test_the_report_keeps_calibration_beside_the_training_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ScriptedRollouts(tmp_path / "tasks", [0.0, 1.0, 1.0, 0.0])
    _, document = configured_report(
        tmp_path,
        monkeypatch,
        service,
        (1.0, 1.0, 1.0, 1.0),
        calibration="refuse",
        calibration_margin=0.1,
        calibration_model="Qwen/Qwen3-8B",
    )
    assert list(document)[:5] == ["generation", "error", "band", "calibration", "next_position"]
    assert document["calibration"] == {
        "mode": "refuse",
        "rollouts": 4,
        "max_rollouts": 8,
        "margin": 0.1,
        "min_std": 0.1,
        "model": "Qwen/Qwen3-8B",
    }
    assert [request["model"] for request in service.rollout_requests] == ["Qwen/Qwen3-8B"]
    # The calibration kept the task; its training rollouts then put it above the training band.
    [task] = document["tasks"]
    [proposal] = document["proposals"]
    assert task["band"] == "above" and proposal["stage"] == "band"
    calibration = task["checks"][-1]
    assert calibration["check"] == "calibration" and calibration["details"]["result"] == "kept"
    assert calibration["details"]["place"] == "inside" and proposal["checks"] == task["checks"]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"calibration": "maybe"}, "calibration must be one of off, measure, refuse"),
        ({"calibration_rollouts": True}, "calibration_rollouts must be an integer"),
        ({"calibration_rollouts": 1}, "calibration_rollouts must be at least 2"),
        ({"calibration_max_rollouts": 8.0}, "calibration_max_rollouts must be an integer"),
        ({"calibration_max_rollouts": 3}, "calibration_max_rollouts must be at least calibration_rollouts"),
        ({"calibration_margin": "wide"}, "calibration_margin must be a number"),
        ({"calibration_margin": True}, "calibration_margin must be a number"),
        ({"calibration_min_std": False}, "calibration_min_std must be a number"),
        ({"calibration_margin": 1.5}, r"calibration_margin must be in \[0, 1\]"),
        ({"calibration_margin": -0.1}, r"calibration_margin must be in \[0, 1\]"),
        ({"calibration_min_std": 0.6}, r"calibration_min_std must be in \[0, 0.5\]"),
        ({"calibration_min_std": float("nan")}, r"calibration_min_std must be in \[0, 0.5\]"),
        ({"calibration_model": ""}, "calibration_model must be a model name or null"),
        ({"calibration_model": 7}, "calibration_model must be a model name or null"),
        ({"band_high": 1.5}, r"band_high must be in \[0, 1\]"),
        # A calibration that is off still has its keys checked.
        ({"calibration": False, "calibration_rollouts": "junk"}, "calibration_rollouts must be an integer"),
        ({"calibration": "off", "calibration_margin": -5}, r"calibration_margin must be in \[0, 1\]"),
    ],
)
def test_from_config_refuses_a_calibration_that_cannot_run(
    tmp_path: Path, fields: dict[str, object], message: str
) -> None:
    with pytest.raises(PluginConfigError, match=message):
        AutoEnvScalingGenerator.from_config(
            generating_config(tmp_path, **{"calibration": "measure", **fields}),
            scenario="one",
            environ={"E2B_API_KEY": "e2b-test"},
        )


def test_a_round_of_the_generated_task_recipe_runs_autoenvscaling_end_to_end(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", rewards={"plain": 0.5})
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2)
    session = SQLiteScenarioStorage(tmp_path / "records").open("autoenvscaling")
    context = PipelineContext("autoenvscaling", task_storage=session.tasks, samples=TrainerSamples(session.samples))
    generation = GeneratedTaskSamples(
        context,
        RoundSettings(rounds=1, description=DESCRIPTION),
        GeneratedTaskRecord2Record(context, TaskGroupSettings()),
        task_generator=generator,
        generator_service=service,
    )
    generated = asyncio.run(generation.run(GenerationJob(receipt="generation-00000", generation=0)))
    for task in generated.tasks:
        # The end of the task job saves each task's training rewards, from the training component's rollout.
        generation.save_result(TaskRolloutResult(task.metadata["task"]["name"], (0.5, 0.5)))
    delivery_error = asyncio.run(generation.deliver_results())
    results = list(rollout_results(generation.task_store).values())
    generation.close()
    assert len(generated.tasks) == 2 and delivery_error == ""
    assert [result.rollout_rewards for result in results] == [(0.5, 0.5)] * 2
    assert service.rollout_requests == [], "the training component runs the training rollouts"
    assert generation.task_store is not None and len(generation.task_store.tasks(split="train")) == 2
    assert [task["band"] for task in report(tmp_path)["tasks"]] == ["inside", "inside"]


def test_an_outage_that_makes_no_task_leaves_the_round_incomplete_so_a_restart_gives_it_back(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    stand_in = StandInAuthor(unreachable=("session-1", "session-2"))
    generator = autoenvscaling(tmp_path, stand_in, service, count=2)
    session = SQLiteScenarioStorage(tmp_path / "records").open("autoenvscaling")
    context = PipelineContext("autoenvscaling", task_storage=session.tasks, samples=TrainerSamples(session.samples))
    generation = GeneratedTaskSamples(
        context,
        RoundSettings(rounds=1, description=DESCRIPTION, generation_attempts=2),
        GeneratedTaskRecord2Record(context, TaskGroupSettings()),
        task_generator=generator,
        generator_service=service,
    )
    outcome = asyncio.run(generation.run(GenerationJob(receipt="generation-00000", generation=0)))
    generation.close()
    assert not outcome.is_complete and "the sandbox did not start" in outcome.error
    assert [call["source"] for call in stand_in.calls] == ["session-1", "session-2"], "one session for each attempt"
    assert generation.task_store is not None and complete_generation_count(generation.task_store) == 0


def test_a_task_whose_calibration_rollouts_cannot_run_leaves_the_generation_complete(tmp_path: Path) -> None:
    # The first task's rollouts score; none of the second task's calibration rollouts can run.
    service = ScriptedRollouts(tmp_path / "tasks", [0.0, 1.0, 1.0, 0.0] + [None] * 8)
    check = CalibrationCheck(
        service,
        scenario="autoenvscaling",
        rollouts=4,
        max_rollouts=8,
        margin=0.15,
        min_std=0.1,
        band=PassRateBand(0.25, 0.75),
    )
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2, calibration=check)
    session = SQLiteScenarioStorage(tmp_path / "records").open("autoenvscaling")
    context = PipelineContext("autoenvscaling", task_storage=session.tasks, samples=TrainerSamples(session.samples))
    generation = GeneratedTaskSamples(
        context,
        RoundSettings(rounds=1, description=DESCRIPTION),
        GeneratedTaskRecord2Record(context, TaskGroupSettings()),
        task_generator=generator,
        generator_service=service,
    )
    generated = asyncio.run(generation.run(GenerationJob(receipt="generation-00000", generation=0)))
    for task in generated.tasks:
        generation.save_result(TaskRolloutResult(task.metadata["task"]["name"], (0.5, 0.5)))
    asyncio.run(generation.deliver_results())
    generation.close()
    assert generated.is_complete and [task.metadata["task"]["name"] for task in generated.tasks] == [NAMES[0]]
    assert service.requested_counts == [4, 4, 4]
    assert [proposal["stage"] for proposal in report(tmp_path)["proposals"]] == ["", "unchecked"]


def test_the_example_selects_the_generated_task_recipe_and_builds_the_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in {
        "REEF_TOKEN": "t",
        "REEF_AUTOENVSCALING_STATE_DIR": str(tmp_path),
        "REEF_AUTOENVSCALING_SOURCES_PATH": str(sources_file(tmp_path, 2)),
        "REEF_AUTOENVSCALING_DESIGNER_URL": "http://127.0.0.1:8901",
        "REEF_AUTOENVSCALING_DESIGNER_TOKEN": "designer-token",
        "E2B_API_KEY": "e2b-test",
        "TINKER_API_KEY": "tinker-test",
    }.items():
        monkeypatch.setenv(name, value)
    example = Path(__file__).resolve().parents[3] / "recipes/beta/autoenvscaling/examples/tinker/serve.yaml"
    resolved, _ = resolve_deployment_config(yaml.safe_load(example.read_text()), None, example)
    settings = service_config_from_mapping(resolved)
    assert settings.recipe == "reef.recipe.generated_tasks:GeneratedTaskRecipe"
    section = settings.pipeline_settings["record2sample"]["config"]["generator"]
    assert section["module"] == "recipes.beta.autoenvscaling:AutoEnvScalingGenerator"
    # reef serve publishes the generator's endpoint when it starts the services.
    config = {**section["config"], "generator_url": "http://127.0.0.1:8910"}
    generator = AutoEnvScalingGenerator.from_config(config, scenario="autoenvscaling", environ=os.environ)
    assert isinstance(generator.author, AgentAuthor)
    assert generator.state_dir == tmp_path / "generations" / "autoenvscaling"
    assert generator.author.upstream.api_key == "designer-token" and generator.author.upstream.model == "gpt-6-sol"


def test_a_generation_closes_after_a_later_one_started_and_after_a_restart(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", rewards={"plain": 0.5})
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (first,) = asyncio.run(generator.generate(TaskGenerationRequest((), DESCRIPTION, generation=0)))
    assert asyncio.run(generator.validate(first)).is_valid
    (second,) = asyncio.run(generator.generate(TaskGenerationRequest((), DESCRIPTION, generation=1)))
    asyncio.run(generator.record_results((RolledOutTask(second, (0.5, 0.5)),), generation=1))
    assert report(tmp_path, 1)["tasks"][0]["name"] == second.name
    restarted = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    asyncio.run(restarted.record_results((RolledOutTask(first, (0.5, 0.5)),), generation=0))
    document = report(tmp_path, 0)
    assert [task["band"] for task in document["tasks"]] == ["inside"]
    assert document["tasks"][0]["checks"], "the check results saved before the restart are in the report"
    asyncio.run(restarted.record_results((), generation=0))
    assert report(tmp_path, 0) == document, "a second delivery of a closed generation does nothing"


def test_a_task_that_could_not_run_is_an_outage_in_the_report(tmp_path: Path) -> None:
    generator = autoenvscaling(tmp_path, StandInAuthor(), StandInGenerator(tmp_path / "tasks"), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    failure = FailedTask(item, "the sandbox provider is down", True)
    asyncio.run(generator.record_results((failure,), generation=0))
    assert [entry["stage"] for entry in report(tmp_path)["proposals"]] == ["outage"]


def test_the_generator_refuses_a_band_whose_low_exceeds_its_high(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="band_low must not exceed band_high"):
        autoenvscaling(tmp_path, StandInAuthor(), band_low=0.8, band_high=0.2)


def test_from_config_gives_the_calibration_the_configured_band(tmp_path: Path) -> None:
    generator = AutoEnvScalingGenerator.from_config(
        generating_config(tmp_path, calibration="measure", band_low=0.1, band_high=0.9),
        scenario="autoenvscaling",
        environ={"E2B_API_KEY": "key"},
    )
    [calibration] = [check for check in generator.task_checks if isinstance(check, CalibrationCheck)]
    assert (calibration.band.low, calibration.band.high) == (0.1, 0.9)
    assert calibration.band is generator.band
