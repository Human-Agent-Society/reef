"""AutoEnvScaling as a task generator: the session cursor, admission, validation, the band and the reports."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.recipes.autoenvscaling.autoenvscaling_stubs import (
    DESCRIPTION,
    NAMES,
    REQUEST,
    RefusingGenerator,
    StandInAuthor,
    autoenvscaling,
    cursor,
    fenced,
    generating_config,
    played,
    report,
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
    CandidateChecks,
    GeneratorChecks,
    PromptAuthor,
    SessionSource,
    TaskAuthor,
    TaskCheck,
    admitted_task,
    workspace_reply,
)
from recipes.beta.autoenvscaling.author import AgentTools
from recipes.beta.autoenvscaling.checks import CheckResult
from reef.core import AgentRecord, RequestType
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, TaskGenerationRequest, TaskPlayResult, read_harbor_task
from reef.pipeline_context import PipelineContext
from reef.recipe.generated_tasks import GeneratedTaskRecord2Record, GeneratedTaskTasks
from reef.recipe.generated_tasks.processor import RoundJob, RoundSettings, TaskGroupSettings
from reef.record2task.generator import DuplicateTask, GeneratorError, WrittenTask
from reef.record2task.generator.harbor import ORACLE_TIMEOUT_S
from reef.service.deploy.orchestrator import resolve_deployment_config
from reef.service.deploy.service_config import service_config_from_mapping


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
    assert [item.task.name for item in tasks] == NAMES
    task = tasks[0].task
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
    asyncio.run(first.record_results(()))
    again = autoenvscaling(tmp_path, author, source_count=3, count=2)
    tasks = asyncio.run(again.generate(REQUEST))
    # The walk cycles: positions 2 and 3 are the third session and the first.
    assert [call["source"] for call in author.calls] == ["session-1", "session-2", "session-3", "session-1"]
    assert [item.task.name for item in tasks] == [
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


def test_a_task_the_authoring_checks_refuse_is_refused_and_names_the_last_call(tmp_path: Path) -> None:
    author = StandInAuthor(has_tmux=False)
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, author, service, count=1)
    assert played(generator) == ()
    check = author.calls[0]["check"]
    assert check["stage"] == "authoring" and "does not install tmux" in check["errors"]
    assert service.checks == [], "a candidate the authoring checks refuse never reaches Harbor"
    asyncio.run(generator.record_results(()))
    [proposal] = report(tmp_path)["proposals"]
    assert proposal["stage"] == "authoring" and proposal["designer_record_id"] == "agent-2"
    assert proposal["refusal"].startswith("the Designer's task is refused") and "tmux" in proposal["refusal"]


def test_sessions_without_a_task_are_recorded_by_stage_and_an_outage_ends_the_generation(tmp_path: Path) -> None:
    author = StandInAuthor(unusable=("session-1",), unreachable=("session-3",))
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, author, service, source_count=4, count=4)
    results = played(generator)
    assert [result.task.name for result in results] == ["harbor-00000-001-autoenvscaling"]
    assert [call["source"] for call in author.calls] == ["session-1", "session-2", "session-3"]
    asyncio.run(generator.record_results(results))

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
    # The tasks made before the outage played; their manifest is written, and the report names the outage.
    assert [manifest["names"] for manifest in service.manifests] == [[NAMES[1]]]
    assert document["manifest"] == str(tmp_path / "tasks" / "manifest-00000.json")
    assert document["next_position"] == 3 and cursor(tmp_path)["next_position"] == 3, "the outage's session is taken"

    silent = StandInGenerator(tmp_path / "silent")
    generator = autoenvscaling(tmp_path / "silent", StandInAuthor(receipts=()), silent, count=1)
    assert played(generator) == ()
    asyncio.run(generator.record_results(()))
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
    assert asyncio.run(generator.validate(item.task)).is_valid
    assert service.checks[-1] == tmp_path / "tasks" / NAMES[0]
    assert service.deleted[-1] == NAMES[0] and not (tmp_path / "tasks" / NAMES[0]).exists()

    further = RefusingCheck()
    generator = autoenvscaling(tmp_path / "further", StandInAuthor(), task_checks=(further,), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item.task)).errors == ("the verifier checks the wording of the change",)
    assert further.runs == [NAMES[0]]

    refusing = StandInGenerator(tmp_path / "refusing", is_solvable=False)
    further = RefusingCheck()
    generator = autoenvscaling(tmp_path / "refusing", StandInAuthor(), refusing, task_checks=(further,), count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(item.task)).errors == ("oracle check refused: the oracle scored 0",)
    assert further.runs == [], "a check that found a defect ends the validation"
    assert refusing.deleted[-1] == NAMES[0]


def test_a_duplicate_is_invalid(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1)
    (item,) = asyncio.run(generator.generate(REQUEST))
    asyncio.run(service.write_task(item.task))
    assert asyncio.run(generator.validate(item.task)).errors == (
        "refused: duplicate of a task already under the root",
    )


def test_a_check_that_cannot_run_raises_and_the_generation_writes_no_manifest(tmp_path: Path) -> None:
    # Two candidate checks, then the first task's check; the second task's check cannot run.
    service = StandInGenerator(tmp_path / "tasks", check_fail_after=3)
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2)
    first, second = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(first.task)).is_valid
    asyncio.run(service.write_task(first.task))
    with pytest.raises(GeneratorError, match="the harbor command line is not installed"):
        asyncio.run(generator.validate(second.task))
    assert not (tmp_path / "tasks" / NAMES[1]).exists(), "no copy is left"
    asyncio.run(generator.record_results((TaskPlayResult(first.task, (0.0, 1.0)),)))
    document = report(tmp_path)
    assert service.manifests == [] and document["manifest"] is None
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
def test_the_band_keeps_a_played_task_only_when_its_pass_rate_is_inside_it(
    tmp_path: Path, training: tuple[float, ...], band: dict[str, float | None], position: str
) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1, **band)
    asyncio.run(generator.record_results(played(generator, training)))
    document = report(tmp_path)
    pass_rate = sum(training) / len(training)
    [task] = document["tasks"]
    assert (task["name"], task["pass_rate"], task["band"]) == (NAMES[0], pass_rate, position)
    # Inside the band or not, the task stays under the tasks root, and the report names where.
    assert task["path"] == str(tmp_path / "tasks" / NAMES[0]) and (tmp_path / "tasks" / NAMES[0]).is_dir()
    assert task["training_rewards"] == list(training) and task["source_record_id"] == "session-1"
    [proposal] = document["proposals"]
    if position == "inside":
        assert proposal["stage"] == "" and service.manifests[0]["names"] == [NAMES[0]]
    else:
        assert (
            proposal["stage"] == "band"
            and proposal["refusal"] == f"outside the solver band: pass rate {pass_rate:.2f}"
        )
        assert proposal["task_name"] == NAMES[0] and service.manifests == []
        assert service.deleted == [f"{NAMES[0]}-check-1", NAMES[0]], "the candidate and the validated copy only"


def test_the_results_name_where_each_refused_task_stopped(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, eval_fraction=0.3, seed=7)
    first, second, third = asyncio.run(generator.generate(REQUEST))
    assert asyncio.run(generator.validate(first.task)).is_valid
    assert asyncio.run(generator.validate(third.task)).is_valid
    asyncio.run(service.write_task(first.task))
    service.is_solvable = False
    errors = asyncio.run(generator.validate(second.task)).errors
    results = (
        TaskPlayResult(first.task, (0.0, 1.0)),
        TaskPlayResult(second.task, error="; ".join(errors)),
        TaskPlayResult(third.task, error="the agent could not play the task: no sandbox"),
    )
    asyncio.run(generator.record_results(results))

    document = report(tmp_path)
    assert [(entry["stage"], entry["refusal"]) for entry in document["proposals"]] == [
        ("", ""),
        ("check", "oracle check refused: the oracle scored 0"),
        ("play", "the agent could not play the task: no sandbox"),
    ]
    assert service.reports == [], "the generator sends no Designer report"
    assert service.manifests == [{"generation": 0, "names": [NAMES[0]], "eval_fraction": 0.3, "seed": 7}]
    assert document["manifest"] == str(tmp_path / "tasks" / "manifest-00000.json")


class GoneAfterPlays(StandInGenerator):
    """A generator service that went away after the plays: the manifest does not reach it."""

    async def write_manifest(self, *, generation: int, names: Sequence[str], eval_fraction: float, seed: int) -> Path:
        raise GeneratorError("POST /manifests did not reach the generator: connection refused")


def test_the_report_is_written_when_the_manifest_does_not_reach_the_service(tmp_path: Path) -> None:
    service = GoneAfterPlays(tmp_path / "tasks")
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=1, band_low=0.0, band_high=1.0)
    asyncio.run(generator.record_results(played(generator, (0.0, 1.0))))

    document = report(tmp_path)
    assert [entry["name"] for entry in document["tasks"]] == [NAMES[0]]
    assert document["tasks"][0]["training_rewards"] == [0.0, 1.0]
    assert document["manifest"] is None and "did not reach the generator" in document["manifest_error"]


def test_a_request_with_source_records_is_refused_and_named_in_the_report(tmp_path: Path) -> None:
    author = StandInAuthor()
    generator = autoenvscaling(tmp_path, author)
    record = AgentRecord.create(scenario="autoenvscaling", request_type=RequestType.INFERENCE, payload={})
    with pytest.raises(ValueError, match="takes its sessions from its sources file"):
        asyncio.run(generator.generate(TaskGenerationRequest((record,), DESCRIPTION)))
    asyncio.run(generator.record_results(()))
    assert author.calls == [] and "not from source records" in report(tmp_path)["error"]
    assert cursor(tmp_path) == {"next_generation": 1, "next_position": 0}


def polled(tools: AgentTools) -> dict[str, Any]:
    """What the agent's task_check does: start a check with /check, then poll /trial "check <id>" until it is done."""
    check_id = tools.check()["check_id"]
    while (result := tools.trial(f"check {check_id}"))["state"] == "running":
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
        results = played(generator, (1.0, 1.0))
        asyncio.run(generator.record_results(results))
    finally:
        reef.shutdown()
        reef.server_close()
    assert len(calls) == 1 and calls[0]["headers"]["x-reef-tag-index"] == "0"
    assert service.checks == [tmp_path / "tasks" / NAMES[0]], "the prompt Designer checks no candidate"
    task = read_harbor_task(tmp_path / "tasks" / NAMES[0])
    assert task.source_agent_record_ids == ("rec-1", "session-1")
    assert task.config["environment"]["build_timeout_sec"] == 1200
    assert task.instruction == DOCUMENT["instruction"] + "\n"
    # Every play passed: outside the default band, inside with the band off.
    assert [entry["band"] for entry in report(tmp_path)["tasks"]] == ["inside"]
    assert service.manifests[0]["names"] == [NAMES[0]]
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
    assert (generator.band_low, generator.band_high) == (0.25, None)
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


def test_a_round_of_the_generated_task_recipe_runs_autoenvscaling_end_to_end(tmp_path: Path) -> None:
    service = StandInGenerator(tmp_path / "tasks", rewards={"plain": 0.5})
    generator = autoenvscaling(tmp_path, StandInAuthor(), service, count=2)
    context = PipelineContext("autoenvscaling")
    generation = GeneratedTaskTasks(
        context,
        RoundSettings(rounds=1, description=DESCRIPTION, state_dir=str(tmp_path / "rounds")),
        GeneratedTaskRecord2Record(context, TaskGroupSettings(plays_per_task=2)),
        task_generator=generator,
        task_player=service,
    )
    outcome = asyncio.run(generation.run_round(RoundJob(receipt="round-00000", round_number=0)))
    generation.close()
    assert outcome.error == "" and [result.training_rewards for result in outcome.results] == [(0.5, 0.5)] * 2
    assert [(call["task"], call["arm"], call["plays"]) for call in service.plays] == [
        (NAMES[0], "plain", 2),
        (NAMES[1], "plain", 2),
    ]
    assert service.manifests[0]["names"] == NAMES[:2]
    assert [task["band"] for task in report(tmp_path)["tasks"]] == ["inside", "inside"]
    assert read_harbor_task(tmp_path / "tasks" / NAMES[0]).name == NAMES[0]


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
    }.items():
        monkeypatch.setenv(name, value)
    example = Path(__file__).resolve().parents[3] / "recipes/beta/autoenvscaling/examples/tinker/serve.yaml"
    resolved, _ = resolve_deployment_config(yaml.safe_load(example.read_text()), None, example)
    settings = service_config_from_mapping(resolved)
    assert settings.recipe == "reef.recipe.generated_tasks:GeneratedTaskRecipe"
    section = settings.pipeline_settings["record2task"]["config"]["generator"]
    assert section["module"] == "recipes.beta.autoenvscaling:AutoEnvScalingGenerator"
    # reef serve publishes the generator's endpoint when it starts the services.
    config = {**section["config"], "generator_url": "http://127.0.0.1:8910"}
    generator = AutoEnvScalingGenerator.from_config(config, scenario="autoenvscaling", environ=os.environ)
    assert isinstance(generator.author, AgentAuthor)
    assert generator.state_dir == tmp_path / "generations" / "autoenvscaling"
    assert generator.author.upstream.api_key == "designer-token" and generator.author.upstream.model == "gpt-6-sol"
