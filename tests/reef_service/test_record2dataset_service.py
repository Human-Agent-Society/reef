"""The generator service and its client: proposals, written tasks, checks, plays and manifests over HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer
from reef_client.client import ReefClientError

from reef.core.tasks import HarborTask, read_harbor_task, read_split_manifest
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset import (
    Designer,
    DesignerAnswer,
    DesignerError,
    DesignerRequest,
    DuplicateTask,
    GeneratorError,
    GeneratorService,
    HarborChecks,
    HarborRuns,
    HttpGenerator,
    JobRunner,
    OracleResult,
    OracleUnavailable,
    ReadinessProbe,
    TaskChecks,
    TaskNameConflict,
    TaskPlays,
    readiness_probes,
)
from reef.record2dataset.service import CLOSE_GRACE_S, DockerProbe, HarborProbe, ModuleProbe
from reef.record2dataset.wire import WireError, play_document, play_from_document, task_document, task_from_document
from reef.service.deploy.generator import generator_settings

pytestmark = pytest.mark.unit

DOCUMENT = {
    "instruction": (
        "A service on this machine writes the port it listens on under /var/run. Find that file and write the "
        "port number, and nothing else, to /workspace/port.txt."
    ),
    "environment": {
        "Dockerfile": "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y tmux && echo 8471 > /var/run/app.port\nWORKDIR /workspace\n"
    },
    "tests": {
        "test.sh": '#!/bin/sh\nmkdir -p /logs/verifier\ntest "$(cat /workspace/port.txt)" = 8471 && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt\n'
    },
    "solution": {"solve.sh": "#!/bin/sh\ncat /var/run/app.port > /workspace/port.txt\n"},
    "hint": "Look under /var/run for what the service left behind.",
}


def reply_for(port: int) -> str:
    document = json.loads(json.dumps(DOCUMENT).replace("8471", str(port)))
    return "```json\n" + json.dumps(document) + "\n```\n"


class StandInDesigner(Designer):
    def __init__(self, scripted: Sequence[str] = (), *, report_failure: Exception | None = None) -> None:
        self.scripted = list(scripted)
        self.report_failure = report_failure
        self.calls: list[dict[str, object]] = []
        self.reports: list[dict[str, object]] = []

    def answer(self, messages, *, scenario, model, tags) -> DesignerAnswer:
        self.calls.append({"messages": list(messages), "scenario": scenario, "model": model, "tags": dict(tags)})
        text = self.scripted.pop(0) if self.scripted else reply_for(8471 + len(self.calls))
        return DesignerAnswer(text=text, record_id=f"designer-{len(self.calls)}")

    def report(self, record_id, *, scenario, score, metadata) -> str:
        if self.report_failure is not None:
            raise self.report_failure
        self.reports.append({"record_id": record_id, "scenario": scenario, "score": score, "metadata": dict(metadata)})
        return f"report-{len(self.reports)}"


class StandInChecks(TaskChecks):
    def __init__(self, *, is_solvable: bool = True, raises: bool = False, gate: threading.Event | None = None) -> None:
        self.is_solvable = is_solvable
        self.raises = raises
        self.gate = gate
        self.calls: list[Path] = []

    def oracle(self, task_path: Path) -> OracleResult:
        self.calls.append(task_path)
        if self.gate is not None:
            self.gate.wait()
        if self.raises:
            raise OracleUnavailable("harbor run -a oracle exited 2: docker is not running")
        if not self.is_solvable:
            return OracleResult(is_solvable=False, reason="the oracle scored 0", oracle_reward=0.0)
        return OracleResult(is_solvable=True, reason="", oracle_reward=1.0, nop_reward=0.0)


class StandInPlays(TaskPlays):
    def __init__(self, reward: float = 0.5) -> None:
        self.reward = reward
        self.calls: list[dict[str, object]] = []

    def play(self, task_path, *, scenario, model, arm, plays, is_reporting, extra_instruction_paths, tags):
        self.calls.append(
            {
                "task": task_path.name,
                "scenario": scenario,
                "model": model,
                "arm": arm,
                "plays": plays,
                "is_reporting": is_reporting,
                "extra": [Path(path) for path in extra_instruction_paths],
                "tags": dict(tags),
            }
        )
        return tuple(
            TaskPlay(
                task_path,
                task_path.name,
                f"e{n}",
                self.reward,
                {"reward": self.reward},
                "",
                ("rec",),
                0,
                ("rep",),
                None,
            )
            for n in range(plays)
        )


class StandInProbe(ReadinessProbe):
    def __init__(self, name: str, reason: str = "") -> None:
        self.name = name
        self.reason = reason
        self.calls = 0

    def missing(self) -> str:
        self.calls += 1
        return self.reason


def service(tmp_path: Path, **parts: object) -> tuple[GeneratorService, StandInDesigner, StandInChecks, StandInPlays]:
    designer = parts.get("designer") or StandInDesigner()
    checks = parts.get("checks") or StandInChecks()
    plays = parts.get("plays") or StandInPlays()
    built = GeneratorService(
        tasks_root=tmp_path / "tasks",
        designer=designer,  # type: ignore[arg-type]
        checks=checks,  # type: ignore[arg-type]
        plays=plays,  # type: ignore[arg-type]
        default_model=parts.get("default_model", "served"),  # type: ignore[arg-type]
        designer_model=parts.get("designer_model"),  # type: ignore[arg-type]
        probes=parts.get("probes"),  # type: ignore[arg-type]
        jobs=parts.get("jobs"),  # type: ignore[arg-type]
    )
    return built, designer, checks, plays  # type: ignore[return-value]


def run_with(built: GeneratorService, body: Callable[[HttpGenerator], Awaitable[object]]) -> object:
    async def run() -> object:
        async with TestServer(built.app()) as server:
            return await body(HttpGenerator(str(server.make_url("")), poll_s=0.01))

    return asyncio.run(run())


def request() -> DesignerRequest:
    return DesignerRequest(target="shell tasks with hidden state", skill="inspection", experience_text="LAST TIME: ok")


def test_a_proposal_comes_back_as_a_task_with_the_designers_record_id(tmp_path: Path) -> None:
    built, designer, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        return await generator.propose(
            request(), scenario="spade", generation=3, index=1, tags={"role": "designer"}, model="m"
        )

    proposed = run_with(built, body)
    assert proposed.task is not None and proposed.refusal == "" and proposed.record_id == "designer-1"
    assert proposed.task.name == "harbor-00003-001-inspection"
    assert (
        proposed.task.source_agent_record_ids == ("designer-1",) and proposed.task.metadata["difficulty"] == "medium"
    )
    assert proposed.task.solution["hint.txt"] == DOCUMENT["hint"] + "\n"
    call = designer.calls[0]
    assert call["scenario"] == "spade" and call["model"] == "m" and call["tags"] == {"role": "designer"}
    assert "LAST TIME: ok" in call["messages"][1]["content"] and "inspection" in call["messages"][1]["content"]


def test_a_reply_that_is_no_task_is_a_refusal_with_the_record_id(tmp_path: Path) -> None:
    built, _, _, _ = service(tmp_path, designer=StandInDesigner(scripted=["no json here"]))

    async def body(generator: HttpGenerator) -> object:
        return await generator.propose(request(), scenario="spade", generation=0, index=0, tags={})

    proposed = run_with(built, body)
    assert proposed.task is None and proposed.record_id == "designer-1"
    assert proposed.refusal.startswith("reply refused: the reply holds no ```json block")


def test_a_proposal_without_a_model_uses_the_services_default_and_none_is_refused(tmp_path: Path) -> None:
    built, designer, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        return await generator.propose(request(), scenario="spade", generation=0, index=0, tags={})

    run_with(built, body)
    assert designer.calls[0]["model"] == "served"
    built, _, _, _ = service(tmp_path, default_model=None)
    with pytest.raises(GeneratorError, match=r"refused \(400\).*model must name the served model"):
        run_with(built, body)


def test_a_service_with_a_designer_model_asks_the_designer_for_it_whatever_the_body_says(tmp_path: Path) -> None:
    built, designer, _, plays = service(tmp_path, designer_model="strong")

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={}, model="m")
        await generator.propose(request(), scenario="spade", generation=1, index=1, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        await generator.play(
            written.path,
            scenario="spade",
            arm="bare",
            plays=1,
            is_reporting=False,
            extra_instruction_files=(),
            tags={},
        )
        return None

    run_with(built, body)
    assert [call["model"] for call in designer.calls] == ["strong", "strong"]
    assert plays.calls[0]["model"] == "served", "the designer model is the Designer's alone; plays keep the served one"
    built, designer, _, _ = service(tmp_path / "without")

    async def body_without(generator: HttpGenerator) -> object:
        return await generator.propose(request(), scenario="spade", generation=1, index=0, tags={}, model="m")

    run_with(built, body_without)
    assert designer.calls[0]["model"] == "m"


def test_a_task_is_written_once_and_a_duplicate_or_a_conflict_is_refused(tmp_path: Path) -> None:
    built, _, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        assert written.path == tmp_path / "tasks" / proposed.task.name and written.digest == proposed.task.digest
        assert read_harbor_task(written.path) == proposed.task
        with pytest.raises(DuplicateTask, match="duplicate"):
            await generator.write_task(
                HarborTask(
                    name="other-name",
                    instruction=proposed.task.instruction,
                    tests=proposed.task.tests,
                    environment=proposed.task.environment,
                    solution=proposed.task.solution,
                )
            )
        with pytest.raises(GeneratorError, match="a different task holds the name"):
            await generator.write_task(
                HarborTask(
                    name=proposed.task.name,
                    instruction=proposed.task.instruction + "Hurry.\n",
                    tests=proposed.task.tests,
                    environment=proposed.task.environment,
                )
            )
        await generator.delete_task(proposed.task.name)
        assert not written.path.exists()
        with pytest.raises(GeneratorError, match=r"refused \(404\)"):
            await generator.delete_task(proposed.task.name)
        with pytest.raises(GeneratorError, match=r"refused \(400\).*not a task name"):
            await generator.delete_task("bad name")
        return None

    run_with(built, body)


def test_a_taken_name_is_a_name_conflict_told_apart_from_a_duplicate(tmp_path: Path) -> None:
    built, _, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        await generator.write_task(proposed.task)
        with pytest.raises(TaskNameConflict, match="a different task holds the name harbor-00001-000-inspection"):
            await generator.write_task(
                HarborTask(
                    name=proposed.task.name,
                    instruction=proposed.task.instruction + "Hurry.\n",
                    tests=proposed.task.tests,
                    environment=proposed.task.environment,
                )
            )
        return None

    run_with(built, body)
    assert issubclass(TaskNameConflict, GeneratorError) and not issubclass(TaskNameConflict, DuplicateTask)


def test_a_check_runs_the_oracle_on_a_task_under_the_root_only(tmp_path: Path) -> None:
    built, _, checks, _ = service(tmp_path, checks=StandInChecks(is_solvable=False))

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        result = await generator.check(written.path)
        assert not result.is_solvable and result.reason == "the oracle scored 0" and result.oracle_reward == 0.0
        assert result == await generator.check(Path(proposed.task.name)), "a name under the root is the same task"
        with pytest.raises(GeneratorError, match="not a task directory under"):
            await generator.check(tmp_path)
        return None

    run_with(built, body)
    assert checks.calls == [tmp_path / "tasks" / "harbor-00001-000-inspection"] * 2


def test_a_check_that_could_not_run_is_a_failed_job_the_client_raises(tmp_path: Path) -> None:
    built, _, checks, _ = service(tmp_path, checks=StandInChecks(raises=True))
    error = "OracleUnavailable: harbor run -a oracle exited 2: docker is not running"

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        submitted = await generator.call("POST", "/checks", body={"path": str(written.path)})
        with pytest.raises(GeneratorError, match=f"job {submitted['job']} failed: {error}"):
            await generator.job_result(submitted)
        job = await generator.call("GET", f"/jobs/{submitted['job']}")
        assert job["state"] == "failed" and job["error"] == error and job["result"] is None
        with pytest.raises(GeneratorError, match=f"failed: {error}"):
            await generator.check(written.path)
        return None

    run_with(built, body)
    assert len(checks.calls) == 2, "the runner went on to the next job after the failed one"


def test_closing_the_service_stops_the_harbor_run_of_the_job_in_flight(tmp_path: Path) -> None:
    harbor = tmp_path / "harbor"
    harbor.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    harbor.chmod(0o755)
    runs = HarborRuns()
    built, _, _, _ = service(tmp_path, checks=HarborChecks(harbor=str(harbor), runs=runs), jobs=JobRunner(runs))

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        submitted = await generator.call("POST", "/checks", body={"path": str(written.path)})
        deadline = time.monotonic() + 10.0
        while not runs.pids() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        return submitted["job"], runs.pids()

    started = time.monotonic()
    job_id, pids = run_with(built, body)  # type: ignore[misc]
    assert time.monotonic() - started < CLOSE_GRACE_S, "the child honours the term, so the close never waits it out"
    (pid,) = pids
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    job = built.jobs.get(job_id)
    # A check the generator stopped could not run: the job fails with the reason, never a refused task.
    assert job is not None and job.state == "failed" and job.result is None
    assert job.error == "OracleUnavailable: harbor run -a oracle was stopped with the generator"
    assert runs.is_closed and runs.pids() == ()


def test_a_job_that_never_finishes_is_given_up_after_the_job_timeout(tmp_path: Path) -> None:
    gate = threading.Event()
    built, _, checks, _ = service(tmp_path, checks=StandInChecks(gate=gate))

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        impatient = HttpGenerator(generator.url, poll_s=0.01, job_timeout_s=0.05)
        try:
            with pytest.raises(GeneratorError, match=r"check job \w+ is still running after [\d.]+ s"):
                await impatient.check(written.path)
        finally:
            gate.set()
        return None

    run_with(built, body)
    assert len(checks.calls) == 1


def test_a_play_runs_the_arm_with_its_files_and_comes_back_as_episodes(tmp_path: Path) -> None:
    built, _, _, plays = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        played = await generator.play(
            written.path,
            scenario="spade",
            arm="hint",
            plays=2,
            is_reporting=True,
            extra_instruction_files=("solution/hint.txt",),
            tags={"generation": "1"},
        )
        assert [play.reward for play in played] == [0.5, 0.5] and played[0].report_agent_record_ids == ("rep",)
        with pytest.raises(GeneratorError, match="extra instruction files do not exist"):
            await generator.play(
                written.path,
                scenario="spade",
                arm="hint",
                plays=1,
                is_reporting=False,
                extra_instruction_files=("solution/missing.txt",),
                tags={},
            )
        return None

    run_with(built, body)
    call = plays.calls[0]
    assert call["task"] == "harbor-00001-000-inspection" and call["arm"] == "hint" and call["plays"] == 2
    assert call["model"] == "served" and call["scenario"] == "spade" and call["tags"] == {"generation": "1"}
    assert call["extra"] == [tmp_path / "tasks" / "harbor-00001-000-inspection" / "solution" / "hint.txt"]


def test_a_manifest_splits_the_named_tasks_under_the_root(tmp_path: Path) -> None:
    built, _, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        names = []
        for index in range(3):
            proposed = await generator.propose(request(), scenario="spade", generation=2, index=index, tags={})
            assert proposed.task is not None
            names.append((await generator.write_task(proposed.task)).name)
        path = await generator.write_manifest(generation=2, names=names, eval_fraction=0.3, seed=1)
        assert path == tmp_path / "tasks" / "manifest-00002.json"
        manifest = read_split_manifest(path)
        assert sorted([*manifest.train, *manifest.eval]) == sorted(names) and len(manifest.eval) == 1
        with pytest.raises(GeneratorError, match="not a task directory"):
            await generator.write_manifest(generation=2, names=["missing"], eval_fraction=0.0, seed=1)
        return None

    run_with(built, body)


def test_a_proposal_report_reaches_the_designer(tmp_path: Path) -> None:
    built, designer, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        return await generator.report_proposal("designer-9", scenario="spade", score=0.25, metadata={"regret": 0.25})

    assert run_with(built, body) == "report-1"
    assert designer.reports == [
        {"record_id": "designer-9", "scenario": "spade", "score": 0.25, "metadata": {"regret": 0.25}}
    ]


@pytest.mark.parametrize(
    "failure",
    [
        DesignerError("the designer report was refused (404): no record designer-9"),
        ReefClientError(404, "no record designer-9"),
    ],
    ids=["designer_error", "client_error"],
)
def test_a_refused_proposal_report_answers_with_the_reason(tmp_path: Path, failure: Exception) -> None:
    built, designer, _, _ = service(tmp_path, designer=StandInDesigner(report_failure=failure))

    async def body(generator: HttpGenerator) -> object:
        with pytest.raises(GeneratorError, match=r"refused \(502\).*no record designer-9"):
            await generator.report_proposal("designer-9", scenario="spade", score=0.0, metadata={})
        return None

    run_with(built, body)
    assert designer.reports == []


def test_bad_requests_and_unknown_jobs_are_refused_with_a_reason(tmp_path: Path) -> None:
    built, _, _, _ = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        with pytest.raises(GeneratorError, match=r"refused \(400\).*target"):
            await generator.call("POST", "/proposals", body={"scenario": "s", "request": {}})
        with pytest.raises(GeneratorError, match=r"refused \(400\).*scenario"):
            await generator.call("POST", "/proposals", body={"request": {"target": "x"}})
        with pytest.raises(GeneratorError, match=r"refused \(404\).*no such job"):
            await generator.job_result({"job": "nope"})
        with pytest.raises(GeneratorError, match=r"refused \(400\).*not JSON"):
            await generator.call("POST", "/tasks", body=None)
        return None

    run_with(built, body)
    with pytest.raises(GeneratorError, match="did not reach the generator"):
        asyncio.run(HttpGenerator("http://127.0.0.1:9", timeout_s=1.0).delete_task("x"))


def test_healthz_is_503_naming_what_is_missing_and_200_once_every_probe_passes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harbor = StandInProbe("harbor", "the harbor command line is not on PATH")
    docker = StandInProbe("docker", "docker version exited 1: the daemon is down")
    probes = [StandInProbe("reef_eval"), harbor, docker]
    built, _, _, _ = service(tmp_path, probes=probes)

    async def body(generator: HttpGenerator) -> object:
        return [await generator.request("GET", "/healthz") for _ in range(2)]

    with caplog.at_level(logging.WARNING, logger="reef.record2dataset.service"):
        first, second = run_with(built, body)
    assert (
        first
        == second
        == (
            503,
            {
                "ok": False,
                "missing": ["harbor", "docker"],
                "reasons": {"harbor": harbor.reason, "docker": docker.reason},
                "tasks_root": str(tmp_path / "tasks"),
            },
        )
    )
    assert [probe.calls for probe in probes] == [2, 2, 2], "every probe runs on every call, so a fix is seen"
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [f"the generator is not ready: harbor: {harbor.reason}; docker: {docker.reason}"]
    harbor.reason = docker.reason = ""
    assert run_with(built, body) == [(200, {"ok": True, "tasks_root": str(tmp_path / "tasks")})] * 2
    assert not caplog.records[len(warnings) :], "a ready generator adds nothing to the log"


def test_the_stack_probes_look_for_reef_eval_harbor_and_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert [probe.name for probe in readiness_probes(harbor="/opt/harbor")] == ["reef_eval", "harbor", "docker"]
    assert ModuleProbe("json").missing() == ""
    reason = ModuleProbe("no_such_module_anywhere", hint="install it").missing()
    assert (
        reason.startswith(f"no_such_module_anywhere does not import under {sys.executable}") and "install it" in reason
    )
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert HarborProbe().missing() == "the harbor command line is not on PATH"
    assert HarborProbe(harbor=str(tmp_path / "harbor")).missing() == f"{tmp_path / 'harbor'} is not an executable"
    assert DockerProbe(docker="docker").missing() == "the docker command line is not on PATH"
    reason = DockerProbe(docker=sys.executable).missing()
    assert reason.startswith(f"{sys.executable} version exited 2:") and "version" in reason
    if os.name != "posix":
        return
    harbor = tmp_path / "harbor"
    harbor.write_text("#!/bin/sh\nexit 0\n")
    harbor.chmod(harbor.stat().st_mode | stat.S_IXUSR)
    assert HarborProbe(harbor=str(harbor)).missing() == ""
    docker = tmp_path / "docker"
    docker.write_text('#!/bin/sh\ntest "$1" = version || exit 3\nexit 0\n')
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    assert DockerProbe(docker=str(docker)).missing() == ""
    docker.write_text("#!/bin/sh\necho 'Cannot connect to the Docker daemon' >&2\nexit 1\n")
    assert (
        DockerProbe(docker=str(docker)).missing() == f"{docker} version exited 1: Cannot connect to the Docker daemon"
    )


def test_the_wire_forms_round_trip(tmp_path: Path) -> None:
    task = HarborTask(
        name="t",
        instruction="Do the thing described here in enough words to pass.",
        tests={"test.sh": "echo 1 > /logs/verifier/reward.txt"},
        environment={"Dockerfile": "FROM x\nRUN true\n"},
        config={"agent": {"timeout_sec": 30}},
        metadata={"generation": 1},
        source_agent_record_ids=("d1",),
    )
    assert task_from_document(task_document(task)) == task
    with pytest.raises(ValueError, match="not a Harbor task"):
        task_from_document({**task_document(task), "tests": {}})
    with pytest.raises(ValueError, match="must be a JSON object"):
        task_from_document("task")
    play = TaskPlay(tmp_path / "t", "t", "e1", None, {"score": 0.5}, "the trial raised", (), 2, (), None)
    assert play_from_document(play_document(play)) == play
    with pytest.raises(ValueError, match="failed_calls"):
        play_from_document({**play_document(play), "failed_calls": "2"})
    with pytest.raises(WireError, match="rewards must map names to numbers"):
        play_from_document({**play_document(play), "rewards": {"score": "high"}})
    with pytest.raises(WireError, match="rewards must map names to numbers"):
        play_from_document({**play_document(play), "rewards": {"score": True}})


def test_the_generator_section_is_parsed_in_either_spelling_and_unknown_fields_are_refused() -> None:
    settings = generator_settings({"tasks-root": "/tmp/t", "designer-timeout-s": 60, "agent": {"name": "terminus-2"}})
    assert settings.tasks_root == "/tmp/t" and settings.designer_timeout_s == 60.0 and settings.port == 8910
    assert generator_settings({"tasks_root": "/tmp/t"}).tasks_root == "/tmp/t"
    for section, message in (
        ({}, "tasks_root is required"),
        ({"tasks-root": "/tmp/t", "bogus": 1}, "unknown config fields: bogus"),
        ({"tasks-root": "/tmp/t", "concurrency": 0}, "concurrency must be at least 1"),
        ({"tasks-root": "/tmp/t", "agent": {"kwargs": {}}}, "Harbor agent name"),
        ({"tasks-root": "/tmp/t", "port": 0}, "port must be"),
        ({"tasks-root": "/tmp/t", "tasks_root": "/tmp/u"}, "twice"),
    ):
        with pytest.raises(ValueError, match=message):
            generator_settings(section)
    with pytest.raises(ValueError, match="must be an object"):
        generator_settings("tasks")  # type: ignore[arg-type]
