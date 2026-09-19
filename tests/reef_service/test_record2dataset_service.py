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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aiohttp.test_utils import TestServer
from reef_client.client import ReefClient, ReefClientError
from reef_service.test_record2dataset_designer import StandInHarness, served_tree

from reef.core.tasks import HarborTask, read_harbor_task, read_split_manifest, write_harbor_task
from reef.harness.client.tasks import TaskPlay, TaskPlayer
from reef.record2dataset import (
    Designer,
    DesignerAnswer,
    DesignerError,
    DesignerPrompt,
    DesignerRequest,
    DesignerTurn,
    DuplicateTask,
    FixedPrompt,
    GeneratorError,
    GeneratorService,
    HarborChecks,
    HarborRuns,
    HarnessPrompt,
    HttpGenerator,
    JobRunner,
    OracleResult,
    OracleUnavailable,
    ReadinessProbe,
    ReefTaskPlays,
    TaskChecks,
    TaskNameConflict,
    TaskPlays,
    readiness_probes,
)
from reef.record2dataset.designer import TREE_PATH
from reef.record2dataset.service import CLOSE_GRACE_S, DockerProbe, HarborProbe, ModuleProbe
from reef.record2dataset.wire import play_document, play_from_document, task_document, task_from_document
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

    def report(self, record_id, *, scenario, score, metadata, feedback=None) -> str:
        if self.report_failure is not None:
            raise self.report_failure
        self.reports.append(
            {
                "record_id": record_id,
                "scenario": scenario,
                "score": score,
                "metadata": dict(metadata),
                "feedback": feedback,
            }
        )
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
        self.reports: list[dict[str, object]] = []

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
                ("rep",) if is_reporting else (),
                None,
                {**tags, "arm": arm},
            )
            for n in range(plays)
        )

    def report(self, plays, *, scenario, model, score_of, metadata):
        self.reports.append(
            {
                "plays": tuple(plays),
                "scenario": scenario,
                "model": model,
                "scores": dict(score_of),
                "metadata": dict(metadata),
            }
        )
        return tuple(
            (
                replace(play, report_agent_record_ids=(f"rep-{play.episode_id}",))
                if play.episode_id in score_of and play.receipts
                else play
            )
            for play in plays
        )


class StandInReef:
    """A Reef service that keeps every report it gets; one that references ``rec-refused`` is refused."""

    def __init__(self) -> None:
        self.reports: list[dict[str, object]] = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {name.lower(): value for name, value in self.headers.items()}
                service.reports.append({"headers": headers, "body": body})
                if self.path != "/reef/report":
                    self.answer(404, {"error": self.path})
                elif "rec-refused" in body.get("references", []):
                    self.answer(400, {"error": "references must identify an existing inference"})
                else:
                    self.answer(200, {"agent_record_id": f"rep-{len(service.reports)}"})

            def answer(self, status: int, document: dict[str, object]) -> None:
                payload = json.dumps(document).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return None

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class CountingPlays(ReefTaskPlays):
    """The real task plays, keeping the labels of every player it builds."""

    def __init__(self, *, reef_url: str, work_dir: Path, token: str | None) -> None:
        super().__init__(reef_url=reef_url, work_dir=work_dir, token=token)
        self.built: list[dict[str, str]] = []

    def player(
        self,
        *,
        scenario: str,
        model: str,
        labels: Mapping[str, str],
        extra_instruction_paths: Sequence[Path],
        is_reporting: bool,
    ) -> TaskPlayer:
        self.built.append(dict(labels))
        return super().player(
            scenario=scenario,
            model=model,
            labels=labels,
            extra_instruction_paths=extra_instruction_paths,
            is_reporting=is_reporting,
        )


class StandInDesignerService(ReefClient):
    """A Reef client whose harness and status answers follow a script, one entry per call; the last entry repeats."""

    def __init__(
        self,
        *,
        releases: Sequence[str | ReefClientError | None] = (),
        load_ids: Sequence[str | tuple[str, int] | ReefClientError | None] = (),
    ) -> None:
        super().__init__("http://127.0.0.1:1", token="t")
        self.releases = list(releases)
        self.load_ids = list(load_ids)
        self.calls: list[dict[str, str]] = []

    def get(self, path: str, *, extra_headers: Mapping[str, str] | None = None) -> dict[str, object]:
        self.calls.append({"path": path, **dict(extra_headers or {})})
        script = self.releases if path == "/reef/harness" else self.load_ids
        entry = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(entry, ReefClientError):
            raise entry
        if path != "/reef/harness":
            if isinstance(entry, tuple):
                return {"scenarios": {"designer": {"current_runtime_load_id": entry[0], "scenario_step": entry[1]}}}
            return {"scenarios": {"designer": {"current_runtime_load_id": entry}}}
        if entry is None:
            raise ReefClientError(404, "no files")
        return served_tree(DesignerPrompt(system=f"System of {entry}.").entries(), entry)


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
        designer_scenario=parts.get("designer_scenario"),  # type: ignore[arg-type]
        prompts=parts.get("prompts"),  # type: ignore[arg-type]
        turn=parts.get("turn"),  # type: ignore[arg-type]
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


def test_a_service_with_a_designer_scenario_sends_the_designers_calls_and_reports_there(tmp_path: Path) -> None:
    built, designer, _, plays = service(tmp_path, designer_scenario="designer")

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        await generator.play(
            written.path,
            scenario="spade",
            arm="plain",
            plays=1,
            is_reporting=True,
            extra_instruction_files=(),
            tags={},
        )
        await generator.report_proposal(proposed.record_id, scenario="spade", score=0.5, metadata={})
        # The deployment owns the Designer's scenario, so a proposal needs none of its own.
        await generator.job_result(await generator.call("POST", "/proposals", body={"request": {"target": "x"}}))
        return None

    run_with(built, body)
    assert [call["scenario"] for call in designer.calls] == ["designer", "designer"]
    assert designer.reports[0]["scenario"] == "designer"
    assert plays.calls[0]["scenario"] == "spade", "the Designer's scenario is the Designer's alone"
    built, designer, _, _ = service(tmp_path / "without")

    async def body_without(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        await generator.report_proposal(proposed.record_id, scenario="spade", score=0.5, metadata={})
        return None

    run_with(built, body_without)
    assert designer.calls[0]["scenario"] == "spade" and designer.reports[0]["scenario"] == "spade"


def test_the_service_asks_the_designer_with_the_prompt_its_generation_gets(tmp_path: Path) -> None:
    evolved = DesignerPrompt(system="Evolved system.", rules="RULES:\n- {turn_limit} commands, keep {state}.")
    client = StandInHarness(served_tree(evolved.entries()))
    built, designer, _, _ = service(tmp_path, prompts=HarnessPrompt(client, "designer"))

    async def body(generator: HttpGenerator) -> object:
        for index in range(2):
            await generator.propose(request(), scenario="spade", generation=4, index=index, tags={})
        return None

    run_with(built, body)
    assert [call["messages"][0]["content"] for call in designer.calls] == ["Evolved system."] * 2
    assert "- 12 commands, keep {state}." in designer.calls[0]["messages"][1]["content"]
    assert client.pulls == [{"path": "/reef/harness", "x-reef-scenario": "designer"}], "one pull per generation"
    assert isinstance(service(tmp_path / "fixed")[0].prompts, FixedPrompt), "the fixed prompt unless a source is given"


def test_a_generation_waits_for_the_release_the_last_generations_reports_produced_before_it_pulls_the_prompt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # r1 answers generation 0's look and pull, the read its first report makes, and generation 1's first two polls;
    # r2 the third poll and the pull after it.
    client = StandInDesignerService(releases=["r1", "r1", "r1", "r1", "r1", "r2"])
    turn = DesignerTurn(client, is_harness_prompt=True, poll_s=0.001, wait_s=5.0)
    built, designer, _, _ = service(
        tmp_path, designer_scenario="designer", prompts=HarnessPrompt(client, "designer"), turn=turn
    )

    async def body(generator: HttpGenerator) -> object:
        first = await generator.propose(request(), scenario="spade", generation=0, index=0, tags={})
        await generator.propose(request(), scenario="spade", generation=0, index=1, tags={})
        await generator.report_proposal(first.record_id, scenario="spade", score=0.5, metadata={})
        await generator.report_proposal("never-proposed", scenario="spade", score=0.5, metadata={})
        for index in range(2):
            await generator.propose(request(), scenario="spade", generation=1, index=index, tags={})
        return None

    with caplog.at_level(logging.INFO, logger="reef.record2dataset.designer"):
        run_with(built, body)
    systems = [call["messages"][0]["content"] for call in designer.calls]
    assert (
        systems == ["System of r1."] * 2 + ["System of r2."] * 2
    ), "generation 1 asks with the release its wait ended on"
    assert (
        client.calls == [{"path": "/reef/harness", "x-reef-scenario": "designer"}] * 7
    ), "generation 0: one look, one pull, one read at its first report; generation 1: three polls, then the pull"
    assert turn.generation == 1 and turn.version == "r2"
    assert turn.report_counts == {0: 1}, "a report for a record the service never proposed counts for no generation"
    messages = [record.getMessage() for record in caplog.records]
    assert "generation 0 asks the Designer at version r1" in messages
    assert (
        "generation 1 waits at Designer version r1 for the deployment to take generation 0 in (1 reported)" in messages
    )
    assert any(message.startswith("generation 1 asks the Designer at version r2 after") for message in messages)
    assert service(tmp_path / "without")[0].turn is None, "no turn unless the deployment gives one"


def test_a_release_that_appeared_before_the_reports_went_out_does_not_end_the_wait(tmp_path: Path) -> None:
    # Generation 0 looks and pulls before the scenario exists (404, the fixed prompt); its first call creates the
    # scenario and the creation release c1; its first report reads c1; generation 1 must wait for t1, the rewrite.
    client = StandInDesignerService(releases=[None, None, "c1", "c1", "t1"])
    turn = DesignerTurn(client, is_harness_prompt=True, poll_s=0.001, wait_s=5.0)
    built, designer, _, _ = service(
        tmp_path, designer_scenario="designer", prompts=HarnessPrompt(client, "designer"), turn=turn
    )

    async def body(generator: HttpGenerator) -> object:
        first = await generator.propose(request(), scenario="spade", generation=0, index=0, tags={})
        await generator.report_proposal(first.record_id, scenario="spade", score=-1.0, metadata={})
        await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        return None

    run_with(built, body)
    systems = [call["messages"][0]["content"] for call in designer.calls]
    assert systems[0] != "System of c1." and systems[1] == "System of t1.", "generation 1 asks with the rewrite"
    assert turn.reported_versions == {0: "c1"} and turn.version == "t1"


def test_a_generation_waits_for_the_designers_runtime_load_id_to_change(caplog: pytest.LogCaptureFixture) -> None:
    client = StandInDesignerService(load_ids=["load-1", "load-1", ReefClientError(503, "busy"), "load-2"])
    turn = DesignerTurn(client, poll_s=0.001, wait_s=5.0)
    turn.begin(0, "designer")
    turn.begin(0, "designer")
    assert turn.version == "load-1" and client.calls == [{"path": "/reef/status"}], "one look per generation"
    turn.proposed(0, "designer-1")
    turn.reported("designer-1", "designer")
    turn.reported("designer-1", "designer")
    with caplog.at_level(logging.INFO, logger="reef.record2dataset.designer"):
        turn.begin(1, "designer")
    assert turn.generation == 1 and turn.version == "load-2" and len(client.calls) == 4
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0] == (
        "generation 1 waits at Designer version load-1 for the deployment to take generation 0 in (2 reported)"
    )
    assert messages[1] == "the Designer's version could not be read (503): busy; the wait goes on"
    assert messages[2].startswith("generation 1 asks the Designer at version load-2 after") and len(messages) == 3


def test_a_weight_designers_skipped_step_moves_its_version() -> None:
    client = StandInDesignerService(load_ids=[("load-1", 1), ("load-1", 1), ("load-1", 1), ("load-1", 2)])
    turn = DesignerTurn(client, poll_s=0.001, wait_s=5.0)
    turn.begin(0, "designer")
    turn.proposed(0, "designer-1")
    turn.reported("designer-1", "designer")
    turn.begin(1, "designer")
    assert turn.version == "load-1@2" and turn.reported_versions == {0: "load-1@1"} and len(client.calls) == 4


def test_a_fixed_designer_is_looked_at_once_per_generation_and_never_waited_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    for client, is_harness_prompt in (
        (StandInDesignerService(load_ids=[None]), False),
        (StandInDesignerService(releases=[None]), True),
        (StandInDesignerService(load_ids=[ReefClientError(500, "down")]), False),
    ):
        turn = DesignerTurn(client, is_harness_prompt=is_harness_prompt, poll_s=0.001, wait_s=5.0)
        with caplog.at_level(logging.WARNING, logger="reef.record2dataset.designer"):
            turn.begin(0, "designer")
        turn.proposed(0, "designer-1")
        turn.reported("designer-1", "designer")
        started = time.monotonic()
        turn.begin(1, "designer")
        assert turn.version is None and len(client.calls) == 3 and time.monotonic() - started < 1.0
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings == [
        "the Designer's version could not be read (500): down; generation 0 proceeds without one",
        "the Designer's version could not be read (500): down; generation 0's reports are timed at its start version",
        "the Designer's version could not be read (500): down; generation 1 proceeds without one",
    ]


def test_a_designer_that_never_moves_is_waited_on_up_to_the_limit_and_the_generation_proceeds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = StandInDesignerService(load_ids=["load-1"])
    turn = DesignerTurn(client, poll_s=0.002, wait_s=0.02)
    turn.begin(0, "designer")
    turn.proposed(0, "designer-1")
    turn.reported("designer-1", "designer")
    with caplog.at_level(logging.WARNING, logger="reef.record2dataset.designer"):
        started = time.monotonic()
        turn.begin(1, "designer")
    elapsed = time.monotonic() - started
    assert turn.generation == 1 and turn.version == "load-1" and 0.02 <= elapsed < 1.0
    assert len(client.calls) >= 3, "the wait polled more than once before it ran out"
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].endswith("; generation 1 proceeds")
    assert warnings[0].startswith("the Designer's deployment did not move past version load-1 within")


def test_a_generation_after_one_that_sent_no_report_does_not_wait(caplog: pytest.LogCaptureFixture) -> None:
    client = StandInDesignerService(load_ids=["load-1"])
    turn = DesignerTurn(client, poll_s=0.001, wait_s=5.0)
    turn.begin(0, "designer")
    turn.proposed(0, "designer-1")
    with caplog.at_level(logging.INFO, logger="reef.record2dataset.designer"):
        turn.begin(1, "designer")
    assert turn.version == "load-1" and len(client.calls) == 2
    assert [record.getMessage() for record in caplog.records] == ["generation 1 asks the Designer at version load-1"]
    with pytest.raises(DesignerError, match="poll_s must be a positive number"):
        DesignerTurn(client, poll_s=0)
    with pytest.raises(DesignerError, match="wait_s must be a positive number"):
        DesignerTurn(client, wait_s=-1.0)


def test_a_served_tree_the_prompt_cannot_read_fails_the_proposal_naming_it(tmp_path: Path) -> None:
    client = StandInHarness({"release_id": "r1", "files": {TREE_PATH: "{not json"}})
    built, designer, _, _ = service(tmp_path, prompts=HarnessPrompt(client, "designer"))

    async def body(generator: HttpGenerator) -> object:
        with pytest.raises(
            GeneratorError, match=r"DesignerError: native/tree\.json of scenario 'designer' is not JSON"
        ):
            await generator.propose(request(), scenario="spade", generation=0, index=0, tags={})
        return None

    run_with(built, body)
    assert designer.calls == [], "no call with a prompt nobody chose"


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


def test_held_plays_are_reported_through_the_service_with_their_scores(tmp_path: Path) -> None:
    built, _, _, plays = service(tmp_path)

    async def body(generator: HttpGenerator) -> object:
        proposed = await generator.propose(request(), scenario="spade", generation=1, index=0, tags={})
        assert proposed.task is not None
        written = await generator.write_task(proposed.task)
        held = await generator.play(
            written.path,
            scenario="spade",
            arm="plain",
            plays=2,
            is_reporting=False,
            extra_instruction_files=(),
            tags={"generation": "1"},
        )
        assert [play.is_reported for play in held] == [False, False]
        assert held[0].labels == {"generation": "1", "arm": "plain"}, "the labels travel back with the play"
        reported = await generator.report_plays(
            held,
            scenario="spade",
            score_of={held[0].episode_id: 1.0},
            metadata={"round": "generation-00001", "round_plays": 1},
            model="m",
        )
        assert [play.report_agent_record_ids for play in reported] == [("rep-e0",), ()]
        assert reported[0].labels == held[0].labels and reported[0].receipts == ("rec",)
        with pytest.raises(GeneratorError, match=r"refused \(400\).*scores must map"):
            await generator.call(
                "POST", "/plays/report", body={"scenario": "spade", "plays": [], "scores": {"e": "1"}}
            )
        with pytest.raises(GeneratorError, match=r"refused \(400\).*labels"):
            await generator.call(
                "POST",
                "/plays/report",
                body={"scenario": "spade", "plays": [{**play_document(held[0]), "labels": {"arm": 1}}]},
            )
        with pytest.raises(GeneratorError, match=r"refused \(400\).*plays must be a list"):
            await generator.call("POST", "/plays/report", body={"scenario": "spade", "plays": {}})
        return None

    run_with(built, body)
    call = plays.reports[0]
    assert call["scenario"] == "spade" and call["model"] == "m" and call["scores"] == {"e0": 1.0}
    assert call["metadata"] == {"round": "generation-00001", "round_plays": 1}
    assert [play.episode_id for play in call["plays"]] == ["e0", "e1"]
    assert call["plays"][0].receipts == ("rec",) and call["plays"][0].labels == {"generation": "1", "arm": "plain"}


def test_reef_task_plays_report_held_plays_with_one_player_per_label_set(tmp_path: Path) -> None:
    task_path = write_harbor_task(
        HarborTask(
            name="t-held",
            instruction="List the files in the working directory and write their count to /app/count.txt.",
            tests={"test.sh": "#!/bin/sh\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n"},
            environment={"Dockerfile": "FROM python:3.12-slim\nWORKDIR /app\n"},
        ),
        tmp_path / "tasks",
    )

    def held(episode_id: str, labels: dict[str, str], receipts: tuple[str, ...] = ("rec-1",)) -> TaskPlay:
        return TaskPlay(task_path, "t-held", episode_id, 1.0, {"reward": 1.0}, "", receipts, 0, (), "trials/x", labels)

    plain = {"arm": "plain", "generation": "1"}
    hint = {"arm": "hint", "generation": "1"}
    reef = StandInReef()
    try:
        plays = CountingPlays(reef_url=reef.url, work_dir=tmp_path / "play", token="tok")
        built, _, _, _ = service(tmp_path, plays=plays)

        async def body(generator: HttpGenerator) -> object:
            return await generator.report_plays(
                [
                    held("e1", plain),
                    held("e2", plain, ("rec-2", "rec-3")),
                    held("e3", hint),
                    held("e4", {"arm": "plain"}),
                    held("e5", plain, ()),
                    held("e6", plain, ("rec-refused",)),
                    held("e7", plain),
                ],
                scenario="spade",
                score_of={"e1": 1.0, "e2": 0.0, "e3": 0.5, "e4": 1.0, "e5": 1.0, "e6": 1.0},
                metadata={"round": "generation-00001", "round_plays": 2, "task": "never this"},
            )

        reported = run_with(built, body)
    finally:
        reef.close()
    assert isinstance(reported, tuple)
    assert [play.report_agent_record_ids for play in reported] == [
        ("rep-1",),
        ("rep-2",),
        ("rep-3",),
        ("rep-4",),
        (),
        (),
        (),
    ]
    assert reported[4].error == "" and reported[6].error == "", "no receipt or no score: unreported, no failure"
    assert reported[5].error.startswith("the report for t-held was refused (400)")
    assert plays.built == [plain, hint, {"arm": "plain"}], "one player per distinct label set"
    bodies = [report["body"] for report in reef.reports]
    assert [body["score"] for body in bodies] == [1.0, 0.0, 0.5, 1.0, 1.0]
    assert [body["references"] for body in bodies] == [
        ["rec-1"],
        ["rec-2", "rec-3"],
        ["rec-1"],
        ["rec-1"],
        ["rec-refused"],
    ]
    assert [body["metadata"]["episode"]["labels"] for body in bodies] == [plain, plain, hint, {"arm": "plain"}, plain]
    first = bodies[0]
    assert first["metadata"]["round"] == "generation-00001" and first["metadata"]["round_plays"] == 2
    assert first["metadata"]["task"]["name"] == "t-held", "extra metadata never replaces the task"
    assert first["metadata"]["episode"]["id"] == "e1" and first["metadata"]["episode"]["trial_uri"] == "trials/x"
    assert first["feedback"] == "verifier reward 1.0 on t-held"
    headers = reef.reports[0]["headers"]
    assert headers["x-reef-scenario"] == "spade" and headers["authorization"] == "Bearer tok"


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
        first = await generator.report_proposal("designer-9", scenario="spade", score=0.25, metadata={"regret": 0.25})
        await generator.report_proposal(
            "designer-9",
            scenario="spade",
            score=-1.0,
            metadata={"refusal": "no json"},
            feedback={"task": None, "round": {"generation": 2, "previous": None}},
        )
        with pytest.raises(GeneratorError, match=r"refused \(400\).*feedback must be a string or an object"):
            await generator.call(
                "POST", "/proposals/designer-9/report", body={"scenario": "spade", "score": 0.0, "feedback": 3}
            )
        return first

    assert run_with(built, body) == "report-1"
    assert designer.reports == [
        {
            "record_id": "designer-9",
            "scenario": "spade",
            "score": 0.25,
            "metadata": {"regret": 0.25},
            "feedback": None,
        },
        {
            "record_id": "designer-9",
            "scenario": "spade",
            "score": -1.0,
            "metadata": {"refusal": "no json"},
            "feedback": {"task": None, "round": {"generation": 2, "previous": None}},
        },
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
    play = TaskPlay(tmp_path / "t", "t", "e1", None, {}, "the trial raised", (), 2, (), None)
    assert play_from_document(play_document(play)) == play
    with pytest.raises(ValueError, match="failed_calls"):
        play_from_document({**play_document(play), "failed_calls": "2"})
    labelled = TaskPlay(tmp_path / "t", "t", "e2", 1.0, {"reward": 1.0}, "", ("r1",), 0, (), None, {"arm": "plain"})
    assert play_document(labelled)["labels"] == {"arm": "plain"}
    assert play_from_document(play_document(labelled)) == labelled
    assert play_from_document(play_document(play)).labels == {}
    with pytest.raises(ValueError, match="labels must map names to text"):
        play_from_document({**play_document(play), "labels": {"arm": 1}})


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
        ({"tasks-root": "/tmp/t", "designer-scenario": " "}, "designer-scenario must name a scenario"),
    ):
        with pytest.raises(ValueError, match=message):
            generator_settings(section)
    with pytest.raises(ValueError, match="must be an object"):
        generator_settings("tasks")  # type: ignore[arg-type]
