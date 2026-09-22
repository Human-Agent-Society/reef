"""The generator service: the slow, container bound steps of task generation behind a small HTTP API.

A processor that generates tasks runs inside the Reef service and must not block the trainer; it also
must not need Docker, the ``harbor`` command line or reef-eval in that process. So the steps that need
them run here, in a process ``reef serve`` starts beside the HTTP service (see ``reef.service.deploy``):

- ``POST /proposals``: ask the designer's model for one task (a job): the service's designer model when it
  has one, else the model the request names; the reply parsed and held to the task contract, the
  designer's record id beside the task or the refusal. A generation's first proposal waits for the
  Designer's deployment to take the last generation's reports in (``DesignerTurn``) before the prompt pull.
- ``POST /proposals/{record_id}/report``: report a score against the designer's receipt, with the method's
  metadata and feedback (text or an object) passed through to the report.
- Both go to the service's designer scenario when the deployment names one, else to the scenario the
  request names, so a Designer on its own Reef service keeps its own scenario.
- ``POST /tasks``: write a task under the tasks root, refusing a duplicate (by content hash) or a
  different task under a taken name; ``DELETE /tasks/{name}`` removes one.
- ``POST /checks``: Harbor's oracle and nop agents on a written task (a job).
- ``POST /plays``: an agent plays a written task through the task player, episodes reported to Reef (a job).
- ``POST /plays/report``: held episodes reported after the fact, each against its receipts with the score
  the caller names (a direct call, so it never waits behind a play in flight).
- ``POST /manifests``: the split manifest of one generation's tasks under the tasks root.
- ``GET /jobs/{id}``: a job's state and result.
- ``GET /healthz``: ok once reef-eval imports, the harbor command line resolves and Docker answers; until
  then 503 naming what is missing, so ``reef serve`` waits on a generator that cannot check or play.

Jobs run one after another on a private thread; closing the service stops the harbor runs a job has in
flight, so ``reef serve`` leaves no container behind. Everything here is stateless beyond the tasks
root: a restarted service serves the same root.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import queue
import shutil
import subprocess
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from aiohttp import web
from reef_client.client import ReefClientError

from reef.core.tasks import (
    HarborTaskConflict,
    HarborTaskError,
    read_harbor_task,
    write_harbor_task,
    write_split_manifest,
)
from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.harness.client.tasks import TaskPlay, TaskPlayer, TaskPlayError
from reef.record2dataset.designer import (
    Designer,
    DesignerError,
    DesignerReplyError,
    DesignerRequest,
    DesignerTurn,
    FixedPrompt,
    PromptSource,
    designer_messages,
    parse_harbor_reply,
)
from reef.record2dataset.harbor import (
    GeneratedHarborTask,
    HarborRuns,
    OracleResult,
    content_hash,
    harbor_task,
    oracle_check,
    split_generation,
)
from reef.record2dataset.wire import (
    WireError,
    checked_object,
    checked_string,
    oracle_document,
    play_document,
    play_from_document,
    task_document,
    task_from_document,
)

logger = logging.getLogger(__name__)

CLOSE_JOIN_S = 10.0
DOCKER_PROBE_TIMEOUT_S = 5.0


# ------------------------------------------------------------- the readiness


class ReadinessProbe(ABC):
    """One thing the generator needs before it can check and play tasks."""

    name: str = ""

    @abstractmethod
    def missing(self) -> str:
        """Why the need is not met, or an empty string when it is."""


class ModuleProbe(ReadinessProbe):
    """A module imports under this interpreter; ``find_spec`` looks without importing."""

    def __init__(self, module: str, *, hint: str = "") -> None:
        self.name = module
        self.module = module
        self.hint = hint

    def missing(self) -> str:
        if importlib.util.find_spec(self.module) is not None:
            return ""
        reason = f"{self.module} does not import under {sys.executable}"
        return f"{reason}; {self.hint}" if self.hint else reason


class HarborProbe(ReadinessProbe):
    """The harbor command line resolves: the configured path, or ``harbor`` on PATH."""

    name = "harbor"

    def __init__(self, *, harbor: str | None = None) -> None:
        self.harbor = harbor

    def missing(self) -> str:
        if self.harbor is None:
            return "" if shutil.which("harbor") is not None else "the harbor command line is not on PATH"
        return "" if shutil.which(self.harbor) is not None else f"{self.harbor} is not an executable"


class DockerProbe(ReadinessProbe):
    """The Docker daemon answers ``docker version``, so Harbor can build and run a task's container."""

    name = "docker"

    def __init__(self, *, docker: str = "docker", timeout_s: float = DOCKER_PROBE_TIMEOUT_S) -> None:
        self.docker = docker
        self.timeout_s = timeout_s

    def missing(self) -> str:
        if shutil.which(self.docker) is None:
            return f"the {self.docker} command line is not on PATH"
        try:
            completed = subprocess.run(
                [self.docker, "version"], capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"{self.docker} version did not answer: {exc}"
        if completed.returncode != 0:
            return f"{self.docker} version exited {completed.returncode}: {completed.stderr.strip()[:500]}"
        return ""


def readiness_probes(*, harbor: str | None = None) -> tuple[ReadinessProbe, ...]:
    """What playing and checking a task takes: reef-eval, the harbor command line and Docker."""
    return (
        ModuleProbe("reef_eval", hint="a dependency of reef-infra; reinstall reef-infra"),
        HarborProbe(harbor=harbor),
        DockerProbe(),
    )


# reef serve kills the generator 30 s after its term; the harbor grace and the join must both fit in that window.
CLOSE_GRACE_S = 10.0


# ------------------------------------------------------------------ the work


class TaskChecks(ABC):
    """The check before a written task is played: Harbor's oracle and nop agents through the harbor command line."""

    @abstractmethod
    def oracle(self, task_path: Path) -> OracleResult: ...


class HarborChecks(TaskChecks):
    """The checks through the harbor command line; ``runs`` shared with the job runner so closing it stops a check."""

    def __init__(self, *, harbor: str | None = None, runs: HarborRuns | None = None) -> None:
        self.harbor = harbor
        self.runs = runs

    def oracle(self, task_path: Path) -> OracleResult:
        return oracle_check(task_path, harbor=self.harbor, runs=self.runs)


class TaskPlays(ABC):
    """Who plays a written task for one arm: the served model through a Harbor agent and the task player."""

    @abstractmethod
    def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        model: str,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_paths: Sequence[Path],
        tags: Mapping[str, str],
    ) -> tuple[TaskPlay, ...]: ...

    @abstractmethod
    def report(
        self,
        plays: Sequence[TaskPlay],
        *,
        scenario: str,
        model: str,
        score_of: Mapping[str, float],
        metadata: Mapping[str, object],
    ) -> tuple[TaskPlay, ...]:
        """Report held episodes after the fact, each with the score ``score_of`` names by episode id; the plays with their report ids."""


class ReefTaskPlays(TaskPlays):
    """A task player per call, its episodes tagged with the arm; a held play is reported by a player with its labels."""

    def __init__(
        self,
        *,
        reef_url: str,
        work_dir: Path,
        token: str | None = None,
        agent: Mapping[str, object] | None = None,
        agent_host: str | None = None,
        concurrency: int = 2,
    ) -> None:
        self.reef_url = reef_url
        self.work_dir = Path(work_dir)
        self.token = token
        self.agent = agent
        self.agent_host = agent_host
        self.concurrency = concurrency

    def play(
        self,
        task_path: Path,
        *,
        scenario: str,
        model: str,
        arm: str,
        plays: int,
        is_reporting: bool,
        extra_instruction_paths: Sequence[Path],
        tags: Mapping[str, str],
    ) -> tuple[TaskPlay, ...]:
        if plays < 1:
            return ()
        player = self.player(
            scenario=scenario,
            model=model,
            labels={**tags, "arm": arm},
            extra_instruction_paths=extra_instruction_paths,
            is_reporting=is_reporting,
        )
        return player.play_concurrently([task_path] * plays, concurrency=min(self.concurrency, plays))

    def report(
        self,
        plays: Sequence[TaskPlay],
        *,
        scenario: str,
        model: str,
        score_of: Mapping[str, float],
        metadata: Mapping[str, object],
    ) -> tuple[TaskPlay, ...]:
        # One player per distinct label set: the labels a play carries are the ones its episode block names.
        players: dict[tuple[tuple[str, str], ...], TaskPlayer] = {}
        reported = []
        for play in plays:
            score = score_of.get(play.episode_id)
            if score is None or not play.receipts:
                reported.append(play)
                continue
            key = tuple(sorted(play.labels.items()))
            if key not in players:
                players[key] = self.player(
                    scenario=scenario, model=model, labels=play.labels, extra_instruction_paths=(), is_reporting=True
                )
            try:
                report_ids = players[key].report_play(play, score=score, metadata=metadata)
            except TaskPlayError as exc:
                # A refused report is this play's failure; the others still go out.
                reported.append(replace(play, error=str(exc)))
                continue
            reported.append(replace(play, report_agent_record_ids=report_ids))
        return tuple(reported)

    def player(
        self,
        *,
        scenario: str,
        model: str,
        labels: Mapping[str, str],
        extra_instruction_paths: Sequence[Path],
        is_reporting: bool,
    ) -> TaskPlayer:
        return TaskPlayer(
            reef_url=self.reef_url,
            scenario=scenario,
            model=model,
            work_dir=self.work_dir,
            token=self.token,
            agent=self.agent,
            agent_host=self.agent_host,
            labels=labels,
            extra_instruction_paths=extra_instruction_paths,
            is_reporting=is_reporting,
        )


# ------------------------------------------------------------------ the jobs


class Job(ABC):
    """One unit of slow work: queued, running, then done with a result or failed with an error."""

    kind: str = ""

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.state = "queued"
        self.result: dict[str, object] | None = None
        self.error = ""

    @abstractmethod
    def run(self) -> dict[str, object]: ...

    def document(self) -> dict[str, object]:
        return {"id": self.id, "kind": self.kind, "state": self.state, "result": self.result, "error": self.error}


class ProposalJob(Job):
    """One designer call with the prompt its generation gets; the reply parsed and held to the task contract."""

    kind = "proposal"

    def __init__(
        self,
        designer: Designer,
        request: DesignerRequest,
        *,
        prompts: PromptSource,
        scenario: str,
        model: str,
        generation: int,
        index: int,
        tags: Mapping[str, str],
        turn: DesignerTurn | None = None,
    ) -> None:
        super().__init__()
        self.designer = designer
        self.request = request
        self.prompts = prompts
        self.scenario = scenario
        self.model = model
        self.generation = generation
        self.index = index
        self.tags = dict(tags)
        self.turn = turn

    def run(self) -> dict[str, object]:
        # On the job thread: the turn's wait and a harness source's pull go over HTTP, which the event loop must not wait on.
        if self.turn is not None:
            self.turn.begin(self.generation, self.scenario)
        prompt = self.prompts.prompt(self.generation)
        answer = self.designer.answer(
            designer_messages(self.request, prompt), scenario=self.scenario, model=self.model, tags=self.tags
        )
        if self.turn is not None:
            self.turn.proposed(self.generation, answer.record_id)
        try:
            reply = parse_harbor_reply(answer.text)
            task = harbor_task(
                GeneratedHarborTask(
                    reply=reply,
                    generation=self.generation,
                    index=self.index,
                    source_record_id=answer.record_id,
                    skill=self.request.skill,
                    difficulty=self.request.difficulty,
                )
            )
        except (DesignerReplyError, ValueError) as exc:
            return {"record_id": answer.record_id, "task": None, "refusal": f"reply refused: {exc}"}
        return {"record_id": answer.record_id, "task": task_document(task), "refusal": ""}


class CheckJob(Job):
    kind = "check"

    def __init__(self, checks: TaskChecks, task_path: Path) -> None:
        super().__init__()
        self.checks = checks
        self.task_path = task_path

    def run(self) -> dict[str, object]:
        return oracle_document(self.checks.oracle(self.task_path))


class PlayJob(Job):
    kind = "play"

    def __init__(
        self,
        plays: TaskPlays,
        task_path: Path,
        *,
        scenario: str,
        model: str,
        arm: str,
        count: int,
        is_reporting: bool,
        extra_instruction_paths: tuple[Path, ...],
        tags: Mapping[str, str],
    ) -> None:
        super().__init__()
        self.plays = plays
        self.task_path = task_path
        self.scenario = scenario
        self.model = model
        self.arm = arm
        self.count = count
        self.is_reporting = is_reporting
        self.extra_instruction_paths = extra_instruction_paths
        self.tags = dict(tags)

    def run(self) -> dict[str, object]:
        played = self.plays.play(
            self.task_path,
            scenario=self.scenario,
            model=self.model,
            arm=self.arm,
            plays=self.count,
            is_reporting=self.is_reporting,
            extra_instruction_paths=self.extra_instruction_paths,
            tags=self.tags,
        )
        return {"plays": [play_document(play) for play in played]}


class JobRunner:
    """Runs jobs one after another on a private thread; closing stops the harbor runs in flight, then the thread."""

    def __init__(self, runs: HarborRuns | None = None) -> None:
        self.runs = runs if runs is not None else HarborRuns()
        self._queue: queue.SimpleQueue[Job | None] = queue.SimpleQueue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, job: Job) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("the job runner is closed")
            self._jobs[job.id] = job
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="record2dataset-jobs", daemon=True)
                self._thread.start()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self.runs.terminate_all(grace_s=CLOSE_GRACE_S)
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=CLOSE_JOIN_S)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            job.state = "running"
            try:
                job.result = job.run()
            except Exception as exc:  # the job's own failure is its result; the runner goes on
                logger.exception("%s job %s failed", job.kind, job.id)
                job.error = f"{type(exc).__name__}: {exc}"[:1000]
                job.state = "failed"
            else:
                job.state = "done"


# --------------------------------------------------------------- the service


def error_response(status: int, message: str, **fields: object) -> web.Response:
    return web.json_response({"error": message, **fields}, status=status)


def checked_tags(value: object) -> dict[str, str]:
    if value is None:
        return {}
    tags = checked_object(value, "tags")
    if any(not isinstance(item, str) for item in tags.values()):
        raise WireError("tags must map names to strings")
    return {name: str(item) for name, item in tags.items()}


def checked_count(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WireError(f"{label} must be an integer of at least {minimum}")
    return value


def checked_scores(value: object) -> dict[str, float]:
    scores: dict[str, float] = {}
    for episode_id, score in checked_object(value if value is not None else {}, "scores").items():
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise WireError("scores must map episode ids to numbers")
        scores[episode_id] = float(score)
    return scores


class GeneratorService:
    """Writes, checks and plays Harbor tasks under one tasks root, for whoever asks over HTTP."""

    def __init__(
        self,
        *,
        tasks_root: Path,
        designer: Designer,
        checks: TaskChecks,
        plays: TaskPlays,
        default_model: str | None = None,
        designer_model: str | None = None,
        designer_scenario: str | None = None,
        prompts: PromptSource | None = None,
        turn: DesignerTurn | None = None,
        jobs: JobRunner | None = None,
        probes: Sequence[ReadinessProbe] | None = None,
    ) -> None:
        self.tasks_root = Path(tasks_root)
        self.designer = designer
        self.checks = checks
        self.plays = plays
        self.default_model = default_model
        # The deployment's generator section owns the Designer's model and scenario; the request's own are the fallback.
        self.designer_model = designer_model
        self.designer_scenario = designer_scenario
        self.prompts = prompts if prompts is not None else FixedPrompt()
        self.turn = turn
        self.jobs = jobs if jobs is not None else JobRunner()
        self.probes = tuple(probes) if probes is not None else readiness_probes()
        self.reported_missing: tuple[str, ...] = ()

    def app(self) -> web.Application:
        app = web.Application()
        app.add_routes(
            [
                web.get("/healthz", self.healthz),
                web.post("/proposals", self.propose),
                web.post("/proposals/{record_id}/report", self.report_proposal),
                web.post("/tasks", self.write_task),
                web.delete("/tasks/{name}", self.delete_task),
                web.post("/checks", self.check),
                web.post("/plays", self.play),
                web.post("/plays/report", self.report_plays),
                web.post("/manifests", self.write_manifest),
                web.get("/jobs/{job_id}", self.job),
            ]
        )
        app.on_cleanup.append(self.close)
        return app

    async def close(self, app: web.Application) -> None:
        self.jobs.close()

    # ------------------------------------------------------------- helpers

    def known_hashes(self) -> set[str]:
        """The content hashes of the tasks under the root, so one is never written twice."""
        hashes: set[str] = set()
        if not self.tasks_root.is_dir():
            return hashes
        for entry in sorted(self.tasks_root.iterdir()):
            if entry.is_dir() and (entry / "task.toml").is_file():
                try:
                    hashes.add(content_hash(read_harbor_task(entry)))
                except HarborTaskError:
                    continue
        return hashes

    def task_path_for(self, value: object) -> Path:
        """A task directory under the root, named by path or by name; anything else is refused."""
        if not isinstance(value, str) or not value:
            raise WireError("path must name a task directory under the tasks root")
        root = self.tasks_root.resolve()
        path = (root / value).resolve() if "/" not in value else Path(value).resolve()
        if path.parent != root or not (path / "task.toml").is_file():
            raise WireError(f"{value} is not a task directory under {self.tasks_root}")
        return path

    def model_for(self, body: Mapping[str, object]) -> str:
        model = body.get("model", self.default_model)
        if not isinstance(model, str) or not model:
            raise WireError("model must name the served model; the service has no default")
        return model

    async def body_of(self, request: web.Request) -> dict[str, object]:
        try:
            return checked_object(await request.json(), "the request body")
        except json.JSONDecodeError as exc:
            raise WireError(f"the request body is not JSON: {exc}") from exc

    def submitted(self, job: Job) -> web.Response:
        self.jobs.submit(job)
        return web.json_response({"job": job.id}, status=202)

    def missing_needs(self) -> dict[str, str]:
        """Every probe's reason for not being met, by name; empty when the generator can check and play."""
        reasons: dict[str, str] = {}
        for probe in self.probes:
            reason = probe.missing()
            if reason:
                reasons[probe.name] = reason
        return reasons

    # ------------------------------------------------------------ handlers

    async def healthz(self, request: web.Request) -> web.Response:
        reasons = await asyncio.to_thread(self.missing_needs)
        missing = tuple(reasons)
        # The readiness wait probes several times a second; the reasons go to the log once per change.
        if missing and missing != self.reported_missing:
            logger.warning(
                "the generator is not ready: %s", "; ".join(f"{name}: {reason}" for name, reason in reasons.items())
            )
        self.reported_missing = missing
        document: dict[str, object] = {"ok": not missing, "tasks_root": str(self.tasks_root)}
        if missing:
            return web.json_response({**document, "missing": list(missing), "reasons": reasons}, status=503)
        return web.json_response(document)

    async def propose(self, request: web.Request) -> web.Response:
        try:
            body = await self.body_of(request)
            fields = checked_object(body.get("request"), "request")
            skill = fields.get("skill")
            grounding = fields.get("grounding")
            designer_request = DesignerRequest(
                target=checked_string(fields, "target", label="request"),
                skill=skill if isinstance(skill, str) else None,
                difficulty=str(fields.get("difficulty", "medium")),
                turn_limit=checked_count(
                    fields.get("turn_limit", DesignerRequest.turn_limit), "turn_limit", minimum=2
                ),
                grounding=grounding if isinstance(grounding, str) else None,
                experience_text=str(fields.get("experience_text", "")),
            )
            job = ProposalJob(
                self.designer,
                designer_request,
                prompts=self.prompts,
                scenario=self.designer_scenario or checked_string(body, "scenario", label="a proposal"),
                model=self.designer_model or self.model_for(body),
                generation=checked_count(body.get("generation", 0), "generation"),
                index=checked_count(body.get("index", 0), "index"),
                tags=checked_tags(body.get("tags")),
                turn=self.turn,
            )
        except (WireError, ValueError) as exc:
            return error_response(400, str(exc))
        return self.submitted(job)

    async def report_proposal(self, request: web.Request) -> web.Response:
        record_id = request.match_info["record_id"]
        try:
            body = await self.body_of(request)
            scenario = self.designer_scenario or checked_string(body, "scenario", label="a report")
            score = body.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise WireError("score must be a number")
            metadata = checked_object(body.get("metadata", {}), "metadata")
            feedback = body.get("feedback")
            if feedback is not None and not isinstance(feedback, (str, Mapping)):
                raise WireError("feedback must be a string or an object")
        except WireError as exc:
            return error_response(400, str(exc))
        try:
            report_id = await asyncio.to_thread(
                self.designer.report,
                record_id,
                scenario=scenario,
                score=float(score),
                metadata=metadata,
                feedback=dict(feedback) if isinstance(feedback, Mapping) else feedback,
            )
        except (DesignerError, ReefClientError, OSError) as exc:
            return error_response(502, str(exc))
        if self.turn is not None:
            await asyncio.to_thread(self.turn.reported, record_id, scenario)
        return web.json_response({"agent_record_id": report_id})

    async def write_task(self, request: web.Request) -> web.Response:
        try:
            task = task_from_document(await self.body_of(request))
        except WireError as exc:
            return error_response(400, str(exc))
        digest = content_hash(task)
        if digest in await asyncio.to_thread(self.known_hashes):
            return error_response(409, "duplicate of a task already under the root", duplicate=True)
        try:
            path = await asyncio.to_thread(write_harbor_task, task, self.tasks_root)
        except HarborTaskConflict as exc:
            return error_response(409, f"a different task holds the name {task.name}: {exc}", duplicate=False)
        return web.json_response({"path": str(path), "name": task.name, "digest": task.digest}, status=201)

    async def delete_task(self, request: web.Request) -> web.Response:
        name = request.match_info["name"]
        if not TASK_NAME_PATTERN.fullmatch(name) or ".." in name:
            return error_response(400, f"{name!r} is not a task name")
        path = self.tasks_root / name
        if not path.is_dir():
            return error_response(404, f"no task {name} under {self.tasks_root}")
        await asyncio.to_thread(shutil.rmtree, path, True)
        return web.Response(status=204)

    async def check(self, request: web.Request) -> web.Response:
        try:
            body = await self.body_of(request)
            job = CheckJob(self.checks, self.task_path_for(body.get("path")))
        except WireError as exc:
            return error_response(400, str(exc))
        return self.submitted(job)

    async def play(self, request: web.Request) -> web.Response:
        try:
            body = await self.body_of(request)
            task_path = self.task_path_for(body.get("path"))
            extra = body.get("extra_instruction_files", [])
            if not isinstance(extra, list) or any(not isinstance(item, str) or not item for item in extra):
                raise WireError("extra_instruction_files must be a list of paths relative to the task")
            extra_paths = tuple(task_path / item for item in extra)
            missing = [str(path) for path in extra_paths if not path.is_file()]
            if missing:
                raise WireError(f"extra instruction files do not exist: {', '.join(missing)}")
            is_reporting = body.get("is_reporting", True)
            if not isinstance(is_reporting, bool):
                raise WireError("is_reporting must be a boolean")
            job = PlayJob(
                self.plays,
                task_path,
                scenario=checked_string(body, "scenario", label="a play"),
                model=self.model_for(body),
                arm=checked_string(body, "arm", label="a play"),
                count=checked_count(body.get("plays", 1), "plays", minimum=1),
                is_reporting=is_reporting,
                extra_instruction_paths=extra_paths,
                tags=checked_tags(body.get("tags")),
            )
        except WireError as exc:
            return error_response(400, str(exc))
        return self.submitted(job)

    async def report_plays(self, request: web.Request) -> web.Response:
        try:
            body = await self.body_of(request)
            scenario = checked_string(body, "scenario", label="a play report")
            model = self.model_for(body)
            rows = body.get("plays")
            if not isinstance(rows, list):
                raise WireError("plays must be a list of played episodes")
            plays = tuple(play_from_document(row) for row in rows)
            score_of = checked_scores(body.get("scores"))
            metadata = checked_object(body.get("metadata", {}), "metadata")
        except WireError as exc:
            return error_response(400, str(exc))
        reported = await asyncio.to_thread(
            self.plays.report, plays, scenario=scenario, model=model, score_of=score_of, metadata=metadata
        )
        return web.json_response({"plays": [play_document(play) for play in reported]})

    async def write_manifest(self, request: web.Request) -> web.Response:
        try:
            body = await self.body_of(request)
            generation = checked_count(body.get("generation", 0), "generation")
            names = body.get("names")
            if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
                raise WireError("names must be a non-empty list of task names")
            eval_fraction = body.get("eval_fraction", 0.0)
            if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, (int, float)):
                raise WireError("eval_fraction must be a number")
            seed = checked_count(body.get("seed", 0), "seed")
            tasks = [read_harbor_task(self.task_path_for(name)) for name in names]
            split = split_generation(tasks, eval_fraction=float(eval_fraction), seed=seed)
        except (WireError, HarborTaskError, ValueError) as exc:
            return error_response(400, str(exc))
        path = self.tasks_root / f"manifest-{generation:05d}.json"
        await asyncio.to_thread(write_split_manifest, path, split)
        return web.json_response({"path": str(path), "train": list(split.train), "eval": list(split.eval)})

    async def job(self, request: web.Request) -> web.Response:
        job = self.jobs.get(request.match_info["job_id"])
        if job is None:
            return error_response(404, "no such job")
        return web.json_response(job.document())
