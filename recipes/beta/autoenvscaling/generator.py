"""The AutoEnvScaling task generator: recorded coding sessions rebuilt as Harbor tasks, and a pass-rate band.

``generate`` has the author rebuild the next ``count`` sessions of the sources file. Admission
(:func:`admitted_task`) holds each reply to the generator service's parser and authoring checks and appends a
Dockerfile step that keeps the repository's history up to the base commit only. ``validate`` runs the task checks
on a copy under the generator service's tasks root. ``record_results`` places each played task against the band
``[band_low, band_high]`` and writes the generation report. An eval task is not played and gets no band position.
The generator reports nothing to the Designer, which is not trained here. A cursor under the state directory lets
a restart continue with the next session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import statistics
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Self

from recipes.beta.autoenvscaling.author import (
    AgentAuthor,
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    SourceError,
    TaskAuthor,
    workspace_reply,
)
from recipes.beta.autoenvscaling.checks import CheckResult, OracleCheck, ReviewCheck, TaskCheck, check_mode
from recipes.beta.autoenvscaling.sources import SessionSource, load_sources
from reef.core.plugins import PluginConfigError
from reef.core.tasks import (
    GeneratedTask,
    HarborTask,
    HarborTaskError,
    TaskGenerationRequest,
    TaskGenerator,
    TaskPlayResult,
    TaskValidationResult,
)
from reef.record2task.generator.client import DuplicateTask, Generator, GeneratorError, HttpGenerator, TaskNameConflict
from reef.record2task.generator.designer import parse_harbor_reply
from reef.record2task.generator.harbor import GeneratedHarborTask, OracleResult, dockerfile_logical_lines, harbor_task
from reef.sandbox.e2b import E2BExecutor
from reef.sandbox.process import SandboxUnavailable

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 8
DEFAULT_BAND_LOW = 0.25
DEFAULT_BAND_HIGH = 0.75
DEFAULT_MAX_CHECKS = 6
DEFAULT_MAX_MODEL_CALLS = 250
DEFAULT_AGENT_TIMEOUT_SECONDS = 2700.0
DEFAULT_AGENT_API = "openai"
DEFAULT_AGENT_TOKEN_ENV = "REEF_TOKEN"
# The Designer: the coding agent, or the generator service's one-reply prompt Designer on the same sessions.
AUTHORS = ("agent", "prompt")
DEFAULT_AUTHOR = "agent"
DEFAULT_REVIEW = "refuse"
# A repository task builds a toolchain and the repository: more room and time than a generated shell task gets. The
# build (1200 s) and the verifier (450 s) stay inside the service's 1800 s bound on one oracle or nop trial
# (ORACLE_TIMEOUT_S), so a slow build or verifier refuses the task instead of stopping the round.
REPOSITORY_TASK_CONFIG: dict[str, dict[str, object]] = {
    "agent": {"timeout_sec": 3600},
    "verifier": {"timeout_sec": 450, "user": "root"},
    "environment": {
        "cpus": 4,
        "memory_mb": 4096,
        "storage_mb": 10240,
        "gpus": 0,
        "network_mode": "no-network",
        "build_timeout_sec": 1200,
    },
}
ERROR_CHARS = 3000
#: A generator may hold several recipes' tasks; this suffix keeps this recipe's names apart from SPADE's
#: ``harbor-<generation>-<index>``.
TASK_NAME_SUFFIX = "autoenvscaling"
# The last step of every task's Dockerfile, added at admission, so no later commit (the change the task asks for)
# reaches the solver. It runs as root: it trusts the checkout whoever owns it, and gives .git back to its owner.
HISTORY_STEP = """# Reef: the repository keeps its history up to the checked-out commit only.
RUN set -e; cd {path}; if [ -d .git ]; then \\
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'; \\
    commit=$(git rev-parse HEAD); branch=$(git symbolic-ref -q HEAD || true); \\
    for remote in $(git remote); do git remote remove "$remote"; done; \\
    git update-ref --no-deref HEAD "$commit"; \\
    git for-each-ref --format='delete %(refname)' | git update-ref --stdin; \\
    rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/packed-refs; \\
    if [ -n "$branch" ]; then git update-ref "$branch" "$commit"; git symbolic-ref HEAD "$branch"; fi; \\
    git reflog expire --expire=now --all; git gc --prune=now --quiet; \\
    chown -R "$(stat -c %u:%g .)" .git; \\
  fi"""
CURSOR_FILE = "cursor.json"
# Where a session that became no task in the band stopped; a task inside the band has no stage.
STAGE_SOURCE = "source"  # the source holds no session, or its repository or base commit is not there
STAGE_AUTHORING = "authoring"  # the author made no recorded call, or admission refused its task
STAGE_OUTAGE = "outage"  # the author could not run; the generation ends here
STAGE_CHECK = "check"  # a check found a defect in the task, or the task is a duplicate
STAGE_UNCHECKED = "unchecked"  # a refusing check could not run, so validate refused the task
STAGE_PLAY = "play"  # the recipe could not play the task
STAGE_BAND = "band"  # the task played, and its pass rate is outside the band
STAGE_UNREACHED = "unreached"  # the round ended before it played the task
STAGE_HELD_OUT = "held_out"  # the task is in the eval split, so the recipe did not play it
# Where a pass rate falls against the band.
BAND_BELOW = "below"
BAND_INSIDE = "inside"
BAND_ABOVE = "above"
CONFIG_KEYS = frozenset(
    {
        "generator_url",
        "state_dir",
        "sources_path",
        "author",
        "count",
        "band_low",
        "band_high",
        "max_checks",
        "agent_model",
        "agent_api",
        "agent_reef_url",
        "agent_token_env",
        "agent_timeout_seconds",
        "max_model_calls",
        "behaviour_rule",
        "review",
        "review_model",
    }
)


def admitted_task(reply: str, *, generation: int, index: int, record_id: str, checkout_path: str) -> HarborTask:
    """A Designer's reply as a Harbor task: the one admission for every author's task and the agent's candidates.

    The reply goes through the generator service's parser and authoring checks, the rules a prompt Designer's
    reply is held to; a task they refuse raises ``ValueError`` (a
    :class:`~reef.record2task.generator.designer.DesignerReplyError` from the parser). The Dockerfile gets
    :data:`HISTORY_STEP` for the repository at ``checkout_path`` as its last step. The step runs as root; a final
    stage that switched to another user switches back after it.
    """
    harbor_reply = parse_harbor_reply(reply)
    dockerfile = harbor_reply.environment["Dockerfile"]
    user = None
    for line in dockerfile_logical_lines(dockerfile):
        keyword, _, argument = line.partition(" ")
        if keyword.upper() == "FROM":
            user = None
        elif keyword.upper() == "USER":
            user = argument.strip()
    step = HISTORY_STEP.format(path=shlex.quote(checkout_path))
    if user is not None and user.split(":")[0] not in ("root", "0"):
        last_steps = f"USER root\n{step}\nUSER {user}"
    else:
        last_steps = step
    environment = {**harbor_reply.environment, "Dockerfile": f"{dockerfile.rstrip()}\n{last_steps}\n"}
    harbor_reply = replace(harbor_reply, environment=environment)
    generated = GeneratedHarborTask(
        reply=harbor_reply, generation=generation, index=index, source_record_id=record_id, skill=TASK_NAME_SUFFIX
    )
    return replace(harbor_task(generated), config=REPOSITORY_TASK_CONFIG)


class GeneratorChecks(CandidateChecks):
    """``task_check`` through the generator service: the candidate written under a name of its own, checked by
    Harbor's oracle and nop agents, then deleted.

    The agent calls :meth:`check` on a gateway thread while the generation's event loop waits for the agent. The
    generator client is asynchronous and belongs to that loop, so each check is handed to it with
    ``run_coroutine_threadsafe`` and the gateway thread waits for the answer. Checks run one at a time; a check
    that could not run uses up none of ``max_checks``.

    With a ``reviewer``, a candidate that passes the oracle check also gets the test review, and the check passes
    only when the review finds nothing. A review that could not run says so and does not fail the check.
    """

    def __init__(
        self,
        generator: Generator,
        loop: asyncio.AbstractEventLoop,
        *,
        generation: int,
        index: int,
        source_record_id: str,
        checkout_path: str,
        max_checks: int,
        reviewer: ReviewCheck | None = None,
    ) -> None:
        super().__init__(max_checks, is_reviewing=reviewer is not None)
        self.generator = generator
        self.loop = loop
        self.generation = generation
        self.index = index
        self.source_record_id = source_record_id
        self.checkout_path = checkout_path
        self.reviewer = reviewer
        self.used_check_count = 0
        self.is_closed = False
        self.lock = threading.Lock()

    def check(self, task_dir: Path) -> dict[str, object]:
        with self.lock:
            if self.is_closed:
                return {"ran": False, "error": "the agent's run is over; no more checks"}
            if self.used_check_count >= self.max_checks:
                return {
                    "ran": False,
                    "error": f"all {self.max_checks} checks are used; finish with the task as it stands",
                    "checks_left": 0,
                }
            try:
                answer = self.checked(task_dir, self.used_check_count + 1)
            except GeneratorError as exc:
                # A check that could not run uses up none.
                checks_left = self.max_checks - self.used_check_count
                return {"ran": False, "error": str(exc)[:ERROR_CHARS], "checks_left": checks_left}
            self.used_check_count += 1
            checks_left = self.max_checks - self.used_check_count
            return {"check": self.used_check_count, "ran": True, **answer, "checks_left": checks_left}

    def checked(self, task_dir: Path, number: int) -> dict[str, object]:
        """The workspace's task through admission, the oracle check and, once that passes, the test review.

        Raises ``GeneratorError`` when the generator service could not check it.
        """
        try:
            task = admitted_task(
                workspace_reply(task_dir),
                generation=self.generation,
                index=self.index,
                record_id=self.source_record_id,
                checkout_path=self.checkout_path,
            )
        except (ValueError, HarborTaskError) as exc:
            return {"passed": False, "stage": "authoring", "errors": str(exc)[:ERROR_CHARS]}
        candidate = replace(task, name=f"{task.name}-check-{number}")
        try:
            result = asyncio.run_coroutine_threadsafe(self.oracle(candidate), self.loop).result()
        except DuplicateTask as exc:
            error = f"the task is the same as one the generator holds already: {exc}"
            return {"passed": False, "stage": "write", "errors": error[:ERROR_CHARS]}
        answer: dict[str, object] = {
            "passed": result.is_solvable,
            "stage": "oracle",
            "oracle_reward": result.oracle_reward,
            "nop_reward": result.nop_reward,
            "reason": result.reason[:ERROR_CHARS],
        }
        if self.reviewer is None or not result.is_solvable:
            return answer
        tags = {
            "role": "test-review",
            "generation": str(self.generation),
            "index": str(self.index),
            "check": str(number),
        }
        review = self.reviewer.review(task, tags=tags)
        if review.could_not_run:
            return {**answer, "review": {"ran": False, "error": review.could_not_run[:ERROR_CHARS]}}
        return {
            **answer,
            "passed": not review.errors,
            "stage": "review",
            "review": {"ran": True, "findings": review.details["findings"]},
        }

    async def oracle(self, candidate: HarborTask) -> OracleResult:
        """The candidate through the generator's check; it is deleted again whatever the check says."""
        written = await self.generator.write_task(candidate)
        try:
            return await self.generator.check(written.path)
        finally:
            await self.generator.delete_task(candidate.name)

    def close(self) -> None:
        """Refuse further checks, once the check in flight has finished and deleted its candidate."""
        with self.lock:
            self.is_closed = True


class ProposalRefused(ValueError):
    """A session became no task: ``stage`` says where it stopped; ``record_id`` is the author's last recorded call."""

    def __init__(self, reason: str, *, stage: str, record_id: str = "") -> None:
        super().__init__(reason)
        self.stage = stage
        self.record_id = record_id


@dataclass(frozen=True)
class GenerationCursor:
    """Where the next generation starts: its number, and the walk's position of its first session.

    The position counts every session taken so far. The walk takes ``sources[position % len(sources)]``, so it
    cycles through the sources file.
    """

    next_generation: int = 0
    next_position: int = 0


@dataclass(frozen=True)
class SessionProposal:
    """One session a generation took, and what became of it.

    ``task_name`` is set once the author's task was admitted. ``stage`` says where a session that became no
    task inside the band stopped (``STAGE_*``); it is empty for a task inside the band. ``designer_record_id`` is
    the author's last recorded call, the one that wrote the task.
    """

    index: int
    position: int
    source_record_id: str
    designer_record_id: str = ""
    task_name: str | None = None
    stage: str = ""
    refusal: str = ""


@dataclass(frozen=True)
class TaskOutcome:
    """A played task: its rewards, the solver's pass rate (the mean training reward), and its place in the band.

    ``path`` is where the task is under the generator service's tasks root.
    """

    name: str
    digest: str
    path: str
    source_record_id: str
    training_rewards: tuple[float, ...]
    pass_rate: float
    band: str


def band_bound(value: object, label: str) -> float | None:
    """A band bound from the config: a number, or null for no bound."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0, 1] or null")
    return float(value)


def task_author(
    config: Mapping[str, Any], environ: Mapping[str, str], *, scenario: str, cache_dir: Path
) -> TaskAuthor:
    """The author the config names, its model calls recorded under ``scenario``: the coding agent, or with
    ``author: prompt`` the one-reply prompt Designer on the agent's model settings.

    Credentials come from ``environ``: the Reef token from the variable ``agent_token_env`` names, the E2B key
    from ``E2B_API_KEY``.
    """
    kind = config.get("author", DEFAULT_AUTHOR)
    if kind not in AUTHORS:
        raise ValueError(f"author must be one of {AUTHORS}")
    model = str(config.get("agent_model", "") or "")
    reef_url = str(config.get("agent_reef_url", "") or "")
    if not model or not reef_url or "${" in reef_url:
        raise ValueError("the author needs agent_model and agent_reef_url: the Designer's model and the Reef it calls")
    api = str(config.get("agent_api", DEFAULT_AGENT_API))
    token = environ.get(str(config.get("agent_token_env", DEFAULT_AGENT_TOKEN_ENV))) or None
    has_behaviour_rule = bool(config.get("behaviour_rule", True))
    if kind == "prompt":
        return PromptAuthor(
            reef_url=reef_url,
            token=token,
            scenario=scenario,
            model=model,
            api=api,
            has_behaviour_rule=has_behaviour_rule,
        )
    executor = E2BExecutor(api_key=environ.get("E2B_API_KEY", ""))
    # Fails now, not mid-generation, without a key or the e2b package.
    executor.preflight()
    timeout_seconds = float(config.get("agent_timeout_seconds", DEFAULT_AGENT_TIMEOUT_SECONDS))
    max_model_calls = int(config.get("max_model_calls", DEFAULT_MAX_MODEL_CALLS))
    if timeout_seconds <= 0 or max_model_calls < 1:
        raise ValueError("agent_timeout_seconds and max_model_calls must be positive")
    return AgentAuthor(
        reef_url=reef_url,
        token=token,
        scenario=scenario,
        model=model,
        api=api,
        executor=executor,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
        max_model_calls=max_model_calls,
        has_behaviour_rule=has_behaviour_rule,
    )


def review_check(config: Mapping[str, Any], environ: Mapping[str, str], *, scenario: str) -> ReviewCheck | None:
    """The test review the config's ``review`` mode asks for, on the author's Reef and dialect; none when off.

    The model is ``review_model``, else ``agent_model``. :func:`task_author` checked the shared settings.
    """
    mode = check_mode(config.get("review", DEFAULT_REVIEW), "review")
    if mode == "off":
        return None
    return ReviewCheck(
        reef_url=str(config["agent_reef_url"]),
        token=environ.get(str(config.get("agent_token_env", DEFAULT_AGENT_TOKEN_ENV))) or None,
        scenario=scenario,
        model=str(config.get("review_model") or config["agent_model"]),
        api=str(config.get("agent_api", DEFAULT_AGENT_API)),
        is_refusing=mode == "refuse",
    )


class AutoEnvScalingGenerator(TaskGenerator):
    """Recorded sessions rebuilt as Harbor tasks by the author, checked through the generator service, and banded
    by the solver's pass rate.

    ``task_checks`` run in ``validate`` after Harbor's oracle and nop check, in order. The first refusing check
    that finds a defect, or cannot run, ends the validation. When the test review refuses, the agent's own check
    also reviews the tests. A review that only measures records its result in ``validate`` and nowhere else.
    """

    def __init__(
        self,
        *,
        scenario: str,
        service: Generator,
        author: TaskAuthor,
        sources: tuple[SessionSource, ...],
        state_dir: Path,
        count: int = DEFAULT_COUNT,
        band_low: float | None = DEFAULT_BAND_LOW,
        band_high: float | None = DEFAULT_BAND_HIGH,
        max_checks: int = DEFAULT_MAX_CHECKS,
        task_checks: tuple[TaskCheck, ...] = (),
    ) -> None:
        if count < 1 or max_checks < 1:
            raise ValueError("count and max_checks must be positive")
        for label, bound in (("band_low", band_low), ("band_high", band_high)):
            if bound is not None and not 0 <= bound <= 1:
                raise ValueError(f"{label} must be in [0, 1] or null")
        if band_low is not None and band_high is not None and band_low > band_high:
            raise ValueError("band_low must not exceed band_high")
        self.scenario = scenario
        self.service = service
        self.author = author
        self.sources = sources
        self.state_dir = state_dir
        self.count = count
        self.band_low = band_low
        self.band_high = band_high
        self.max_checks = max_checks
        self.task_checks: tuple[TaskCheck, ...] = (OracleCheck(service), *task_checks)
        # Only a refusing review may fail the agent's own check.
        self.tool_reviewer = next(
            (check for check in task_checks if isinstance(check, ReviewCheck) and check.is_refusing), None
        )
        cursor_path = state_dir / CURSOR_FILE
        self.cursor = (
            GenerationCursor(**json.loads(cursor_path.read_text(encoding="utf-8")))
            if cursor_path.is_file()
            else GenerationCursor()
        )
        # The open generation: what generate made and record_results has not closed yet.
        self.generation = self.cursor.next_generation
        self.refused: list[SessionProposal] = []
        self.pending: dict[str, SessionProposal] = {}
        # What validate found for each task: each check's result, and the stage of a refused task ("" when valid).
        self.check_results: dict[str, tuple[CheckResult, ...]] = {}
        self.validation_stages: dict[str, str] = {}
        # Where validate wrote each task; the recipe writes it again under the same name to play it.
        self.task_paths: dict[str, Path] = {}
        # The generation's error for its report.
        self.error = ""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], *, scenario: str, environ: Mapping[str, str]) -> Self:
        unknown = sorted(str(key) for key in config if key not in CONFIG_KEYS)
        if unknown:
            raise PluginConfigError(f"AutoEnvScalingGenerator does not take {', '.join(unknown)}")
        generator_url = str(config.get("generator_url", "") or "")
        if not generator_url or "${" in generator_url:
            raise PluginConfigError(
                "AutoEnvScalingGenerator needs generator_url: ${endpoints.generator} when the deployment has a "
                "generator section"
            )
        state_dir = str(config.get("state_dir", "") or "")
        sources_path = str(config.get("sources_path", "") or "")
        if not state_dir or not sources_path:
            raise PluginConfigError("AutoEnvScalingGenerator needs a state_dir and a sources_path")
        # Each scenario keeps its own cursor, reports, runs and repository snapshots.
        scenario_dir = Path(state_dir) / scenario
        try:
            author = task_author(config, environ, scenario=scenario, cache_dir=scenario_dir / "repositories")
            review = review_check(config, environ, scenario=scenario)
            return cls(
                scenario=scenario,
                service=HttpGenerator(generator_url),
                author=author,
                sources=load_sources(Path(sources_path)),
                state_dir=scenario_dir,
                count=int(config.get("count", DEFAULT_COUNT)),
                band_low=band_bound(config.get("band_low", DEFAULT_BAND_LOW), "band_low"),
                band_high=band_bound(config.get("band_high", DEFAULT_BAND_HIGH), "band_high"),
                max_checks=int(config.get("max_checks", DEFAULT_MAX_CHECKS)),
                task_checks=() if review is None else (review,),
            )
        except (OSError, ValueError, SandboxUnavailable) as exc:
            raise PluginConfigError(f"invalid AutoEnvScalingGenerator config: {exc}") from exc

    def save_cursor(self, cursor: GenerationCursor) -> None:
        """Keep ``cursor`` in the state directory, replacing the file whole."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / CURSOR_FILE
        partial = path.with_name(f"{CURSOR_FILE}.partial")
        partial.write_text(json.dumps(asdict(cursor)) + "\n", encoding="utf-8")
        os.replace(partial, path)
        self.cursor = cursor

    async def generate(self, request: TaskGenerationRequest) -> tuple[GeneratedTask, ...]:
        """One generation: the next ``count`` sessions, each rebuilt and admitted; an outage ends it with what ran.

        The sources file says what the tasks are, so the request's description goes unused. A request that
        carries source records is refused: this generator chooses its own sessions.
        """
        self.generation = self.cursor.next_generation
        # A generation number is used once, so no attempt meets the task names of an attempt a reload cancelled.
        self.save_cursor(replace(self.cursor, next_generation=self.generation + 1))
        self.refused, self.pending, self.error = [], {}, ""
        self.check_results, self.validation_stages, self.task_paths = {}, {}, {}
        if request.source_records:
            self.error = "AutoEnvScalingGenerator takes its sessions from its sources file, not from source records"
            raise ValueError(self.error)
        tasks: list[GeneratedTask] = []
        for index in range(self.count):
            position = self.cursor.next_position
            source = self.sources[position % len(self.sources)]
            proposal = SessionProposal(index, position, source.record_id)
            try:
                task = await self.rebuilt(source, index)
            except ProposalRefused as exc:
                self.refused.append(
                    replace(proposal, designer_record_id=exc.record_id, stage=exc.stage, refusal=str(exc))
                )
            except AuthorError as exc:
                # An outage must not pass for the author's failure: it ends the generation, and the report names it.
                self.error = f"the author could not rebuild {source.record_id}: {exc}"
                self.refused.append(replace(proposal, stage=STAGE_OUTAGE, refusal=self.error))
            else:
                self.pending[task.name] = replace(
                    proposal, designer_record_id=task.source_agent_record_ids[0], task_name=task.name
                )
                tasks.append(GeneratedTask(task))
            # The session is taken; a restart continues with the next one. A cancelled run takes none.
            self.save_cursor(replace(self.cursor, next_position=position + 1))
            if self.error:
                logger.error("AutoEnvScaling generation %d stopped: %s", self.generation, self.error)
                break
        return tuple(tasks)

    async def rebuilt(self, source: SessionSource, index: int) -> HarborTask:
        """The author's task from one session, admitted; its sources are the author's last recorded call and the
        session.

        Raises :class:`ProposalRefused` when the session becomes no task, and ``AuthorError`` when the author
        could not run.
        """
        checks = GeneratorChecks(
            self.service,
            asyncio.get_running_loop(),
            generation=self.generation,
            index=index,
            source_record_id=source.record_id,
            checkout_path=source.repository.path,
            max_checks=self.max_checks,
            reviewer=self.tool_reviewer,
        )
        tags = {"role": "designer", "generation": str(self.generation), "index": str(index)}
        work_dir = self.state_dir / "runs" / f"{self.generation:05d}-{index:03d}"
        stop = threading.Event()
        try:
            authored = await asyncio.to_thread(self.author.author, source, work_dir, checks, tags=tags, stop=stop)
        except asyncio.CancelledError:
            # The thread outlives the cancelled await; the stop signal ends the agent's run in it.
            stop.set()
            raise
        except SourceError as exc:
            raise ProposalRefused(
                f"the source {source.record_id} cannot be rebuilt: {exc}", stage=STAGE_SOURCE
            ) from exc
        finally:
            # A check the agent left running finishes, and deletes its candidate, before the task is written.
            await asyncio.to_thread(checks.close)
        if not authored.record_ids:
            raise ProposalRefused("the Designer made no model call that Reef recorded", stage=STAGE_AUTHORING)
        record_id = authored.record_ids[-1]
        try:
            task = admitted_task(
                authored.reply,
                generation=self.generation,
                index=index,
                record_id=record_id,
                checkout_path=source.repository.path,
            )
        except (ValueError, HarborTaskError) as exc:
            raise ProposalRefused(
                f"the Designer's task is refused: {exc}", stage=STAGE_AUTHORING, record_id=record_id
            ) from exc
        return replace(
            task,
            source_agent_record_ids=(record_id, source.record_id),
            metadata={**task.metadata, "designer_record_id": record_id},
        )

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """The task checks on a written copy, which is removed again; a duplicate or a taken name is invalid.

        A refusing check that finds a defect, or cannot run, makes the task invalid, and the report keeps the two
        apart. An oracle check that cannot run raises: the generator service is not usable for the round.
        """
        try:
            written = await self.service.write_task(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            self.validation_stages[task.name] = STAGE_CHECK
            return TaskValidationResult((f"refused: {exc}",))
        self.task_paths[task.name] = written.path
        results: list[CheckResult] = []
        errors: tuple[str, ...] = ()
        stage = ""
        try:
            for check in self.task_checks:
                result = await check.run(task, written)
                results.append(result)
                if not check.is_refusing:
                    continue
                if result.errors:
                    errors, stage = result.errors, STAGE_CHECK
                    break
                if result.could_not_run:
                    errors, stage = (f"the {check.name} check could not run: {result.could_not_run}",), STAGE_UNCHECKED
                    break
        except GeneratorError as exc:
            self.error = f"the check of {task.name} could not run: {exc}"
            raise
        finally:
            self.check_results[task.name] = tuple(results)
            await self.service.delete_task(task.name)
        self.validation_stages[task.name] = stage
        return TaskValidationResult(errors)

    async def record_results(self, results: tuple[TaskPlayResult, ...]) -> None:
        """Close the generation: each played task's pass rate and band position, and the report.

        An eval task gets the stage ``held_out``. The band is only recorded in the report. It does not change the
        recipe's task store.
        """
        proposals = list(self.refused)
        outcomes: dict[str, TaskOutcome] = {}
        for result in results:
            proposal = self.pending.pop(result.task.name)
            if result.split == "eval":
                proposals.append(replace(proposal, stage=STAGE_HELD_OUT))
                continue
            if not result.is_played:
                stage = self.validation_stages[result.task.name] or STAGE_PLAY
                proposals.append(replace(proposal, stage=stage, refusal=result.error))
                continue
            pass_rate = statistics.fmean(result.training_rewards) if result.training_rewards else 0.0
            # Where the pass rate falls against the band; a null bound is no bound.
            if self.band_low is not None and pass_rate < self.band_low:
                band = BAND_BELOW
            elif self.band_high is not None and pass_rate > self.band_high:
                band = BAND_ABOVE
            else:
                band = BAND_INSIDE
            outcome = TaskOutcome(
                name=result.task.name,
                digest=result.task.digest,
                path=str(self.task_paths[result.task.name]),
                source_record_id=proposal.source_record_id,
                training_rewards=result.training_rewards,
                pass_rate=pass_rate,
                band=band,
            )
            outcomes[outcome.name] = outcome
            if outcome.band == BAND_INSIDE:
                proposals.append(proposal)
                continue
            proposals.append(
                replace(proposal, stage=STAGE_BAND, refusal=f"outside the solver band: pass rate {pass_rate:.2f}")
            )
        proposals.extend(
            replace(proposal, stage=STAGE_UNREACHED, refusal="the round ended before it played the task")
            for proposal in self.pending.values()
        )
        proposals.sort(key=lambda proposal: proposal.index)
        document = {
            "generation": self.generation,
            "error": self.error,
            "band": {"low": self.band_low, "high": self.band_high},
            "next_position": self.cursor.next_position,
            "proposals": [
                {**asdict(proposal), "checks": self.checks_document(proposal.task_name)} for proposal in proposals
            ],
            "tasks": [
                {**asdict(outcome), "checks": self.checks_document(outcome.name)} for outcome in outcomes.values()
            ],
        }
        report_path = self.state_dir / f"generation-{self.generation:05d}.json"
        report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "AutoEnvScaling generation %d on scenario %r: %d tasks inside the band of %d played, %d sessions; %s",
            self.generation,
            self.scenario,
            sum(outcome.band == BAND_INSIDE for outcome in outcomes.values()),
            len(outcomes),
            len(proposals),
            report_path,
        )

    def checks_document(self, task_name: str | None) -> list[dict[str, object]]:
        """Each check's result for a task, as the report keeps it; none for a session that became no task."""
        return [asdict(result) for result in self.check_results.get(task_name or "", ())]
