"""The AutoEnvScaling task generator: recorded coding sessions rebuilt as Harbor tasks, and a pass-rate band.

``generate`` has the author rebuild the next ``count`` sessions of the sources file. Admission
(:func:`admitted_task`) holds each reply to the generator service's parser and authoring checks and appends a
Dockerfile step that keeps the repository's history up to the base commit only. ``validate`` runs the task checks
on a copy under the generator service's tasks root. With ``calibration`` on, the last check runs the current
solver's rollouts on the task and holds their mean reward to the band. ``generate`` keeps the open generation in
``open-<generation>.json`` under the state directory, named by the recipe's generation number, and ``validate``
adds each task's check results to it. ``record_results`` closes it, also after later generations started or after
a restart: it places each task that ran against the band ``[band_low, band_high]`` and writes the generation report.
A second call for a closed generation does nothing.
Eval tasks run no training rollouts and receive no band measurement. A task that could not run is an outage, not a
band measurement. The generator reports nothing to the Designer, which is not trained here. A cursor under the state
directory lets a restart continue with the next session.

With ``adaptive`` on, ``record_results`` also keeps each training task whose pass rate fell outside the band, and a
later generation gives up to ``variant_share`` of its slots to variants of those tasks: harder for a task the solver
passes too often, simpler for one it passes too rarely. A variant comes from the same session, so it shares the
parent's split. Eval tasks, outages and tasks whose mean reward is outside [0, 1] never become parents, and each
parent is varied once. The task checks run in ``validate``: Harbor's oracle and nop check, then the contamination
check, the test review and the solver calibration when they are on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shlex
import shutil
import statistics
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import TypeAdapter

from recipes.beta.autoenvscaling.author import (
    VARIANT_HARDER,
    VARIANT_SIMPLER,
    AgentAuthor,
    AuthorError,
    CandidateChecks,
    PromptAuthor,
    SourceError,
    TaskAuthor,
    TaskVariant,
    VariantAuthor,
    VariantDirection,
    workspace_reply,
)
from recipes.beta.autoenvscaling.checks import (
    BAND_ABOVE,
    BAND_BELOW,
    BAND_INSIDE,
    CONTAMINATION_N,
    BandPlace,
    CalibrationCheck,
    CheckResult,
    ContaminationCheck,
    OracleCheck,
    PassRateBand,
    ReviewCheck,
    TaskCheck,
    band_place,
    check_mode,
    reference_instructions,
    validate_calibration,
)
from recipes.beta.autoenvscaling.claude_author import (
    CLAUDE_HARNESS,
    TOKEN_PREFIX,
    ClaudeCodeAuthor,
    CredentialHeld,
    overrides_run,
)
from recipes.beta.autoenvscaling.sources import SessionSource, load_sources
from reef.core.plugins import PluginConfigError
from reef.core.tasks import HarborTask, HarborTaskError, write_harbor_task
from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.execution.sandbox.config import remote_executor
from reef.execution.sandbox.process import SandboxUnavailable
from reef.execution.sandbox.remote import RemoteExecutor
from reef.recipe.generated_tasks.generator import (
    FailedTask,
    GeneratedTaskResult,
    HeldOutTask,
    RejectedTask,
    TaskGenerator,
    TaskValidationResult,
)
from reef.record2sample.generator.client import (
    DuplicateTask,
    Generator,
    GeneratorError,
    HttpGenerator,
    TaskNameConflict,
)
from reef.record2sample.generator.designer import parse_harbor_reply
from reef.record2sample.generator.harbor import (
    GeneratedHarborTask,
    OracleResult,
    dockerfile_logical_lines,
    harbor_task,
)
from reef.record2sample.generator.inputs import TaskGenerationRequest

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 8
DEFAULT_VARIANT_SHARE = 0.5
# The most parents an adaptive generator keeps; the oldest go first.
MAX_PARENTS = 256
FEEDBACK_FILE = "adaptive.json"
PARENTS_DIR = "parents"
# A variant takes no session from the walk, so its proposal has no walk position.
VARIANT_POSITION = -1
DEFAULT_BAND_LOW = 0.25
DEFAULT_BAND_HIGH = 0.75
DEFAULT_BAND = PassRateBand(DEFAULT_BAND_LOW, DEFAULT_BAND_HIGH)
DEFAULT_MAX_CHECKS = 6
DEFAULT_MAX_MODEL_CALLS = 250
DEFAULT_AGENT_TIMEOUT_SECONDS = 2700.0
DEFAULT_AGENT_API = "openai"
DEFAULT_AGENT_TOKEN_ENV = "REEF_TOKEN"
# The cloud sandbox the agent runs in: a provider of reef.execution.sandbox.config.REMOTE_EXECUTORS.
DEFAULT_AGENT_SANDBOX = "e2b"
# The Designer: the coding agent, or the generator service's one-reply prompt Designer on the same sessions.
AUTHORS = ("agent", "prompt")
DEFAULT_AUTHOR = "agent"
# The coding agent that is the agent author: Reefine's pi, or Claude Code signed in with its owner's OAuth token.
AUTHOR_HARNESSES = ("pi", CLAUDE_HARNESS)
DEFAULT_AUTHOR_HARNESS = "pi"
CLAUDE_KEYS = ("claude_model", "claude_token_env")
DEFAULT_CLAUDE_MODEL = "opus"
DEFAULT_CLAUDE_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
DEFAULT_REVIEW = "refuse"
DEFAULT_CALIBRATION = "off"
DEFAULT_CALIBRATION_ROLLOUTS = 4
DEFAULT_CALIBRATION_MAX_ROLLOUTS = 8
DEFAULT_CALIBRATION_MARGIN = 0.15
DEFAULT_CALIBRATION_MIN_STD = 0.1
DEFAULT_CONTAMINATION = "off"
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
OPEN_PREFIX = "open-"
# Where a session that became no task in the band stopped; a task inside the band has no stage.
STAGE_SOURCE = "source"  # the source holds no session, or its repository or base commit is not there
STAGE_AUTHORING = "authoring"  # the author made no recorded call, or admission refused its task
STAGE_OUTAGE = "outage"  # the author could not run; the generation ends here
STAGE_CREDENTIAL = "credential"  # the author's task held its Claude Code token; the generation ends here
STAGE_CHECK = "check"  # a check found a defect in the task, or the task is a duplicate
STAGE_UNCHECKED = "unchecked"  # a refusing check could not run, so validate refused the task
STAGE_CALIBRATION = "calibration"  # the solver's calibration rollouts put the task outside the band, or barely vary
STAGE_RUN = "run"  # the recipe could not run the task
STAGE_BAND = "band"  # the task ran, and its pass rate is outside the band
STAGE_UNREACHED = "unreached"  # the round ended before it ran the task
STAGE_HELD_OUT = "held_out"  # the task is in the eval split, so the recipe did not run it
CONFIG_KEYS = frozenset(
    {
        "generator_url",
        "state_dir",
        "sources_path",
        "author",
        "author_harness",
        "claude_model",
        "claude_token_env",
        "count",
        "band_low",
        "band_high",
        "max_checks",
        "agent_model",
        "agent_api",
        "agent_reef_url",
        "agent_token_env",
        "agent_timeout_seconds",
        "agent_sandbox",
        "agent_sandbox_options",
        "max_model_calls",
        "behaviour_rule",
        "review",
        "review_model",
        "adaptive",
        "variant_share",
        "calibration",
        "calibration_rollouts",
        "calibration_max_rollouts",
        "calibration_margin",
        "calibration_min_std",
        "calibration_model",
        "contamination",
        "contamination_tasks_path",
        "contamination_ngram",
    }
)


def admitted_task(reply: str, *, generation: int, index: int, record_id: str, checkout_path: str) -> HarborTask:
    """A Designer's reply as a Harbor task: the one admission for every author's task and the agent's candidates.

    The reply goes through the generator service's parser and authoring checks, the rules a prompt Designer's
    reply is held to; a task they refuse raises ``ValueError`` (a
    :class:`~reef.record2sample.generator.designer.DesignerReplyError` from the parser). The Dockerfile gets
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

    Values: index is a non-negative integer; source_record_id is non-empty text.
    Position is a non-negative walk index or -1 exactly when parent_task and direction
    identify a variant. Designer_record_id and refusal are text, empty when unavailable.
    Purpose: generate saves proposals; record_results updates them for the generation report.
    ``task_name`` is a Harbor name or None, set once the author's task was admitted.
    ``stage`` says where a session that became no task inside the band stopped (``STAGE_*``); it is empty for a task inside the band. ``designer_record_id`` is
    the author's last recorded call, the one that wrote the task, or empty for an author that Reef records no call
    of (Claude Code; its task names the transcript instead). A variant has ``position`` -1, the name of the
    task it varies in ``parent_task`` and ``harder`` or ``simpler`` in ``direction``; both are empty otherwise.
    """

    index: int
    position: int
    source_record_id: str
    designer_record_id: str = ""
    task_name: str | None = None
    stage: str = ""
    refusal: str = ""
    parent_task: str = ""
    direction: VariantDirection | Literal[""] = ""

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0 or type(self.position) is not int:
            raise ValueError("a proposal needs a non-negative index and an integer position")
        if not isinstance(self.source_record_id, str) or not self.source_record_id.strip():
            raise ValueError("a proposal needs a source_record_id")
        if not all(
            isinstance(value, str)
            for value in (self.designer_record_id, self.refusal, self.parent_task, self.direction, self.stage)
        ):
            raise ValueError("proposal record, refusal, variant and stage fields must be text")
        if self.position == VARIANT_POSITION:
            if not TASK_NAME_PATTERN.fullmatch(self.parent_task) or ".." in self.parent_task:
                raise ValueError("a variant proposal needs a parent task name")
            if self.direction not in (VARIANT_HARDER, VARIANT_SIMPLER):
                raise ValueError("a variant proposal needs direction harder or simpler")
        elif self.position < 0 or self.parent_task or self.direction:
            raise ValueError("a session proposal needs a non-negative position and no variant fields")
        if self.stage not in (
            "",
            STAGE_SOURCE,
            STAGE_AUTHORING,
            STAGE_OUTAGE,
            STAGE_CREDENTIAL,
            STAGE_CHECK,
            STAGE_UNCHECKED,
            STAGE_CALIBRATION,
            STAGE_RUN,
            STAGE_BAND,
            STAGE_UNREACHED,
            STAGE_HELD_OUT,
        ):
            raise ValueError("a proposal has an unknown stage")
        if self.task_name is not None and (
            not isinstance(self.task_name, str)
            or not TASK_NAME_PATTERN.fullmatch(self.task_name)
            or ".." in self.task_name
        ):
            raise ValueError("a proposal's task_name must be a task name or None")


@dataclass(frozen=True)
class OpenGeneration:
    """A generation that ``generate`` made and ``record_results`` has not closed yet.

    Values: ``number`` is the generator's number of the generation. ``refused`` holds the sessions that made
    no task, and ``pending`` the sessions that became tasks, by task name. ``error`` is the error that stopped
    the generation, or empty. ``checks``, ``stages`` and ``paths`` are keyed by task name: the results of the
    task checks, the stage of a refused task (empty when it passed), and where ``validate`` wrote the task.
    Purpose: ``generate`` writes it to ``open-<generation>.json`` and ``validate`` adds to it;
    ``record_results`` reads it to close the generation, also after later generations and after a restart.
    """

    number: int
    refused: tuple[SessionProposal, ...]
    pending: dict[str, SessionProposal]
    error: str = ""
    checks: dict[str, tuple[CheckResult, ...]] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)


OPEN_GENERATION = TypeAdapter(OpenGeneration)


@dataclass(frozen=True)
class ParentTask:
    """A training task retained for an adaptive generation to vary.

    Values: ``name`` is the task's name (a Harbor task name) and ``source_record_id`` its session's record id.
    ``source_record_id`` must be non-empty. ``generation`` is a non-negative integer;
    ``rollouts`` is a positive integer. Neither accepts bool. ``pass_rate`` is a non-boolean
    number in [0, 1], representing the mean rollout reward. ``direction`` is harder or simpler.
    ``varied`` is a bool, true once a generation gave this parent a variant slot.
    Purpose: ``record_results`` adds one per training task outside the band and keeps the task's files under
    ``parents/<name>``; ``generate`` picks the newest unvaried parents for its variant slots.
    """

    name: str
    source_record_id: str
    generation: int
    pass_rate: float
    rollouts: int
    direction: VariantDirection
    varied: bool = False

    def __post_init__(self) -> None:
        # The name names the parent's directory, which a bounded list removes.
        if not isinstance(self.name, str) or not TASK_NAME_PATTERN.fullmatch(self.name) or ".." in self.name:
            raise ValueError(f"a parent's name must be a task name matching {TASK_NAME_PATTERN.pattern}")
        if self.direction not in (VARIANT_HARDER, VARIANT_SIMPLER):
            raise ValueError(f"a parent's direction must be {VARIANT_HARDER} or {VARIANT_SIMPLER}")
        if not isinstance(self.source_record_id, str) or not self.source_record_id.strip():
            raise ValueError("a parent needs a non-empty source_record_id")
        if type(self.varied) is not bool:
            raise ValueError("a parent's varied flag must be a bool")
        if (
            type(self.pass_rate) not in (int, float)
            or not 0 <= self.pass_rate <= 1
            or type(self.rollouts) is not int
            or self.rollouts < 1
            or type(self.generation) is not int
            or self.generation < 0
        ):
            raise ValueError("a parent needs a pass rate in [0, 1], positive rollouts and a generation")


PARENTS = TypeAdapter(list[ParentTask])


@dataclass(frozen=True)
class TaskOutcome:
    """A task that ran: its rewards, the solver's pass rate (the mean training reward), and its place in the band.

    Values: name is a Harbor task name; digest is its content identifier; path is the
    task location or empty when unwritten. Source_record_id is non-empty text.
    Rollout_rewards is a non-empty ordered tuple of finite numbers (duplicates allowed),
    and pass_rate is their arithmetic mean. Band is below, inside or above: the label
    assigned against the report's configured band, which is not stored in this value.
    Purpose: record_results creates one per trained task and serializes it in the report.
    """

    name: str
    digest: str
    path: str
    source_record_id: str
    rollout_rewards: tuple[float, ...]
    pass_rate: float
    band: BandPlace

    def __post_init__(self) -> None:
        if not TASK_NAME_PATTERN.fullmatch(self.name) or ".." in self.name:
            raise ValueError("a task outcome needs a Harbor task name")
        if not self.source_record_id.strip():
            raise ValueError("a task outcome needs a source_record_id")
        if not self.rollout_rewards or any(not math.isfinite(reward) for reward in self.rollout_rewards):
            raise ValueError("a task outcome needs finite rollout rewards")
        if self.pass_rate != statistics.fmean(self.rollout_rewards):
            raise ValueError("a task outcome's pass_rate must equal its mean reward")
        if self.band not in (BAND_BELOW, BAND_INSIDE, BAND_ABOVE):
            raise ValueError("a task outcome needs a band place")


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

    The agent is pi, or Claude Code with ``author_harness: claude-code`` (:func:`claude_code_author`). It runs in
    the cloud sandbox ``agent_sandbox`` names (``e2b`` by default, or ``modal``), with that provider's options from
    ``agent_sandbox_options``. Credentials come from ``environ``: the Reef token from the variable
    ``agent_token_env`` names, and the sandbox provider's own (``E2B_API_KEY``, or Modal's token).
    """
    kind = config.get("author", DEFAULT_AUTHOR)
    if kind not in AUTHORS:
        raise ValueError(f"author must be one of {AUTHORS}")
    harness = config.get("author_harness", DEFAULT_AUTHOR_HARNESS)
    if harness not in AUTHOR_HARNESSES:
        raise ValueError(f"author_harness must be one of {AUTHOR_HARNESSES}")
    if harness != CLAUDE_HARNESS and any(key in config for key in CLAUDE_KEYS):
        raise ValueError(f"{' and '.join(CLAUDE_KEYS)} need author_harness: {CLAUDE_HARNESS}")
    if harness == CLAUDE_HARNESS and kind == "prompt":
        raise ValueError(f"author_harness: {CLAUDE_HARNESS} needs author: agent")
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
    options = config.get("agent_sandbox_options") or {}
    if not isinstance(options, Mapping):
        raise ValueError("agent_sandbox_options must be a mapping of the sandbox provider's options")
    executor = remote_executor(str(config.get("agent_sandbox", DEFAULT_AGENT_SANDBOX)), options, environ)
    # Fails now, not mid-generation, without the provider's credentials or package.
    executor.preflight()
    timeout_seconds = float(config.get("agent_timeout_seconds", DEFAULT_AGENT_TIMEOUT_SECONDS))
    max_model_calls = int(config.get("max_model_calls", DEFAULT_MAX_MODEL_CALLS))
    if timeout_seconds <= 0 or max_model_calls < 1:
        raise ValueError("agent_timeout_seconds and max_model_calls must be positive")
    if harness == CLAUDE_HARNESS:
        return claude_code_author(
            config,
            environ,
            reef_url=reef_url,
            token=token,
            executor=executor,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            max_turns=max_model_calls,
            has_behaviour_rule=has_behaviour_rule,
        )
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


def claude_code_author(
    config: Mapping[str, object],
    environ: Mapping[str, str],
    *,
    reef_url: str,
    token: str | None,
    executor: RemoteExecutor,
    cache_dir: Path,
    timeout_seconds: float,
    max_turns: int,
    has_behaviour_rule: bool,
) -> ClaudeCodeAuthor:
    """Claude Code as the agent author, on ``claude_model``, signed in with the OAuth token in the variable
    ``claude_token_env`` names.

    Raises ``ValueError`` when that variable is unset or empty, or holds no Claude Code OAuth token, and when the
    sandbox's ``env_from`` forwards a variable that wins over the run's own setup (:func:`overrides_run`: an
    ``ANTHROPIC_*`` or ``CLAUDE_*`` variable, a base URL, a proxy, or one the run sets), or one whose value holds
    the token, which would put it on the jail's command line. No message holds the token.
    """
    token_env = config.get("claude_token_env", DEFAULT_CLAUDE_TOKEN_ENV)
    if not isinstance(token_env, str) or not token_env.isidentifier():
        raise ValueError("claude_token_env must name an environment variable")
    claude_token = environ.get(token_env, "").strip()
    if not claude_token:
        raise ValueError(f"author_harness: {CLAUDE_HARNESS} needs a Claude Code OAuth token in {token_env}")
    if not claude_token.startswith(TOKEN_PREFIX):
        raise ValueError(
            f"{token_env} must hold a Claude Code OAuth token, which starts with {TOKEN_PREFIX} "
            "(`claude setup-token` makes one)"
        )
    model = config.get("claude_model", DEFAULT_CLAUDE_MODEL)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("claude_model must be a Claude model name or alias")
    for name, value in executor.env.items():
        if overrides_run(name) or claude_token in value:
            raise ValueError(
                f"agent_sandbox_options.env_from must not forward {name} to Claude Code: it would win over the "
                f"run's own setup or carry the token, which comes from {token_env} alone"
            )
    return ClaudeCodeAuthor(
        reef_url=reef_url,
        token=token,
        claude_token=claude_token,
        model=model.strip(),
        executor=executor,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
        max_turns=max_turns,
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


def int_key(config: Mapping[str, object], label: str, default: int) -> int:
    """Read an integer config field, excluding booleans."""
    value = config.get(label, default)
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def number_key(config: Mapping[str, object], label: str, default: float) -> float:
    """Read a numeric config field, excluding booleans."""
    value = config.get(label, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def calibration_check(
    config: Mapping[str, object],
    service: Generator,
    *,
    scenario: str,
    band: PassRateBand,
) -> CalibrationCheck | None:
    """Validate every calibration key; construct a check only when its mode is enabled."""
    mode = check_mode(config.get("calibration", DEFAULT_CALIBRATION), "calibration")
    rollouts = int_key(config, "calibration_rollouts", DEFAULT_CALIBRATION_ROLLOUTS)
    max_rollouts = int_key(config, "calibration_max_rollouts", DEFAULT_CALIBRATION_MAX_ROLLOUTS)
    margin = number_key(config, "calibration_margin", DEFAULT_CALIBRATION_MARGIN)
    min_std = number_key(config, "calibration_min_std", DEFAULT_CALIBRATION_MIN_STD)
    model = config.get("calibration_model")
    if model is not None and not (isinstance(model, str) and model.strip()):
        raise ValueError("calibration_model must be a model name or null")
    if mode == "off":
        validate_calibration(rollouts, max_rollouts, margin, min_std)
        return None
    return CalibrationCheck(
        service,
        scenario=scenario,
        rollouts=rollouts,
        max_rollouts=max_rollouts,
        margin=margin,
        min_std=min_std,
        band=band,
        model=model,
        is_refusing=mode == "refuse",
    )


def contamination_check(config: Mapping[str, object]) -> ContaminationCheck | None:
    """The contamination check the config's ``contamination`` mode asks for, against the instructions under
    ``contamination_tasks_path``; none when off.

    Raises ``ValueError`` for a key of the wrong type, also when the mode is off. When the mode is not off, it also
    raises when ``contamination_tasks_path`` is null, is not a directory or holds no ``instruction.md``.
    """
    mode = check_mode(config.get("contamination", DEFAULT_CONTAMINATION), "contamination")
    n = config.get("contamination_ngram", CONTAMINATION_N)
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("contamination_ngram must be an integer of at least 1")
    tasks_path = config.get("contamination_tasks_path")
    if tasks_path is not None and not (isinstance(tasks_path, str) and tasks_path.strip()):
        raise ValueError("contamination_tasks_path must be a directory path or null")
    if mode == "off":
        if n < 1:
            raise ValueError("contamination_ngram must be an integer of at least 1")
        return None
    if tasks_path is None:
        raise ValueError("contamination needs contamination_tasks_path: the directory of the reference tasks")
    return ContaminationCheck(
        reference_instructions(Path(tasks_path).expanduser()), n=n, tasks_path=tasks_path, is_refusing=mode == "refuse"
    )


class AutoEnvScalingGenerator(TaskGenerator):
    """Recorded sessions rebuilt as Harbor tasks by the author, checked through the generator service, and banded
    by the solver's pass rate.

    ``task_checks`` run in ``validate`` after Harbor's oracle and nop check, in order. The first refusing check
    that finds a defect, or cannot run, ends the validation. When the test review refuses, the agent's own check
    also reviews the tests. A review that only measures records its result in ``validate`` and nowhere else. A
    refusing calibration that finds the task outside the band gives it the stage ``calibration``.

    Values: ``count`` and ``max_checks`` are positive. ``band`` is a PassRateBand, also used by
    ``calibration`` when present. Checks run in order: oracle, task_checks, then calibration.
    ``calibration`` is a CalibrationCheck or None. ``cursor`` is the cursor
    saved under ``state_dir``, and ``generation`` is the generation that ``generate`` makes now. ``is_adaptive``
    is a bool and ``variant_share`` in (0, 1]; ``variant_slots``, the most variant slots of a generation, is
    ``variant_share`` of ``count`` rounded down after adding a 1e-9 slot tolerance.
    It is at least 1 when ``is_adaptive``. ``parents`` are the :class:`ParentTask` entries
    saved in ``adaptive.json``, in insertion order, with unique names and at most
    MAX_PARENTS entries; empty unless ``is_adaptive``.
    Purpose: the generated-task recipe builds it with ``from_config`` and calls ``generate``, ``validate`` and
    ``record_results`` in each round.
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
        band: PassRateBand = DEFAULT_BAND,
        calibration: CalibrationCheck | None = None,
        max_checks: int = DEFAULT_MAX_CHECKS,
        task_checks: tuple[TaskCheck, ...] = (),
        is_adaptive: bool = False,
        variant_share: float = DEFAULT_VARIANT_SHARE,
    ) -> None:
        if count < 1 or max_checks < 1:
            raise ValueError("count and max_checks must be positive")
        if not 0 < variant_share <= 1:
            raise ValueError("variant_share must be in (0, 1]")
        # The tolerance keeps a share such as 0.29 of 100 slots at 29, not 28.
        variant_slots = math.floor(count * variant_share + 1e-9)
        if is_adaptive and variant_slots < 1:
            raise ValueError(f"variant_share {variant_share} of count {count} gives no variant slot")
        if calibration is not None and calibration.band != band:
            raise ValueError("calibration and training must use the same pass-rate band")
        self.scenario = scenario
        self.service = service
        self.author = author
        if is_adaptive and not isinstance(author, VariantAuthor):
            raise ValueError("adaptive generation needs a VariantAuthor")
        self.variant_author = author if isinstance(author, VariantAuthor) else None
        self.sources = sources
        self.state_dir = state_dir
        self.count = count
        self.band = band
        self.max_checks = max_checks
        self.task_checks: tuple[TaskCheck, ...] = (OracleCheck(service), *task_checks)
        if calibration is not None:
            self.task_checks = (*self.task_checks, calibration)
        # Only a refusing review may fail the agent's own check.
        self.tool_reviewer = next(
            (check for check in task_checks if isinstance(check, ReviewCheck) and check.is_refusing), None
        )
        self.calibration = calibration
        cursor_path = state_dir / CURSOR_FILE
        self.cursor = (
            GenerationCursor(**json.loads(cursor_path.read_text(encoding="utf-8")))
            if cursor_path.is_file()
            else GenerationCursor()
        )
        # The generation that generate makes now; generate takes the next number from the cursor.
        self.generation = self.cursor.next_generation
        self.is_adaptive = is_adaptive
        # The most slots of a generation that go to variants: variant_share of count, rounded down.
        self.variant_slots = variant_slots
        feedback_path = state_dir / FEEDBACK_FILE
        self.parents: list[ParentTask] = (
            PARENTS.validate_json(feedback_path.read_bytes(), strict=True)
            if is_adaptive and feedback_path.is_file()
            else []
        )
        if len(self.parents) > MAX_PARENTS or len({parent.name for parent in self.parents}) != len(self.parents):
            raise ValueError(f"adaptive.json must hold at most {MAX_PARENTS} parents with unique names")

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
        is_adaptive = config.get("adaptive", False)
        if not isinstance(is_adaptive, bool):
            raise PluginConfigError("adaptive must be true or false")
        variant_share = config.get("variant_share", DEFAULT_VARIANT_SHARE)
        if isinstance(variant_share, bool) or not isinstance(variant_share, (int, float)):
            raise PluginConfigError("variant_share must be in (0, 1]")
        if is_adaptive is True and config.get("author", DEFAULT_AUTHOR) != "agent":
            raise PluginConfigError("AutoEnvScalingGenerator adaptive generation needs author: agent")
        if "variant_share" in config and is_adaptive is not True:
            raise PluginConfigError("AutoEnvScalingGenerator variant_share needs adaptive: true")
        # Each scenario keeps its own cursor, reports, runs and repository snapshots.
        scenario_dir = Path(state_dir) / scenario
        try:
            author = task_author(config, environ, scenario=scenario, cache_dir=scenario_dir / "repositories")
            service = HttpGenerator(generator_url)
            band_low = band_bound(config.get("band_low", DEFAULT_BAND_LOW), "band_low")
            band_high = band_bound(config.get("band_high", DEFAULT_BAND_HIGH), "band_high")
            band = PassRateBand(band_low, band_high)
            # The contamination check runs right after the oracle check: it makes no model call and runs no rollout.
            contamination = contamination_check(config)
            review = review_check(config, environ, scenario=scenario)
            # The calibration runs last: its rollouts are the costliest check.
            calibration = calibration_check(config, service, scenario=scenario, band=band)
            return cls(
                scenario=scenario,
                service=service,
                author=author,
                sources=load_sources(Path(sources_path)),
                state_dir=scenario_dir,
                count=int(config.get("count", DEFAULT_COUNT)),
                band=band,
                calibration=calibration,
                max_checks=int(config.get("max_checks", DEFAULT_MAX_CHECKS)),
                task_checks=tuple(check for check in (contamination, review) if check is not None),
                is_adaptive=is_adaptive,
                variant_share=variant_share,
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

    def save_parents(self) -> None:
        """Keep the parents in the state directory, replacing the file whole."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / FEEDBACK_FILE
        partial = path.with_name(f"{FEEDBACK_FILE}.partial")
        partial.write_bytes(PARENTS.dump_json(self.parents))
        os.replace(partial, path)

    def variants(self) -> list[tuple[ParentTask, SessionSource]]:
        """The parents this generation varies, newest first, with their sessions; none unless adaptive.

        At most ``variant_slots``; a parent whose session left the sources file, or whose files are gone, is
        skipped. They still occupy parent capacity until dropped as the oldest entries.
        """
        if not self.is_adaptive:
            return []
        sessions = {source.record_id: source for source in self.sources}
        chosen: list[tuple[ParentTask, SessionSource]] = []
        for parent in sorted(self.parents, key=lambda parent: (-parent.generation, parent.name)):
            if len(chosen) >= self.variant_slots:
                break
            source = sessions.get(parent.source_record_id)
            if parent.varied or source is None or not (self.state_dir / PARENTS_DIR / parent.name).is_dir():
                continue
            chosen.append((parent, source))
        return chosen

    async def generate(self, request: TaskGenerationRequest) -> tuple[HarborTask, ...]:
        """One generation: the next ``count`` sessions, each rebuilt and admitted; an outage ends it with what ran.

        The sources file says what the tasks are, so the request's description goes unused. A request that
        carries source records is refused: this generator chooses its own sessions.
        """
        self.generation = self.cursor.next_generation
        # A generation number is used once, so no attempt meets the task names of an attempt a reload cancelled.
        self.save_cursor(replace(self.cursor, next_generation=self.generation + 1))
        if request.source_records:
            error = "AutoEnvScalingGenerator takes its sessions from its sources file, not from source records"
            self.save_open(request.generation, OpenGeneration(self.generation, (), {}, error))
            raise ValueError(error)
        refused: list[SessionProposal] = []
        pending: dict[str, SessionProposal] = {}
        error = ""
        tasks: list[HarborTask] = []
        variants = self.variants()
        for index in range(self.count):
            variant: TaskVariant | None = None
            if index < len(variants):
                parent, source = variants[index]
                position = VARIANT_POSITION
                variant = TaskVariant(
                    parent=parent.name,
                    parent_dir=self.state_dir / PARENTS_DIR / parent.name,
                    direction=parent.direction,
                    pass_rate=parent.pass_rate,
                    rollouts=parent.rollouts,
                )
                proposal = SessionProposal(
                    index, position, source.record_id, parent_task=parent.name, direction=parent.direction
                )
            else:
                position = self.cursor.next_position
                source = self.sources[position % len(self.sources)]
                proposal = SessionProposal(index, position, source.record_id)
            try:
                task = await self.rebuilt(source, index, variant=variant)
            except ProposalRefused as exc:
                refused.append(replace(proposal, designer_record_id=exc.record_id, stage=exc.stage, refusal=str(exc)))
            except AuthorError as exc:
                # An outage must not pass for the author's failure: it ends the generation, and the report names it.
                # So does a task that held the token, which may have left the sandbox.
                error = f"the author could not rebuild {source.record_id}: {exc}"
                stage = STAGE_CREDENTIAL if isinstance(exc, CredentialHeld) else STAGE_OUTAGE
                refused.append(replace(proposal, stage=stage, refusal=error))
            else:
                pending[task.name] = replace(
                    proposal, designer_record_id=str(task.metadata.get("designer_record_id", "")), task_name=task.name
                )
                tasks.append(task)
            # The session or the parent is taken; a restart continues with the next one. A cancelled run takes
            # none, so a parent whose variant was cancelled keeps its slot. Otherwise each parent gets one variant
            # slot, whatever becomes of the variant.
            if variant is None:
                self.save_cursor(replace(self.cursor, next_position=position + 1))
            else:
                self.parents = [
                    replace(kept, varied=kept.varied or kept.name == variant.parent) for kept in self.parents
                ]
                self.save_parents()
            if error:
                logger.error("AutoEnvScaling generation %d stopped: %s", self.generation, error)
                break
        self.save_open(request.generation, OpenGeneration(self.generation, tuple(refused), pending, error))
        return tuple(tasks)

    def open_path(self, generation: int) -> Path:
        return self.state_dir / f"{OPEN_PREFIX}{generation:05d}.json"

    def save_open(self, generation: int, state: OpenGeneration) -> None:
        """Keep an open generation under the state directory, replacing the file whole."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.open_path(generation)
        partial = path.with_name(f"{path.name}.partial")
        partial.write_bytes(OPEN_GENERATION.dump_json(state))
        os.replace(partial, path)

    def save_validation(
        self, task_name: str, checks: tuple[CheckResult, ...], stage: str, path: str, *, error: str = ""
    ) -> None:
        """Add what validate found for a task, and an error that stopped it, to the open generation of the task."""
        for open_path in sorted(self.state_dir.glob(f"{OPEN_PREFIX}*.json")):
            state = OPEN_GENERATION.validate_json(open_path.read_bytes())
            if task_name in state.pending:
                generation = int(open_path.stem[len(OPEN_PREFIX) :])
                self.save_open(
                    generation,
                    replace(
                        state,
                        checks={**state.checks, task_name: checks},
                        stages={**state.stages, task_name: stage},
                        paths={**state.paths, task_name: path} if path else state.paths,
                        error=error or state.error,
                    ),
                )
                return

    async def rebuilt(self, source: SessionSource, index: int, *, variant: TaskVariant | None = None) -> HarborTask:
        """The author's task from one session, or a variant of a parent from it, admitted; its sources are the
        author's last recorded call and the session, so a variant shares its parent's split. An author with no
        recorded call but a transcript (Claude Code) gives a task whose one source is the session, and whose
        metadata names the transcript under the state directory.

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
            if variant is None:
                authored = await asyncio.to_thread(self.author.author, source, work_dir, checks, tags=tags, stop=stop)
            else:
                if self.variant_author is None:
                    raise ValueError("adaptive generation needs a VariantAuthor")
                authored = await asyncio.to_thread(
                    self.variant_author.vary, source, work_dir, checks, tags=tags, stop=stop, variant=variant
                )
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
        if not authored.record_ids and not authored.transcript:
            raise ProposalRefused("the Designer made no model call that Reef recorded", stage=STAGE_AUTHORING)
        record_id = authored.record_ids[-1] if authored.record_ids else ""
        try:
            task = admitted_task(
                authored.reply,
                generation=self.generation,
                index=index,
                record_id=record_id or source.record_id,
                checkout_path=source.repository.path,
            )
        except (ValueError, HarborTaskError) as exc:
            raise ProposalRefused(
                f"the Designer's task is refused: {exc}", stage=STAGE_AUTHORING, record_id=record_id
            ) from exc
        lineage = {} if variant is None else {"parent_task": variant.parent, "variant": variant.direction}
        if record_id:
            return replace(
                task,
                source_agent_record_ids=(record_id, source.record_id),
                metadata={**task.metadata, "designer_record_id": record_id, **lineage},
            )
        designer = {
            "designer_transcript": f"runs/{work_dir.name}/{authored.transcript}",
            "designer_harness": CLAUDE_HARNESS,
        }
        return replace(
            task, source_agent_record_ids=(source.record_id,), metadata={**task.metadata, **designer, **lineage}
        )

    async def validate(self, task: HarborTask) -> TaskValidationResult:
        """The task checks on a written copy, which is removed again; a duplicate or a taken name is invalid.

        A refusing check that finds a defect, or cannot run, makes the task invalid, and the report keeps the two
        apart. An oracle check that cannot run raises: the generator service is not usable for the round. So does
        a refusing calibration whose rollout request the generator service fails.
        """
        try:
            written = await self.service.write_task(task)
        except (DuplicateTask, TaskNameConflict) as exc:
            self.save_validation(task.name, (), STAGE_CHECK, "")
            return TaskValidationResult((f"refused: {exc}",))
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
                    errors = result.errors
                    stage = check.stage
                    break
                if result.could_not_run:
                    errors, stage = (f"the {check.name} check could not run: {result.could_not_run}",), STAGE_UNCHECKED
                    break
        except GeneratorError as exc:
            self.save_validation(
                task.name,
                tuple(results),
                stage,
                str(written.path),
                error=f"the check of {task.name} could not run: {exc}",
            )
            raise
        finally:
            # A copy that was already staged belongs to its writer, which may be running it.
            if written.created:
                await self.service.delete_task(task.name)
        self.save_validation(task.name, tuple(results), stage, str(written.path))
        return TaskValidationResult(errors)

    def keep_parents(
        self, generation: int, outcomes: Mapping[str, TaskOutcome], ran: Mapping[str, HarborTask]
    ) -> None:
        """Keep each training task outside the band as a parent, with its files; the oldest beyond the bound go.

        Only rolled out training tasks reach ``outcomes``: eval tasks and outages never become parents. A task
        whose mean reward is outside [0, 1] has no pass rate to tell a variant, so it becomes no parent either.
        """
        direction_by_place: dict[BandPlace, VariantDirection] = {
            BAND_ABOVE: VARIANT_HARDER,
            BAND_BELOW: VARIANT_SIMPLER,
        }
        known = {parent.name for parent in self.parents}
        root = self.state_dir / PARENTS_DIR
        for name, outcome in outcomes.items():
            if outcome.band == BAND_INSIDE or name in known:
                continue
            if not 0 <= outcome.pass_rate <= 1:
                logger.warning(
                    "AutoEnvScaling keeps no parent for %s: its mean reward %s is outside [0, 1]",
                    name,
                    outcome.pass_rate,
                )
                continue
            parent = ParentTask(
                name=name,
                source_record_id=outcome.source_record_id,
                generation=generation,
                pass_rate=outcome.pass_rate,
                rollouts=len(outcome.rollout_rewards),
                direction=direction_by_place[outcome.band],
            )
            write_harbor_task(ran[name], root)
            self.parents.append(parent)
        while len(self.parents) > MAX_PARENTS:
            dropped = self.parents.pop(0)
            shutil.rmtree(root / dropped.name, ignore_errors=True)
        self.save_parents()

    async def record_results(self, results: tuple[GeneratedTaskResult, ...], *, generation: int) -> None:
        """Close an open generation: save training pass rates, band measurements and held-out outcomes in its report.

        An eval task gets the stage ``held_out``. The band is only recorded in the report. It does not change the
        recipe's task store. With ``is_adaptive``, each training task outside the band also becomes a parent
        (:meth:`keep_parents`).
        """
        open_path = self.open_path(generation)
        if not open_path.is_file():
            logger.info("AutoEnvScaling generation %d of scenario %r is already closed", generation, self.scenario)
            return
        state = OPEN_GENERATION.validate_json(open_path.read_bytes())
        pending = dict(state.pending)
        proposals = list(state.refused)
        outcomes: dict[str, TaskOutcome] = {}
        ran: dict[str, HarborTask] = {}
        for result in results:
            proposal = pending.pop(result.task.name)
            if isinstance(result, HeldOutTask):
                proposals.append(replace(proposal, stage=STAGE_HELD_OUT))
                continue
            if isinstance(result, RejectedTask):
                stage = state.stages.get(result.task.name) or STAGE_RUN
                proposals.append(replace(proposal, stage=stage, refusal=result.reason))
                continue
            if isinstance(result, FailedTask):
                if result.infrastructure_failure:
                    stage = STAGE_OUTAGE
                else:
                    stage = state.stages.get(result.task.name) or STAGE_RUN
                proposals.append(replace(proposal, stage=stage, refusal=result.error))
                continue
            pass_rate = statistics.fmean(result.rollout_rewards)
            band = band_place(pass_rate, self.band.low, self.band.high)
            outcome = TaskOutcome(
                name=result.task.name,
                digest=result.task.digest,
                path=state.paths.get(result.task.name, ""),
                source_record_id=proposal.source_record_id,
                rollout_rewards=result.rollout_rewards,
                pass_rate=pass_rate,
                band=band,
            )
            outcomes[outcome.name] = outcome
            ran[outcome.name] = result.task
            if outcome.band == BAND_INSIDE:
                proposals.append(proposal)
                continue
            proposals.append(
                replace(proposal, stage=STAGE_BAND, refusal=f"outside the solver band: pass rate {pass_rate:.2f}")
            )
        proposals.extend(
            replace(proposal, stage=STAGE_UNREACHED, refusal="the round ended before it ran the task")
            for proposal in pending.values()
        )
        proposals.sort(key=lambda proposal: proposal.index)
        document = {
            "generation": state.number,
            "error": state.error,
            "band": {"low": self.band.low, "high": self.band.high},
            # A calibration that is off leaves no trace in the report.
            **({} if self.calibration is None else {"calibration": self.calibration.settings()}),
            "next_position": self.cursor.next_position,
            "proposals": [
                {**proposal_document(proposal), "checks": checks_document(state, proposal.task_name)}
                for proposal in proposals
            ],
            "tasks": [
                {**asdict(outcome), "checks": checks_document(state, outcome.name)} for outcome in outcomes.values()
            ],
        }
        if self.is_adaptive:
            self.keep_parents(state.number, outcomes, ran)
        report_path = self.state_dir / f"generation-{state.number:05d}.json"
        report_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        open_path.unlink()
        logger.info(
            "AutoEnvScaling generation %d on scenario %r: %d tasks inside the band of %d that ran, %d sessions; %s",
            state.number,
            self.scenario,
            sum(outcome.band == BAND_INSIDE for outcome in outcomes.values()),
            len(outcomes),
            len(proposals),
            report_path,
        )


def proposal_document(proposal: SessionProposal) -> dict[str, object]:
    """A proposal as the report keeps it; a session rebuilt from scratch has no variant fields."""
    document = asdict(proposal)
    if not proposal.parent_task:
        document.pop("parent_task")
        document.pop("direction")
    return document


def checks_document(state: OpenGeneration, task_name: str | None) -> list[dict[str, object]]:
    """Each check's result for a task, as the report keeps it; none for a session that became no task."""
    return [asdict(result) for result in state.checks.get(task_name or "", ())]
