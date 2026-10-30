"""Claude Code as the AutoEnvScaling author: Reef's pinned Claude Code in a cloud sandbox, signed in with OAuth.

:class:`ClaudeCodeAuthor` runs Claude Code headless (``claude -p``) on the inputs, rules and workspace that pi
gets (:class:`~recipes.beta.autoenvscaling.author.AgentAuthor`). Claude Code sends every model call itself, straight
to Anthropic, with the token in its process environment, so Reef records none of them: the task names its session
and Claude Code's transcript instead of a model-call record. The agent checks its task with the ``task_check.py``
command, which reaches a loopback gateway (:class:`CheckGateway`) that runs the checks and answers every model
route with 404.

The token stays out of every command line, log, error text, file of the run's directory and task. The sandbox
gets it through the sandbox process environment, without placing values in command arguments. Claude Code removes it
from the environment of its Bash, hook and MCP subprocesses (``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB``). The host redacts every
text it logs or raises and every file and file name of the run's directory, refuses a check of a task that holds
the token, and admits no such task (:class:`CredentialHeld`).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from reef_client.client import ReefClient

from recipes.beta.autoenvscaling.author import (
    AGENT_PROMPT,
    FINISH_SECONDS,
    RECORD_READ_TIMEOUT_SECONDS,
    VARIANT_INPUTS_TEXT,
    AgentTools,
    AuthoredTask,
    AuthorError,
    CandidateChecks,
    SeedVariantAuthor,
    TaskVariant,
    agent_prompt,
    agent_rules,
    close_when_stopped,
    repository_snapshot,
    workspace_reply,
    write_inputs,
)
from recipes.beta.autoenvscaling.author_credentials import CredentialChecks, CredentialHeld, clean_run, scrubbed
from recipes.beta.autoenvscaling.check_gateway import CheckGateway
from recipes.beta.autoenvscaling.sources import SessionSource
from reef.execution.sandbox.environment import execute
from reef.execution.sandbox.process import ProgramExecutor, RolloutLaunchError, RolloutTimeout, SandboxUnavailable
from reef.harness.adapters import get_adapter
from reef.harness.rollouts.vendor_install import AdapterProgram
from reef.harness.tree.render import render_composition

logger = logging.getLogger(__name__)

#: The author_harness name of this author, and the designer_harness of its tasks.
CLAUDE_HARNESS = "claude-code"
#: The variable Claude Code reads its OAuth token from, in the sandbox.
TOKEN_VARIABLE = "CLAUDE_CODE_OAUTH_TOKEN"
#: How every Claude Code OAuth token (``claude setup-token``) starts.
TOKEN_PREFIX = "sk-ant-oat"
#: The tools the agent has, all of them allowed: the scrub switch keeps the permission mode at default.
AUTHOR_TOOLS = "Read,Glob,Grep,Write,Edit,Bash"
CHECK_SCRIPT = Path(__file__).with_name("task_check.py")
BASH_TIMEOUT_MS = "600000"
#: The variables a run sets for Claude Code beside its adapter's own (``CLAUDE_CONFIG_DIR`` and the switches that
#: turn off its updates and telemetry). A deployment's ``env_from`` would override them in the sandbox.
RUN_VARIABLES = frozenset(
    {
        "HOME",
        "DISABLE_ERROR_REPORTING",
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB",
        "BASH_DEFAULT_TIMEOUT_MS",
        "BASH_MAX_TIMEOUT_MS",
        "REEF_AUTHOR_URL",
        "REEF_AUTHOR_DEADLINE_MS",
        "REEF_AUTHOR_FINISH_SECONDS",
        "REEF_AUTHOR_CHECK",
    }
)
#: The most characters of Claude Code's own text that an error message keeps.
ERROR_TEXT_CHARS = 500
#: A Claude Code session id names its transcript file.
SESSION_ID = re.compile(r"[0-9A-Za-z-]+")
# Appended to the agent's rules: task_check is a command here, not a tool.
CLAUDE_CHECK_TEXT = """
HOW TO CHECK:
- task_check is a command: run `python3 "$REEF_AUTHOR_CHECK"` with Bash. It prints the check's result as JSON, then
  your remaining time.
- When it prints `state: running`, the check still runs: run the command it prints
  (`python3 "$REEF_AUTHOR_CHECK" --wait <check_id>`) to wait for that check's result. While a check runs,
  `python3 "$REEF_AUTHOR_CHECK"` also waits for it and starts no other check.
- When it prints "Finish now", stop with the best task you have in ./task/.
"""


def overrides_run(name: str) -> bool:
    """Whether a deployment's sandbox variable ``name``, which wins over the run's own, would change how Claude Code
    signs in, where it sends its calls or how the run sets it up: an ``ANTHROPIC_*`` or ``CLAUDE_*`` variable, a
    base URL, a proxy, or a variable the run sets."""
    upper = name.upper()
    return (
        upper.startswith(("ANTHROPIC_", "CLAUDE_"))
        or "BASE_URL" in upper
        or "PROXY" in upper
        or upper in RUN_VARIABLES
        or upper in get_adapter("claude").env
    )


def claude_result(stdout: str) -> dict[str, object] | None:
    """The result object ``claude -p --output-format json`` printed last, or ``None`` when there is none: the whole
    output as one object, else its lines from the last."""
    for text in (stdout, *reversed(stdout.strip().splitlines())):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") == "result":
            return value
    return None


def transcript_path(work_dir: Path, session_id: str) -> str:
    """The path, relative to ``work_dir``, of Claude Code's transcript of ``session_id``, or of its newest session
    when that one is not there; ``""`` when Claude Code wrote none."""
    projects = work_dir / "claude" / "projects"
    found = sorted(projects.glob(f"*/{session_id}.jsonl")) if SESSION_ID.fullmatch(session_id) else []
    if not found:
        found = sorted(projects.glob("*/*.jsonl"), key=lambda path: path.stat().st_mtime)
    return found[-1].relative_to(work_dir).as_posix() if found else ""


class ClaudeCodeAuthor(SeedVariantAuthor):
    """Reef's pinned Claude Code in a cloud sandbox, signed in with ``claude_token`` and asking for ``model``.

    Values: ``reef_url`` and ``token`` read the source sessions, and ``cache_dir`` keeps the repository snapshots.
    ``claude_token`` is a Claude Code OAuth token (it starts with ``sk-ant-oat``). One run lasts at most
    ``timeout_seconds`` (positive) and at most ``max_turns`` (positive) agent turns, Claude Code's ``--max-turns``.
    With ``has_behaviour_rule`` the rules include the behaviour rule.
    Purpose: the generator builds one for ``author_harness: claude-code``; :meth:`author` runs Claude Code on one
    session, :meth:`vary` on a parent of that session and :meth:`vary_seed` on a seeded parent, and each returns its
    task with no recorded model call and the path of its transcript.
    """

    def __init__(
        self,
        *,
        reef_url: str,
        token: str | None,
        claude_token: str,
        model: str,
        executor: ProgramExecutor,
        cache_dir: Path,
        timeout_seconds: float,
        max_turns: int,
        has_behaviour_rule: bool = True,
    ) -> None:
        if not claude_token.startswith(TOKEN_PREFIX):
            raise ValueError(f"the Claude Code token must be an OAuth token, which starts with {TOKEN_PREFIX}")
        if not model or timeout_seconds <= 0 or max_turns < 1:
            raise ValueError("the Claude Code author needs a model, a positive timeout and positive turns")
        self.reef = ReefClient(reef_url, token=token, timeout_s=RECORD_READ_TIMEOUT_SECONDS)
        self.claude_token = claude_token
        self.model = model
        self.executor = executor
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.max_turns = max_turns
        self.has_behaviour_rule = has_behaviour_rule
        self.descriptor = get_adapter("claude")

    def author(
        self,
        source: SessionSource,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
    ) -> AuthoredTask:
        return self.write_task(source, work_dir, checks, tags=tags, stop=stop)

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
        return self.write_task(source, work_dir, checks, tags=tags, stop=stop, variant=variant)

    def vary_seed(
        self,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant,
    ) -> AuthoredTask:
        return self.write_task(None, work_dir, checks, tags=tags, stop=stop, variant=variant)

    def write_task(
        self,
        source: SessionSource | None,
        work_dir: Path,
        checks: CandidateChecks,
        *,
        tags: Mapping[str, str],
        stop: threading.Event,
        variant: TaskVariant | None = None,
    ) -> AuthoredTask:
        """Claude Code's task from ``source``, or with ``variant`` a variant of its parent; a ``source`` of None is a
        variant of a seeded parent, written from the parent alone. ``tags`` go unused, since Reef records none of its
        model calls.

        Raises :class:`CredentialHeld` when the task holds the token: the task is not admitted.
        """
        inputs = work_dir / "inputs"
        task_dir = work_dir / "workspace" / "task"
        write_inputs(self.reef, source, work_dir, variant)
        try:
            if source is not None:
                shutil.copytree(repository_snapshot(source, self.cache_dir), inputs / "repo", symlinks=True)
            tools = AgentTools(work_dir, CredentialChecks(checks, token=self.claude_token))
            gateway = CheckGateway(tools)
            gateway.start()
            try:
                transcript = self.run(
                    work_dir,
                    gateway,
                    tools,
                    stop,
                    extra_rules="" if variant is None or source is None else VARIANT_INPUTS_TEXT,
                    prompt=agent_prompt(variant, source is None),
                    is_seeded=source is None,
                )
            finally:
                gateway.stop()
        finally:
            is_held = clean_run(work_dir, self.claude_token)
        if is_held:
            raise CredentialHeld("the agent's task held the run's Claude Code token, so it is not admitted")
        # The transcript's path names the file as renamed.
        transcript = scrubbed(transcript, self.claude_token)
        return AuthoredTask(reply=workspace_reply(task_dir), record_ids=(), transcript=transcript)

    def run(
        self,
        work_dir: Path,
        gateway: CheckGateway,
        tools: AgentTools,
        stop: threading.Event,
        *,
        extra_rules: str = "",
        prompt: str = AGENT_PROMPT,
        is_seeded: bool = False,
    ) -> str:
        """Claude Code in a sandbox over the workspace until it stops, its time runs out or ``stop`` is set;
        ``extra_rules`` extends the authoring rules and ``prompt`` states the task to write. With ``is_seeded`` the
        authoring rules are the seed rules (:func:`agent_rules`): the inputs hold no session. Answers the path of its
        transcript, relative to ``work_dir``, or ``""``.

        Raises :class:`AuthorError` when the run could not start or was stopped, and when Claude Code ended with an
        error other than its turn limit: an outage (a spent plan, a lost sign-in) must not pass for the agent's
        failure. Every message is redacted.
        """
        if stop.is_set():
            raise AuthorError("the agent's run was stopped before it started")
        executor, binary = self.executor.for_program(AdapterProgram(self.descriptor))
        rules = agent_rules(tools.checks, self.has_behaviour_rule, extra_rules, is_seeded)
        # The rules are Claude Code's CLAUDE.md, and settings.json its defaults: no model binding, so no endpoint.
        files = render_composition((("rules", {"text": rules + CLAUDE_CHECK_TEXT}),), self.descriptor)
        files["tools/task_check.py"] = CHECK_SCRIPT.read_text(encoding="utf-8")
        for relative, text in files.items():
            target = work_dir / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        # Claude Code's config directory and a home of its own are the run's writable state.
        state = (work_dir / "claude", work_dir / "home")
        for path in state:
            path.mkdir(parents=True, exist_ok=True)
        env = {key: value.replace("{root}", str(work_dir)) for key, value in self.descriptor.env.items()}
        env.update(
            {
                TOKEN_VARIABLE: self.claude_token,
                "HOME": str(work_dir / "home"),
                "DISABLE_ERROR_REPORTING": "1",
                # Claude Code removes its token from its Bash, hook and MCP subprocesses.
                "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
                "BASH_DEFAULT_TIMEOUT_MS": BASH_TIMEOUT_MS,
                "BASH_MAX_TIMEOUT_MS": BASH_TIMEOUT_MS,
                "REEF_AUTHOR_URL": gateway.base_url,
                "REEF_AUTHOR_DEADLINE_MS": str(int((time.time() + self.timeout_seconds) * 1000)),
                "REEF_AUTHOR_FINISH_SECONDS": str(FINISH_SECONDS),
                "REEF_AUTHOR_CHECK": str(work_dir / "tools" / "task_check.py"),
            }
        )
        argv = [
            binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            self.model,
            "--max-turns",
            str(self.max_turns),
            "--tools",
            AUTHOR_TOOLS,
            "--allowedTools",
            AUTHOR_TOOLS,
            "--disable-slash-commands",
            "--strict-mcp-config",
        ]
        started = time.monotonic()
        session: ProgramExecutor | None = None
        finished = threading.Event()
        try:
            session = executor.open_session(forward_ports=(gateway.port,), timeout_s=self.timeout_seconds)
            tools.session = session
            threading.Thread(
                target=close_when_stopped, args=(stop, finished, session), name="autoenvscaling-stop", daemon=True
            ).start()
            outcome = execute(
                session,
                argv,
                root=work_dir,
                workspace=work_dir / "workspace",
                env=env,
                timeout=self.timeout_seconds,
                writable_paths=state,
                readonly_paths=tuple(work_dir / PurePosixPath(relative) for relative in files),
            )
        except RolloutTimeout:
            # The sandbox copied the workspace back: the task stands as the agent left it.
            logger.info("AutoEnvScaling Claude Code timed out after %.0f s", time.monotonic() - started)
            return transcript_path(work_dir, "")
        except (RolloutLaunchError, SandboxUnavailable) as exc:
            if stop.is_set():
                raise AuthorError("the agent's run was stopped") from None
            raise AuthorError(f"Claude Code could not run: {scrubbed(str(exc), self.claude_token)}") from None
        finally:
            finished.set()
            if session is not None:
                try:
                    session.close()
                except RolloutLaunchError as exc:
                    logger.warning(
                        "the AutoEnvScaling agent's sandbox did not stop; its provider timeout applies: %s",
                        scrubbed(str(exc), self.claude_token),
                    )
        result = claude_result(outcome.stdout)
        if result is None:
            tail = scrubbed(outcome.stderr.strip() or outcome.stdout.strip(), self.claude_token)[-ERROR_TEXT_CHARS:]
            raise AuthorError(f"Claude Code exited {outcome.exit_code} without a result: {tail}")
        # The result holds the agent's own text, so each field logged or raised is redacted.
        subtype = scrubbed(str(result.get("subtype")), self.claude_token)
        errors = result.get("errors")
        text = str(result.get("result") or ("; ".join(map(str, errors)) if isinstance(errors, list) else ""))
        logger.info(
            "AutoEnvScaling Claude Code ended with %s after %.0f s and %s turns",
            subtype,
            time.monotonic() - started,
            scrubbed(str(result.get("num_turns")), self.claude_token),
        )
        transcript = transcript_path(work_dir, str(result.get("session_id") or ""))
        # The turn limit is the agent's own: its workspace stands, as after a timeout.
        if subtype == "error_max_turns":
            return transcript
        is_api_error = text.startswith("API Error") or "Please run /login" in text
        if (
            subtype != "success"
            or result.get("is_error") is not False
            or result.get("api_error_status") is not None
            or is_api_error
        ):
            cut = scrubbed(text, self.claude_token)[:ERROR_TEXT_CHARS]
            raise AuthorError(f"Claude Code's run ended with an error ({subtype}): {cut}")
        return transcript
