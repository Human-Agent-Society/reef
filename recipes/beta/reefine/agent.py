"""Reefine's agent: a pi session that makes the change a request asks for in the step's tree, and tries it.

The text answer (:mod:`recipes.beta.reefine.answer`) writes a request's files
in a few model calls and never runs them. The agent gives the request to a
coding agent instead: pi, the harness being evolved, working in the step's
``harness/`` directory itself, with the network and three tools of its own.
``harness_check`` runs the tree through the step's validation;
``harness_trial`` runs an online task or scripted session checks;
``harness_progress`` saves findings that stay available after compaction.
When the agent stops, its tree is read back as the step's candidate and
reviewed like a text answer.

The agent's current directory is ``harness/``. Beside it, in the step's
workspace, it writes ``output/design.md``, ``output/requires.json`` and
``output/progress.md``, and reads the API reference in ``scratch/``; its own
configuration and sessions are under ``scratch/agent``. The run keeps its
checks, its session log and, when the candidate is not the tree the agent
left, that tree in ``output/``.

The agent holds no credential. Online model and provider calls go through a
loopback gateway (:mod:`recipes.beta.reefine.agent_gateway`) that spends from
the step's model-call budget and records into the step's ``calls.json``. Its
isolation is the executor the recipe built for it
(``evolution.agent.sandbox``). Scripted trials use a fixed local
model in that same executor.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from recipes.beta.reefine.agent_gateway import ReefineGateway
from recipes.beta.reefine.answer import (
    EXTENSION_ADAPTER,
    PI_EXTENSION_API,
    Answer,
    client_text,
    failures_section,
    kept_design,
    request_kinds,
    review_answer,
    screened_requires,
    undeclared_env,
)
from recipes.beta.reefine.multimodal import MultimodalProvider
from recipes.beta.reefine.prompts import prompt
from recipes.beta.reefine.trial import trial_script
from reef.core.prompts import untrusted_text
from reef.core.training_request import TrainingRequest
from reef.harness.adapters import get_adapter
from reef.harness.adapters.descriptor import AdapterDescriptor
from reef.harness.episodes.model_binding import ModelBinding
from reef.harness.episodes.run import headless_argv
from reef.harness.episodes.trajectory import final_assistant_text
from reef.harness.gateway import AgentTool
from reef.harness.tree import HarnessTree, TreePatch, locate, render_tree
from reef.sandbox.environment import download_directory
from reef.sandbox.process import EpisodeExecutor, EpisodeLaunchError, EpisodeTimeout, ProcessOutcome
from reef.train.harness_backend import Tools, Workspace
from reef.train.harness_backend.agent.run import AgentExecution, agent_env, launch_agent, read_session

AGENT_RULES = Path(__file__).with_name("agent_rules.md")
AGENT_TOOLS = Path(__file__).with_name("agent_tools.ts")
TRIAL_DRIVER = Path(__file__).with_name("trial_driver.mjs")
#: The agent's own harness: its rules and its tools, as a tree rendered for pi.
AGENT_TOOLS_PATH = "extensions/reef-agent-tools.ts"
#: How much of a trial the agent reads back: enough to see what happened, never a whole transcript.
MAX_TRIAL_TEXT = 4000
MAX_TRIAL_CALLS = 30
MAX_STDERR_CHARS = 2000

#: In the step's ``output/``: what the agent writes beside the tree, and what the run keeps of it.
DESIGN_FILE = "design.md"
REQUIRES_FILE = "requires.json"
CHECKS_FILE = "agent-checks.jsonl"
SESSION_FILE = "agent-session.jsonl"
LEFT_TREE_DIR = "agent-harness"
#: In the step's ``scratch/``: the API reference the agent reads, and pi's own directory for the agent's session.
REFERENCE_FILE = "reference/pi-extension-api.md"
AGENT_HOME = "agent"

#: The prompt that hands the request to the agent; its rules say the rest.
AGENT_PROMPT = prompt("agent_request")


@dataclass(frozen=True)
class ReefineAgent:
    """Where Reefine's agent runs, for how long, and the multimodal gateway its trials reach.

    Values: ``execution`` is ``evolution.agent`` bound to pi: the isolation
    the agent and its trials run under, pi's binary there, and the time limit
    of the whole agent run. ``trial_timeout_s`` > 0 bounds each trial.
    ``provider`` is the deployment's multimodal gateway, ``None`` for none.
    Purpose: ``ReefineRecipe.build_backend`` makes one per scenario;
    ``ReefineMethod`` runs the agent on a request step with it.
    """

    execution: AgentExecution
    trial_timeout_s: float
    provider: MultimodalProvider | None = None

    def __post_init__(self) -> None:
        if self.execution.descriptor.name != EXTENSION_ADAPTER:
            raise ValueError(f"Reefine's agent is {EXTENSION_ADAPTER}, not {self.execution.descriptor.name}")
        if self.trial_timeout_s <= 0:
            raise ValueError("ReefineAgent.trial_timeout_s must be positive")


def agent_rules(provider: MultimodalProvider | None) -> str:
    """The agent's AGENTS.md, with what this deployment's multimodal provider serves and how to list its models."""
    if provider is None:
        note = (
            "This deployment configures no multimodal provider: those routes answer 501. A change that needs one "
            "cannot be delivered here; say so in design.md."
        )
    else:
        routes = ", ".join(f"`{route}`" for route in provider.preset.paths)
        if "/v1/decisions" in provider.preset.paths:
            other_modalities = "`image`, `embeddings`, `decisions`"
        else:
            other_modalities = "`image`, `embeddings`"
        note = (
            f"This deployment's multimodal provider is {provider.preset.name} ({provider.base_url}); it serves "
            f"{routes}, and "
            "any other of these routes answers 501. List its models with "
            f'`curl -s "$REEF_AGENT_URL/models?modality=speech"` (or {other_modalities}).'
        )
    return AGENT_RULES.read_text(encoding="utf-8").replace("<!-- provider -->", note)


def trial_env(gateway_url: str) -> dict[str, str]:
    """Reef's address as a session sees it, pointed at the gateway: an extension calls the multimodal routes at
    ``REEF_SERVICE_URL`` with the scenario and token headers as it will in a user's session, and the gateway
    answers them without reading either."""
    return {
        "REEF_SERVICE_URL": gateway_url,
        "REEF_SCENARIO": "reef-agent-trial",
        "REEF_TOKEN": "reef-agent-trial",
    }


def rendered_files(tree: HarnessTree, binding: ModelBinding, descriptor: AdapterDescriptor) -> dict[str, str]:
    """``tree`` rendered for ``descriptor``'s harness with ``binding`` as its model, root-relative path to text."""
    return render_tree(tree, descriptor, binding.config_overlays(descriptor))


def launch_pi(
    descriptor: AdapterDescriptor,
    binary: str,
    executor: EpisodeExecutor,
    files: Mapping[str, str],
    prompt_text: str,
    env: Mapping[str, str],
    *,
    root: Path,
    timeout: float,
    script: dict[str, object] | None = None,
    home: Path | None = None,
    workspace: Path | None = None,
    writable: Sequence[Path] = (),
) -> tuple[ProcessOutcome, list[Mapping[str, Any]]]:
    """Run pi (``descriptor``'s ``binary``) online over ``files``: the outcome and the session log.

    ``root`` holds everything the process may see. ``home``, ``root`` unless
    given, is pi's own directory: ``files`` are written there, read-only, and
    pi keeps its sessions there. ``workspace``, ``root/workspace`` unless
    given, is the session's working directory, and ``writable`` names other
    directories under ``root`` it may write. ``script`` runs the scripted
    trial driver in place of pi.
    """
    home = root if home is None else home
    session_env = agent_env(descriptor, binary, home)
    # An agent and its trials run online: pi's offline switch is for benchmark episodes.
    session_env.pop("PI_OFFLINE", None)
    argv = headless_argv(descriptor, binary, prompt_text)
    if script is not None:
        home.mkdir(parents=True, exist_ok=True)
        (home / "trial-script.json").write_text(json.dumps(script), encoding="utf-8")
        driver = home / "trial-driver.mjs"
        shutil.copyfile(TRIAL_DRIVER, driver)
        argv = ["node", str(driver)]
    outcome = launch_agent(
        descriptor,
        executor,
        argv,
        files,
        {**session_env, **env},
        root=root,
        home=home,
        cwd=root / "workspace" if workspace is None else workspace,
        timeout=timeout,
        writable=writable,
    )
    return outcome, read_session(descriptor, home)


def tool_calls(trajectory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The tools a pi session called and the ones that failed, in order, each cut to what a reader needs."""
    calls: list[dict[str, Any]] = []
    for event in trajectory:
        inner = event.get("message")
        message: Mapping[str, Any] = inner if isinstance(inner, Mapping) else event
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "toolCall":
                    arguments = json.dumps(part.get("arguments"), ensure_ascii=False, default=str)
                    calls.append({"tool": part.get("name"), "arguments": arguments[:300]})
        elif role == "toolResult" and message.get("isError"):
            text = json.dumps(content, ensure_ascii=False, default=str)
            calls.append({"tool": message.get("toolName"), "error": text[:300]})
    return calls[-MAX_TRIAL_CALLS:]


def agent_requires(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The requires items the agent wrote to ``path``, through the same screens as a text answer's, and the refused
    ones."""
    try:
        items = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    except json.JSONDecodeError as error:
        return [], [{"item": REQUIRES_FILE, "reason": f"not JSON: {error}"}]
    if not isinstance(items, list):
        return [], [{"item": REQUIRES_FILE, "reason": "not a JSON array"}]
    return screened_requires(items)


def unread_lines(left_out: Mapping[str, str], paths: Iterable[str]) -> list[str]:
    """One line for each path of ``left_out`` with its reason, in order. A path inside a directory that holds none
    of ``paths`` on the layout is named by the outermost such directory instead, once for the directory."""
    layout_directories = {str(parent) for path in paths if locate(path) for parent in PurePosixPath(path).parents}
    lines: dict[str, None] = {}
    for relative, reason in left_out.items():
        # The parents from the outermost down; the tree's root holds the layout itself, so it is never named.
        parents = reversed(PurePosixPath(relative).parents[:-1])
        outside = next((str(parent) for parent in parents if str(parent) not in layout_directories), None)
        if outside is None:
            lines[f"harness/{relative} {reason}, so it is not read"] = None
        else:
            lines[f"harness/{outside}/ holds no file of the harness layout, so nothing in it is read"] = None
    return list(lines)


class AgentRun:
    """One agent run in a step's workspace, and what its two tools, ``check`` and ``trial``, do with its tree.

    Values: ``workspace`` is the step's workspace, whose ``harness/`` the
    agent edits; ``tools`` checks trees and meters the step; ``head`` is the
    tree the step started from; ``agent`` says where and how long the agent
    and its trials run; ``descriptor`` is pi's. ``served`` is the step's
    served model. ``gateway`` and ``session`` are ``None`` until the run
    starts them. ``trials`` ≥ 0 counts the trials run; ``deadline`` (on
    ``time.monotonic``) ends the run, and no trial runs into its last
    ``finish_seconds``. ``checks`` holds every check and trial result in
    order.
    Purpose: ``answer_with_agent`` makes one per request step; the gateway's
    ``CheckTool`` and ``TrialTool`` call ``check`` and ``trial`` while the
    agent works, and the step reads the candidate with ``read`` when the
    agent stops.
    """

    def __init__(self, workspace: Workspace, tools: Tools, head: HarnessTree, agent: ReefineAgent) -> None:
        self.workspace = workspace
        self.tools = tools
        self.head = head
        self.agent = agent
        self.descriptor = get_adapter(EXTENSION_ADAPTER)
        self.served = tools.models.served
        self.gateway: ReefineGateway | None = None
        self.session: EpisodeExecutor | None = None
        #: pi runs the tool calls of one turn concurrently, and the gateway serves each on its own thread; the
        #: pull replaces the tree's directory, so one call at a time reads it.
        self.workspace_lock = threading.Lock()
        self.trials = 0
        self.deadline = time.monotonic() + agent.execution.timeout_s
        self.finish_seconds = min(60.0, agent.execution.timeout_s * 0.1)
        self.checks: list[dict[str, Any]] = []

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def record_check(self, kind: str, result: dict[str, Any]) -> dict[str, Any]:
        """Keep runner observations outside the model's compacted conversation."""
        result["remaining_seconds"] = round(self.remaining_seconds(), 1)
        with self.workspace_lock:
            self.checks.append({"kind": kind, **result})
        return result

    def write_checks(self) -> None:
        """Write the checks and trials to ``output/agent-checks.jsonl``, once the agent's files are back: a cloud
        sandbox's copy replaces the step's directory when the agent stops."""
        if self.checks:
            lines = "".join(json.dumps(check, ensure_ascii=False) + "\n" for check in self.checks)
            (self.workspace.output / CHECKS_FILE).write_text(lines, encoding="utf-8")

    def executor(self) -> EpisodeExecutor:
        """The run's session, which reaches the gateway; the agent's executor before the run starts."""
        return self.agent.execution.executor if self.session is None else self.session

    def snapshot(self) -> tuple[HarnessTree, list[str]]:
        """Pull the agent's tree from its session and read it (see ``read``)."""
        with self.workspace_lock:
            # The agent may edit a copy off the Reef host; a check reads that copy as it stands now.
            download_directory(self.executor(), self.workspace.root, self.workspace.harness.name)
            return self.read()

    def read(self) -> tuple[HarnessTree, list[str]]:
        """The candidate in the agent's tree, and one line for each changed path it leaves out.

        The candidate is the head tree with the agent's changes to the files a
        request may write on pi. Every other change is left out and named: a
        path off the layout, a file of a kind a request may not change, a
        symbolic link and a special file. A directory that holds no file of
        the layout, such as the ``node_modules/`` of an install, is named
        once. ``ValueError`` when a file a request may write is a symbolic
        link, is not a regular file, or is not UTF-8 text.
        """
        kinds = request_kinds(self.descriptor.name)
        root = self.workspace.harness
        found: dict[str, Path] = {}
        for directory, subdirectories, names in os.walk(root):
            base = Path(directory)
            # A link to a directory is one path of the tree: os.walk lists it with the directories, never enters it.
            for name in (*names, *(entry for entry in subdirectories if (base / entry).is_symlink())):
                found[(base / name).relative_to(root).as_posix()] = base / name
        kept: dict[str, str | None] = {}
        left_out: dict[str, str] = {}
        for relative in sorted(found.keys() | self.head.files.keys()):
            location = locate(relative)
            path = found.get(relative)
            head_text = self.head.files.get(relative)
            if location is not None and location.kind in kinds:
                if path is None:
                    kept[relative] = None
                    continue
                if path.is_symlink() or not path.is_file():
                    shape = "a symbolic link" if path.is_symlink() else "not a regular file"
                    raise ValueError(f"harness/{relative} is {shape}")
                try:
                    text = path.read_bytes().decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"harness/{relative} is not UTF-8 text: {error}") from error
                if text != head_text:
                    kept[relative] = text
            elif path is not None and path.is_symlink():
                left_out[relative] = "is a symbolic link"
            elif path is not None and not path.is_file():
                left_out[relative] = "is not a regular file"
            elif location is None:
                # The head tree is valid, so it holds no path off the layout: every such path is the agent's.
                left_out[relative] = "is not a path of the harness layout"
            elif path is None or head_text is None or path.read_bytes() != head_text.encode("utf-8"):
                left_out[relative] = "is a file a request may not change"
        return self.head.apply(TreePatch(kept)), unread_lines(left_out, (*found, *self.head.files))

    def check(self) -> dict[str, Any]:
        try:
            candidate, unread = self.snapshot()
        except ValueError as error:
            self.tools.calls.note("check", f"the tree cannot be read: {error}", failed=True)
            return self.record_check("check", {"valid": False, "problems": [str(error)]})
        # The tree as it stands, the head included: the step skips an unchanged tree, the agent's checks do not.
        validation = self.tools.check(candidate)
        changes = self.head.diff(candidate).changes
        result: dict[str, Any] = {
            "valid": validation.passed,
            "files": {path: "removed" if text is None else "written" for path, text in changes.items()},
            "candidate": candidate.checksum() if validation.passed else None,
        }
        if not validation.passed:
            result["problems"] = list(validation.problems)
            self.tools.calls.note("check", f"validation refused the tree: {validation.message()}", failed=True)
        else:
            self.tools.calls.note("check", f"validation passed: {', '.join(changes) or 'the head tree, unchanged'}")
        if unread:
            result["unread"] = unread
        return self.record_check("check", result)

    def trial(self, task: str, script: dict[str, object] | None = None) -> dict[str, Any]:
        if self.gateway is None:
            raise RuntimeError("the agent gateway is not running")
        try:
            candidate, unread = self.snapshot()
        except ValueError as error:
            self.tools.calls.note("trial", f"not run, the tree cannot be read: {error}", failed=True)
            return self.record_check("trial", {"ran": False, "refusal": str(error)})
        validation = self.tools.check(candidate)
        if not validation.passed:
            self.tools.calls.note(
                "trial", f"not run, validation refused the tree: {validation.message()}", failed=True
            )
            return self.record_check("trial", {"ran": False, "refusal": validation.message(), "unread": unread})
        timeout = min(self.agent.trial_timeout_s, self.remaining_seconds() - self.finish_seconds)
        if timeout <= 0:
            return self.record_check(
                "trial", {"ran": False, "refusal": "finish the current candidate; trial budget exhausted"}
            )
        self.trials += 1
        self.tools.calls.note(
            "trial", f"trial {self.trials} running the changed harness: {task or 'scripted session'}"
        )
        binding = ModelBinding(base_url=self.gateway.base_url, model=self.served.model, api=self.served.api)
        files = rendered_files(candidate, binding, self.descriptor)
        before = self.gateway.provider_call_count()
        # A trial runs where a session of the person's would: outside the step's directory.
        root = Path(tempfile.mkdtemp(prefix="reef-agent-trial-"))
        started = time.monotonic()
        checksum = candidate.checksum()
        try:
            outcome, trajectory = launch_pi(
                self.descriptor,
                self.agent.execution.binary,
                self.executor(),
                files,
                task,
                trial_env(self.gateway.base_url),
                root=root,
                timeout=timeout,
                script=script,
            )
            if script is not None:
                report_path = root / "sessions" / "trial-result.json"
                if not report_path.is_file():
                    return self.record_check(
                        "trial",
                        {
                            "ran": True,
                            "passed": False,
                            "candidate": checksum,
                            "error": "scripted driver produced no result",
                            "stderr_tail": outcome.stderr[-MAX_STDERR_CHARS:],
                        },
                    )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result = {"ran": True, "mode": "scripted", **report, "exit_code": outcome.exit_code}
                self.tools.calls.note(
                    "trial",
                    f"trial {self.trials}: scripted assertions {'passed' if report['passed'] else 'failed'}",
                    failed=not report["passed"],
                )
                return self.record_check(
                    "trial", {**result, "candidate": checksum, "seconds": round(time.monotonic() - started, 1)}
                )
        except EpisodeTimeout:
            limit = f"the trial ran past its {timeout:g} s limit"
            self.tools.calls.note("trial", f"trial {self.trials}: {limit}", failed=True)
            return self.record_check(
                "trial",
                {
                    "ran": True,
                    "candidate": checksum,
                    "timed_out": limit,
                    "provider_calls": self.gateway.provider_calls_since(before),
                },
            )
        except EpisodeLaunchError as error:
            self.tools.calls.note("trial", f"trial {self.trials} could not start: {error}", failed=True)
            return self.record_check("trial", {"ran": False, "candidate": checksum, "error": str(error)})
        finally:
            shutil.rmtree(root, ignore_errors=True)
        final = final_assistant_text(trajectory) or ""
        made = self.gateway.provider_calls_since(before)
        refused = [call for call in made if int(call.get("status") or 0) >= 400]
        summary = f"trial {self.trials} exited {outcome.exit_code} in {time.monotonic() - started:.0f} s"
        if made:
            summary += f", {len(made)} multimodal call{'s' if len(made) != 1 else ''}"
            summary += f" ({len(refused)} refused)" if refused else " (all answered)"
        self.tools.calls.note("trial", summary, failed=outcome.exit_code != 0 or bool(refused))
        return self.record_check(
            "trial",
            {
                "ran": True,
                "candidate": checksum,
                "seconds": round(time.monotonic() - started, 1),
                "exit_code": outcome.exit_code,
                "final_text": final[:MAX_TRIAL_TEXT],
                "tool_calls": tool_calls(trajectory),
                "provider_calls": made,
                "stderr_tail": outcome.stderr[-MAX_STDERR_CHARS:],
                "changed_files": list(self.head.diff(candidate).changes),
            },
        )


class CheckTool(AgentTool):
    """``check``: the agent's tree through the step's validation, with what a request may not change named."""

    name = "check"
    description = "Validate the tree in harness/ as the step validates a request's change."
    input_schema: ClassVar[Mapping[str, Any]] = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, run: AgentRun) -> None:
        self.run = run

    def call(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.run.check()


class TrialTool(AgentTool):
    """``trial``: one online task, or one scripted session, of the candidate harness."""

    name = "trial"
    description = "Run the candidate harness on one online task, or on one scripted session of checks."
    input_schema: ClassVar[Mapping[str, Any]] = {
        "type": "object",
        "properties": {"task": {"type": "string"}, "script": {"type": "object"}},
        "additionalProperties": False,
    }

    def __init__(self, run: AgentRun) -> None:
        self.run = run

    def call(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        task, script = arguments.get("task"), arguments.get("script")
        if script is not None:
            if task is not None:
                raise ValueError("choose task or script, not both")
            return self.run.trial("", trial_script(script))
        if not isinstance(task, str) or not task.strip():
            raise ValueError("a trial needs a task or script")
        return self.run.trial(task.strip())


def answer_with_agent(
    workspace: Workspace, tools: Tools, head: HarnessTree, request: TrainingRequest, agent: ReefineAgent
) -> Answer:
    """Run the agent on one request in the step's tree and read its work back: the change with the notes the step
    records, or the empty change with the reason under ``failure``. When the answer's tree is not the one the agent
    left, that tree stays in ``output/agent-harness/`` for diagnosis; it is never published."""
    (workspace.output / REQUIRES_FILE).write_text("[]\n", encoding="utf-8")
    reference = workspace.scratch / REFERENCE_FILE
    reference.parent.mkdir(parents=True)
    shutil.copyfile(PI_EXTENSION_API, reference)
    home = workspace.scratch / AGENT_HOME
    run = AgentRun(workspace, tools, head, agent)
    gateway = ReefineGateway(run.served, tools.calls, (CheckTool(run), TrialTool(run)), agent.provider)
    gateway.start()
    run.gateway = gateway
    record: dict[str, Any] = {}
    answer: Answer | None = None
    started = time.monotonic()
    try:
        agent_tree = HarnessTree(
            {"rules.md": agent_rules(agent.provider), AGENT_TOOLS_PATH: AGENT_TOOLS.read_text(encoding="utf-8")}
        )
        binding = ModelBinding(base_url=gateway.base_url, model=run.served.model, api=run.served.api)
        agent_prompt = AGENT_PROMPT.format(
            request=untrusted_text(request.text, "user request"),
            machine=client_text(request),
            failures=failures_section(workspace.traces()),
        )
        env = {"REEF_AGENT_URL": gateway.base_url, **trial_env(gateway.base_url)}
        limit = agent.execution.timeout_s
        run.session = agent.execution.executor.open_session(forward_ports=(gateway.port,), timeout_s=limit)
        run.deadline = time.monotonic() + limit
        env["REEF_AGENT_DEADLINE_MS"] = str(int((time.time() + limit) * 1000))
        env["REEF_AGENT_FINISH_SECONDS"] = str(run.finish_seconds)
        tools.calls.note("agent", f"the coding agent started on the request (at most {limit:g} s)")
        outcome, trajectory = launch_pi(
            run.descriptor,
            agent.execution.binary,
            run.executor(),
            rendered_files(agent_tree, binding, run.descriptor),
            agent_prompt,
            env,
            root=workspace.root,
            timeout=limit,
            home=home,
            workspace=workspace.harness,
            writable=(workspace.output,),
        )
        record["exit_code"] = outcome.exit_code
        if outcome.exit_code != 0:
            record["stderr_tail"] = outcome.stderr[-MAX_STDERR_CHARS:]
        # Pi can exit zero after a failed model response; its final assistant event carries the error.
        for event in reversed(trajectory):
            message = event.get("message")
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            if message.get("stopReason") in ("error", "aborted"):
                record["error"] = str(message.get("errorMessage") or "the model response was interrupted")
            elif message.get("stopReason") == "length":
                # The reply budget ran out mid-answer, so the turn carried no tool call and pi stopped: a
                # reasoning model spends the budget on its reasoning first. Say that, not "changed no file".
                record["error"] = (
                    "the model's reply hit its token budget and carried no answer; raise the harness's "
                    "maxTokens (Reef renders the model binding's max_output_tokens into it)"
                )
            break
    except EpisodeTimeout:
        record["timed_out"] = True
    except EpisodeLaunchError as error:
        tools.calls.note("agent", f"the coding agent run failed: {error}", failed=True)
        answer = failed({}, f"the agent run failed: {error}")
    finally:
        try:
            if run.session is not None:
                run.session.close()
        finally:
            gateway.stop()
            # The executor copies the agent's files back even after a timeout; the run's own records follow them.
            record["seconds"] = round(time.monotonic() - started, 1)
            record["trials"] = run.trials
            run.write_checks()
            keep_session(home / run.descriptor.trajectory_path, workspace.output / SESSION_FILE)
    if answer is None:
        ended = "ran past its limit" if record.get("timed_out") else f"exited {record.get('exit_code')}"
        tools.calls.note(
            "agent",
            f"the coding agent {ended} after {record['seconds']:g} s and {run.trials} trials; reading its tree",
            failed=bool(record.get("timed_out") or record.get("error")) or record.get("exit_code", 0) != 0,
        )
        answer = read_answer(run, request, record)
    keep_left_tree(workspace, head.apply(answer.patch))
    return answer


def read_answer(run: AgentRun, request: TrainingRequest, record: dict[str, Any]) -> Answer:
    """The agent's work as the step's answer: its change, design and requires, reviewed."""
    notes: dict[str, Any] = {"agent": record}
    design_path = run.workspace.output / DESIGN_FILE
    design = design_path.read_text(encoding="utf-8", errors="replace").strip() if design_path.is_file() else ""
    if design:
        notes["design"] = kept_design(design)
    if record.get("timed_out"):
        return failed(notes, f"the agent ran past its {record['seconds']:g} s limit")
    if record.get("error"):
        return failed(notes, f"the agent's model response failed: {record['error']}")
    try:
        candidate, unread = run.read()
    except ValueError as error:
        return failed(notes, f"the agent's tree cannot be read: {error}")
    if unread:
        notes["unread"] = unread
    patch = run.head.diff(candidate)
    if not patch.changes:
        if record.get("exit_code", 0) != 0:
            tail = str(record.get("stderr_tail", "")).strip().splitlines()[-1:] or ["no stderr"]
            return failed(notes, f"the agent exited with {record['exit_code']}: {tail[0]}")
        return failed(notes, "the agent changed no file")
    own = [dict(item) for item in request.requires]
    added, refused = agent_requires(run.workspace.output / REQUIRES_FILE)
    if refused:
        notes["refused_requires"] = refused
    review, review_failure = review_answer(
        run.tools.models, request.text, design or None, patch, [*own, *added], adapter=run.descriptor.name
    )
    if review is not None:
        notes["review"] = review
    else:
        notes["review_failure"] = review_failure or "the review did not run"
    undeclared = undeclared_env(patch, [*own, *added])
    if undeclared:
        notes["undeclared_env"] = undeclared
    if review is not None and review.get("delivers") is False:
        reason = review["uncovered"][0] if review["uncovered"] else "the files only imitate the behavior"
        return failed(notes, f"the change does not deliver the request: {reason}")
    return Answer(patch, requires=tuple(added), notes=notes)


def failed(notes: Mapping[str, Any], reason: str) -> Answer:
    """An answer that changes nothing, its notes saying why under ``failure``."""
    return Answer(TreePatch({}), notes={**notes, "failure": reason})


def keep_session(sessions: Path, target: Path) -> None:
    """Copy the agent's session logs under ``sessions`` into ``target``, one after another."""
    if not sessions.is_dir():
        return
    logs = sorted(sessions.rglob("*.jsonl"))
    if not logs:
        return
    with open(target, "x", encoding="utf-8") as kept:
        for log in logs:
            kept.write(log.read_text(encoding="utf-8", errors="replace"))


def keep_left_tree(workspace: Workspace, candidate: HarnessTree) -> None:
    """Copy the tree the agent left to ``output/agent-harness/`` when the step's candidate is another one."""
    try:
        left = workspace.tree()
    except ValueError:
        # A symbolic link or a binary file the agent left is no candidate: the copy keeps it as it was.
        left = None
    if left != candidate:
        shutil.copytree(workspace.harness, workspace.output / LEFT_TREE_DIR, symlinks=True)


__all__ = [
    "AGENT_TOOLS",
    "AGENT_TOOLS_PATH",
    "AgentRun",
    "CheckTool",
    "ReefineAgent",
    "TrialTool",
    "answer_with_agent",
    "launch_pi",
    "rendered_files",
    "tool_calls",
]
