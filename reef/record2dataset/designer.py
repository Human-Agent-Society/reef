"""The task designer: what a model is asked for a Harbor task, and how its reply becomes one.

One call asks the served model for one Harbor task: an instruction, a container, a verifier and a
reference solution. The prompt carries the task contract (the rules every task must meet so any Harbor
agent can play it and Harbor can score it), the target the task should test, and whatever text the
method wants the designer to know about earlier tasks (``DesignerRequest.experience_text``). Grounding and
earlier text enter the prompt fenced as data, never as instructions. The reply's ``json`` block is read
into the instruction, the three file mappings and an optional hint. The call goes through Reef, so every
proposal is an inference record with a receipt the method can report against.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from reef_client.client import ReefClient, ReefClientError

from reef.core.tasks.harbor import HarborTaskError, checked_files
from reef.train.cordis_backend.strategies import untrusted_text

DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_TURN_LIMIT = 12
GROUNDING_CHARS = 6000
DESIGNER_TIMEOUT_S = 1800.0
CHAT_PATH = "/v1/chat/completions"
SKILL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
JSON_BLOCK = re.compile(r"^[ \t]*```[^\n{]*(?:\r?\n)?[ \t]*(\{.*?\})[ \t]*(?:\r?\n)?[ \t]*```", re.S | re.M)
HARBOR_FILE_PATTERN = re.compile(r"^\.?[A-Za-z0-9][A-Za-z0-9._-]{0,99}(/\.?[A-Za-z0-9][A-Za-z0-9._-]{0,99}){0,3}$")

SYSTEM_PROMPT = (
    "You are an expert programmer and environment designer. You write executable environments that train "
    "a language model agent by finding the edge of what it can do."
)


class DesignerReplyError(ValueError):
    """The designer's reply holds no usable task."""


class DesignerError(RuntimeError):
    """A designer call or report did not go through."""


@dataclass(frozen=True)
class DesignerRequest:
    """One designer call: what to test (a target, an optional skill), how hard, a grounding text, earlier results as text."""

    target: str
    skill: str | None = None
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_TURN_LIMIT
    grounding: str | None = None
    experience_text: str = ""

    def __post_init__(self) -> None:
        if self.skill is not None and (not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill)):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be non-empty text")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if isinstance(self.turn_limit, bool) or not isinstance(self.turn_limit, int) or self.turn_limit < 2:
            raise ValueError("turn_limit must be an integer of at least 2")
        if self.grounding is not None and (not isinstance(self.grounding, str) or not self.grounding.strip()):
            raise ValueError("grounding must be non-empty text when set")
        if not isinstance(self.experience_text, str):
            raise ValueError("experience_text must be text")


@dataclass(frozen=True)
class HarborReply:
    """A usable reply: the task's files by directory, and a hint for the agent when the designer gave one."""

    instruction: str
    environment: dict[str, str]
    tests: dict[str, str]
    solution: dict[str, str]
    hint: str = ""


def designer_messages(request: DesignerRequest) -> list[dict[str, str]]:
    """The chat messages for one designer call."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": designer_prompt(request)}]


def designer_prompt(request: DesignerRequest) -> str:
    """The user turn of a designer call: the target, the method's experience text, the grounding, the rules, the output."""
    target = request.target.strip()
    if request.skill is not None:
        target = f"{request.skill} ({target})"
    parts = [
        f"Create ONE Harbor task, a container with files, an instruction and a verifier, that tests: {target}.",
        (
            f"DIFFICULTY: {request.difficulty}. The agent has at most {request.turn_limit} turns; a careful agent "
            "finishes in fewer, a careless one fails."
        ),
    ]
    if request.experience_text.strip():
        parts.append(request.experience_text.strip())
    if request.grounding is not None:
        parts.append(
            "GROUNDING: the environment must make the agent execute a technique or operate a system from this "
            "document. Never mention the document in the environment's text.\n"
            + untrusted_text(request.grounding.strip()[:GROUNDING_CHARS], "reference document")
        )
    parts.extend([HARBOR_RULES_TEXT.format(turn_limit=request.turn_limit), HARBOR_OUTPUT_TEXT])
    return "\n\n".join(parts)


HARBOR_RULES_TEXT = """RULES:
- The agent gets a shell in a container built from environment/Dockerfile and the text of instruction.md; it has at most {turn_limit} commands. It never sees tests/ or solution/.
- TWO NETWORK PHASES: the build of environment/Dockerfile has network, so install there every package the task and the verifier need and COPY every fixture from environment/ into the image; the agent's container and the verifier have no network, so nothing may download at solve or grade time, and a task whose intended solution downloads anything is refused.
- environment/Dockerfile starts FROM a public image (python:3.12-slim, ubuntu:24.04) and is read by the CLASSIC Docker parser: no heredocs (a heredoc body is read as instructions and the build fails); to create a file, put it next to the Dockerfile and COPY it, or write it on one line with printf. The image installs tmux (the agent runs inside it; `apt-get install -y tmux`), creates every directory the instruction or the scripts write to, and the agent starts in the image's WORKDIR. COPY sources are paths relative to environment/ (`COPY app.conf /etc/app.conf`, never `COPY environment/app.conf`), and every source must be a file in the reply's environment. Install only package names the base image's package manager has (the base image already has coreutils, grep, sed and find; netcat is netcat-openbsd on Debian), and nothing the build cannot do (chattr, mount, systemctl).
- NO PROCESS SURVIVES THE BUILD: Harbor starts the agent's container with `sleep infinity`, so a RUN that launches a program, a CMD or an ENTRYPOINT run nothing by the time the agent arrives, and a log such a program would have written does not exist. Hidden state lives in files the build wrote (configs, logs, a database file, a git history). A task that needs a running service has the instruction name the command that starts it (the reference solution starts it the same way, and the verifier checks the outcome after), or declares the service as a second container in environment/docker-compose.yaml.
- instruction.md is at least 80 characters, self contained, and never contains the answer.
- tests/test.sh is the verifier: it runs after the agent, with /tests holding the tests/ files, and writes one number in [0, 1] to /logs/verifier/reward.txt (1 for success). It checks the outcome, never the transcript, and needs nothing the image lacks. It writes the reward on every path, the failure path included (`echo 0 > /logs/verifier/reward.txt`), and the file holds the number only.
- solution/solve.sh is a reference solution: the commands that complete the task from the same starting point. The task is accepted only if this script scores 1 and doing nothing scores below 1.
- HIDDEN STATE: the task needs the agent to inspect the container (files, logs, a running process, a database) before it can act. The agent runs as the image's user: root unless the Dockerfile adds a user and switches to it with USER; the verifier always runs as root. An environment that answers the agent step by step (a game, a puzzle, a simulated tool) is a program in the image whose state the agent cannot read: keep the state under a root only path, run the agent as a non root user, and let a sudoers rule for that one command drive it.
- TARGET: an agent at the frontier completes the task in one of four to three of four attempts; too easy or out of reach is refused later.
- Files are plain text; paths are relative, no directories above the task, at most four levels."""

HARBOR_OUTPUT_TEXT = """OUTPUT exactly one fenced json block and nothing else, with these keys:
```json
{
  "instruction": "<the text the agent reads>",
  "environment": {"Dockerfile": "<the image>", "<other file>": "<text>"},
  "tests": {"test.sh": "<the verifier, a POSIX shell script>", "<other file>": "<text>"},
  "solution": {"solve.sh": "<the reference solution, a POSIX shell script>"},
  "hint": "<one to three sentences for the agent: the key strategy, without the answer itself>"
}
```"""


def parse_harbor_reply(text: str) -> HarborReply:
    """The ``json`` block of a reply: the instruction, the three file mappings and the hint, all checked."""
    if not isinstance(text, str) or not text.strip():
        raise DesignerReplyError("the reply is empty")
    match = JSON_BLOCK.search(text)
    if match is None:
        raise DesignerReplyError("the reply holds no ```json block with an object")
    try:
        # strict=False: a model writes real line breaks inside the file strings as often as escaped ones.
        document = json.loads(match.group(1), strict=False)
    except json.JSONDecodeError as exc:
        raise DesignerReplyError(f"the ```json block is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise DesignerReplyError("the ```json block must hold an object")
    unknown = sorted(key for key in document if key not in ("instruction", "environment", "tests", "solution", "hint"))
    if unknown:
        raise DesignerReplyError(f"the reply carries keys the task has no place for: {', '.join(unknown)}")
    instruction = checked_text(document.get("instruction"), "instruction")
    hint = checked_text(document["hint"], "hint") if document.get("hint") is not None else ""
    files = {label: checked_harbor_files(document.get(label), label) for label in ("environment", "tests", "solution")}
    for label, required in (("environment", "Dockerfile"), ("tests", "test.sh"), ("solution", "solve.sh")):
        if not files[label].get(required, "").strip():
            raise DesignerReplyError(f"the reply's {label} must hold a non-empty {required}")
    if "hint.txt" in files["solution"]:
        raise DesignerReplyError("the reply's solution must not name hint.txt; the hint has its own key")
    return HarborReply(
        instruction=instruction.strip() + "\n",
        environment=files["environment"],
        tests=files["tests"],
        solution=files["solution"],
        hint=" ".join(hint.split()),
    )


def checked_text(value: object, label: str) -> str:
    """Non-empty text that encodes as UTF-8, with its line ends folded."""
    if not isinstance(value, str) or not value.strip():
        raise DesignerReplyError(f"the reply's {label} must be non-empty text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DesignerReplyError(f"the reply's {label} is not valid text: {exc}") from exc
    return value.replace("\r\n", "\n")


def checked_harbor_files(value: object, label: str) -> dict[str, str]:
    """A mapping of relative file paths to text, under the task writer's own rules and a depth of four."""
    if not isinstance(value, dict):
        raise DesignerReplyError(f"the reply's {label} must be an object of file paths to text")
    for path, text in value.items():
        if not isinstance(path, str) or not HARBOR_FILE_PATTERN.fullmatch(path) or ".." in path.split("/"):
            raise DesignerReplyError(f"the reply's {label} names a file the task cannot hold: {path!r}")
        if not isinstance(text, str):
            raise DesignerReplyError(f"the reply's {label}/{path} must be text")
    try:
        files = checked_files(label, {path: text.replace("\r\n", "\n") for path, text in value.items()})
    except HarborTaskError as exc:
        raise DesignerReplyError(f"the reply's {label}: {exc}") from exc
    return files


@dataclass(frozen=True)
class DesignerAnswer:
    """What the designer said and the record its call left on Reef."""

    text: str
    record_id: str


class Designer(ABC):
    """Who answers a designer prompt and takes a report about the proposal: the served model through Reef."""

    @abstractmethod
    def answer(
        self, messages: Sequence[Mapping[str, str]], *, scenario: str, model: str, tags: Mapping[str, str]
    ) -> DesignerAnswer: ...

    @abstractmethod
    def report(self, record_id: str, *, scenario: str, score: float, metadata: Mapping[str, object]) -> str: ...


class ReefDesigner(Designer):
    """The served model behind a Reef service; each proposal is an inference record, each outcome a report."""

    def __init__(
        self,
        *,
        reef_url: str,
        token: str | None = None,
        request_options: Mapping[str, object] | None = None,
        timeout_s: float = DESIGNER_TIMEOUT_S,
    ) -> None:
        # Extra fields of the chat request, e.g. {"reasoning_effort": "none"} for a model that would think for
        # thousands of tokens before writing an environment and run past the service's inference deadline.
        self.request_options = dict(request_options or {})
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise DesignerError("timeout_s must be a positive number of seconds")
        # A large local model writes an environment in minutes; the service's own inference deadline must allow it too.
        self.client = ReefClient(reef_url, token=token, timeout_s=float(timeout_s))

    def answer(
        self, messages: Sequence[Mapping[str, str]], *, scenario: str, model: str, tags: Mapping[str, str]
    ) -> DesignerAnswer:
        headers = {f"x-reef-tag-{name}": value for name, value in tags.items()}
        payload = {**self.request_options, "model": model, "messages": [dict(message) for message in messages]}
        try:
            body, record_id = self.client.inference_with_record(scenario, CHAT_PATH, payload, extra_headers=headers)
        except ReefClientError as exc:
            raise DesignerError(f"the designer call was refused ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise DesignerError(
                f"the designer call did not complete within {self.client.timeout_s:g} s: {exc}"
            ) from exc
        choices = body.get("choices")
        message = choices[0].get("message") if isinstance(choices, list) and choices else None
        text = message.get("content") if isinstance(message, Mapping) else None
        return DesignerAnswer(text=text if isinstance(text, str) else "", record_id=record_id)

    def report(self, record_id: str, *, scenario: str, score: float, metadata: Mapping[str, object]) -> str:
        payload = {
            "score": score,
            "feedback": "task designer score",
            # The report shares the scenario with the agent's episodes; the role tells a processor them apart.
            "metadata": {**dict(metadata), "role": "designer"},
        }
        try:
            answer = self.client.report(scenario, payload, references=[record_id])
        except ReefClientError as exc:
            raise DesignerError(f"the designer report was refused ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise DesignerError(f"the designer report did not reach Reef: {exc}") from exc
        return str(answer.get("agent_record_id", ""))
