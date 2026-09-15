"""The Environment Designer: what it is asked and how its reply becomes a Harbor task.

The Designer is the served model in a second role (SPADE Sec. 4.1). One call asks for one environment
at the edge of what the agent can do today: the request carries what the agent did on the last
generation's environments, sorted by the hint based regret of Sec. 4.2 into the frontier (the hint
turns losses into wins), the mastered (won without it) and the out of reach (lost even with it), so the
next environment lands where the agent fails without a hint and passes with one. The environment is a
Harbor task written directly, an instruction, a container, a verifier and a reference solution, and Harbor
running the reference solution validates it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from reef.core.tasks.harbor import TASK_NAME_PATTERN, HarborTaskError, checked_files
from reef.train.cordis_backend.strategies import untrusted_text

DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_TURN_LIMIT = 12
MAX_EXPERIENCE_RECORDS = 12
DESIGNER_TIMEOUT_S = 1800.0
#: The harness tree entries a Designer prompt is made of, by entry id: the text each one carries.
DESIGNER_SYSTEM_ENTRY = "designer-system"
DESIGNER_RULES_ENTRY = "designer-rules"
PROMPT_ENTRY_FIELDS = {DESIGNER_SYSTEM_ENTRY: "system", DESIGNER_RULES_ENTRY: "rules"}
INSTRUCTION_EXCERPT_CHARS = 1200
GROUNDING_CHARS = 6000
MASTERED_RETURN = 0.9
TOO_HARD_RETURN = 0.1
SKILL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
JSON_BLOCK = re.compile(r"^[ \t]*```[^\n{]*(?:\r?\n)?[ \t]*(\{.*?\})[ \t]*(?:\r?\n)?[ \t]*```", re.S | re.M)
HARBOR_FILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}(/[A-Za-z0-9][A-Za-z0-9._-]{0,99}){0,3}$")

SYSTEM_PROMPT = (
    "You are an expert programmer and environment designer. You write executable environments that train "
    "a language model agent by finding the edge of what it can do."
)


class DesignerReplyError(ValueError):
    """The Designer's reply holds no usable environment."""


@dataclass(frozen=True)
class PlayRecord:
    """What the agent did on one earlier environment: its mean return over the rollouts without and with the hint."""

    name: str
    return_without_hint: float
    return_with_hint: float
    skill: str | None = None
    instruction_excerpt: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TASK_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"a play record's name must be a task name matching {TASK_NAME_PATTERN.pattern}")
        if self.skill is not None and (not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill)):
            raise ValueError(f"a play record's skill must match {SKILL_PATTERN.pattern}")
        for label, value in (
            ("return_without_hint", self.return_without_hint),
            ("return_with_hint", self.return_with_hint),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{label} must be a number in [-1, 1]")
        if not isinstance(self.instruction_excerpt, str):
            raise ValueError("instruction_excerpt must be text")

    @property
    def regret(self) -> float:
        """How much the hint helped: the with hint return minus the without hint return (Sec. 4.2)."""
        return float(self.return_with_hint) - float(self.return_without_hint)

    @property
    def outcome(self) -> str:
        """The reference memory's bands on the no hint return: ``mastered`` above 0.9, ``out_of_reach`` below 0.1, else ``frontier``."""
        if self.return_without_hint > MASTERED_RETURN:
            return "mastered"
        if self.return_without_hint < TOO_HARD_RETURN:
            return "out_of_reach"
        return "frontier"


@dataclass(frozen=True)
class DesignerRequest:
    """One Designer call: what to test (a description, an optional skill), how hard, what the agent did last time, a grounding text."""

    skill_description: str
    skill: str | None = None
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_TURN_LIMIT
    grounding: str | None = None
    experience: tuple[PlayRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.skill is not None and (not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill)):
            raise ValueError(f"skill {self.skill!r} must match {SKILL_PATTERN.pattern}")
        if not isinstance(self.skill_description, str) or not self.skill_description.strip():
            raise ValueError("skill_description must be non-empty text")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
        if isinstance(self.turn_limit, bool) or not isinstance(self.turn_limit, int) or self.turn_limit < 2:
            raise ValueError("turn_limit must be an integer of at least 2")
        if self.grounding is not None and (not isinstance(self.grounding, str) or not self.grounding.strip()):
            raise ValueError("grounding must be non-empty text when set")
        if not isinstance(self.experience, tuple) or not all(isinstance(r, PlayRecord) for r in self.experience):
            raise ValueError("experience must be a tuple of PlayRecord")
        if len(self.experience) > MAX_EXPERIENCE_RECORDS:
            raise ValueError(f"experience holds at most {MAX_EXPERIENCE_RECORDS} records; pick the most recent")


@dataclass(frozen=True)
class HarborReply:
    """A usable reply: the task's files by directory, and the hint for the agent."""

    instruction: str
    environment: dict[str, str]
    tests: dict[str, str]
    solution: dict[str, str]
    hint: str


def designer_messages(request: DesignerRequest, prompt: DesignerPrompt | None = None) -> list[dict[str, str]]:
    """The chat messages for one Designer call, in the shape ``ModelBinding.chat`` takes."""
    prompt = prompt if prompt is not None else DesignerPrompt()
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": designer_prompt(request, prompt)},
    ]


def designer_prompt(request: DesignerRequest, prompt: DesignerPrompt | None = None) -> str:
    """The user turn of a Designer call: target, what the agent did last time, grounding, the rules, output."""
    prompt = prompt if prompt is not None else DesignerPrompt()
    target = request.skill_description.strip()
    if request.skill is not None:
        target = f"{request.skill} ({target})"
    parts = [
        f"Create ONE Harbor task, a container with files, an instruction and a verifier, that tests: {target}.",
        (
            f"DIFFICULTY: {request.difficulty}. The agent has at most {request.turn_limit} turns; a careful agent "
            "finishes in fewer, a careless one fails."
        ),
        experience_text(request.experience),
    ]
    if request.grounding is not None:
        parts.append(
            "GROUNDING: the environment must make the agent execute a technique or operate a system from this "
            "document. Never mention the document in the environment's text.\n"
            + untrusted_text(request.grounding.strip()[:GROUNDING_CHARS], "reference document")
        )
    # A replace, not str.format: an evolved rules text may carry braces of its own.
    parts.extend([prompt.rules.replace("{turn_limit}", str(request.turn_limit)), HARBOR_OUTPUT_TEXT])
    return "\n\n".join(parts)


def experience_text(experience: Sequence[PlayRecord]) -> str:
    """The agent's results on the last environments, sorted into what to write more of and what to avoid."""
    if not experience:
        return (
            "WHAT THE AGENT DID LAST TIME: nothing recorded yet. Aim for an environment a careful agent completes "
            "and a hasty one fails."
        )
    frontier = sorted((r for r in experience if r.outcome == "frontier"), key=lambda r: r.regret, reverse=True)
    mastered = [r for r in experience if r.outcome == "mastered"]
    out_of_reach = [r for r in experience if r.outcome == "out_of_reach"]
    lines = [
        (
            "WHAT THE AGENT DID LAST TIME (mean episode returns in [-1, 1] over its attempts; a hint is a few "
            "sentences of strategy the agent was given on a second set of attempts):"
        )
    ]
    if frontier:
        lines.append(
            "- Within reach but not mastered, the ones the hint helped most first. Write environments like these, "
            "varied, not copies:"
        )
        lines.extend(record_lines(frontier, is_instruction_shown=True))
    if mastered:
        lines.append("- Mastered without any hint. Too easy; do not write environments like these:")
        lines.extend(record_lines(mastered, is_instruction_shown=False))
    if out_of_reach:
        lines.append(
            "- Won fewer than one attempt in ten without the hint. Out of reach or broken; do not write environments "
            "like these, and make sure the instruction gives the agent enough to act on:"
        )
        lines.extend(record_lines(out_of_reach, is_instruction_shown=False))
    return "\n".join(lines)


def record_lines(records: Sequence[PlayRecord], *, is_instruction_shown: bool) -> list[str]:
    lines = []
    for record in records:
        label = record.name if record.skill is None else f"{record.name} ({record.skill})"
        lines.append(
            f"  {label}: without hint {record.return_without_hint:+.2f}, with hint {record.return_with_hint:+.2f}"
        )
        if is_instruction_shown and record.instruction_excerpt.strip():
            lines.append(
                untrusted_text(record.instruction_excerpt.strip()[:INSTRUCTION_EXCERPT_CHARS], "earlier instruction")
            )
    return lines


HARBOR_RULES_TEXT = """RULES:
- The agent gets a shell in a container built from environment/Dockerfile and the text of instruction.md; it has at most {turn_limit} commands. It never sees tests/ or solution/.
- TWO NETWORK PHASES: the build of environment/Dockerfile has network, so install there every package the task and the verifier need and COPY every fixture from environment/ into the image; the agent's container and the verifier have no network, so nothing may download at solve or grade time, and a task whose intended solution downloads anything is refused.
- environment/Dockerfile starts FROM a public image (python:3.12-slim, ubuntu:24.04) and is read by the CLASSIC Docker parser: no heredocs (a heredoc body is read as instructions and the build fails); to create a file, put it next to the Dockerfile and COPY it, or write it on one line with printf. The image installs tmux (the agent runs inside it; `apt-get install -y tmux`), creates every directory the instruction or the scripts write to, and the agent starts in the image's WORKDIR.
- NO PROCESS SURVIVES THE BUILD: Harbor starts the agent's container with `sleep infinity`, so a RUN that launches a program, a CMD or an ENTRYPOINT run nothing by the time the agent arrives, and a log such a program would have written does not exist. Hidden state lives in files the build wrote (configs, logs, a database file, a git history). A task that needs a running service has the instruction name the command that starts it (the reference solution starts it the same way, and the verifier checks the outcome after), or declares the service as a second container in environment/docker-compose.yaml.
- instruction.md is at least 80 characters, self contained, and never contains the answer.
- tests/test.sh is the verifier: it runs after the agent, with /tests holding the tests/ files, and writes one number in [0, 1] to /logs/verifier/reward.txt (1 for success). It checks the outcome, never the transcript, and needs nothing the image lacks.
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


@dataclass(frozen=True)
class DesignerPrompt:
    """The Designer's harness: the system turn and the rules block, the two texts a harness release may evolve."""

    system: str = SYSTEM_PROMPT
    rules: str = HARBOR_RULES_TEXT
    request_options: Mapping[str, object] = field(default_factory=dict)
    timeout_s: float = DESIGNER_TIMEOUT_S

    def __post_init__(self) -> None:
        for label, value in (("system", self.system), ("rules", self.rules)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text")
        if not isinstance(self.request_options, Mapping):
            raise ValueError("request_options must be a mapping of chat request fields")
        if isinstance(self.timeout_s, bool) or not isinstance(self.timeout_s, (int, float)) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number of seconds")

    def entries(self) -> tuple[dict[str, object], ...]:
        """The prompt as harness tree entries: one skill per text, its config name the entry id."""
        return tuple(
            {"id": entry_id, "name": "skill", "config": {"name": entry_id, "text": text}}
            for entry_id, text in ((DESIGNER_SYSTEM_ENTRY, self.system), (DESIGNER_RULES_ENTRY, self.rules))
        )

    def with_entries(self, entries: Sequence[Mapping[str, object]]) -> DesignerPrompt:
        """The prompt with the texts a served tree carries under the two entry ids; other entries are ignored."""
        texts = {"system": self.system, "rules": self.rules}
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("a harness entry must be an object with an id and a config")
            entry_id = entry.get("id")
            if not isinstance(entry_id, str) or entry_id not in PROMPT_ENTRY_FIELDS:
                continue
            config = entry.get("config")
            text = config.get("text") if isinstance(config, Mapping) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"harness entry {entry_id} must carry non-empty text")
            texts[PROMPT_ENTRY_FIELDS[entry_id]] = text
        return replace(self, system=texts["system"], rules=texts["rules"])


def parse_harbor_reply(text: str) -> HarborReply:
    """The ``json`` block of a reply: instruction, the three file mappings and the hint, all checked."""
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
    hint = checked_text(document.get("hint"), "hint")
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
