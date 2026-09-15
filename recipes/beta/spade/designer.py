"""The Environment Designer: what it is asked and how its reply becomes an environment of one kind.

The Designer is the served model in a second role (SPADE Sec. 4.1). One call asks for one environment
of one kind at the edge of what the agent can do today: the request carries what the agent did on the
last generation's environments, sorted by the hint based regret of Sec. 4.2 into the frontier (the hint
turns losses into wins), the mastered (won without it) and the out of reach (lost even with it), so the
next environment lands where the agent fails without a hint and passes with one. Two kinds so far, one
Harbor task each, all playable by any Harbor agent: ``gym``, a Python class with the Gym interface (a game, a simulated
tool use setting) served inside the container behind the ``observe`` and ``act`` commands and validated by running it in
child interpreters on the host; ``harbor``, a Harbor task written directly (an instruction, a container, a
verifier, a reference solution) validated by Harbor running the reference solution; ``openenv``, an
OpenEnv environment package served inside the container behind ``serve`` and validated by a reset and one
step against that server.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field

from recipes.beta.spade.environment_loader import environment_class_name
from recipes.beta.spade.process import EnvironmentProcess, EnvironmentProcessError
from recipes.beta.spade.tasks import DEFAULT_MAX_TURNS, SKILL_PATTERN
from reef.core.tasks.harbor import TASK_NAME_PATTERN, HarborTaskError, checked_files
from reef.train.cordis_backend.strategies import untrusted_text

KINDS = ("harbor", "gym", "openenv")
DIFFICULTIES = ("easy", "medium", "hard")
MAX_EXPERIENCE_RECORDS = 12
CODE_EXCERPT_CHARS = 1200
GROUNDING_CHARS = 6000
MASTERED_RETURN = 0.9
TOO_HARD_RETURN = 0.1
SMOKE_INSTANCES = 3
SMOKE_PROBES = ("\\boxed{probe}", "\\boxed{1}")
# The rules forbid files, processes, the network and the interpreter; the smoke test runs on the host, so these are refused first.
FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "http",
        "urllib",
        "ctypes",
        "importlib",
        "io",
        "signal",
        "threading",
        "multiprocessing",
        "asyncio",
        "tempfile",
        "glob",
        "webbrowser",
        "ftplib",
        "smtplib",
        "xmlrpc",
        "pickle",
        "marshal",
        "code",
        "codeop",
        "runpy",
        "pty",
        "resource",
        "gc",
        "inspect",
        "builtins",
        "sysconfig",
        "platform",
        "time",
        "datetime",
        "secrets",
        "uuid",
        "posix",
        "nt",
        "fcntl",
        "_posixsubprocess",
        "_io",
        "_socket",
        "_thread",
        "_signal",
    }
)
FORBIDDEN_CALLS = frozenset(
    {"open", "__import__", "exec", "eval", "compile", "input", "breakpoint", "globals", "getattr", "setattr"}
)
PYTHON_BLOCK = re.compile(r"^[ \t]*```(?:python3?|py)\b[^\n]*\r?\n(.*?)\r?\n[ \t]*```", re.S | re.M | re.I)
HINT_BLOCK = re.compile(r"^[ \t]*```hint\b[^\n]*\r?\n(.*?)\r?\n[ \t]*```", re.S | re.M | re.I)
HINT_LINE = re.compile(r"^HINT:[ \t]*(.+)$", re.M)
JSON_BLOCK = re.compile(r"^[ \t]*```[^\n{]*(?:\r?\n)?[ \t]*(\{.*?\})[ \t]*(?:\r?\n)?[ \t]*```", re.S | re.M)
HARBOR_FILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}(/[A-Za-z0-9][A-Za-z0-9._-]{0,99}){0,3}$")

SYSTEM_PROMPT = (
    "You are an expert Python programmer and environment designer. You write executable environments that train "
    "a language model agent by finding the edge of what it can do."
)


class DesignerReplyError(ValueError):
    """The Designer's reply holds no usable environment."""


@dataclass(frozen=True)
class PlayRecord:
    """What the agent did on one earlier environment: its mean return over the rollouts without and with the hint."""

    name: str
    skill: str
    return_without_hint: float
    return_with_hint: float
    code_excerpt: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TASK_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(f"a play record's name must be a task name matching {TASK_NAME_PATTERN.pattern}")
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
            raise ValueError(f"a play record's skill must match {SKILL_PATTERN.pattern}")
        for label, value in (
            ("return_without_hint", self.return_without_hint),
            ("return_with_hint", self.return_with_hint),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{label} must be a number in [-1, 1]")
        if not isinstance(self.code_excerpt, str):
            raise ValueError("code_excerpt must be text")

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
    """One Designer call: the kind and skill to test, how hard, what the agent did last time, and a grounding text."""

    kind: str
    skill: str
    skill_description: str
    difficulty: str = "medium"
    turn_limit: int = DEFAULT_MAX_TURNS
    grounding: str | None = None
    experience: tuple[PlayRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if not isinstance(self.skill, str) or not SKILL_PATTERN.fullmatch(self.skill):
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
class GymReply:
    """A usable reply of the ``gym`` kind: the class and the hint for the agent."""

    code: str
    hint: str


@dataclass(frozen=True)
class HarborReply:
    """A usable reply of the ``harbor`` kind: the task's files by directory, and the hint for the agent."""

    instruction: str
    environment: dict[str, str]
    tests: dict[str, str]
    solution: dict[str, str]
    hint: str


@dataclass(frozen=True)
class OpenEnvReply:
    """A usable reply of the ``openenv`` kind: the goal, the models and environment modules, an example action, the hint."""

    instruction: str
    models: str
    environment: str
    action_example: dict[str, object]
    hint: str


@dataclass(frozen=True)
class SmokeResult:
    """Whether a ``gym`` class runs as an environment; ``reason`` names the first contract break."""

    is_runnable: bool
    reason: str
    first_observation: str = ""


def designer_messages(request: DesignerRequest) -> list[dict[str, str]]:
    """The chat messages for one Designer call, in the shape ``ModelBinding.chat`` takes."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": designer_prompt(request)}]


def designer_prompt(request: DesignerRequest) -> str:
    """The user turn of a Designer call: target, what the agent did last time, grounding, the kind's rules, output."""
    if request.kind == "gym":
        opening = (
            "Create ONE interactive, multi turn environment as a Python class with the Gym interface that tests: "
            f"{request.skill} ({request.skill_description.strip()})."
        )
        rules = GYM_RULES_TEXT.format(turn_limit=request.turn_limit)
        output = GYM_OUTPUT_TEXT
    elif request.kind == "openenv":
        opening = (
            "Create ONE interactive, multi turn environment as an OpenEnv environment package that tests: "
            f"{request.skill} ({request.skill_description.strip()})."
        )
        rules = OPENENV_RULES_TEXT.format(turn_limit=request.turn_limit)
        output = OPENENV_OUTPUT_TEXT
    else:
        opening = (
            "Create ONE Harbor task, a container with files, an instruction and a verifier, that tests: "
            f"{request.skill} ({request.skill_description.strip()})."
        )
        rules = HARBOR_RULES_TEXT.format(turn_limit=request.turn_limit)
        output = HARBOR_OUTPUT_TEXT
    parts = [
        opening,
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
    parts.extend([rules, output])
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
        lines.extend(record_lines(frontier, is_code_shown=True))
    if mastered:
        lines.append("- Mastered without any hint. Too easy; do not write environments like these:")
        lines.extend(record_lines(mastered, is_code_shown=False))
    if out_of_reach:
        lines.append(
            "- Won fewer than one attempt in ten without the hint. Out of reach or broken; do not write environments "
            "like these, and make sure the observation gives the agent enough to act on:"
        )
        lines.extend(record_lines(out_of_reach, is_code_shown=False))
    return "\n".join(lines)


def record_lines(records: Sequence[PlayRecord], *, is_code_shown: bool) -> list[str]:
    lines = []
    for record in records:
        lines.append(
            f"  {record.name} ({record.skill}): without hint {record.return_without_hint:+.2f}, "
            f"with hint {record.return_with_hint:+.2f}"
        )
        if is_code_shown and record.code_excerpt.strip():
            lines.append(untrusted_text(record.code_excerpt.strip()[:CODE_EXCERPT_CHARS], "earlier environment code"))
    return lines


GYM_RULES_TEXT = """RULES:
- One class whose name ends in Env, standard library only (random, json, re, math, itertools, collections), no input(), no files, no network, no printing.
- reset(self, seed=None) -> (observation: str, info: dict). All randomness comes from the seed: the same seed gives the same environment, in reset() and in every step(). reset() generates ONE task for the episode; step() never generates a new one.
- step(self, action: str) -> (observation: str, reward: float, terminated: bool, truncated: bool, info: dict). Every code path returns that 5-tuple; info is a dict, never a string.
- step() receives the agent's answer as \\boxed{{action}}. Extract the action with re.search(r"\\\\boxed\\{{([^}}]*)\\}}", action); when there is no box, return a reminder of the format with reward 0.0 and do not end the episode. An action the environment does not understand gets a specific error observation and does not end the episode.
- HIDDEN STATE: the goal cannot be reached in one action; the agent must probe, remember and plan. The observation never states the answer or the rule behind it.
- Every observation shows the current state, the result of the last action, what actions are possible, and reminds the agent to answer with \\boxed{{action}}.
- REWARD: success returns 1.0 with terminated=True; failure returns 0.0 with terminated=True; every other step returns 0.0; after {turn_limit} turns return truncated=True with 0.0.
- SELF CHECK before you answer, in your head, never as a code block: trace two different action sequences from reset(seed=0) and confirm the returns above."""

GYM_OUTPUT_TEXT = """OUTPUT exactly two fenced blocks and nothing else:
```python
<the complete environment code>
```
```hint
<one to three sentences for the agent: the key strategy and the answer format, without the answer itself; mention only what the agent can see>
```"""

HARBOR_RULES_TEXT = """RULES:
- The agent gets a shell in a container built from environment/Dockerfile and the text of instruction.md; it has at most {turn_limit} commands. It never sees tests/ or solution/.
- TWO NETWORK PHASES: the build of environment/Dockerfile has network, so install there every package the task and the verifier need and COPY every fixture from environment/ into the image; the agent's container and the verifier have no network, so nothing may download at solve or grade time, and a task whose intended solution downloads anything is refused.
- environment/Dockerfile starts FROM a public image (python:3.12-slim, ubuntu:24.04) and is read by the CLASSIC Docker parser: no heredocs (a heredoc body is read as instructions and the build fails); to create a file, put it next to the Dockerfile and COPY it, or write it on one line with printf. The image creates every directory the instruction or the scripts write to, and the agent starts in the image's WORKDIR.
- instruction.md is at least 80 characters, self contained, and never contains the answer.
- tests/test.sh is the verifier: it runs after the agent, with /tests holding the tests/ files, and writes one number in [0, 1] to /logs/verifier/reward.txt (1 for success). It checks the outcome, never the transcript, and needs nothing the image lacks.
- solution/solve.sh is a reference solution: the commands that complete the task from the same starting point. The task is accepted only if this script scores 1 and doing nothing scores below 1.
- HIDDEN STATE: the task needs the agent to inspect the container (files, logs, a running process, a database) before it can act.
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


def checked_code(code: str) -> str:
    """The environment class name of ``code`` after the rules' bans: no file, process, network or interpreter access."""
    name = environment_class_name(code)
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules = [node.module.split(".")[0]]
        else:
            modules = []
        for module in modules:
            if module in FORBIDDEN_MODULES:
                raise ValueError(f"environment code imports {module!r}, which the rules forbid")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise ValueError(f"environment code calls {node.func.id}(), which the rules forbid")
    return name


OPENENV_RULES_TEXT = """RULES:
- Two Python modules of the OpenEnv framework (huggingface/OpenEnv). models.py defines exactly one class inheriting Action and exactly one inheriting Observation, imported with `from openenv.core.env_server.types import Action, Observation`, with pydantic fields; the observation carries reward: float and done: bool. environment.py defines exactly one class inheriting Environment, imported with `from openenv.core.env_server import Environment`, and imports the models with `from openenv_task.models import ...`; it has reset(self, seed=None, **kwargs) -> the observation, step(self, action) -> the observation, both plain methods (no async), and a state property returning State(episode_id=..., step_count=...) from openenv.core.env_server.types. No other class may subclass it.
- Standard library, openenv and pydantic only; no files, no processes, no network, no printing. All randomness comes from the seed: the same seed gives the same episode. reset() generates ONE task for the episode; step() never generates a new one.
- HIDDEN STATE: the goal cannot be reached in one action; the agent must probe, remember and plan. The observation never states the answer or the rule behind it, and every observation shows the state, the result of the last action and what actions are possible.
- REWARD: success sets reward 1.0 and done True; failure sets reward 0.0 and done True; every other step sets reward 0.0 and done False. The server ends the episode after {turn_limit} steps on its own.
- Give one valid example action as a JSON object with the action's fields, so the agent and the check can take a first step."""

OPENENV_OUTPUT_TEXT = """OUTPUT exactly one fenced json block and nothing else, with these keys:
```json
{
  "instruction": "<the goal the agent reads, at least 80 characters, without the answer>",
  "models": "<the complete models.py>",
  "environment": "<the complete environment.py>",
  "action_example": {"<field>": "<value>"},
  "hint": "<one to three sentences for the agent: the key strategy, without the answer itself>"
}
```"""


def parse_openenv_reply(text: str) -> OpenEnvReply:
    """The ``json`` block of an openenv reply: the goal, both modules, the example action and the hint, all checked."""
    if not isinstance(text, str) or not text.strip():
        raise DesignerReplyError("the reply is empty")
    blocks = list(JSON_BLOCK.finditer(text))
    if not blocks:
        raise DesignerReplyError("the reply holds no ```json block with an object")
    chosen = None
    first_error = ""
    for block in blocks:
        try:
            # strict=False: a model writes real line breaks inside the code strings as often as escaped ones.
            document = json.loads(textwrap.dedent(block.group(1)), strict=False)
        except json.JSONDecodeError as exc:
            first_error = first_error or f"the ```json block is not valid JSON: {exc}"
            continue
        if not isinstance(document, dict):
            first_error = first_error or "the ```json block must hold an object"
            continue
        if chosen is None or "models" in document:
            chosen = (block, document)
        if "models" in document:
            break
    if chosen is None:
        raise DesignerReplyError(first_error)
    block, document = chosen
    unknown = sorted(
        key for key in document if key not in ("instruction", "models", "environment", "action_example", "hint")
    )
    if unknown:
        raise DesignerReplyError(f"the reply carries keys the task has no place for: {', '.join(unknown)}")
    if "hint" not in document:
        hint_match = HINT_BLOCK.search(text, block.end()) or HINT_LINE.search(text, block.end())
        if hint_match is not None:
            document["hint"] = hint_match.group(1)
    instruction = checked_text(document.get("instruction"), "instruction")
    models = checked_text(document.get("models"), "models")
    environment = checked_text(document.get("environment"), "environment")
    hint = checked_text(document.get("hint"), "hint")
    action_example = document.get("action_example")
    if isinstance(action_example, str):
        try:
            action_example = json.loads(action_example, strict=False)
        except json.JSONDecodeError:
            action_example = None
    if not isinstance(action_example, dict) or not action_example:
        raise DesignerReplyError("the reply's action_example must be a non-empty object")
    return OpenEnvReply(
        instruction=instruction.strip() + "\n",
        models=models.strip("\n") + "\n",
        environment=environment.strip("\n") + "\n",
        action_example=action_example,
        hint=" ".join(hint.split()),
    )


def parse_gym_reply(text: str) -> GymReply:
    """The first ``python`` block that holds an environment class, and the hint after it."""
    if not isinstance(text, str) or not text.strip():
        raise DesignerReplyError("the reply is empty")
    blocks = list(PYTHON_BLOCK.finditer(text))
    if not blocks:
        raise DesignerReplyError("the reply holds no ```python block")
    chosen = None
    first_error = ""
    for block in blocks:
        code = textwrap.dedent(block.group(1).replace("\r\n", "\n")).strip("\n") + "\n"
        if not code.strip():
            continue
        try:
            environment_class_name(code)
        except ValueError as exc:
            if not first_error:
                first_error = str(exc)
            continue
        chosen = (block, code)
        break
    if chosen is None:
        if first_error:
            raise DesignerReplyError(f"no ```python block holds an environment class: {first_error}")
        raise DesignerReplyError("the ```python block is empty")
    block, code = chosen
    try:
        checked_code(code)
    except ValueError as exc:
        raise DesignerReplyError(str(exc)) from exc
    hint_match = HINT_BLOCK.search(text, block.end()) or HINT_LINE.search(text, block.end())
    if hint_match is None:
        raise DesignerReplyError("the reply holds no ```hint block after the code")
    hint = " ".join(hint_match.group(1).split())
    if not hint:
        raise DesignerReplyError("the hint is empty")
    return GymReply(code=code, hint=hint)


def parse_harbor_reply(text: str) -> HarborReply:
    """The ``json`` block of a harbor reply: instruction, the three file mappings and the hint, all checked."""
    if not isinstance(text, str) or not text.strip():
        raise DesignerReplyError("the reply is empty")
    match = JSON_BLOCK.search(text)
    if match is None:
        raise DesignerReplyError("the reply holds no ```json block with an object")
    try:
        document = json.loads(match.group(1))
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


def smoke_test(
    code: str, *, seed: int = 0, max_turns: int = DEFAULT_MAX_TURNS, timeout_s: float = 20.0
) -> SmokeResult:
    """Run a ``gym`` class the way the container will, in three child interpreters: it loads, reset and step agree, the 5-tuple holds."""
    try:
        checked_code(code)
    except ValueError as exc:
        return SmokeResult(is_runnable=False, reason=str(exc))
    instances = [EnvironmentProcess(code, max_turns=max_turns) for _ in range(SMOKE_INSTANCES)]
    try:
        observations = [instance.reset(seed, timeout_s=timeout_s) for instance in instances]
        if not observations[0].strip():
            return SmokeResult(is_runnable=False, reason="reset(seed) must return a non-empty observation")
        if len(set(observations)) != 1:
            return SmokeResult(
                is_runnable=False, reason="reset(seed) is not deterministic: resets with one seed differ"
            )
        for probe in SMOKE_PROBES:
            results = [instance.step(probe, timeout_s=timeout_s) for instance in instances]
            if len(set(results)) != 1:
                return SmokeResult(
                    is_runnable=False,
                    reason="step(action) is not deterministic: instances with one seed and one action differ",
                )
            if results[0][2] or results[0][3]:
                break
    except EnvironmentProcessError as exc:
        return SmokeResult(is_runnable=False, reason=str(exc))
    finally:
        for instance in instances:
            instance.close()
    return SmokeResult(is_runnable=True, reason="", first_observation=observations[0])
