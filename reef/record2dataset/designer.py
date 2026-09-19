"""The task designer: what a model is asked for a Harbor task, and how its reply becomes one.

One call asks the served model for one Harbor task: an instruction, a container, a verifier and a
reference solution. The prompt carries the task contract (the rules every task must meet so any Harbor
agent can play it and Harbor can score it), the target the task should test, and whatever text the
method wants the designer to know about earlier tasks (``DesignerRequest.experience_text``). Grounding and
earlier text enter the prompt fenced as data, never as instructions. The reply's ``json`` block is read
into the instruction, the three file mappings and an optional hint. The call goes through Reef, so every
proposal is an inference record with a receipt the method can report against.

The two texts of the prompt that are the Designer's own, the system turn and the rules block, are a value
(``DesignerPrompt``) the generator takes from a ``PromptSource``: the fixed texts here, or the tree a
harness evolution release serves, pulled once per generation, so a method can evolve the Designer's prompt
as a harness while the generator keeps asking the same way.

The Designer's deployment evolves between generations: a harness evolution deployment publishes a rewrite
as a new release once it has consumed a generation's reports, a weight training deployment commits a step
and serves a new runtime load id. Either takes time after the last report lands, and a generation that asks
the Designer before then uses the version the previous generation used, so the rewrite or the step its
reports produced is never used. ``DesignerTurn`` closes that gap: it reads the Designer's version (the served
release id under a harness prompt, else ``current_runtime_load_id`` of the Designer's scenario in
``GET /reef/status`` with the scenario's step, so a step that skips still counts) when a generation's first
report goes out, and holds the next generation's first proposal until the version moved past that one,
polling every ``poll_s`` seconds up to ``wait_s``. The version at report time is the one that matters: a
release that appeared earlier in the generation, the scenario's creation release for one, is not the rewrite
those reports produced. A deployment with neither a release nor a runtime load id is fixed and never
waited on; a wait that runs out logs a warning and the generation proceeds, so no generation blocks forever
or fails on the wait.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from reef_client.client import ReefClient, ReefClientError

from reef.core.tasks.harbor import HarborTaskError, checked_files
from reef.train.cordis_backend.strategies import untrusted_text

logger = logging.getLogger(__name__)

DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_TURN_LIMIT = 12
GROUNDING_CHARS = 6000
DESIGNER_TIMEOUT_S = 1800.0
CHAT_PATH = "/v1/chat/completions"
HARNESS_PATH = "/reef/harness"
STATUS_PATH = "/reef/status"
DESIGNER_POLL_S = 5.0
DESIGNER_WAIT_S = 1800.0
#: The harness tree entries a Designer prompt is made of, by entry id: the text each one carries.
DESIGNER_SYSTEM_ENTRY = "designer-system"
DESIGNER_RULES_ENTRY = "designer-rules"
PROMPT_ENTRY_FIELDS = {DESIGNER_SYSTEM_ENTRY: "system", DESIGNER_RULES_ENTRY: "rules"}
# Where the native adapter's descriptor puts the entries list in every served tree.
TREE_PATH = "native/tree.json"
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


def designer_messages(request: DesignerRequest, prompt: DesignerPrompt | None = None) -> list[dict[str, str]]:
    """The chat messages for one designer call, with the fixed prompt texts unless a source gave others."""
    prompt = prompt if prompt is not None else DesignerPrompt()
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": designer_prompt(request, prompt)},
    ]


def designer_prompt(request: DesignerRequest, prompt: DesignerPrompt | None = None) -> str:
    """The user turn of a designer call: the target, the method's experience text, the grounding, the rules, the output."""
    prompt = prompt if prompt is not None else DesignerPrompt()
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
    # A replace, not str.format: an evolved rules text may carry braces of its own.
    parts.extend([prompt.rules.replace("{turn_limit}", str(request.turn_limit)), HARBOR_OUTPUT_TEXT])
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


@dataclass(frozen=True)
class DesignerPrompt:
    """The Designer's own texts, the system turn and the rules block: what a harness release may evolve."""

    system: str = SYSTEM_PROMPT
    rules: str = HARBOR_RULES_TEXT

    def __post_init__(self) -> None:
        for label, value in (("system", self.system), ("rules", self.rules)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be non-empty text")

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
    def report(
        self,
        record_id: str,
        *,
        scenario: str,
        score: float,
        metadata: Mapping[str, object],
        feedback: str | Mapping[str, object] | None = None,
    ) -> str: ...


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

    def report(
        self,
        record_id: str,
        *,
        scenario: str,
        score: float,
        metadata: Mapping[str, object],
        feedback: str | Mapping[str, object] | None = None,
    ) -> str:
        payload = {
            "score": score,
            "feedback": "task designer score" if feedback is None else feedback,
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


class PromptSource(ABC):
    """Where the Designer's prompt for a generation comes from: the fixed texts, or a harness release."""

    @abstractmethod
    def prompt(self, generation: int) -> DesignerPrompt: ...


class FixedPrompt(PromptSource):
    """One prompt for every generation."""

    def __init__(self, fixed: DesignerPrompt | None = None) -> None:
        self.fixed = fixed if fixed is not None else DesignerPrompt()

    def prompt(self, generation: int) -> DesignerPrompt:
        return self.fixed


class HarnessPrompt(PromptSource):
    """The prompt a harness release serves: ``GET /reef/harness`` pulled once per generation, its tree over the fixed texts.

    A scenario that serves no files yet (404) leaves the fixed prompt in place with a warning; a served tree
    the prompt cannot read is a ``DesignerError``, so the generation stops naming it instead of writing tasks
    with a prompt nobody chose.
    """

    def __init__(self, client: ReefClient, scenario: str, fallback: DesignerPrompt | None = None) -> None:
        if not isinstance(scenario, str) or not scenario:
            raise DesignerError("a harness prompt needs the scenario whose tree it pulls")
        self.client = client
        self.scenario = scenario
        self.fallback = fallback if fallback is not None else DesignerPrompt()
        self.generation: int | None = None
        self.cached: DesignerPrompt | None = None

    def prompt(self, generation: int) -> DesignerPrompt:
        if self.cached is not None and self.generation == generation:
            return self.cached
        self.cached = self.pull(generation)
        self.generation = generation
        return self.cached

    def pull(self, generation: int) -> DesignerPrompt:
        """The served tree's texts over the fixed ones, or the fixed ones when the scenario serves no files."""
        try:
            manifest = self.client.get(HARNESS_PATH, extra_headers={"x-reef-scenario": self.scenario})
        except ReefClientError as exc:
            if exc.status != 404:
                raise DesignerError(
                    f"the harness pull for scenario {self.scenario!r} was refused ({exc.status}): {exc.body[:300]}"
                ) from exc
            logger.warning(
                "scenario %r serves no harness tree yet; generation %d asks the Designer with the fixed prompt",
                self.scenario,
                generation,
            )
            return self.fallback
        except OSError as exc:
            raise DesignerError(f"the harness pull for scenario {self.scenario!r} did not reach Reef: {exc}") from exc
        files = manifest.get("files")
        text = files.get(TREE_PATH) if isinstance(files, Mapping) else None
        if not isinstance(text, str):
            raise DesignerError(f"the harness release of scenario {self.scenario!r} carries no {TREE_PATH}")
        try:
            entries = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DesignerError(f"{TREE_PATH} of scenario {self.scenario!r} is not JSON: {exc}") from exc
        if not isinstance(entries, list):
            raise DesignerError(f"{TREE_PATH} of scenario {self.scenario!r} must hold a list of entries")
        try:
            prompt = self.fallback.with_entries(entries)
        except ValueError as exc:
            raise DesignerError(f"{TREE_PATH} of scenario {self.scenario!r}: {exc}") from exc
        logger.info(
            "generation %d asks the Designer with harness release %s of scenario %r",
            generation,
            manifest.get("release_id"),
            self.scenario,
        )
        return prompt


class DesignerTurn:
    """A generation's turn at the Designer: its first proposal waits until the deployment took the last generation in."""

    def __init__(
        self,
        client: ReefClient,
        *,
        is_harness_prompt: bool = False,
        poll_s: float = DESIGNER_POLL_S,
        wait_s: float = DESIGNER_WAIT_S,
    ) -> None:
        for label, value in (("poll_s", poll_s), ("wait_s", wait_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DesignerError(f"{label} must be a positive number of seconds")
        self.client = client
        self.is_harness_prompt = is_harness_prompt
        self.poll_s = float(poll_s)
        self.wait_s = float(wait_s)
        self.generation: int | None = None
        self.version: str | None = None
        self.generation_of: dict[str, int] = {}
        self.report_counts: dict[int, int] = {}
        # The version the Designer served when a generation's first report went out: what the next one waits past.
        self.reported_versions: dict[int, str | None] = {}
        # Proposals register on the job thread while reports arrive on the event loop thread.
        self.lock = threading.Lock()

    def proposed(self, generation: int, record_id: str) -> None:
        """Remember the generation a Designer record belongs to, so its report counts for that generation."""
        with self.lock:
            self.generation_of[record_id] = generation

    def reported(self, record_id: str, scenario: str) -> None:
        """Count a report for one of this service's proposals; a generation's first report reads the Designer's version.

        The read goes over HTTP, so the caller runs this off the event loop.
        """
        with self.lock:
            generation = self.generation_of.get(record_id)
            if generation is None:
                return
            count = self.report_counts.get(generation, 0) + 1
            self.report_counts[generation] = count
        if count > 1:
            return
        try:
            version = self.version_of(scenario)
        except DesignerError as exc:
            logger.warning("%s; generation %d's reports are timed at its start version", exc, generation)
            version = self.version
        with self.lock:
            self.reported_versions[generation] = version

    def begin(self, generation: int, scenario: str) -> None:
        """Before a proposal: a generation's first waits for the Designer's version to move past the last generation's.

        The version to move past is the one the Designer served when the last generation's reports went out, so a
        release or a load that appeared before those reports (the scenario's creation release) does not count.
        """
        if generation == self.generation:
            return
        with self.lock:
            reported = self.report_counts.get(self.generation, 0) if self.generation is not None else 0
            previous = (
                self.reported_versions.get(self.generation, self.version) if self.generation is not None else None
            )
        if previous is not None and reported > 0:
            version = self.wait_past(scenario, generation=generation, reported=reported, previous=previous)
        else:
            try:
                version = self.version_of(scenario)
            except DesignerError as exc:
                logger.warning("%s; generation %d proceeds without one", exc, generation)
                version = None
            logger.info(
                "generation %d asks the Designer at version %s",
                generation,
                version if version is not None else "none, a fixed deployment",
            )
        self.generation = generation
        self.version = version

    def version_of(self, scenario: str) -> str | None:
        """The Designer's version now: the served release id, else the runtime load id with the scenario's step; None when fixed.

        The step is part of a weight Designer's version because a step that skips (every proposal scored alike)
        leaves the load id where it was and still took the generation in.
        """
        try:
            if self.is_harness_prompt:
                manifest = self.client.get(HARNESS_PATH, extra_headers={"x-reef-scenario": scenario})
                value = manifest.get("release_id")
            else:
                scenarios = self.client.get(STATUS_PATH).get("scenarios")
                block = scenarios.get(scenario) if isinstance(scenarios, Mapping) else None
                value = block.get("current_runtime_load_id") if isinstance(block, Mapping) else None
                step = block.get("scenario_step") if isinstance(block, Mapping) else None
                if isinstance(value, str) and value and isinstance(step, int) and not isinstance(step, bool):
                    value = f"{value}@{step}"
        except ReefClientError as exc:
            # A scenario that serves no files is a fixed Designer, not a failed read.
            if self.is_harness_prompt and exc.status == 404:
                return None
            raise DesignerError(f"the Designer's version could not be read ({exc.status}): {exc.body[:300]}") from exc
        except OSError as exc:
            raise DesignerError(f"the Designer's version could not be read: {exc}") from exc
        return value if isinstance(value, str) and value else None

    def wait_past(self, scenario: str, *, generation: int, reported: int, previous: str) -> str | None:
        """Poll until the Designer serves a version other than ``previous`` or ``wait_s`` runs out; the version then."""
        logger.info(
            "generation %d waits at Designer version %s for the deployment to take generation %s in (%d reported)",
            generation,
            previous,
            self.generation,
            reported,
        )
        started = time.monotonic()
        failure = ""
        while True:
            try:
                version = self.version_of(scenario)
            except DesignerError as exc:
                version = None
                if str(exc) != failure:
                    failure = str(exc)
                    logger.warning("%s; the wait goes on", failure)
            elapsed = time.monotonic() - started
            if version is not None and version != previous:
                logger.info("generation %d asks the Designer at version %s after %.0f s", generation, version, elapsed)
                return version
            if elapsed >= self.wait_s:
                logger.warning(
                    "the Designer's deployment did not move past version %s within %.0f s; generation %d proceeds",
                    previous,
                    elapsed,
                    generation,
                )
                return version
            time.sleep(min(self.poll_s, self.wait_s - elapsed))
