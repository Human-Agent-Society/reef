"""The checks a generated task runs before the recipe plays it, and what each one found.

``validate`` runs them in order on the task's copy under the generator service's tasks root:

- :class:`OracleCheck`: Harbor's oracle and nop agents. The reference solution must score 1, and doing nothing
  must score below 1.
- :class:`ReviewCheck`: one model call reads the instruction, the tests and the reference solution, and lists
  each requirement that the tests check and the instruction does not state.

A refusing check keeps a task out when it finds a defect or cannot run; the two are recorded apart. A check that
only measures records what it found and never keeps a task out.
"""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from recipes.beta.autoenvscaling.author import ModelReplies
from reef.core.prompts import untrusted_text
from reef.core.tasks import HarborTask
from reef.record2task.generator.client import Generator, WrittenTask
from reef.record2task.generator.designer import DesignerError

# A check that is off does not run; one that measures records its result; one that refuses also keeps a task out.
CHECK_MODES = ("off", "measure", "refuse")
REVIEW_KINDS = ("wording", "unnamed_interface", "unstated_requirement")
# The keys of one finding in the review's reply, in ReviewFinding's order.
REVIEW_FIELDS = ("test_file", "kind", "requirement", "reason")
# Each file the review reads keeps this many characters.
REVIEW_FILE_CHARS = 20_000
ERROR_CHARS = 1500
FENCED_REPLY = re.compile(r"```(?:json)?\s*\n(.*)\n```", re.DOTALL)

REVIEW_SYSTEM_PROMPT = """You review the tests of a coding task before an agent is trained on it. The agent that \
solves the task reads only the instruction. It never sees the tests or the reference solution. A correct solution is \
any change that does what the instruction asks, not only the change in the reference solution. Your job is to find \
each requirement in the tests that a correct solution can fail because the instruction does not state it. You reply \
with JSON only."""

REVIEW_PROMPT = """Review the tests of the task below.

INSTRUCTION is all the solver sees. TESTS run after the solver finishes and grade its work. SOLUTION is the \
reference solution: one correct way to do the task. A different correct solution must also pass the tests.

List each requirement that the tests check and the instruction does not state. Give each one a kind:
- "wording": a test matches free text, such as the wording of a report, a doc, a comment, a log line or a message, \
that the instruction does not give word for word.
- "unnamed_interface": a test calls or looks for a function, class, method, flag, option, file name, path or UI label \
that the instruction does not name.
- "unstated_requirement": a test requires something else that the instruction does not state, such as a number of \
commits, a format, an extra file or a side effect.

Do not list:
- behaviour that the instruction asks for, checked by running commands or tests, or by reading files and outputs \
that the instruction names;
- names and text that the instruction gives;
- what every correct solution does anyway, such as keeping the repository's existing tests passing.

Reply with one JSON object and nothing else:
{{"findings": [{{"test_file": "<path under tests/>", "kind": "wording" or "unnamed_interface" or \
"unstated_requirement", "requirement": "<what the test requires>", "reason": "<why the instruction does not state \
it>"}}]}}
When the tests check only what the instruction states, reply {{"findings": []}}.

INSTRUCTION:
{instruction}

TESTS:
{tests}

SOLUTION:
{solution}"""


@dataclass(frozen=True)
class CheckResult:
    """What one check found: the task's defects, or why the check could not run, and what it measured."""

    check: str
    errors: tuple[str, ...] = ()
    could_not_run: str = ""
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ReviewFinding:
    """One requirement that a test checks and the instruction does not state."""

    test_file: str
    kind: str
    requirement: str
    reason: str


class TaskCheck(ABC):
    """One check that ``validate`` runs on a task's copy under the generator service's tasks root.

    With ``is_refusing``, a defect or a check that could not run keeps the task out; without it, the check only
    measures.
    """

    def __init__(self, name: str, *, is_refusing: bool = True) -> None:
        self.name = name
        self.is_refusing = is_refusing

    @abstractmethod
    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        """The check's result for ``task``, whose copy ``written`` is under the tasks root."""


class OracleCheck(TaskCheck):
    """Harbor's oracle and nop agents: the reference solution scores 1, and doing nothing scores below 1.

    A check that cannot run raises ``GeneratorError``: the generator service that plays the tasks is not usable.
    """

    def __init__(self, service: Generator) -> None:
        super().__init__("oracle")
        self.service = service

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        oracle = await self.service.check(written.path)
        details = {"oracle_reward": oracle.oracle_reward, "nop_reward": oracle.nop_reward}
        errors = () if oracle.is_solvable else (f"oracle check refused: {oracle.reason}",)
        return CheckResult(self.name, errors, details=details)


def review_prompt(task: HarborTask) -> str:
    """The review's user turn: the instruction, every test file and every solution file, each cut to
    :data:`REVIEW_FILE_CHARS` characters and fenced as data."""

    def fenced(text: str, label: str) -> str:
        cut = f"\n[... {len(text) - REVIEW_FILE_CHARS} characters cut]" if len(text) > REVIEW_FILE_CHARS else ""
        return untrusted_text(text[:REVIEW_FILE_CHARS] + cut, label)

    def files_text(folder: str, files: Mapping[str, str]) -> str:
        return "\n\n".join(fenced(text, f"{folder}/{name}") for name, text in sorted(files.items())) or "(none)"

    return REVIEW_PROMPT.format(
        instruction=fenced(task.instruction, "instruction.md"),
        tests=files_text("tests", task.tests),
        solution=files_text("solution", task.solution),
    )


def review_findings(reply: str) -> tuple[ReviewFinding, ...]:
    """The findings a review's reply lists. Raises ``ValueError`` when the reply is not the JSON asked for."""
    text = reply.strip()
    fenced = FENCED_REPLY.fullmatch(text)
    document = json.loads(fenced.group(1) if fenced else text)
    entries = document.get("findings") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError("the reply holds no findings list")
    findings = []
    for entry in entries:
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(key), str) or not entry[key].strip() for key in REVIEW_FIELDS
        ):
            raise ValueError("each finding needs test_file, kind, requirement and reason as non-empty text")
        finding = ReviewFinding(**{key: entry[key] for key in REVIEW_FIELDS})
        if finding.kind not in REVIEW_KINDS:
            raise ValueError(f"a finding's kind must be one of {REVIEW_KINDS}, not {finding.kind!r}")
        findings.append(finding)
    return tuple(findings)


class ReviewCheck(TaskCheck):
    """One model call that lists each requirement the tests check and the instruction does not state.

    The call asks for ``model`` through the Reef at ``reef_url``, in the ``api`` dialect (``openai`` or
    ``responses``), and Reef records it under ``scenario``. Any finding is a defect. A refused or failed call, and a
    reply that is not the JSON asked for, mean the check could not run. The agent's own check calls
    :meth:`review` too, so the agent can revise its tests before it finishes.
    """

    def __init__(
        self, *, reef_url: str, token: str | None, scenario: str, model: str, api: str, is_refusing: bool = True
    ) -> None:
        super().__init__("review", is_refusing=is_refusing)
        self.replies = ModelReplies(reef_url=reef_url, token=token, api=api)
        self.scenario = scenario
        self.model = model

    def review(self, task: HarborTask, *, tags: Mapping[str, str]) -> CheckResult:
        """The review of ``task``, a blocking model call; ``tags`` label the call on Reef."""
        try:
            answer = self.replies.reply(
                scenario=self.scenario,
                model=self.model,
                system=REVIEW_SYSTEM_PROMPT,
                prompt=review_prompt(task),
                tags=tags,
            )
        except (DesignerError, ValueError) as exc:
            return CheckResult(self.name, could_not_run=f"the review's call failed: {exc}")
        record = {"record_id": answer.record_id}
        try:
            findings = review_findings(answer.text)
        except ValueError as exc:
            reason = f"the review's reply is not the JSON asked for: {exc}"
            return CheckResult(self.name, could_not_run=reason, details={**record, "reply": answer.text[:ERROR_CHARS]})
        errors = tuple(f"{finding.kind} in {finding.test_file}: {finding.requirement}" for finding in findings)
        return CheckResult(
            self.name, errors, details={**record, "findings": [asdict(finding) for finding in findings]}
        )

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        return await asyncio.to_thread(self.review, task, tags={"role": "test-review", "task": task.name})


def check_mode(value: object, label: str) -> str:
    """A check's mode from the config. YAML reads a bare ``off`` as false, so false is ``off`` too."""
    mode = "off" if value is False else value
    if mode not in CHECK_MODES:
        raise ValueError(f"{label} must be one of {', '.join(CHECK_MODES)}")
    return str(mode)
