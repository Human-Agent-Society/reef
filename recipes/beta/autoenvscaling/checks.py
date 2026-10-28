"""The checks a generated task runs before the recipe runs it, and what each one found.

``validate`` runs them in order on the task's copy under the generator service's tasks root:

- :class:`OracleCheck`: Harbor's oracle and nop agents. The reference solution must score 1, and doing nothing
  must score below 1.
- :class:`ContaminationCheck`: the instruction must share no run of 13 consecutive word tokens (by default) with
  the instruction of a reference task, such as a benchmark task kept for evaluation.
- :class:`ReviewCheck`: one model call reads the instruction, the tests and the reference solution, and lists
  each requirement that the tests check and the instruction does not state.
- :class:`CalibrationCheck`: episodes of the current solver on the task. Their mean reward must lie inside the
  pass-rate band, and their rewards must vary.

A refusing check keeps a task out when it finds a defect or cannot run; the two are recorded apart. A check that
only measures records what it found and never keeps a task out. :func:`band_place` places a pass rate against the
band, for the calibration and for the training pass rates in the generation report.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

from recipes.beta.autoenvscaling.author import ModelReplies
from reef.core.prompts import untrusted_text
from reef.core.tasks import HarborTask
from reef.harness.client.tasks import TaskEpisode, rewards_of
from reef.record2sample.generator.client import Generator, GeneratorError, WrittenTask
from reef.record2sample.generator.designer import DesignerError

# A check that is off does not run; one that measures records its result; one that refuses also keeps a task out.
CHECK_MODES = ("off", "measure", "refuse")
REVIEW_KINDS = ("wording", "unnamed_interface", "unstated_requirement")
# The keys of one finding in the review's reply, in ReviewFinding's order.
REVIEW_FIELDS = ("test_file", "kind", "requirement", "reason")
# Each file the review reads keeps this many characters.
REVIEW_FILE_CHARS = 20_000
ERROR_CHARS = 1500
FENCED_REPLY = re.compile(r"```(?:json)?\s*\n(.*)\n```", re.DOTALL)
# Where a pass rate falls against the band.
BAND_BELOW = "below"
BAND_INSIDE = "inside"
BAND_ABOVE = "above"
# The generator service's arm of the calibration episodes. They are rollouts: Reef keeps no record of them.
CALIBRATION_ARM = "calibration"
# What the calibration found, besides a mean reward below or above the band: the task is kept, its rewards vary
# too little, or the calibration did not finish (fewer than CALIBRATION_MIN_SCORED episodes scored, or the
# generator service failed).
CALIBRATION_KEPT = "kept"
CALIBRATION_NO_VARIANCE = "no_variance"
CALIBRATION_OUTAGE = "outage"
# The calibration's result keeps the errors of this many episodes.
CALIBRATION_ERRORS = 3
# The fewest scored episodes that measure a task: the std of one reward is 0, whatever the task.
CALIBRATION_MIN_SCORED = 2
# A mean's distance from a band edge and the rewards' std are rounded to this many decimals before they are compared,
# so float error does not move a value that equals its bound past it: 0.4 is 0.15 from 0.25, and the population std
# of [0.4, 0.6, 0.4, 0.6] is 0.1, though plain floats make them 0.15000000000000002 and 0.09999999999999998.
COMPARISON_DECIMALS = 9
# The contamination check: an instruction is contaminated when it shares CONTAMINATION_N consecutive word tokens
# with the instruction of a reference task. Its result keeps CONTAMINATION_HITS shared n-grams, each cut to
# CONTAMINATION_NGRAM_CHARS characters.
CONTAMINATION_N = 13
CONTAMINATION_HITS = 5
CONTAMINATION_NGRAM_CHARS = 200
# Benchmark tasks can carry the same canary string; a line that names a canary says nothing about the task.
CANARY_LINE = re.compile(r"canary", re.IGNORECASE)
WORD_TOKEN = re.compile(r"\w+")

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

    A check that cannot run raises ``GeneratorError``: the generator service that runs the tasks is not usable.
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


def validate_band(low: float | None, high: float | None) -> None:
    """Raise ``ValueError`` unless each bound is in [0, 1] or null and ``low`` does not exceed ``high``."""
    for label, bound in (("band_low", low), ("band_high", high)):
        if bound is not None and not 0 <= bound <= 1:
            raise ValueError(f"{label} must be in [0, 1] or null")
    if low is not None and high is not None and low > high:
        raise ValueError("band_low must not exceed band_high")


def band_place(pass_rate: float, low: float | None, high: float | None) -> str:
    """Where ``pass_rate`` falls against the band ``[low, high]``: below, inside or above. A null bound is no bound."""
    if low is not None and pass_rate < low:
        return BAND_BELOW
    if high is not None and pass_rate > high:
        return BAND_ABOVE
    return BAND_INSIDE


class CalibrationCheck(TaskCheck):
    """Episodes of the current solver on the task: their mean reward must lie inside the band, and must vary.

    Values: ``rollouts`` is at least :data:`CALIBRATION_MIN_SCORED`, ``max_rollouts`` at least ``rollouts``,
    ``margin`` in [0, 1] and ``min_std`` in [0, 0.5]. ``band_low`` and ``band_high`` are each in [0, 1] or null, and
    ``band_low`` does not exceed ``band_high``; a null bound is no bound and no edge. The episodes run on the
    generator service ``service`` under ``scenario``, on the arm :data:`CALIBRATION_ARM`, with ``model``: a model
    name, or None for the service's default model.
    Purpose: ``from_config`` builds it from the ``calibration`` keys and puts it last in ``task_checks``.
    ``validate`` runs it once the other refusing checks pass, and ``record_results`` writes :meth:`settings` in
    the generation report.

    The rule: ``rollouts`` episodes run, and the episodes that could not run are asked for once more. When the
    mean reward is within ``margin`` of a band edge, on either side, and fewer than ``max_rollouts`` episodes
    scored, the rest up to ``max_rollouts`` run, with the same one retry. The task is kept when the mean is inside
    the band and the population standard deviation of the rewards is at least ``min_std``. When fewer than
    :data:`CALIBRATION_MIN_SCORED` episodes scored, the task's difficulty is unknown, and the check could not run.
    When the generator service fails an episode request, a refusing check raises ``GeneratorError``, and one that
    measures could not run.
    """

    def __init__(
        self,
        service: Generator,
        *,
        scenario: str,
        rollouts: int,
        max_rollouts: int,
        margin: float,
        min_std: float,
        band_low: float | None,
        band_high: float | None,
        model: str | None = None,
        is_refusing: bool = True,
    ) -> None:
        if rollouts < CALIBRATION_MIN_SCORED:
            raise ValueError(f"calibration_rollouts must be at least {CALIBRATION_MIN_SCORED}")
        if max_rollouts < rollouts:
            raise ValueError("calibration_max_rollouts must be at least calibration_rollouts")
        if not 0 <= margin <= 1:
            raise ValueError("calibration_margin must be in [0, 1]")
        # The population standard deviation of rewards in [0, 1] is at most 0.5.
        if not 0 <= min_std <= 0.5:
            raise ValueError("calibration_min_std must be in [0, 0.5]")
        validate_band(band_low, band_high)
        super().__init__("calibration", is_refusing=is_refusing)
        self.service = service
        self.scenario = scenario
        self.rollouts = rollouts
        self.max_rollouts = max_rollouts
        self.margin = margin
        self.min_std = min_std
        self.band_low = band_low
        self.band_high = band_high
        self.model = model

    def settings(self) -> dict[str, object]:
        """The calibration's mode and rule, as the generation report keeps them."""
        return {
            "mode": "refuse" if self.is_refusing else "measure",
            "rollouts": self.rollouts,
            "max_rollouts": self.max_rollouts,
            "margin": self.margin,
            "min_std": self.min_std,
            "model": self.model,
        }

    def is_near_an_edge(self, mean: float) -> bool:
        """Whether ``mean`` is within ``margin`` of a band edge, on either side of it."""
        return any(
            round(abs(mean - edge), COMPARISON_DECIMALS) <= self.margin
            for edge in (self.band_low, self.band_high)
            if edge is not None
        )

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        episodes: list[TaskEpisode] = []
        asked: list[int] = []

        async def play(count: int) -> None:
            # count episodes, then once more as many as could not run.
            for _ in range(2):
                asked.append(count)
                played = await self.service.run_episodes(
                    written.path,
                    scenario=self.scenario,
                    arm=CALIBRATION_ARM,
                    count=count,
                    extra_instruction_files=(),
                    tags={"role": "calibration"},
                    model=self.model,
                )
                episodes.extend(played)
                count = sum(episode.is_infrastructure_failure for episode in played)
                if count == 0:
                    return

        is_expanded = False
        failure = ""
        try:
            await play(self.rollouts)
            rewards = rewards_of(tuple(episodes))
            if rewards and len(rewards) < self.max_rollouts and self.is_near_an_edge(statistics.fmean(rewards)):
                is_expanded = True
                await play(self.max_rollouts - len(rewards))
        except GeneratorError as exc:
            if self.is_refusing:
                raise
            failure = f"the calibration episodes of {task.name} could not run: {exc}"
        rewards = rewards_of(tuple(episodes))
        episode_errors = [episode.error for episode in episodes if episode.error]
        details: dict[str, object] = {
            # The mode this result was made under: the report's calibration settings are those at its close.
            "mode": "refuse" if self.is_refusing else "measure",
            "rewards": list(rewards),
            "mean": None,
            "std": None,
            "episodes_asked": sum(asked),
            "unscored": sum(episode.is_infrastructure_failure for episode in episodes),
            "expanded": is_expanded,
            "place": None,
            "result": CALIBRATION_OUTAGE,
            "episode_errors": [error[:ERROR_CHARS] for error in episode_errors[:CALIBRATION_ERRORS]],
        }
        if failure:
            return CheckResult(self.name, could_not_run=failure[:ERROR_CHARS], details=details)
        if len(rewards) < CALIBRATION_MIN_SCORED:
            # The task itself may keep its episodes from running, so too few scores end this task's check, not the round.
            first_error = episode_errors[0] if episode_errors else "no episode came back"
            reason = (
                f"{len(rewards)} of {sum(asked)} calibration episodes of {task.name} scored, too few to measure its "
                f"difficulty: {first_error[:ERROR_CHARS]}"
            )
            return CheckResult(self.name, could_not_run=reason, details=details)
        mean = statistics.fmean(rewards)
        std = statistics.pstdev(rewards)
        place = band_place(mean, self.band_low, self.band_high)
        measured = f"over {len(rewards)} calibration episodes"
        if place == BAND_BELOW:
            result = place
            error = f"the solver's mean reward {measured}, {mean:.2f}, is below band_low {self.band_low}"
        elif place == BAND_ABOVE:
            result = place
            error = f"the solver's mean reward {measured}, {mean:.2f}, is above band_high {self.band_high}"
        elif round(std, COMPARISON_DECIMALS) < self.min_std:
            result = CALIBRATION_NO_VARIANCE
            error = f"the population std of the solver's rewards {measured}, {std:.3f}, is below {self.min_std}"
        else:
            result, error = CALIBRATION_KEPT, ""
        details = {**details, "mean": mean, "std": std, "place": place, "result": result}
        return CheckResult(self.name, (error,) if error else (), details=details)


def instruction_tokens(text: str) -> list[str]:
    """The lowercase word tokens of ``text``, without the lines that name a canary.

    Case, punctuation, Markdown and whitespace do not change the tokens.
    """
    kept = "\n".join(line for line in text.splitlines() if not CANARY_LINE.search(line))
    return WORD_TOKEN.findall(kept.lower())


def instruction_ngrams(tokens: list[str], n: int) -> list[str]:
    """Every run of ``n`` consecutive tokens, with stride 1, joined by spaces; none for fewer than ``n`` tokens."""
    return [" ".join(tokens[start : start + n]) for start in range(len(tokens) - n + 1)]


def reference_instructions(root: Path) -> dict[str, str]:
    """Every ``instruction.md`` under ``root``, in path order, by the name of its task.

    A task's name is its directory relative to ``root``, or ``root``'s own name for an ``instruction.md`` in
    ``root``. Raises ``ValueError`` when ``root`` is not a directory or holds no ``instruction.md``.
    """
    if not root.is_dir():
        raise ValueError(f"contamination_tasks_path {root} is not a directory")
    instructions: dict[str, str] = {}
    for path in sorted(root.rglob("instruction.md")):
        relative = path.parent.relative_to(root).as_posix()
        instructions[root.name if relative == "." else relative] = path.read_text(encoding="utf-8", errors="replace")
    if not instructions:
        raise ValueError(f"contamination_tasks_path {root} holds no instruction.md")
    return instructions


class ContaminationCheck(TaskCheck):
    """The task's instruction must share no n-gram of word tokens with the instruction of a reference task.

    Values: ``n`` is at least 1. ``index`` maps each n-gram of a reference instruction to the reference tasks
    whose instruction holds it, in name order. Each reference task is indexed on its own, so no n-gram spans two
    tasks. ``reference_task_count`` is the number of reference tasks, and ``tasks_path`` the directory they were
    read from, as the config gives it.
    Purpose: ``from_config`` builds it from the ``contamination`` keys and puts it in ``task_checks`` right after
    the oracle check. ``validate`` runs it on each task, and the generation report keeps its result.

    The rule is upstream's, after Tmax: the tokens are :func:`instruction_tokens`, and the n-grams
    :func:`instruction_ngrams`. Any shared n-gram is a defect. The index is built once, so the check never fails
    to run. It makes no model call and runs no sandbox.
    """

    def __init__(self, instructions: Mapping[str, str], *, n: int, tasks_path: str, is_refusing: bool = True) -> None:
        if n < 1:
            raise ValueError("contamination_ngram must be an integer of at least 1")
        super().__init__("contamination", is_refusing=is_refusing)
        self.n = n
        self.tasks_path = tasks_path
        self.reference_task_count = len(instructions)
        index: dict[str, list[str]] = {}
        for name in sorted(instructions):
            for ngram in set(instruction_ngrams(instruction_tokens(instructions[name]), n)):
                index.setdefault(ngram, []).append(name)
        self.index: dict[str, tuple[str, ...]] = {ngram: tuple(names) for ngram, names in index.items()}

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        # Each distinct shared n-gram, in the instruction's order, and the first reference task that holds it.
        shared: dict[str, str] = {}
        for ngram in instruction_ngrams(instruction_tokens(task.instruction), self.n):
            if ngram in self.index and ngram not in shared:
                shared[ngram] = self.index[ngram][0]
        hits = [
            {"reference_task": name, "ngram": ngram[:CONTAMINATION_NGRAM_CHARS]}
            for ngram, name in list(shared.items())[:CONTAMINATION_HITS]
        ]
        details = {
            "n": self.n,
            "tasks_path": self.tasks_path,
            "reference_tasks": self.reference_task_count,
            "shared_ngrams": len(shared),
            "hits": hits,
        }
        if not hits:
            return CheckResult(self.name, details=details)
        first = hits[0]
        quoted = f'"{first["ngram"]}"'
        error = f"instruction.md shares a {self.n}-gram with reference task {first['reference_task']}: {quoted}"
        return CheckResult(self.name, (error,), details=details)
