"""The checks a generated task runs before the recipe runs it, and what each one found.

``validate`` runs them in order on the task's copy under the generator service's tasks root:

- :class:`OracleCheck`: Harbor's oracle and nop agents. The reference solution must score 1, and doing nothing
  must score below 1.
- :class:`ContaminationCheck`: the instruction must share no run of 13 consecutive word tokens (by default) with
  the instruction of a reference task, such as a benchmark task kept for evaluation.
- :class:`ReviewCheck`: one model call reads the instruction, the tests and the reference solution, and lists
  each requirement that the tests check and the instruction does not state.
- :class:`CalibrationCheck`: rollouts of the current solver on the task. Their mean reward must lie inside the
  pass-rate band, and their rewards must vary.

A seed task runs :class:`SeedCheck` in place of the oracle check, then the contamination check alone.

A refusing check keeps a task out when it finds a defect or cannot run; the two are recorded apart. A check that
only measures records what it found and never keeps a task out. :func:`band_place` places a pass rate against the
band, for the calibration and for the training pass rates in the generation report.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from recipes.beta.autoenvscaling.designer import ModelReplies
from reef.core.prompts import untrusted_text
from reef.core.tasks import HarborTask
from reef.harness.tasks.rollouts import TaskRollout, rewards_of
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
BandPlace = Literal["below", "inside", "above"]
"""A mean below the lower bound, inside both inclusive bounds, or above the upper bound."""

BAND_BELOW: Literal["below"] = "below"
BAND_INSIDE: Literal["inside"] = "inside"
BAND_ABOVE: Literal["above"] = "above"
# The generator service's arm of the calibration rollouts. Reef keeps no record of them.
CALIBRATION_ARM = "calibration"
# What the calibration found, besides a mean reward below or above the band: the task is kept, its rewards vary
# too little, or the calibration did not finish (fewer than CALIBRATION_MIN_SCORED rollouts scored, or the
# generator service failed).
CalibrationResult = Literal["kept", "below", "above", "no_variance", "outage"]
"""Admission, an out-of-band mean, insufficient variance, or an unmeasured task, respectively."""

CALIBRATION_KEPT: Literal["kept"] = "kept"
CALIBRATION_NO_VARIANCE: Literal["no_variance"] = "no_variance"
CALIBRATION_OUTAGE: Literal["outage"] = "outage"
# The calibration's result keeps the errors of this many rollouts.
CALIBRATION_ERRORS = 3
# The fewest scored rollouts that measure a task: the std of one reward is 0, whatever the task.
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

    Values: name is non-empty text, is_refusing is a bool, and stage is check or calibration.
    Subclasses implement run(task, written), returning a CheckResult for that task's written copy.
    Purpose: validate records the result. When is_refusing is true, defects or failure to run
    reject the task; otherwise the check only measures. Stage labels defects in the report.
    """

    stage: Literal["check", "calibration"] = "check"

    def __init__(self, name: str, *, is_refusing: bool = True) -> None:
        if not name.strip() or type(is_refusing) is not bool or self.stage not in ("check", "calibration"):
            raise ValueError("a task check needs a name, a refusing bool and a valid stage")
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


class SeedCheck(TaskCheck):
    """Harbor's oracle trial on a seed task: its environment builds and its verifier scores, and doing nothing does
    not pass.

    Values: ``service`` is the generator service that runs the trials. The check refuses a seed whose oracle
    reward is None: the environment did not build, or the verifier wrote no reward. It passes a seed whose oracle
    and nop check passes, or whose oracle reward is below 1: the service stops after a failed oracle trial, so for a
    seed whose solution does nothing, as a converted TMax task's does, that trial is the nop measurement. A seed
    whose solution scores 1 while doing nothing also scores 1, or gives no reward, is refused. Its details hold
    ``oracle_reward`` and ``nop_reward``.
    Purpose: ``validate`` runs it in place of :class:`OracleCheck` on a seed task, which comes with no reference
    solution to hold to a score of 1. A check that cannot run raises ``GeneratorError``, as the oracle check does.
    """

    def __init__(self, service: Generator) -> None:
        super().__init__("seed")
        self.service = service

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        oracle = await self.service.check(written.path)
        details = {"oracle_reward": oracle.oracle_reward, "nop_reward": oracle.nop_reward}
        if oracle.oracle_reward is None:
            errors: tuple[str, ...] = (
                f"seed check refused: the environment did not build or its verifier wrote no reward: {oracle.reason}",
            )
        elif oracle.is_solvable or oracle.oracle_reward < 1:
            errors = ()
        else:
            errors = (f"seed check refused: doing nothing passes the seed: {oracle.reason}",)
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


def band_place(pass_rate: float, low: float | None, high: float | None) -> BandPlace:
    """Where ``pass_rate`` falls against the band ``[low, high]``: below, inside or above. A null bound is no bound."""
    if low is not None and pass_rate < low:
        return BAND_BELOW
    if high is not None and pass_rate > high:
        return BAND_ABOVE
    return BAND_INSIDE


@dataclass(frozen=True)
class PassRateBand:
    """The inclusive interval shared by training reports and solver calibration.

    Values: low and high are non-boolean numbers in [0, 1] or None, with low <= high
    when both exist. None removes that bound. Configuration constructs the value;
    calibration and record_results read the same value to classify a mean reward.
    """

    low: float | None
    high: float | None

    def __post_init__(self) -> None:
        for label, bound in (("band_low", self.low), ("band_high", self.high)):
            if bound is not None and (type(bound) not in (int, float) or not 0 <= bound <= 1):
                raise ValueError(f"{label} must be in [0, 1] or null")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("band_low must not exceed band_high")


@dataclass(frozen=True)
class CalibrationSample:
    """The ordered observations from one calibration attempt.

    Values: rewards is a possibly empty tuple of finite numbers, with duplicates allowed.
    Verifiers can return rewards outside [0, 1]; these remain visible in the report.
    Asked and unscored are non-negative integers; len(rewards) + unscored <= asked.
    Expanded is a bool indicating an edge-triggered request. Errors are rollout error
    strings in arrival order. Failure is empty or the service error that ended sampling.
    Purpose: sample returns these observations; CalibrationOutcome classifies them.
    """

    rewards: tuple[float, ...]
    asked: int
    unscored: int
    expanded: bool
    errors: tuple[str, ...]
    failure: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.rewards, tuple) or any(
            type(reward) not in (int, float) or not math.isfinite(reward) for reward in self.rewards
        ):
            raise ValueError("calibration rewards must be a tuple of finite numbers")
        if (
            type(self.asked) is not int
            or type(self.unscored) is not int
            or self.unscored < 0
            or self.asked < len(self.rewards) + self.unscored
        ):
            raise ValueError("calibration counts must cover every observed rollout")
        if type(self.expanded) is not bool or not isinstance(self.failure, str):
            raise ValueError("calibration needs an expanded bool and failure text")
        if not isinstance(self.errors, tuple) or any(not isinstance(error, str) for error in self.errors):
            raise ValueError("calibration errors must be a tuple of strings")


@dataclass(frozen=True)
class CalibrationOutcome:
    """A sample classified against a band and a minimum population standard deviation.

    Values: sample is a CalibrationSample, band a PassRateBand, and min_std a number in [0, 0.5].
    All remaining fields are derived. Failure or fewer than two rewards implies outage
    and mean, std and place are None. Otherwise mean and std are the sample statistics;
    place classifies mean against band. Below/above implies the same result. Inside
    implies no_variance when round(std, 9) < min_std, otherwise kept.
    Purpose: CalibrationCheck.run constructs the outcome and serializes its derived fields.
    """

    sample: CalibrationSample
    band: PassRateBand
    min_std: float
    mean: float | None = field(init=False, default=None)
    std: float | None = field(init=False, default=None)
    place: BandPlace | None = field(init=False, default=None)
    result: CalibrationResult = field(init=False, default=CALIBRATION_OUTAGE)

    def __post_init__(self) -> None:
        if not isinstance(self.sample, CalibrationSample) or not isinstance(self.band, PassRateBand):
            raise ValueError("a calibration outcome needs a sample and band")
        if type(self.min_std) not in (int, float) or not 0 <= self.min_std <= 0.5:
            raise ValueError("calibration_min_std must be in [0, 0.5]")
        if self.sample.failure or len(self.sample.rewards) < CALIBRATION_MIN_SCORED:
            return
        mean = statistics.fmean(self.sample.rewards)
        std = statistics.pstdev(self.sample.rewards)
        place = band_place(mean, self.band.low, self.band.high)
        result: CalibrationResult
        if place in (BAND_BELOW, BAND_ABOVE):
            result = place
        elif round(std, COMPARISON_DECIMALS) < self.min_std:
            result = CALIBRATION_NO_VARIANCE
        else:
            result = CALIBRATION_KEPT
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "place", place)
        object.__setattr__(self, "result", result)


def validate_calibration(rollouts: int, max_rollouts: int, margin: float, min_std: float) -> None:
    """Validate the sampling rule, including configurations whose calibration mode is off."""
    if rollouts < CALIBRATION_MIN_SCORED:
        raise ValueError(f"calibration_rollouts must be at least {CALIBRATION_MIN_SCORED}")
    if max_rollouts < rollouts:
        raise ValueError("calibration_max_rollouts must be at least calibration_rollouts")
    if not 0 <= margin <= 1:
        raise ValueError("calibration_margin must be in [0, 1]")
    if not 0 <= min_std <= 0.5:
        raise ValueError("calibration_min_std must be in [0, 0.5]")


class CalibrationCheck(TaskCheck):
    """Rollouts of the current solver on the task: their mean reward must lie inside the band, and must vary.

    Values: ``rollouts`` is at least :data:`CALIBRATION_MIN_SCORED`, ``max_rollouts`` at least ``rollouts``,
    ``margin`` in [0, 1] and ``min_std`` in [0, 0.5]. ``band`` is the generator's immutable PassRateBand;
    a null bound is no bound and no edge. The rollouts run on the
    generator service ``service`` under ``scenario``, on the arm :data:`CALIBRATION_ARM`, with ``model``: a model
    name, or None for the service's default model.
    Purpose: ``from_config`` builds it from the ``calibration`` keys and puts it last in ``task_checks``.
    ``validate`` runs it once the other refusing checks pass, and ``record_results`` writes :meth:`settings` in
    the generation report.

    The rule: ``rollouts`` rollouts run, and the rollouts that could not run are asked for once more. When the
    mean reward is within ``margin`` of a band edge, on either side, and fewer than ``max_rollouts`` rollouts
    scored, the rest up to ``max_rollouts`` run, with the same one retry. The task is kept when the mean is inside
    the band and the population standard deviation of the rewards is at least ``min_std``. When fewer than
    :data:`CALIBRATION_MIN_SCORED` rollouts scored, the task's difficulty is unknown, and the check could not run.
    When the generator service fails a rollout request, a refusing check raises ``GeneratorError``, and one that
    measures could not run.
    """

    stage: Literal["check", "calibration"] = "calibration"

    def __init__(
        self,
        service: Generator,
        *,
        scenario: str,
        rollouts: int,
        max_rollouts: int,
        margin: float,
        min_std: float,
        band: PassRateBand,
        model: str | None = None,
        is_refusing: bool = True,
    ) -> None:
        validate_calibration(rollouts, max_rollouts, margin, min_std)
        super().__init__("calibration", is_refusing=is_refusing)
        self.service = service
        self.scenario = scenario
        self.rollouts = rollouts
        self.max_rollouts = max_rollouts
        self.margin = margin
        self.min_std = min_std
        self.band = band
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
            for edge in (self.band.low, self.band.high)
            if edge is not None
        )

    async def sample(self, task: HarborTask, written: WrittenTask) -> CalibrationSample:
        """Sample, retry unscored rollouts once per batch, and expand near either band edge."""
        rollouts: list[TaskRollout] = []
        asked = 0
        is_expanded = False
        failure = ""
        count = self.rollouts
        try:
            for batch in range(2):
                for _ in range(2):
                    asked += count
                    played = await self.service.run_rollouts(
                        written.path,
                        scenario=self.scenario,
                        arm=CALIBRATION_ARM,
                        count=count,
                        extra_instruction_files=(),
                        tags={"role": "calibration"},
                        model=self.model,
                    )
                    rollouts.extend(played)
                    count = sum(rollout.is_infrastructure_failure for rollout in played)
                    if count == 0:
                        break
                rewards = rewards_of(tuple(rollouts))
                if batch == 1 or not rewards or len(rewards) >= self.max_rollouts:
                    break
                if not self.is_near_an_edge(statistics.fmean(rewards)):
                    break
                is_expanded = True
                count = self.max_rollouts - len(rewards)
        except GeneratorError as exc:
            if self.is_refusing:
                raise
            failure = f"the calibration rollouts of {task.name} could not run: {exc}"
        return CalibrationSample(
            rewards=rewards_of(tuple(rollouts)),
            asked=asked,
            unscored=sum(rollout.is_infrastructure_failure for rollout in rollouts),
            expanded=is_expanded,
            errors=tuple(rollout.error for rollout in rollouts if rollout.error),
            failure=failure,
        )

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        sample = await self.sample(task, written)
        outcome = CalibrationOutcome(sample, self.band, self.min_std)
        details = {
            "mode": "refuse" if self.is_refusing else "measure",
            "rewards": list(sample.rewards),
            "mean": outcome.mean,
            "std": outcome.std,
            "rollouts_asked": sample.asked,
            "unscored": sample.unscored,
            "expanded": sample.expanded,
            "place": outcome.place,
            "result": outcome.result,
            "rollout_errors": [error[:ERROR_CHARS] for error in sample.errors[:CALIBRATION_ERRORS]],
        }
        if sample.failure:
            return CheckResult(self.name, could_not_run=sample.failure[:ERROR_CHARS], details=details)
        if outcome.result == CALIBRATION_OUTAGE:
            first_error = sample.errors[0] if sample.errors else "no rollout came back"
            reason = (
                f"{len(sample.rewards)} of {sample.asked} calibration rollouts of {task.name} scored, too few to measure its "
                f"difficulty: {first_error[:ERROR_CHARS]}"
            )
            return CheckResult(self.name, could_not_run=reason, details=details)
        measured = f"over {len(sample.rewards)} calibration rollouts"
        if outcome.result == BAND_BELOW:
            error = f"the solver's mean reward {measured}, {outcome.mean:.2f}, is below band_low {self.band.low}"
        elif outcome.result == BAND_ABOVE:
            error = f"the solver's mean reward {measured}, {outcome.mean:.2f}, is above band_high {self.band.high}"
        elif outcome.result == CALIBRATION_NO_VARIANCE:
            error = (
                f"the population std of the solver's rewards {measured}, {outcome.std:.3f}, is below {self.min_std}"
            )
        else:
            error = ""
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

    Values: ``n`` is at least 1. ``index`` maps each n-gram of a reference instruction to the first reference task
    by name whose instruction holds it. Each reference task is indexed on its own, so no n-gram spans two
    tasks. ``reference_task_count`` is the number of reference tasks, and ``tasks_path`` the directory they were
    read from, as the config gives it.
    Purpose: ``from_config`` builds it from the ``contamination`` keys and puts it in ``task_checks`` right after
    the oracle check. ``validate`` runs it on each task, and the generation report keeps its result.

    Tokens follow :func:`instruction_tokens` and windows follow :func:`instruction_ngrams`.
    The recipe README's Contamination section gives the source of the default window size.
    Any shared n-gram is a defect. The index is built once, so the check never fails to run.
    It makes no model call and runs no sandbox.
    """

    def __init__(self, instructions: Mapping[str, str], *, n: int, tasks_path: str, is_refusing: bool = True) -> None:
        if n < 1:
            raise ValueError("contamination_ngram must be an integer of at least 1")
        super().__init__("contamination", is_refusing=is_refusing)
        self.n = n
        self.tasks_path = tasks_path
        self.reference_task_count = len(instructions)
        self.index: dict[str, str] = {}
        for name in sorted(instructions):
            for ngram in set(instruction_ngrams(instruction_tokens(instructions[name]), n)):
                self.index.setdefault(ngram, name)

    async def run(self, task: HarborTask, written: WrittenTask) -> CheckResult:
        # Each distinct shared n-gram, in the instruction's order, and the first reference task that holds it.
        shared: dict[str, str] = {}
        for ngram in instruction_ngrams(instruction_tokens(task.instruction), self.n):
            if ngram in self.index and ngram not in shared:
                shared[ngram] = self.index[ngram]
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
