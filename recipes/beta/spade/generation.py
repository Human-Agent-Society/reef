"""What one SPADE generation records: each proposal's fate, each task's measure, and the experience the next prompt carries.

The Designer is asked for one environment at the edge of what the agent can do today (SPADE Sec. 4.1);
the request carries what the agent did on the last generation's environments, sorted by the hint based
regret of Sec. 4.2 into the frontier (the hint turns losses into wins), the mastered (won without it) and
the out of reach (lost even with it), so the next environment lands where the agent fails without a hint
and passes with one. ``experience_text`` is that section of the prompt; the generic task designer in
``reef.record2sample.generator`` carries it as data. ``blend_score`` is the Designer's score of one task, the
reference code's blend of a flat-top difficulty reward (``plateau_reward``) and the floored regret, and
``centered_advantages`` turns a generation's scores into the advantages that train its replies. The rest of this
module is the bookkeeping of one generation, which ``SpadeTaskGenerator`` writes under its state directory and reads
back after a restart.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from reef.core.prompts import untrusted_text
from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.record2sample.generator.designer import SKILL_PATTERN

MAX_EXPERIENCE_RECORDS = 12
INSTRUCTION_EXCERPT_CHARS = 1200
RECORDED_EXCERPT_CHARS = 600
MASTERED_RETURN = 0.9
TOO_HARD_RETURN = 0.1
REPORT_PREFIX = "generation-"
DEFAULT_MAX_DESIGNER_SCALE = 50.0


class GenerationError(RuntimeError):
    """A generation could not run as asked."""


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


def experience_for(
    records: Sequence[PlayRecord], skill: str | None, limit: int = MAX_EXPERIENCE_RECORDS
) -> tuple[PlayRecord, ...]:
    """The records a prompt gets: this skill's first when there is one, the frontier by regret, then the two bands."""
    rank = {"frontier": 0, "out_of_reach": 1, "mastered": 2}
    ordered = sorted(
        records,
        key=lambda record: (skill is not None and record.skill != skill, rank[record.outcome], -record.regret),
    )
    return tuple(ordered[:limit])


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
            "WHAT THE AGENT DID LAST TIME (mean rollout returns in [0, 1] over its attempts; a hint is a few "
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


@dataclass(frozen=True)
class ProposalRecord:
    """One Designer call and what became of it."""

    index: int
    skill: str | None
    designer_record_id: str
    task_name: str | None
    refusal: str
    designer_report_id: str = ""
    split: Literal["train", "eval"] = "train"
    infrastructure_failure: bool = False


@dataclass(frozen=True)
class TaskMeasure:
    """A train task after both arms played: the rewards, the regret, the band, and the Designer's score of the task.

    Values: ``name``, ``skill`` and ``digest`` are the task's; ``plain_rewards`` are the rewards of its training
    rollouts and ``hint_rewards`` those of its hint plays, possibly none; ``record`` is its play record, whose
    returns are the means of the two arms. ``blend`` is a finite non-negative number, the ``blend_score`` that the
    generator gave the task, 0.0 until it scores the task. The constructor refuses another ``blend``.
    Purpose: ``SpadeTaskGenerator.record_results`` makes one for each train task that ran; the generation report and
    ``centered_advantages`` read it.
    """

    name: str
    skill: str | None
    digest: str
    plain_rewards: tuple[float, ...]
    hint_rewards: tuple[float, ...]
    record: PlayRecord
    blend: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.blend, bool)
            or not isinstance(self.blend, (int, float))
            or not math.isfinite(self.blend)
            or self.blend < 0
        ):
            raise ValueError("a task's blend must be a finite non-negative number")

    @property
    def regret(self) -> float:
        return self.record.regret

    @property
    def outcome(self) -> str:
        return self.record.outcome


@dataclass(frozen=True)
class GenerationRecord:
    """What one generation produced, as its report file keeps it.

    Values: ``generation`` is SPADE's number of the generation; ``proposals`` holds each proposal and what became of
    it, and ``measures`` each train task that ran; ``error`` is empty or names what stopped part of the generation.
    ``recipe_generation`` is the recipe's number of the generation, or None for a report that does not know it.
    ``designer_advantages`` maps the inference record ID of each Designer reply that trains to its advantage, a
    finite number (``centered_advantages``); the constructor refuses another value.
    Purpose: ``SpadeTaskGenerator.record_results`` makes it, ``write_generation_report`` keeps it, and
    ``saved_designer_advantages`` reads its advantages back.
    """

    generation: int
    proposals: tuple[ProposalRecord, ...]
    measures: tuple[TaskMeasure, ...]
    error: str = ""
    recipe_generation: int | None = None
    designer_advantages: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(
            not isinstance(record_id, str)
            or not record_id
            or isinstance(advantage, bool)
            or not isinstance(advantage, (int, float))
            or not math.isfinite(advantage)
            for record_id, advantage in self.designer_advantages.items()
        ):
            raise ValueError("designer_advantages must map record IDs to finite numbers")

    @property
    def experience(self) -> tuple[PlayRecord, ...]:
        return tuple(measure.record for measure in self.measures)


def plateau_reward(win_rate: float, low: float, high: float, ramp: float) -> float:
    """The reference code's flat-top difficulty reward of a task whose mean plain reward is ``win_rate``.

    It is 1.0 on ``[low, high]`` and falls linearly to 0 over ``ramp`` on each side, so it is 0 at or below
    ``low - ramp`` and at or above ``high + ramp``.
    """
    if win_rate < low:
        return max(0.0, 1.0 - (low - win_rate) / ramp)
    if win_rate > high:
        return max(0.0, 1.0 - (win_rate - high) / ramp)
    return 1.0


@dataclass(frozen=True)
class BlendSettings:
    """The weights and the plateau band of the Designer's score of a task (``blend_score``).

    Values: every field is a finite number. ``plateau_low`` and ``plateau_high`` are in [0, 1] with
    ``plateau_low <= plateau_high``; ``plateau_ramp`` and ``regret_scale`` are positive; ``plateau_weight`` and
    ``regret_weight`` are non-negative, so every score is non-negative. The constructor refuses other values. The
    defaults score a task by its regret alone, clipped to [0, 1]; the reference code's paper runs use the weights
    0.6 and 0.4, ``regret_scale`` 0.15, and the band 0.2 to 0.4 with ramp 0.2 for Qwen3-4B, 0.4 to 0.6 with ramp
    0.25 for Qwen3-30B-A3B.
    Purpose: ``SpadeTaskGenerator`` reads it from its config and scores each task that ran with ``blend_score``.
    """

    plateau_low: float = 0.4
    plateau_high: float = 0.6
    plateau_ramp: float = 0.25
    plateau_weight: float = 0.0
    regret_weight: float = 1.0
    regret_scale: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.plateau_low,
            self.plateau_high,
            self.plateau_ramp,
            self.plateau_weight,
            self.regret_weight,
            self.regret_scale,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("the blend settings must be finite numbers")
        if not 0 <= self.plateau_low <= self.plateau_high <= 1:
            raise ValueError("the plateau band needs 0 <= plateau_low <= plateau_high <= 1")
        if self.plateau_ramp <= 0 or self.regret_scale <= 0:
            raise ValueError("plateau_ramp and regret_scale must be positive")
        if self.plateau_weight < 0 or self.regret_weight < 0:
            raise ValueError("plateau_weight and regret_weight must be non-negative")


def blend_score(regret: float, win_rate: float, settings: BlendSettings) -> float:
    """The Designer's score of a task: the reference code's blend with the regret floor.

    ``plateau_weight * plateau_reward(win_rate) + regret_weight * min(1, max(0, regret / regret_scale))``, with the
    plateau band of ``settings``. ``win_rate`` is the task's mean plain reward and ``regret`` its mean hint reward
    minus that.
    """
    plateau = plateau_reward(win_rate, settings.plateau_low, settings.plateau_high, settings.plateau_ramp)
    regret_part = min(1.0, max(0.0, regret / settings.regret_scale))
    return settings.plateau_weight * plateau + settings.regret_weight * regret_part


def centered_advantages(
    proposals: Sequence[ProposalRecord], measures: Sequence[TaskMeasure], *, max_scale: float
) -> dict[str, float]:
    """The advantage of each Designer reply of a generation that trains, by the ID of its inference record.

    A reply trains when its proposal became a train task that ran rollouts, so it has a measure: refused, eval and
    failed proposals get no entry. Its advantage is ``scale * (blend - mean)``, where ``mean`` is the mean
    ``blend`` of the replies of its skill (the replies without a skill form one group), as the reference code
    centers per skill without a deviation. ``scale = min(plays / replies, max_scale)``: ``plays`` is the number of
    training rollouts of the measured tasks and ``replies`` the number of measured tasks. This is the reference
    code's scale with one task job for each generation.
    """
    record_ids = {proposal.task_name: proposal.designer_record_id for proposal in proposals if proposal.task_name}
    measured = [(record_ids[measure.name], measure) for measure in measures if measure.name in record_ids]
    if not measured:
        return {}
    plays = sum(len(measure.plain_rewards) for _, measure in measured)
    scale = min(plays / len(measured), max_scale)
    by_skill: dict[str | None, list[float]] = defaultdict(list)
    for _, measure in measured:
        by_skill[measure.skill].append(measure.blend)
    means = {skill: statistics.fmean(blends) for skill, blends in by_skill.items()}
    return {record_id: scale * (measure.blend - means[measure.skill]) for record_id, measure in measured}


def skill_tag(skill: str | None) -> dict[str, str]:
    """The skill as a tag or metadata entry, absent when the generation has no skill axis."""
    return {} if skill is None else {"skill": skill}


def mean_reward(rewards: Sequence[float]) -> float:
    """The mean reward of an arm's rollouts that ran; an arm without any counts as 0, like a loss."""
    return statistics.fmean(rewards) if rewards else 0.0


def report_path_for(state_dir: Path, generation: int) -> Path:
    return Path(state_dir) / f"{REPORT_PREFIX}{generation:05d}.json"


def write_generation_report(state_dir: Path, record: GenerationRecord) -> Path:
    """Keep every proposal, refusal and measure of one generation, and the experience the next generation's prompts take."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "generation": record.generation,
        "recipe_generation": record.recipe_generation,
        "error": record.error,
        "proposals": [asdict(proposal) for proposal in record.proposals],
        "tasks": [
            {
                "name": measure.name,
                "skill": measure.skill,
                "digest": measure.digest,
                "plain_rewards": list(measure.plain_rewards),
                "hint_rewards": list(measure.hint_rewards),
                "regret": measure.regret,
                "outcome": measure.outcome,
                "blend": measure.blend,
            }
            for measure in record.measures
        ],
        "experience": [asdict(measure.record) for measure in record.measures],
        "designer_advantages": dict(record.designer_advantages),
    }
    path = report_path_for(state_dir, record.generation)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def load_experience(report_path: Path) -> tuple[PlayRecord, ...]:
    """The play records a generation report holds, for the next generation's prompts."""
    try:
        document = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GenerationError(f"{report_path} is not a generation report: {exc}") from exc
    records = document.get("experience") if isinstance(document, dict) else None
    if not isinstance(records, list):
        raise GenerationError(f"{report_path} holds no experience")
    try:
        return tuple(PlayRecord(**record) for record in records)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"{report_path} holds a record the Designer cannot take: {exc}") from exc


def recorded_generations(state_dir: Path) -> tuple[int, ...]:
    """The generations whose reports are under the state directory, in order."""
    state_dir = Path(state_dir)
    if not state_dir.is_dir():
        return ()
    numbers = []
    for path in sorted(state_dir.glob(f"{REPORT_PREFIX}*.json")):
        stem = path.name[len(REPORT_PREFIX) : -len(".json")]
        if stem.isdigit():
            numbers.append(int(stem))
    return tuple(numbers)


def saved_designer_advantages(state_dir: Path, recipe_generation: int) -> dict[str, float]:
    """The Designer advantages that the report of the recipe's generation ``recipe_generation`` keeps.

    The result is empty when no report under ``state_dir`` names that generation, such as a report written before
    reports named it. Raises ``GenerationError`` for a report that cannot be read.
    """
    # The report of a generation usually has the recipe's number; another report is a fallback.
    numbers = recorded_generations(state_dir)
    ordered = sorted(numbers, key=lambda number: number != recipe_generation)
    for number in ordered:
        path = report_path_for(state_dir, number)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GenerationError(f"{path} is not a generation report: {exc}") from exc
        if not isinstance(document, dict) or document.get("recipe_generation") != recipe_generation:
            continue
        advantages = document.get("designer_advantages")
        if not isinstance(advantages, dict):
            raise GenerationError(f"{path} holds no Designer advantages")
        return {str(record_id): float(advantage) for record_id, advantage in advantages.items()}
    return {}
