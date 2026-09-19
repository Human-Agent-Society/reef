"""What one SPADE generation records: each proposal's fate, each task's measure, and the experience the next prompt carries.

The Designer is asked for one environment at the edge of what the agent can do today (SPADE Sec. 4.1);
the request carries what the agent did on the last generation's environments, sorted by the hint based
regret of Sec. 4.2 into the frontier (the hint turns losses into wins), the mastered (won without it) and
the out of reach (lost even with it), so the next environment lands where the agent fails without a hint
and passes with one. ``experience_text`` is that section of the prompt; the generic task designer in
``reef.record2dataset`` carries it as data. The rest of this module is the bookkeeping of one generation,
which ``SpadeProcessor`` writes under its state directory and reads back after a restart.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from reef.core.tasks.harbor import TASK_NAME_PATTERN
from reef.harness.client.tasks import TaskPlay
from reef.record2dataset.designer import SKILL_PATTERN
from reef.train.cordis_backend.strategies import untrusted_text

MAX_EXPERIENCE_RECORDS = 12
INSTRUCTION_EXCERPT_CHARS = 1200
RECORDED_EXCERPT_CHARS = 600
MASTERED_RETURN = 0.9
TOO_HARD_RETURN = 0.1
REPORT_PREFIX = "generation-"


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


@dataclass(frozen=True)
class ProposalRecord:
    """One Designer call and what became of it."""

    index: int
    skill: str | None
    designer_record_id: str
    task_name: str | None
    refusal: str
    designer_report_id: str = ""

    @property
    def is_written(self) -> bool:
        return self.task_name is not None and not self.refusal


@dataclass(frozen=True)
class TaskMeasure:
    """A written task after both arms played: the rewards, the regret and the band; the plays themselves when held."""

    name: str
    skill: str | None
    task_path: Path
    digest: str
    plain_rewards: tuple[float, ...]
    hint_rewards: tuple[float, ...]
    record: PlayRecord
    plain_plays: tuple[TaskPlay, ...] = ()
    hint_plays: tuple[TaskPlay, ...] = ()

    @property
    def regret(self) -> float:
        return self.record.regret

    @property
    def outcome(self) -> str:
        return self.record.outcome


@dataclass(frozen=True)
class GenerationSummary:
    """One generation as a whole: its tasks measured and refused and their mean regret, for the next generation's reports."""

    generation: int
    mean_regret: float | None
    measured: int
    refused: int

    def document(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRecord:
    """What one generation produced, as its report file keeps it."""

    generation: int
    proposals: tuple[ProposalRecord, ...]
    measures: tuple[TaskMeasure, ...]
    manifest_path: Path | None
    error: str = ""
    # How many held plays were reported once the generation landed; None when every play reported as it ended.
    held_plays_reported: int | None = None

    @property
    def experience(self) -> tuple[PlayRecord, ...]:
        return tuple(measure.record for measure in self.measures)

    @property
    def summary(self) -> GenerationSummary:
        return GenerationSummary(
            generation=self.generation,
            mean_regret=statistics.fmean(measure.regret for measure in self.measures) if self.measures else None,
            measured=len(self.measures),
            refused=sum(1 for proposal in self.proposals if proposal.refusal),
        )


def generation_label(generation: int) -> str:
    """The generation's name in file names, receipts and the round its held plays report under."""
    return f"{REPORT_PREFIX}{generation:05d}"


def skill_tag(skill: str | None) -> dict[str, str]:
    """The skill as a tag or metadata entry, absent when the generation has no skill axis."""
    return {} if skill is None else {"skill": skill}


def is_unplayed(play: TaskPlay) -> bool:
    """An episode that never ran: no reward and an error (the agent could not start, the trial raised)."""
    return play.reward is None and bool(play.error)


def mean_reward(plays: Sequence[TaskPlay]) -> float:
    """The mean reward of the episodes of an arm that ran; a run that scored nothing counts as 0, like a loss."""
    ran = [play for play in plays if not is_unplayed(play)]
    if not ran:
        return 0.0
    return statistics.fmean(play.reward if play.reward is not None else 0.0 for play in ran)


def rewards_of(plays: Sequence[TaskPlay]) -> tuple[float, ...]:
    return tuple(play.reward if play.reward is not None else 0.0 for play in plays if not is_unplayed(play))


def report_path_for(state_dir: Path, generation: int) -> Path:
    return Path(state_dir) / f"{generation_label(generation)}.json"


def write_generation_report(state_dir: Path, record: GenerationRecord) -> Path:
    """Keep every proposal, refusal and measure of one generation, and the experience the next generation's prompts take."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "generation": record.generation,
        "manifest": str(record.manifest_path) if record.manifest_path is not None else None,
        "error": record.error,
        "held_plays_reported": record.held_plays_reported,
        "proposals": [asdict(proposal) for proposal in record.proposals],
        "tasks": [
            {
                "name": measure.name,
                "skill": measure.skill,
                "path": str(measure.task_path),
                "digest": measure.digest,
                "plain_rewards": list(measure.plain_rewards),
                "hint_rewards": list(measure.hint_rewards),
                "held_plays": {"plain": len(measure.plain_plays), "hint": len(measure.hint_plays)},
                "regret": measure.regret,
                "outcome": measure.outcome,
            }
            for measure in record.measures
        ],
        "experience": [asdict(measure.record) for measure in record.measures],
    }
    path = report_path_for(state_dir, record.generation)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def read_generation_report(report_path: Path) -> dict[str, object]:
    """The document a generation report holds."""
    try:
        document = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GenerationError(f"{report_path} is not a generation report: {exc}") from exc
    if not isinstance(document, dict):
        raise GenerationError(f"{report_path} is not a generation report: it holds no object")
    return document


def load_experience(report_path: Path) -> tuple[PlayRecord, ...]:
    """The play records a generation report holds, for the next generation's prompts."""
    records = read_generation_report(report_path).get("experience")
    if not isinstance(records, list):
        raise GenerationError(f"{report_path} holds no experience")
    try:
        return tuple(PlayRecord(**record) for record in records)
    except (TypeError, ValueError) as exc:
        raise GenerationError(f"{report_path} holds a record the Designer cannot take: {exc}") from exc


def load_generation_summary(report_path: Path) -> GenerationSummary:
    """What a generation report says of the generation as a whole, for the reports of the next one."""
    document = read_generation_report(report_path)
    generation, tasks, proposals = document.get("generation"), document.get("tasks"), document.get("proposals")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise GenerationError(f"{report_path} holds no generation number")
    if not isinstance(tasks, list) or not isinstance(proposals, list):
        raise GenerationError(f"{report_path} holds no tasks and proposals")
    try:
        regrets = [float(task["regret"]) for task in tasks]
        refused = sum(1 for proposal in proposals if proposal["refusal"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GenerationError(f"{report_path} holds a task or a proposal without its fields: {exc}") from exc
    return GenerationSummary(
        generation=generation,
        mean_regret=statistics.fmean(regrets) if regrets else None,
        measured=len(regrets),
        refused=refused,
    )


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
